"""在短工作线程中处理输入、识别、标准库和设备，并返回有编号的结构化事件。"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Lock

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..domain import RecognitionResult
from ..enrollment import EnrollmentService
from ..errors import OperationCancelled, TaskFailure, classify_error
from ..face_library import InMemoryFaceLibrary
from ..image_input import ImageInput
from ..integrity import verify_library
from ..library_management import LibraryManagementService
from ..recognition import RecognitionService
from ..repository import FaceRepository


class RecognitionInputStream:
    """在线程安全队列中传递陆续到达的摄像头帧。"""

    def __init__(self) -> None:
        """创建尚未结束的输入流。"""

        self._queue: Queue[ImageInput | object] = Queue()
        # 正常结束和取消使用不同哨兵，消费者无需轮询共享状态即可立即被唤醒。
        self._end_marker = object()
        self._cancel_marker = object()
        self._finished = False
        self._lock = Lock()

    def submit(self, image_input: ImageInput) -> None:
        """提交一张可立即被识别线程消费的独立图片输入。"""

        with self._lock:
            if self._finished:
                raise RuntimeError("recognition input stream is already finished")
            self._queue.put(image_input)

    def finish(self) -> None:
        """结束输入流并唤醒正在等待下一帧的识别线程。"""

        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._queue.put(self._end_marker)

    def cancel(self) -> None:
        """取消输入流并唤醒消费者；重复取消不会产生额外消息。"""

        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._queue.put(self._cancel_marker)

    def __iter__(self) -> Iterator[ImageInput]:
        """按提交顺序阻塞读取图片，直到采集端结束输入。"""

        while True:
            item = self._queue.get()
            if item is self._end_marker:
                return
            if item is self._cancel_marker:
                raise OperationCancelled("recognition was cancelled")
            if not isinstance(item, ImageInput):
                raise TypeError("recognition input stream received an invalid item")
            yield item


@dataclass(frozen=True)
class RecognitionOutcome:
    """同一次任务的判定、实际画面和明确来源，不依赖界面共享帧列表。"""

    result: RecognitionResult
    frame: np.ndarray | None
    source: str


@dataclass(frozen=True)
class LibraryOutcome:
    """区分数据库是否已提交、复查是否通过及文件清理警告。"""

    committed: bool
    message: str
    report: object | None
    snapshot: object | None
    failure: TaskFailure | None = None


class TaskWorker(QThread):
    """提供任务编号、协作取消和统一错误出口的短工作线程。"""

    result_ready = Signal(int, object)
    worker_error = Signal(int, object)

    def __init__(self, task_id, *, parent=None):
        """记录任务身份，取消只在安全处理边界生效。"""
        super().__init__(parent)
        self.task_id = task_id
        self.cancel_event = Event()

    def cancel(self):
        """请求取消，不终止正在执行的模型或数据库提交。"""
        self.cancel_event.set()

    def run(self):
        """统一分发结果和错误，线程结束信号只负责资源退出。"""
        try:
            if self.cancel_event.is_set():
                raise OperationCancelled("操作已取消")
            self.result_ready.emit(self.task_id, self.execute())
        except Exception as error:  # noqa: BLE001 - 线程边界必须把任意故障交给界面
            self.worker_error.emit(self.task_id, classify_error(error, scope=self.scope))

    def execute(self):
        """由具体任务实现处理，必须在返回之前释放自己的连接。"""
        raise NotImplementedError


class CameraDiscoveryWorker(TaskWorker):
    """设备扫描也放入后台，避免驱动探测阻塞窗口。"""

    scope = "device"

    def __init__(self, task_id, camera, parent=None):
        """保存设备服务，不打开视频读取线程。"""
        super().__init__(task_id, parent=parent)
        self.camera = camera

    def execute(self):
        """返回能实际采集画面的设备列表。"""
        return self.camera.discover()


class CameraWorker(QThread):
    """独占摄像头读取和释放，将设备事件与识别事件分开。"""

    frame_ready = Signal(object)
    worker_error = Signal(object)
    worker_status = Signal(str)
    opened = Signal()

    def __init__(self, camera, index, parent=None):
        """保存设备编号；真正的打开和关闭都在该线程执行。"""
        super().__init__(parent)
        self.camera, self.index = camera, index
        self._running = True

    def run(self):
        """采集画面并保证退出时由原线程释放句柄。"""
        try:
            self.camera.open(self.index)
            if not self._running:
                return
            self.opened.emit()
            self.worker_status.emit(f"已打开摄像头 {self.index}。")
            while self._running:
                frame = self.camera.read_frame()
                if not self._running:
                    break
                self.frame_ready.emit(frame)
                self.msleep(15)
        except Exception as error:  # noqa: BLE001 - 各平台驱动错误类型不统一
            if self._running:
                self.worker_error.emit(classify_error(error, scope="device"))
        finally:
            self.camera.close()

    def stop(self):
        """仅发出停止请求，读取未结束时不从其他线程释放设备。"""
        self._running = False


class RecognitionWorker(TaskWorker):
    """后台解码本地图片并执行单帧识别。"""

    scope = "system"

    def __init__(
        self,
        task_id,
        *,
        settings,
        face_engine,
        library_snapshot,
        image_inputs=(),
        paths=(),
        parent=None,
    ):
        """保留不可变标准库快照，路径在后台解码。"""
        super().__init__(task_id, parent=parent)
        self.settings, self.engine, self.snapshot = settings, face_engine, library_snapshot
        self.inputs, self.paths = tuple(image_inputs), tuple(paths)

    def execute(self):
        """在自己的数据库连接中执行识别并返回实际参与判定的画面。"""
        inputs = self.inputs or tuple(ImageInput.from_file(path) for path in self.paths)
        repository = FaceRepository(self.settings.database_path)
        try:
            service = RecognitionService(
                repository,
                self.settings,
                self.engine,
                self.snapshot,
                cancelled=self.cancel_event.is_set,
            )
            result = service.compare_input(inputs[0])
            if self.cancel_event.is_set():
                raise OperationCancelled("识别已取消")
            return RecognitionOutcome(result, inputs[0].frame, inputs[0].source_type)
        finally:
            repository.close()


class CameraRecognitionWorker(RecognitionWorker):
    """边接收五帧边检测，最后只检索一次标准库。"""

    def __init__(
        self,
        task_id,
        *,
        settings,
        face_engine,
        library_snapshot,
        frame_count,
        started_at,
        parent=None,
    ):
        """输入队列和实际画面归同一任务所有，避免迟到事件串到下一轮。"""
        super().__init__(
            task_id,
            settings=settings,
            face_engine=face_engine,
            library_snapshot=library_snapshot,
            parent=parent,
        )
        self.frame_count, self.started_at = frame_count, started_at
        self.stream = RecognitionInputStream()
        self.frames = []

    def submit(self, image_input):
        """先记录本任务画面，再提交给阻塞消费者。"""
        self.frames.append(image_input)
        self.stream.submit(image_input)

    def finish_inputs(self):
        """结束正常输入，不与取消混淆。"""
        self.stream.finish()

    def cancel(self):
        """取消尚未处理完的识别并唤醒队列。"""
        super().cancel()
        self.stream.cancel()

    def execute(self):
        """返回选中帧与同一判定；全部帧无效时仍返回实际输入来源。"""
        repository = FaceRepository(self.settings.database_path)
        try:
            service = RecognitionService(
                repository,
                self.settings,
                self.engine,
                self.snapshot,
                cancelled=self.cancel_event.is_set,
            )
            result = service.compare_inputs(
                self.stream, frame_count=self.frame_count, started_at=self.started_at
            )
            if self.cancel_event.is_set():
                raise OperationCancelled("识别已取消")
            index = result.selected_frame_index or 0
            frame = self.frames[index].frame if self.frames else None
            return RecognitionOutcome(result, frame, "camera")
        finally:
            repository.close()


class LibraryWorker(TaskWorker):
    """后台执行录入、管理和复查，返回同源矩阵与展示元信息。"""

    scope = "library"

    def __init__(
        self,
        task_id,
        *,
        settings,
        face_engine,
        operation="check",
        paths=(),
        inputs=(),
        person_id=None,
        sample_id=None,
        name=None,
        parent=None,
    ):
        """保存命令参数，构造时不读取数据库或文件。"""
        super().__init__(task_id, parent=parent)
        self.settings, self.engine, self.operation = settings, face_engine, operation
        self.paths, self.inputs = tuple(paths), tuple(inputs)
        self.person_id, self.sample_id, self.name = person_id, sample_id, name

    def execute(self):
        """提交后复查失败也保留已提交事实，不能报告成操作回滚。"""
        repository = FaceRepository(self.settings.database_path)
        committed, message = False, "标准库检查已完成。"
        try:
            management = LibraryManagementService(repository, self.settings)
            if self.operation in {"create", "append"}:
                inputs = list(self.inputs) or [ImageInput.from_file(path) for path in self.paths]
                service = EnrollmentService(
                    repository=repository,
                    settings=self.settings,
                    face_engine=self.engine,
                    image_saver=_save_bgr_image,
                    cancelled=self.cancel_event.is_set,
                )
                if self.operation == "create":
                    person = service.create_from_inputs(self.name, inputs)
                    message = f"“{person.display_name}”已录入标准库。"
                else:
                    count = service.append_from_inputs(self.person_id, inputs)
                    message = f"已追加 {count} 张有效单人脸图片。"
                committed = True
            elif self.operation == "rename":
                person = management.rename_person(self.person_id, self.name)
                committed, message = True, f"人员姓名已修改为“{person.display_name}”。"
            elif self.operation in {"delete_person", "delete_sample"}:
                result = (
                    management.delete_person(self.person_id)
                    if self.operation == "delete_person"
                    else management.delete_sample(self.person_id, self.sample_id)
                )
                committed, message = (
                    True,
                    "人员已删除。" if self.operation == "delete_person" else "样本已删除。",
                )
                if result.cleanup_failures:
                    message += (
                        f"有 {len(result.cleanup_failures)} 个图片文件未能清理，详见复查警告。"
                    )
            elif self.operation != "check":
                raise ValueError("无效的标准库操作")
            try:
                report = verify_library(repository, self.settings)
                snapshot = (
                    InMemoryFaceLibrary.from_samples(report.people, report.samples).snapshot()
                    if report.is_valid
                    else None
                )
                return LibraryOutcome(committed, message, report, snapshot)
            except Exception as error:
                if not committed:
                    raise
                return LibraryOutcome(
                    True,
                    "修改已保存，但标准库复查失败，暂时不能识别。",
                    None,
                    None,
                    classify_error(error, scope="library"),
                )
        finally:
            repository.close()


def _save_bgr_image(path: Path, frame: np.ndarray):
    """保存本次录入图片，失败由服务负责回滚与补偿。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"图片保存失败：{path}")
