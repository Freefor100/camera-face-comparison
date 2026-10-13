"""集中管理任务、标准库健康和设备生命周期，不直接更新控件。"""

from dataclasses import dataclass, replace
from time import perf_counter

from PySide6.QtCore import QObject, QTimer, Signal

from ..face_library import InMemoryFaceLibrary
from ..image_input import ImageInput
from .workers import (
    CameraDiscoveryWorker,
    CameraRecognitionWorker,
    CameraWorker,
    LibraryWorker,
    RecognitionWorker,
)

CAMERA_FRAME_COUNT = 5
CAMERA_FRAME_INTERVAL_MS = 80


@dataclass(frozen=True)
class ApplicationState:
    """按钮状态的唯一依据，设备事件不能冒充业务任务状态。"""

    camera: str = "stopped"
    camera_ready: bool = False
    library: str = "unchecked"
    task: str = "idle"
    closing: bool = False
    scanning: bool = False


class TaskController(QObject):
    """所有请求经过统一入口，迟到通知按任务编号丢弃。"""

    state_changed = Signal(object)
    frame_ready = Signal(object)
    devices_ready = Signal(object)
    result_ready = Signal(object)
    result_reset = Signal()
    recognition_status = Signal(str)
    recognition_error = Signal(object)
    recognition_cancelled = Signal(str)
    library_status = Signal(str)
    device_status = Signal(str)
    catalog_ready = Signal(object)
    backend_changed = Signal(object)
    ready_to_close = Signal()

    def __init__(self, settings, face_engine, camera, parent=None):
        """构造时不打开数据库，连接页面后再异步检查。"""
        super().__init__(parent)
        self.settings, self.engine, self.camera = settings, face_engine, camera
        self.state = ApplicationState()
        self.face_library = InMemoryFaceLibrary.empty()
        self.catalog = self.current_frame = self.capture_timer = None
        self.worker = self.camera_worker = self.scan_worker = None
        self._sequence = self._task_id = self._captured = 0
        self._cancelled = False
        self._device_failed = False
        self.request_started_at = None

    @property
    def idle(self):
        """业务线程完全退出后才返回空闲。"""
        return self.state.task == "idle" and not self.state.scanning

    @property
    def library_valid(self):
        """检查通过并发布矩阵后，标准库才健康。"""
        return self.state.library == "healthy"

    @property
    def can_close(self):
        """所有线程实际退出后才允许销毁窗口。"""
        return self.worker is None and self.camera_worker is None and self.scan_worker is None

    def _set_state(self, **changes):
        """发布完整新状态，由页面统一计算按钮可用性。"""
        self.state = replace(self.state, **changes)
        self.state_changed.emit(self.state)

    def _admit(self, kind, require_library=True):
        """在业务入口校验状态，不依赖按钮是否恰好被禁用。"""
        if (
            self.state.closing
            or self.state.task != "idle"
            or (require_library and not self.library_valid)
        ):
            return False
        self._sequence += 1
        self._task_id = self._sequence
        self._cancelled = False
        self._set_state(task=kind)
        return True

    def _start(self, worker):
        """连接带编号的终态和退出事件，退出不覆盖显示结果。"""
        self.worker = worker
        worker.result_ready.connect(self._on_result)
        worker.worker_error.connect(self._on_error)
        worker.finished.connect(lambda: self._on_finished(worker.task_id, worker))
        worker.start()

    def recheck_library(self):
        """用新连接复查，失败分支不再读取同一份坏数据。"""
        if self._admit("check", False):
            self._set_state(library="checking")
            self.library_status.emit("正在检查标准库，请稍候。")
            self._start(
                LibraryWorker(
                    self._task_id, settings=self.settings, face_engine=self.engine, parent=self
                )
            )

    def library_operation(self, operation, **parameters):
        """录入、改名和删除共用入口，图片在后台解码。"""
        if self._admit(operation):
            self.library_status.emit("正在处理标准库，请稍候。")
            self._start(
                LibraryWorker(
                    self._task_id,
                    settings=self.settings,
                    face_engine=self.engine,
                    operation=operation,
                    parent=self,
                    **parameters,
                )
            )

    def recognize_file(self, path):
        """本地图片只进行单帧识别，计时从业务请求开始。"""
        if self._admit("recognition"):
            self.request_started_at = perf_counter()
            self.result_reset.emit()
            self.recognition_status.emit("正在识别本地图片。")
            self._start(
                RecognitionWorker(
                    self._task_id,
                    settings=self.settings,
                    face_engine=self.engine,
                    library_snapshot=self.face_library.snapshot(),
                    paths=(path,),
                    parent=self,
                )
            )

    def recognize_camera(self):
        """立即检测第一帧，并按原来的 80 毫秒间隔补齐五帧。"""
        if self.state.camera != "running" or self.current_frame is None:
            self.device_status.emit("尚未获得可用的摄像头画面。")
            return
        if not self._admit("camera_recognition"):
            return
        self.request_started_at = perf_counter()
        self.result_reset.emit()
        self._captured = 1
        worker = CameraRecognitionWorker(
            self._task_id,
            settings=self.settings,
            face_engine=self.engine,
            library_snapshot=self.face_library.snapshot(),
            frame_count=5,
            started_at=self.request_started_at,
            parent=self,
        )
        self._start(worker)
        worker.submit(ImageInput.from_camera(self.current_frame))
        self.recognition_status.emit("正在采集 1/5 帧。")
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(CAMERA_FRAME_INTERVAL_MS)
        self.capture_timer.timeout.connect(self._collect_frame)
        self.capture_timer.start()

    def _collect_frame(self):
        """仅向当前摄像头任务提交画面，失败后不会继续采集。"""
        if not isinstance(self.worker, CameraRecognitionWorker) or self._cancelled:
            self._stop_collection()
            return
        if self.current_frame is not None:
            self.worker.submit(ImageInput.from_camera(self.current_frame))
            self._captured += 1
            self.recognition_status.emit(f"正在采集 {self._captured}/5 帧。")
            if self._captured == CAMERA_FRAME_COUNT:
                self.worker.finish_inputs()
                self._stop_collection()

    def _stop_collection(self):
        """释放采集定时器，不擅自解除线程占用。"""
        if self.capture_timer is not None:
            self.capture_timer.stop()
            self.capture_timer.deleteLater()
            self.capture_timer = None

    def _on_result(self, task_id, outcome):
        """当前任务才能发布结果，提交事实与复查结果分别处理。"""
        if task_id != self._task_id or self.worker is None or self._cancelled:
            return
        if self.state.task in {"recognition", "camera_recognition"}:
            if outcome.result.storage_error_kind == "storage_unavailable":
                self._invalidate("unavailable")
                self.library_status.emit("日志未保存，数据库无法使用，请重新检查。")
            self.result_ready.emit(outcome)
            self.backend_changed.emit(getattr(self.engine, "backend", None))
            return
        # 复查也可能发布外部修改后的库，旧判定不能冒充新标准库的结果。
        if outcome.committed or outcome.snapshot is not None:
            self.result_reset.emit()
        if outcome.report is not None:
            self.catalog = outcome.report
            self.catalog_ready.emit(outcome.report)
        if outcome.snapshot is not None:
            self.face_library.publish(outcome.snapshot)
            self._set_state(library="healthy")
            message = outcome.message
            if outcome.report.warnings:
                message += f" 有 {len(outcome.report.warnings)} 项标准库提示，详见诊断详情。"
                message += "\n" + "\n".join(outcome.report.warnings)
        else:
            self._invalidate("unavailable" if outcome.failure else "invalid")
            message = (
                "修改已保存，但标准库复查失败，暂时不能识别。"
                if outcome.committed
                else "标准库检查未通过，暂时不能识别或修改。"
            )
            detail = (
                outcome.failure.detail
                if outcome.failure
                else "；".join(f"{item.kind}：{item.detail}" for item in outcome.report.failures)
            )
            message += "\n" + detail
        self.library_status.emit(message)

    def _invalidate(self, state):
        """撤销健康标记并发布空矩阵，旧快照不会原地改变。"""
        self.face_library.clear()
        self.result_reset.emit()
        self._set_state(library=state)

    def _on_error(self, task_id, failure):
        """失败立即停止采集，线程退出之后才恢复按钮。"""
        if task_id != self._task_id or self.worker is None:
            return
        self._stop_collection()
        if failure.kind == "storage_unavailable" or self.state.task == "check":
            self._invalidate("unavailable")
        if failure.kind == "storage_unavailable":
            # 取消只取消本次识别显示，不能掩盖当前线程发现的标准库故障。
            self.library_status.emit(failure.message + "\n" + failure.detail)
        if not self._cancelled and failure.kind != "cancelled":
            if self.state.task in {"recognition", "camera_recognition"}:
                self.recognition_error.emit(failure)
            elif failure.kind != "storage_unavailable":
                self.library_status.emit(failure.message + "\n" + failure.detail)

    def _on_finished(self, task_id, worker):
        """退出只释放资源，不覆盖成功、失败或取消文字。"""
        if task_id == self._task_id and self.worker is worker:
            if self._cancelled and self.state.task in {"recognition", "camera_recognition"}:
                self.recognition_cancelled.emit("识别已取消。")
            self._stop_collection()
            self.worker = None
            worker.deleteLater()
            self._set_state(task="idle")
            self.backend_changed.emit(getattr(self.engine, "backend", None))
            self._check_close_ready()

    def refresh_devices(self):
        """停止状态下异步扫描，不与读取线程争用设备。"""
        if self.state.closing or self.camera_worker is not None or self.scan_worker is not None:
            return
        self._sequence += 1
        worker = CameraDiscoveryWorker(self._sequence, self.camera, self)
        self.scan_worker = worker
        self._set_state(scanning=True)
        worker.result_ready.connect(lambda _, devices: self._devices_result(worker, devices))
        worker.worker_error.connect(
            lambda _, error: self.device_status.emit(error.message + " " + error.detail)
        )
        worker.finished.connect(lambda: self._scan_finished(worker))
        worker.start()

    def _devices_result(self, worker, devices):
        """仅当前扫描可以更新设备列表。"""
        if self.scan_worker is worker and not self.state.closing:
            self.devices_ready.emit(devices)
            self.device_status.emit("请选择摄像头并启动预览。" if devices else "未发现可采集设备。")

    def _scan_finished(self, worker):
        """扫描实际退出后，才恢复设备按钮。"""
        if self.scan_worker is worker:
            self.scan_worker = None
            self._set_state(scanning=False)
            worker.deleteLater()
            self._check_close_ready()

    def start_camera(self, index):
        """设备线程独占句柄，预览不依赖标准库健康状态。"""
        if self.state.closing or self.camera_worker is not None or self.scan_worker is not None:
            return
        self._device_failed = False
        worker = CameraWorker(self.camera, index, self)
        self.camera_worker = worker
        self._set_state(camera="starting")
        worker.opened.connect(lambda: self._camera_opened(worker))
        worker.frame_ready.connect(lambda frame: self._camera_frame(worker, frame))
        worker.worker_status.connect(lambda message: self._camera_status(worker, message))
        worker.worker_error.connect(lambda error: self._camera_error(worker, error))
        worker.finished.connect(lambda: self._camera_finished(worker))
        worker.start()

    def _camera_opened(self, worker):
        """打开事件只改变设备状态，不重新启用忙碌中的识别按钮。"""
        if self.camera_worker is worker and self.state.camera == "starting":
            self._set_state(camera="running")

    def _camera_status(self, worker, message):
        """已停止或关闭中的设备通知不能覆盖当前设备说明。"""
        if (
            self.camera_worker is worker
            and self.state.camera == "running"
            and not self.state.closing
        ):
            self.device_status.emit(message)

    def _camera_frame(self, worker, frame):
        """停止后丢弃迟到画面，避免最后一帧重新填回预览。"""
        if self.camera_worker is worker and self.state.camera == "running":
            self.current_frame = frame.copy()
            if not self.state.camera_ready:
                self._set_state(camera_ready=True)
            self.frame_ready.emit(self.current_frame)

    def _camera_error(self, worker, error):
        """摄像头错误只显示在设备区，停止不覆盖错误原因。"""
        if self.camera_worker is worker:
            self._device_failed = True
            self.device_status.emit(error.message + " " + error.detail)
            self.stop_camera()

    def stop_camera(self):
        """立即清空预览，等原读取线程自己释放设备。"""
        self.current_frame = None
        self.frame_ready.emit(None)
        if self.state.task == "camera_recognition" and self.capture_timer is not None:
            self._cancelled = True
            self._stop_collection()
            self.worker.cancel()
            self.recognition_cancelled.emit("识别已取消，正在等待任务安全退出。")
        if self.camera_worker is None:
            self._set_state(camera="stopped", camera_ready=False)
        else:
            self._set_state(camera="stopping", camera_ready=False)
            self.camera_worker.stop()

    def _camera_finished(self, worker):
        """真正退出后才能重新打开，不在主线程阻塞等待。"""
        if self.camera_worker is worker:
            self.camera_worker = None
            self._set_state(camera="stopped", camera_ready=False)
            if not self._device_failed and not self.state.closing:
                self.device_status.emit("预览已停止。")
            worker.deleteLater()
            self._check_close_ready()

    def request_close(self):
        """协作取消并等待，绝不强制销毁仍运行的线程。"""
        self._set_state(closing=True)
        self.stop_camera()
        if self.worker is not None:
            self._cancelled = True
            self._stop_collection()
            self.worker.cancel()
        if self.scan_worker is not None:
            self.scan_worker.cancel()
        self.device_status.emit("正在等待后台任务与设备安全退出。")
        self._check_close_ready()

    def _check_close_ready(self):
        """所有线程实际退出后，才通知窗口再次关闭。"""
        if self.state.closing and self.can_close:
            self.ready_to_close.emit()
