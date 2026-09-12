from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..camera import CameraService
from ..config import Settings
from ..enrollment import EnrollmentService
from ..face_engine import FaceEngine
from ..face_library import FaceLibrarySnapshot
from ..image_input import ImageInput
from ..recognition import RecognitionService
from ..repository import FaceRepository


class CameraWorker(QThread):
    """在后台线程中持续读取摄像头并发送画面与状态信号。"""

    frame_ready = Signal(object)
    worker_error = Signal(str)
    worker_status = Signal(str)

    def __init__(self, camera: CameraService, index: int) -> None:
        """创建摄像头读取线程。"""
        super().__init__()
        self._camera = camera
        self._index = index
        self._running = True

    def run(self) -> None:
        """打开设备、循环发送帧，并在线程结束时释放设备。"""
        try:
            self._camera.open(self._index)
            self.worker_status.emit(f"已打开摄像头 {self._index}")
            while self._running:
                self.frame_ready.emit(self._camera.read_frame())
                self.msleep(15)
        except Exception as error:  # noqa: BLE001 - 工作线程异常必须展示给用户
            self.worker_error.emit(str(error))
        finally:
            self._camera.close()
            self.worker_status.emit("摄像头已停止")

    def stop(self) -> None:
        """请求读取循环停止；真正释放设备由 `run()` 的 finally 完成。"""
        self._running = False


class RecognitionWorker(QThread):
    """在后台线程中执行一次人脸检测和基于内存快照的 1:N 比对。"""

    result_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(
        self,
        *,
        database_path: Path,
        settings: Settings,
        face_engine: FaceEngine,
        image_inputs: tuple[ImageInput, ...],
        library_snapshot: FaceLibrarySnapshot,
    ) -> None:
        """创建一次单帧或多帧识别任务。"""
        super().__init__()
        self._database_path = database_path
        self._settings = settings
        self._face_engine = face_engine
        self._image_inputs = image_inputs
        self._library_snapshot = library_snapshot

    def run(self) -> None:
        """在线程中打开独立仓库执行识别，并保证结束时关闭连接。"""
        repository: FaceRepository | None = None
        try:
            repository = FaceRepository(self._database_path)
            service = RecognitionService(
                repository,
                self._settings,
                self._face_engine,
                self._library_snapshot,
            )
            result = (
                service.compare_input(self._image_inputs[0])
                if len(self._image_inputs) == 1
                else service.compare_inputs(self._image_inputs)
            )
            self.result_ready.emit(result)
        except Exception as error:  # noqa: BLE001 - 工作线程异常必须展示给用户
            self.worker_error.emit(str(error))
        finally:
            if repository is not None:
                repository.close()


class EnrollmentWorker(QThread):
    """在独立短线程中完成图片特征提取和标准库写入。"""

    result_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(
        self,
        *,
        database_path: Path,
        settings: Settings,
        face_engine: FaceEngine,
        operation: str,
        image_inputs: tuple[ImageInput, ...],
        display_name: str | None = None,
        person_id: str | None = None,
    ) -> None:
        """创建一次新增或追加样本任务。"""
        super().__init__()
        self._database_path = database_path
        self._settings = settings
        self._face_engine = face_engine
        self._operation = operation
        self._image_inputs = image_inputs
        self._display_name = display_name
        self._person_id = person_id

    def run(self) -> None:
        """在线程中执行录入服务，并在结束时关闭独立数据库连接。"""
        repository: FaceRepository | None = None
        try:
            repository = FaceRepository(self._database_path)
            service = EnrollmentService(
                repository=repository,
                settings=self._settings,
                face_engine=self._face_engine,
                image_saver=_save_bgr_image,
            )
            if self._operation == "create":
                if self._display_name is None:
                    raise ValueError("display name is required for person creation")
                person = service.create_from_inputs(self._display_name, self._image_inputs)
                self.result_ready.emit((self._operation, person))
            elif self._operation == "append":
                if self._person_id is None:
                    raise ValueError("person id is required for sample append")
                count = service.append_from_inputs(self._person_id, self._image_inputs)
                self.result_ready.emit((self._operation, self._person_id, count))
            else:
                raise ValueError(f"unknown enrollment operation: {self._operation}")
        except Exception as error:  # noqa: BLE001 - 录入异常必须反馈给界面
            self.worker_error.emit(str(error))
        finally:
            if repository is not None:
                repository.close()


def _save_bgr_image(path: Path, frame: np.ndarray) -> None:
    """使用 OpenCV 将 BGR 图像保存到指定路径。"""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not save image to {path}")
