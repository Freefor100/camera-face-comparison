from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..camera import CameraService
from ..config import Settings
from ..domain import RecognitionResult
from ..enrollment import REQUIRED_POSES, EnrollmentService
from ..face_engine import FaceEngine, FaceInputError
from ..recognition import RecognitionService
from ..repository import FaceRepository

POSE_LABELS = {
    "front": "正脸",
    "left": "向左转脸",
    "right": "向右转脸",
    "up": "轻度抬头",
    "down": "轻度低头",
}


class CameraWorker(QThread):
    frame_ready = Signal(object)
    worker_error = Signal(str)
    worker_status = Signal(str)

    def __init__(self, camera: CameraService, index: int) -> None:
        super().__init__()
        self._camera = camera
        self._index = index
        self._running = True

    def run(self) -> None:
        try:
            self._camera.open(self._index)
            self.worker_status.emit(f"已打开摄像头 {self._index}")
            while self._running:
                self.frame_ready.emit(self._camera.read_frame())
                self.msleep(15)
        except Exception as error:  # noqa: BLE001 - worker errors must be shown to the user
            self.worker_error.emit(str(error))
        finally:
            self._camera.close()
            self.worker_status.emit("摄像头已停止")

    def stop(self) -> None:
        self._running = False


class RecognitionWorker(QThread):
    result_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(
        self,
        *,
        database_path: Path,
        settings: Settings,
        face_engine: FaceEngine,
        frame: np.ndarray,
    ) -> None:
        super().__init__()
        self._database_path = database_path
        self._settings = settings
        self._face_engine = face_engine
        self._frame = frame

    def run(self) -> None:
        repository: FaceRepository | None = None
        try:
            repository = FaceRepository(self._database_path)
            result = RecognitionService(repository, self._settings, self._face_engine).compare(
                self._frame
            )
            self.result_ready.emit(result)
        except Exception as error:  # noqa: BLE001 - worker errors must be shown to the user
            self.worker_error.emit(str(error))
        finally:
            if repository is not None:
                repository.close()


class EnrollmentDialog(QDialog):
    """Five-step enrollment dialog that captures from the main live-preview frame."""

    def __init__(
        self,
        *,
        repository: FaceRepository,
        settings: Settings,
        face_engine: FaceEngine,
        frame_supplier: Callable[[], np.ndarray | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新增人员并采集五种姿态")
        self._frame_supplier = frame_supplier
        self._service = EnrollmentService(
            repository=repository,
            settings=settings,
            face_engine=face_engine,
            image_saver=_save_bgr_image,
        )
        self._session = None
        self._pose_index = 0

        self.name_input = QLineEdit()
        self.pose_label = QLabel()
        self.status_label = QLabel("请先输入姓名并保持画面中只有一个清晰人脸。")
        self.capture_button = QPushButton("抓拍当前姿态")
        self.capture_button.clicked.connect(self.capture_pose)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("人员姓名：", self.name_input)
        form.addRow("当前步骤：", self.pose_label)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(buttons)
        self._refresh_pose_label()

    def capture_pose(self) -> None:
        frame = self._frame_supplier()
        if frame is None:
            self.status_label.setText("请先启动摄像头预览。")
            return
        if self._session is None:
            try:
                self._session = self._service.begin(self.name_input.text())
            except ValueError as error:
                self.status_label.setText(str(error))
                return
            self.name_input.setEnabled(False)
        pose = REQUIRED_POSES[self._pose_index]
        try:
            self._session.capture(pose, frame)
        except (FaceInputError, ValueError) as error:
            self.status_label.setText(f"采集失败：{error}")
            return

        self._pose_index += 1
        if self._pose_index == len(REQUIRED_POSES):
            try:
                person = self._session.commit()
            except Exception as error:  # noqa: BLE001 - storage errors must stay inside the dialog
                self.status_label.setText(f"保存失败：{error}")
                return
            self.status_label.setText(f"已录入 {person.display_name} 的五张样本。")
            self.capture_button.setEnabled(False)
            self.accept()
            return
        self.status_label.setText("采集成功，请完成下一种姿态。")
        self._refresh_pose_label()

    def _refresh_pose_label(self) -> None:
        if self._pose_index < len(REQUIRED_POSES):
            pose = REQUIRED_POSES[self._pose_index]
            self.pose_label.setText(f"{self._pose_index + 1}/5：{POSE_LABELS[pose]}")


class MainWindow(QMainWindow):
    """The desktop application shell for preview, recognition, and library enrollment."""

    def __init__(
        self,
        *,
        settings: Settings,
        face_engine: FaceEngine,
        camera: CameraService,
    ) -> None:
        super().__init__()
        self.setWindowTitle("摄像头人脸比对系统")
        self.resize(1000, 700)
        self._settings = settings
        self._face_engine = face_engine
        self._camera = camera
        self._repository = FaceRepository(settings.database_path)
        self._camera_worker: CameraWorker | None = None
        self._recognition_worker: RecognitionWorker | None = None
        self._current_frame: np.ndarray | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None

        tabs = QTabWidget()
        tabs.addTab(self._build_recognition_tab(), "实时比对")
        tabs.addTab(self._build_library_tab(), "标准人脸库")
        self.setCentralWidget(tabs)
        self.refresh_cameras()
        self.refresh_people()

    def _build_recognition_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.start_button = QPushButton("启动预览")
        self.stop_button = QPushButton("停止预览")
        self.compare_button = QPushButton("抓拍并比对")
        self.stop_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.compare_button.clicked.connect(self.compare_current_frame)
        controls.addWidget(QLabel("摄像头："))
        controls.addWidget(self.camera_combo, 1)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.compare_button)

        self.preview_label = QLabel("尚未启动摄像头")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background: #171717; color: #eeeeee;")
        self.result_label = QLabel("识别结果：等待抓拍")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.status_label = QLabel("状态：未启动")
        layout.addLayout(controls)
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.result_label)
        layout.addWidget(self.status_label)
        return page

    def _build_library_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.people_list = QListWidget()
        self.add_person_button = QPushButton("新增人员（五步采样）")
        self.append_sample_button = QPushButton("为选中人员追加样本")
        self.add_person_button.clicked.connect(self.open_enrollment)
        self.append_sample_button.clicked.connect(self.append_sample_to_selected_person)
        layout.addWidget(QLabel("人员与已入库样本数量"))
        layout.addWidget(self.people_list, 1)
        layout.addWidget(self.add_person_button)
        layout.addWidget(self.append_sample_button)
        return page

    def refresh_cameras(self) -> None:
        self.camera_combo.clear()
        try:
            devices = self._camera.discover()
        except Exception as error:  # noqa: BLE001 - OpenCV backend errors vary by platform
            self.status_label.setText(f"设备扫描失败：{error}")
            return
        for device in devices:
            self.camera_combo.addItem(device.label, device.index)
        if not devices:
            self.status_label.setText("未发现可打开的摄像头，请检查连接和系统权限。")

    def start_camera(self) -> None:
        if self.camera_combo.currentIndex() < 0:
            self.status_label.setText("请先刷新并选择摄像头。")
            return
        self.stop_camera()
        self._camera_worker = CameraWorker(self._camera, int(self.camera_combo.currentData()))
        self._camera_worker.frame_ready.connect(self.on_frame)
        self._camera_worker.worker_error.connect(self.on_camera_error)
        self._camera_worker.worker_status.connect(self.status_label.setText)
        self._camera_worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.compare_button.setEnabled(True)

    def stop_camera(self) -> None:
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker.wait(2000)
            self._camera_worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.compare_button.setEnabled(False)

    def on_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame.copy()
        self._render_frame(self._current_frame, self._last_bbox)

    def compare_current_frame(self) -> None:
        if self._current_frame is None:
            self.status_label.setText("尚未获得摄像头画面。")
            return
        if self._recognition_worker is not None and self._recognition_worker.isRunning():
            return
        self.compare_button.setEnabled(False)
        self.status_label.setText("正在进行人脸检测与比对…")
        self._recognition_worker = RecognitionWorker(
            database_path=self._settings.database_path,
            settings=self._settings,
            face_engine=self._face_engine,
            frame=self._current_frame.copy(),
        )
        self._recognition_worker.result_ready.connect(self.on_recognition_result)
        self._recognition_worker.worker_error.connect(self.on_recognition_error)
        self._recognition_worker.finished.connect(self.on_recognition_finished)
        self._recognition_worker.start()

    def on_recognition_result(self, result: RecognitionResult) -> None:
        self._last_bbox = result.bbox
        if result.status == "matched":
            self.result_label.setText(
                f"识别成功：{result.display_name}｜相似度 {result.top_score:.3f}"
            )
        elif result.status == "unknown":
            self.result_label.setText(
                f"未知人员｜最高相似度 {result.top_score or 0.0:.3f}｜原因：{result.reason}"
            )
        else:
            self.result_label.setText(f"无法识别当前画面：{result.reason}")
        self.status_label.setText(f"本次处理耗时：{result.latency_ms:.0f} ms")
        if self._current_frame is not None:
            self._render_frame(self._current_frame, self._last_bbox)

    def on_recognition_error(self, message: str) -> None:
        self.result_label.setText(f"比对失败：{message}")
        self.status_label.setText("比对任务异常结束。")

    def on_recognition_finished(self) -> None:
        if self._camera_worker is not None:
            self.compare_button.setEnabled(True)

    def on_camera_error(self, message: str) -> None:
        self.status_label.setText(f"摄像头错误：{message}")
        self.stop_camera()

    def open_enrollment(self) -> None:
        if self._current_frame is None:
            QMessageBox.information(self, "需要摄像头画面", "请先在实时比对页启动摄像头预览。")
            return
        dialog = EnrollmentDialog(
            repository=self._repository,
            settings=self._settings,
            face_engine=self._face_engine,
            frame_supplier=lambda: self._current_frame.copy()
            if self._current_frame is not None
            else None,
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            self.refresh_people()

    def append_sample_to_selected_person(self) -> None:
        if self._current_frame is None:
            QMessageBox.information(self, "需要摄像头画面", "请先在实时比对页启动摄像头预览。")
            return
        item = self.people_list.currentItem()
        if item is None:
            QMessageBox.information(self, "请选择人员", "请先在标准人脸库中选择一名已录入人员。")
            return
        pose_labels = [POSE_LABELS[pose] for pose in REQUIRED_POSES]
        pose_label, accepted = QInputDialog.getItem(
            self,
            "追加样本",
            "本次姿态：",
            pose_labels,
            0,
            False,
        )
        if not accepted:
            return
        pose = REQUIRED_POSES[pose_labels.index(pose_label)]
        try:
            EnrollmentService(
                repository=self._repository,
                settings=self._settings,
                face_engine=self._face_engine,
                image_saver=_save_bgr_image,
            ).append_sample(
                str(item.data(Qt.UserRole)),
                pose,
                self._current_frame.copy(),
            )
        except (FaceInputError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "追加样本失败", str(error))
            return
        self.refresh_people()
        self.status_label.setText("已为选中人员追加一张合格样本。")

    def refresh_people(self) -> None:
        people = self._repository.list_people()
        counts: dict[str, int] = {}
        for sample in self._repository.list_samples():
            counts[sample.person_id] = counts.get(sample.person_id, 0) + 1
        self.people_list.clear()
        for person in people:
            item = QListWidgetItem(f"{person.display_name}（{counts.get(person.id, 0)} 张样本）")
            item.setData(Qt.UserRole, person.id)
            self.people_list.addItem(item)

    def _render_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        if bbox is not None:
            left, top, right, bottom = (int(value) for value in bbox)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.green, 3))
            painter.drawRect(left, top, right - left, bottom - top)
            painter.end()
        self.preview_label.setPixmap(
            pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.stop_camera()
        self._repository.close()
        event.accept()


def _save_bgr_image(path: Path, frame: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not save image to {path}")
