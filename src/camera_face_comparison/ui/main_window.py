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
    QFileDialog,
    QFormLayout,
    QFrame,
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
from ..domain import Person, RecognitionResult
from ..enrollment import REQUIRED_POSES, EnrollmentService
from ..face_engine import FaceEngine, FaceInputError
from ..image_input import ImageInput
from ..integrity import verify_library
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
        image_input: ImageInput,
    ) -> None:
        super().__init__()
        self._database_path = database_path
        self._settings = settings
        self._face_engine = face_engine
        self._image_input = image_input

    def run(self) -> None:
        repository: FaceRepository | None = None
        try:
            repository = FaceRepository(self._database_path)
            result = RecognitionService(repository, self._settings, self._face_engine).compare_input(
                self._image_input
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
        self.resize(1180, 760)
        self.setStyleSheet(APP_STYLE_SHEET)
        self._settings = settings
        self._face_engine = face_engine
        self._camera = camera
        self._repository = FaceRepository(settings.database_path)
        self._camera_worker: CameraWorker | None = None
        self._recognition_worker: RecognitionWorker | None = None
        self._current_frame: np.ndarray | None = None
        self._display_frame: np.ndarray | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None

        tabs = QTabWidget()
        tabs.addTab(self._build_recognition_tab(), "实时比对")
        tabs.addTab(self._build_library_tab(), "标准人脸库")
        self.setCentralWidget(tabs)
        self.refresh_cameras()
        self.refresh_people()

    def _build_recognition_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("recognitionPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        title = QLabel("现场识别")
        title.setObjectName("pageTitle")
        subtitle = QLabel("摄像头或本地图片均通过同一套开放集 1:N 规则，证据不足时明确拒识。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        controls = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.start_button = QPushButton("启动预览")
        self.stop_button = QPushButton("停止预览")
        self.compare_button = QPushButton("抓拍并比对")
        self.import_compare_button = QPushButton("选择本地图片")
        self.stop_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.compare_button.clicked.connect(self.compare_current_frame)
        self.import_compare_button.clicked.connect(self.compare_local_image)
        controls.addWidget(QLabel("摄像头："))
        controls.addWidget(self.camera_combo, 1)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.compare_button)
        controls.addWidget(self.import_compare_button)

        self.preview_label = QLabel("尚未启动摄像头")
        self.preview_label.setObjectName("previewSurface")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_card)
        self.result_label = QLabel("等待输入图片")
        self.result_label.setObjectName("resultTitle")
        self.status_label = QLabel("状态：未启动")
        self.status_label.setObjectName("statusText")
        self.integrity_label = QLabel()
        self.integrity_label.setObjectName("integrityText")
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.status_label)
        result_layout.addWidget(self.integrity_label)
        layout.addLayout(controls)
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(result_card)
        return page

    def _build_library_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("libraryPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(14)
        title = QLabel("标准人脸库")
        title.setObjectName("pageTitle")
        subtitle = QLabel("样本达到 3 张合格图片后自动激活；图片可来自本地文件或当前摄像头画面。")
        subtitle.setObjectName("pageSubtitle")
        self.people_list = QListWidget()
        self.people_list.setObjectName("peopleList")
        self.add_person_from_files_button = QPushButton("从本地图片新增人员")
        self.add_person_button = QPushButton("从当前画面新增人员")
        self.append_local_button = QPushButton("为选中人员导入图片")
        self.append_sample_button = QPushButton("为选中人员添加当前画面")
        self.add_person_from_files_button.clicked.connect(self.add_person_from_files)
        self.add_person_button.clicked.connect(self.add_person_from_current_frame)
        self.append_local_button.clicked.connect(self.append_local_images_to_selected_person)
        self.append_sample_button.clicked.connect(self.append_sample_to_selected_person)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.people_list, 1)
        layout.addWidget(self.add_person_from_files_button)
        layout.addWidget(self.add_person_button)
        layout.addWidget(self.append_local_button)
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
        self._current_frame = None
        self._display_frame = None
        self._last_bbox = None
        self.preview_label.clear()
        self.preview_label.setText("预览已停止")

    def on_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame.copy()
        self._display_frame = self._current_frame
        self._render_frame(self._display_frame, self._last_bbox)

    def compare_current_frame(self) -> None:
        if self._current_frame is None:
            self.status_label.setText("尚未获得摄像头画面。")
            return
        self._start_recognition(ImageInput.from_camera(self._current_frame))

    def compare_local_image(self) -> None:
        paths = self._select_local_image_paths(multiple=False)
        if not paths:
            return
        try:
            image_input = ImageInput.from_file(paths[0])
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "打开图片失败", str(error))
            return
        self._display_frame = image_input.frame.copy()
        self._last_bbox = None
        self._render_frame(self._display_frame, None)
        self._start_recognition(image_input)

    def _start_recognition(self, image_input: ImageInput) -> None:
        if self._recognition_worker is not None and self._recognition_worker.isRunning():
            self.status_label.setText("上一张图片仍在比对，请稍候。")
            return
        self.compare_button.setEnabled(False)
        self.status_label.setText("正在进行人脸检测与开放集比对…")
        self._recognition_worker = RecognitionWorker(
            database_path=self._settings.database_path,
            settings=self._settings,
            face_engine=self._face_engine,
            image_input=image_input,
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
        if self._display_frame is not None:
            self._render_frame(self._display_frame, self._last_bbox)

    def on_recognition_error(self, message: str) -> None:
        self.result_label.setText(f"比对失败：{message}")
        self.status_label.setText("比对任务异常结束。")

    def on_recognition_finished(self) -> None:
        if self._camera_worker is not None:
            self.compare_button.setEnabled(True)

    def on_camera_error(self, message: str) -> None:
        self.status_label.setText(f"摄像头错误：{message}")
        self.stop_camera()

    def add_person_from_files(self) -> None:
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if not accepted:
            return
        inputs = self._select_local_inputs()
        if not inputs:
            return
        try:
            person = self._enrollment_service().create_from_inputs(name, inputs)
        except (FaceInputError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "新增人员失败", str(error))
            return
        self.refresh_people()
        self.status_label.setText(self._enrollment_message(person))

    def add_person_from_current_frame(self) -> None:
        if self._current_frame is None:
            QMessageBox.information(self, "需要摄像头画面", "请先在实时比对页启动摄像头预览。")
            return
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if not accepted:
            return
        try:
            person = self._enrollment_service().create_from_inputs(
                name,
                [ImageInput.from_camera(self._current_frame)],
            )
        except (FaceInputError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "新增人员失败", str(error))
            return
        self.refresh_people()
        self.status_label.setText(self._enrollment_message(person))

    def append_local_images_to_selected_person(self) -> None:
        person_id = self._selected_person_id()
        if person_id is None:
            return
        inputs = self._select_local_inputs()
        if not inputs:
            return
        try:
            count = self._enrollment_service().append_from_inputs(person_id, inputs)
        except (FaceInputError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "追加样本失败", str(error))
            return
        self.refresh_people()
        self.status_label.setText(f"已追加 {count} 张合格图片。")

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
        person_id = self._selected_person_id()
        if person_id is None:
            return
        try:
            self._enrollment_service().append_from_inputs(
                person_id,
                [ImageInput.from_camera(self._current_frame)],
            )
        except (FaceInputError, RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "追加样本失败", str(error))
            return
        self.refresh_people()
        self.status_label.setText("已为选中人员追加一张合格图片。")

    def _enrollment_service(self) -> EnrollmentService:
        return EnrollmentService(
            repository=self._repository,
            settings=self._settings,
            face_engine=self._face_engine,
            image_saver=_save_bgr_image,
        )

    def _select_local_image_paths(self, *, multiple: bool) -> list[Path]:
        image_filter = "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        if multiple:
            paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", image_filter)
            return [Path(path) for path in paths]
        path, _ = QFileDialog.getOpenFileName(self, "选择待识别图片", "", image_filter)
        return [] if not path else [Path(path)]

    def _select_local_inputs(self) -> list[ImageInput]:
        paths = self._select_local_image_paths(multiple=True)
        if not paths:
            return []
        try:
            return [ImageInput.from_file(path) for path in paths]
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "打开图片失败", str(error))
            return []

    def _selected_person_id(self) -> str | None:
        item = self.people_list.currentItem()
        if item is None:
            QMessageBox.information(self, "请选择人员", "请先在标准人脸库中选择一名人员。")
            return None
        return str(item.data(Qt.UserRole))

    def _enrollment_message(self, person: Person) -> str:
        if person.lifecycle == "active":
            return f"{person.display_name} 已激活，可以参与识别。"
        sample_count = len(self._repository.list_samples(person.id))
        remaining = max(0, self._settings.min_active_samples - sample_count)
        return (
            f"{person.display_name} 已保存为草稿，还需至少 {remaining} 张合格样本才会参与识别。"
        )

    def refresh_people(self) -> None:
        people = self._repository.list_people()
        counts: dict[str, int] = {}
        for sample in self._repository.list_samples():
            counts[sample.person_id] = counts.get(sample.person_id, 0) + 1
        self.people_list.clear()
        for person in people:
            sample_count = counts.get(person.id, 0)
            lifecycle = "已激活" if person.lifecycle == "active" else "草稿"
            item = QListWidgetItem(f"{person.display_name}\n{sample_count} 张样本 · {lifecycle}")
            item.setData(Qt.UserRole, person.id)
            self.people_list.addItem(item)
        report = verify_library(self._repository, self._settings)
        if report.is_valid:
            self.integrity_label.setText("标准库完整性：正常")
            self.integrity_label.setProperty("state", "ok")
        else:
            self.integrity_label.setText(
                f"标准库完整性：异常（{report.failures[0].kind}）"
            )
            self.integrity_label.setProperty("state", "warning")
        self.integrity_label.style().unpolish(self.integrity_label)
        self.integrity_label.style().polish(self.integrity_label)

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


APP_STYLE_SHEET = """
QMainWindow, QWidget {
    background: #0b1020;
    color: #e8edf8;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
QTabWidget::pane { border: 0; }
QTabBar::tab {
    background: transparent;
    color: #8d99b3;
    padding: 14px 22px;
    margin: 0 4px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #f4f7ff; border-bottom-color: #7c8cff; }
#pageTitle { color: #f8faff; font-size: 28px; font-weight: 700; }
#pageSubtitle { color: #98a4bd; font-size: 14px; padding-bottom: 4px; }
QPushButton {
    background: #202b48;
    border: 1px solid #344467;
    border-radius: 8px;
    color: #edf2ff;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #2a385c; border-color: #7084d9; }
QPushButton:pressed { background: #18213a; }
QPushButton:disabled { background: #151c30; border-color: #222c46; color: #65708b; }
QComboBox {
    background: #131b30;
    border: 1px solid #344467;
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 20px;
}
#previewSurface {
    background: #050812;
    border: 1px solid #263452;
    border-radius: 14px;
    color: #74809a;
}
#resultCard {
    background: #121a2e;
    border: 1px solid #263452;
    border-radius: 12px;
    padding: 4px;
}
#resultTitle { color: #eef2ff; font-size: 18px; font-weight: 700; }
#statusText { color: #a8b3c9; }
#integrityText { color: #77e1b5; }
#integrityText[state="warning"] { color: #ffbd72; }
#peopleList {
    background: #11182a;
    border: 1px solid #263452;
    border-radius: 12px;
    padding: 6px;
}
#peopleList::item { padding: 12px; margin: 3px; border-radius: 8px; }
#peopleList::item:selected { background: #26375e; color: #ffffff; }
"""
