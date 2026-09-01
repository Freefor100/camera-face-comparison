from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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
from ..enrollment import EnrollmentService
from ..face_engine import FaceEngine, FaceInputError
from ..image_input import ImageInput
from ..integrity import verify_library
from ..recognition import RecognitionService
from ..repository import FaceRepository


class CameraWorker(QThread):
    """在后台线程中持续读取摄像头并发送画面与状态信号。"""

    frame_ready = Signal(object)
    worker_error = Signal(str)
    worker_status = Signal(str)

    def __init__(self, camera: CameraService, index: int) -> None:
        """创建摄像头读取线程。

        参数：
            camera：摄像头服务实例。
            index：待打开的设备索引。
        """
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
    """在后台线程中执行一次人脸检测、完整性检查和 1:N 比对。"""

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
        """创建一次识别任务。

        参数：
            database_path：标准库 SQLite 文件路径。
            settings：当前运行配置。
            face_engine：人脸检测和特征提取引擎。
            image_input：待识别的独立图片输入。
        """
        super().__init__()
        self._database_path = database_path
        self._settings = settings
        self._face_engine = face_engine
        self._image_input = image_input

    def run(self) -> None:
        """在线程中打开独立仓库执行识别，并保证结束时关闭连接。"""
        repository: FaceRepository | None = None
        try:
            repository = FaceRepository(self._database_path)
            result = RecognitionService(repository, self._settings, self._face_engine).compare_input(
                self._image_input
            )
            self.result_ready.emit(result)
        except Exception as error:  # noqa: BLE001 - 工作线程异常必须展示给用户
            self.worker_error.emit(str(error))
        finally:
            if repository is not None:
                repository.close()


class MainWindow(QMainWindow):
    """负责摄像头预览、识别和标准库录入的桌面应用主窗口。"""

    def __init__(
        self,
        *,
        settings: Settings,
        face_engine: FaceEngine,
        camera: CameraService,
    ) -> None:
        """组装窗口、服务依赖、识别页和标准库页。

        参数：
            settings：运行目录和算法配置。
            face_engine：本地人脸模型适配器。
            camera：跨平台摄像头服务。
        前置条件：
            本地模型已经加载成功，数据库路径可写。
        """
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
        """构造摄像头选择、预览、抓拍比对和结果展示页面。"""
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
        """构造人员列表以及本地图片/当前画面录入操作页面。"""
        page = QWidget()
        page.setObjectName("libraryPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(14)
        title = QLabel("标准人脸库")
        title.setObjectName("pageTitle")
        subtitle = QLabel("一张合格图片即可参与识别；可继续追加不同来源和角度的样本。")
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
        """扫描设备并刷新下拉列表；预览运行时拒绝刷新以保护线程状态。"""
        if self._camera_worker is not None:
            self.status_label.setText("请先停止预览，再刷新摄像头设备。")
            return
        self.camera_combo.clear()
        try:
            devices = self._camera.discover()
        except Exception as error:  # noqa: BLE001 - OpenCV 后端错误因平台而异
            self.status_label.setText(f"设备扫描失败：{error}")
            return
        for device in devices:
            self.camera_combo.addItem(device.label, device.index)
        if not devices:
            self.status_label.setText("未发现可打开的摄像头，请检查连接和系统权限。")

    def start_camera(self) -> None:
        """启动选中摄像头的后台预览线程并更新按钮状态。"""
        if self.camera_combo.currentIndex() < 0:
            self.status_label.setText("请先刷新并选择摄像头。")
            return
        self.stop_camera()
        self._camera_worker = CameraWorker(self._camera, int(self.camera_combo.currentData()))
        self._camera_worker.frame_ready.connect(self.on_frame)
        self._camera_worker.worker_error.connect(self.on_camera_error)
        self._camera_worker.worker_status.connect(self.status_label.setText)
        self._camera_worker.start()
        self.camera_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.compare_button.setEnabled(True)

    def stop_camera(self) -> None:
        """停止预览、释放线程和设备，并清除最后一帧画面及检测框。"""
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker.wait(2000)
            self._camera_worker = None
        self.start_button.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self._current_frame = None
        self._display_frame = None
        self._last_bbox = None
        self.preview_label.clear()
        self.preview_label.setText("预览已停止")

    def on_frame(self, frame: np.ndarray) -> None:
        """接收后台线程的一帧画面，复制后更新预览缓存。"""
        self._current_frame = frame.copy()
        self._display_frame = self._current_frame
        self._render_frame(self._display_frame, self._last_bbox)

    def compare_current_frame(self) -> None:
        """复制当前摄像头帧并异步提交一次识别任务。"""
        if self._current_frame is None:
            self.status_label.setText("尚未获得摄像头画面。")
            return
        self._start_recognition(ImageInput.from_camera(self._current_frame))

    def compare_local_image(self) -> None:
        """从文件选择器读取一张本地图片并异步提交识别任务。"""
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
        """创建并启动识别线程，避免模型推理阻塞主界面。"""
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
        """把服务结果转换为识别标签、耗时和检测框展示。"""
        self._last_bbox = result.bbox
        candidate_gap = _format_candidate_gap(result)
        if result.status == "matched":
            self.result_label.setText(
                f"识别成功：{result.display_name}｜相似度 {result.top_score:.3f}｜{candidate_gap}"
            )
        elif result.status == "unknown":
            self.result_label.setText(
                f"未知人员｜最高相似度 {result.top_score or 0.0:.3f}｜{candidate_gap}"
                f"｜原因：{result.reason}"
            )
        else:
            self.result_label.setText(f"无法识别当前画面：{result.reason}")
        self.status_label.setText(f"本次处理耗时：{result.latency_ms:.0f} ms")
        if self._display_frame is not None:
            self._render_frame(self._display_frame, self._last_bbox)

    def on_recognition_error(self, message: str) -> None:
        """展示识别工作线程抛出的异常信息。"""
        self.result_label.setText(f"比对失败：{message}")
        self.status_label.setText("比对任务异常结束。")

    def on_recognition_finished(self) -> None:
        """识别线程结束后恢复摄像头比对按钮。"""
        if self._camera_worker is not None:
            self.compare_button.setEnabled(True)

    def on_camera_error(self, message: str) -> None:
        """展示摄像头异常并停止当前预览。"""
        self.status_label.setText(f"摄像头错误：{message}")
        self.stop_camera()

    def add_person_from_files(self) -> None:
        """通过文件选择器创建一个至少含一张合格样本的新人员。"""
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
        """使用当前摄像头画面创建一个新人员。"""
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
        """把用户选中的本地图片追加到当前选中人员。"""
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

    def append_sample_to_selected_person(self) -> None:
        """把当前摄像头画面追加到当前选中人员。"""
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
        """根据当前窗口依赖创建标准库录入服务。"""
        return EnrollmentService(
            repository=self._repository,
            settings=self._settings,
            face_engine=self._face_engine,
            image_saver=_save_bgr_image,
        )

    def _select_local_image_paths(self, *, multiple: bool) -> list[Path]:
        """打开图片选择器并返回用户选择的路径列表。"""
        image_filter = "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        if multiple:
            paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", image_filter)
            return [Path(path) for path in paths]
        path, _ = QFileDialog.getOpenFileName(self, "选择待识别图片", "", image_filter)
        return [] if not path else [Path(path)]

    def _select_local_inputs(self) -> list[ImageInput]:
        """将用户选择的本地路径读取为图片输入，失败时弹出提示。"""
        paths = self._select_local_image_paths(multiple=True)
        if not paths:
            return []
        try:
            return [ImageInput.from_file(path) for path in paths]
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "打开图片失败", str(error))
            return []

    def _selected_person_id(self) -> str | None:
        """返回标准库列表当前选中人员的编号。"""
        item = self.people_list.currentItem()
        if item is None:
            QMessageBox.information(self, "请选择人员", "请先在标准人脸库中选择一名人员。")
            return None
        return str(item.data(Qt.UserRole))

    def _enrollment_message(self, person: Person) -> str:
        """生成新人员录入成功后的状态提示。"""
        return f"{person.display_name} 已录入，可以参与识别。"

    def refresh_people(self) -> None:
        """刷新人员及样本数量，并更新标准库完整性状态。"""
        people = self._repository.list_people()
        counts: dict[str, int] = {}
        for sample in self._repository.list_samples():
            counts[sample.person_id] = counts.get(sample.person_id, 0) + 1
        self.people_list.clear()
        for person in people:
            sample_count = counts.get(person.id, 0)
            item = QListWidgetItem(f"{person.display_name}\n{sample_count} 张样本")
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
        """把 BGR 帧转换成 Qt 图片，并可选绘制检测框。"""
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
        """关闭窗口前停止摄像头并释放数据库连接。"""
        self.stop_camera()
        self._repository.close()
        event.accept()


def _save_bgr_image(path: Path, frame: np.ndarray) -> None:
    """使用 OpenCV 将 BGR 图像保存到指定路径。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not save image to {path}")


def _format_candidate_gap(result: RecognitionResult) -> str:
    """将识别结果中的候选差距格式化为界面文本。"""
    if result.top_score is None or result.runner_up_score is None:
        return "候选差距 --"
    return f"候选差距 {result.top_score - result.runner_up_score:.3f}"


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
