from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from ..camera import CameraService
from ..config import Settings
from ..domain import FaceSample, Person, RecognitionResult
from ..face_engine import FaceEngine
from ..face_library import InMemoryFaceLibrary
from ..image_input import ImageInput
from ..integrity import verify_library
from ..repository import FaceRepository
from .library_page import LibraryPage
from .recognition_page import RecognitionPage
from .theme import APP_STYLE_SHEET
from .workers import (
    CameraRecognitionWorker,
    CameraWorker,
    EnrollmentWorker,
    RecognitionWorker,
)

CAMERA_FRAME_COUNT = 5
CAMERA_FRAME_INTERVAL_MS = 80


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
        self._face_library = InMemoryFaceLibrary.empty()
        self._library_valid = False
        self._busy = False
        self._camera_worker: CameraWorker | None = None
        self._recognition_worker: RecognitionWorker | CameraRecognitionWorker | None = None
        self._enrollment_worker: EnrollmentWorker | None = None
        self._capture_timer: QTimer | None = None
        self._captured_frames: list[ImageInput] = []
        self._recognition_frames: list[ImageInput] = []
        self._current_frame: np.ndarray | None = None
        self._display_frame: np.ndarray | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None

        self._recognition_page = RecognitionPage()
        self._library_page = LibraryPage(settings.data_dir)
        self._recognition_page.refresh_requested.connect(self.refresh_cameras)
        self._recognition_page.start_requested.connect(self.start_camera)
        self._recognition_page.stop_requested.connect(self.stop_camera)
        self._recognition_page.compare_camera_requested.connect(self.compare_current_frame)
        self._recognition_page.compare_file_requested.connect(self.compare_local_image)
        self._library_page.create_from_files_requested.connect(self.add_person_from_files)
        self._library_page.create_from_camera_requested.connect(
            self.add_person_from_current_frame
        )
        self._library_page.append_files_requested.connect(
            self.append_local_images_to_selected_person
        )
        self._library_page.append_camera_requested.connect(
            self.append_sample_to_selected_person
        )
        self._library_page.recheck_requested.connect(self.recheck_library)

        # 保留主窗口对关键控件的直接引用，业务方法只负责协调页面和服务。
        self.camera_combo = self._recognition_page.camera_combo
        self.refresh_button = self._recognition_page.refresh_button
        self.start_button = self._recognition_page.start_button
        self.stop_button = self._recognition_page.stop_button
        self.compare_button = self._recognition_page.compare_button
        self.import_compare_button = self._recognition_page.import_compare_button
        self.preview_label = self._recognition_page.preview_label
        self.result_preview_label = self._recognition_page.result_preview_label
        self.result_label = self._recognition_page.result_label
        self.status_label = self._recognition_page.status_label
        self.integrity_label = self._recognition_page.integrity_label
        backend = getattr(self._face_engine, "backend", None)
        backend_name = getattr(backend, "name", "cpu")
        self._recognition_page.model_backend_label.setText(
            f"推理：{str(backend_name).upper()}"
        )
        self._recognition_page.threshold_label.setText(
            f"{self._settings.recognition_policy.minimum_score:.3f}"
        )
        self.people_list = self._library_page.people_list
        self.add_person_from_files_button = self._library_page.add_person_from_files_button
        self.add_person_button = self._library_page.add_person_button
        self.append_local_button = self._library_page.append_local_button
        self.append_sample_button = self._library_page.append_sample_button
        self.recheck_library_button = self._library_page.recheck_library_button

        tabs = QTabWidget()
        tabs.addTab(self._recognition_page, "实时比对")
        tabs.addTab(self._library_page, "标准人脸库")
        self.setCentralWidget(tabs)
        self.refresh_cameras()
        self.recheck_library()

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
        self.compare_button.setEnabled(self._library_valid)
        self._recognition_page.set_camera_running(True)

    def stop_camera(self) -> None:
        """停止预览、释放线程和设备，并清除最后一帧画面及检测框。"""
        self._cancel_frame_capture()
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
        self._recognition_page.set_camera_running(False)

    def on_frame(self, frame: np.ndarray) -> None:
        """接收后台线程的一帧画面，复制后更新预览缓存。"""
        self._current_frame = frame.copy()
        self._render_frame(self._current_frame, None)

    def compare_current_frame(self) -> None:
        """点击后立即启动检测流水线，并按固定间隔继续提交摄像头帧。"""
        if self._current_frame is None:
            self.status_label.setText("尚未获得摄像头画面。")
            return
        if self._busy:
            self.status_label.setText("上一项任务仍在处理，请稍候。")
            return
        request_started_at = perf_counter()
        first_input = ImageInput.from_camera(self._current_frame)
        self._captured_frames = [first_input]
        self._busy = True
        self._set_task_buttons_enabled(False)
        self.status_label.setText(f"正在采集 1/{CAMERA_FRAME_COUNT} 帧…")
        self._start_camera_recognition_stream(first_input, started_at=request_started_at)
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(CAMERA_FRAME_INTERVAL_MS)
        self._capture_timer.timeout.connect(self._collect_next_frame)
        self._capture_timer.start()

    def _collect_next_frame(self) -> None:
        """收集一帧最新摄像头画面，五帧齐全后启动识别线程。"""

        if self._current_frame is not None:
            image_input = ImageInput.from_camera(self._current_frame)
            self._captured_frames.append(image_input)
            final = len(self._captured_frames) >= CAMERA_FRAME_COUNT
            self._submit_camera_recognition_frame(image_input, final=final)
        self.status_label.setText(
            f"正在采集 {len(self._captured_frames)}/{CAMERA_FRAME_COUNT} 帧…"
        )
        if len(self._captured_frames) < CAMERA_FRAME_COUNT:
            return
        self._cancel_frame_capture(restore_buttons=False)

    def _start_camera_recognition_stream(
        self,
        first_input: ImageInput,
        *,
        started_at: float,
    ) -> None:
        """启动摄像头识别线程，并立即提交第一帧以重叠采集和检测。"""

        self._prepare_recognition_ui([first_input])
        self._recognition_worker = CameraRecognitionWorker(
            database_path=self._settings.database_path,
            settings=self._settings,
            face_engine=self._face_engine,
            library_snapshot=self._face_library.snapshot(),
            frame_count=CAMERA_FRAME_COUNT,
            started_at=started_at,
        )
        self._connect_recognition_worker(self._recognition_worker)
        self._recognition_worker.start()
        self._recognition_worker.submit(first_input)

    def _submit_camera_recognition_frame(
        self,
        image_input: ImageInput,
        *,
        final: bool = False,
    ) -> None:
        """向摄像头识别流水线提交一帧，并按需要结束输入。"""

        worker = self._recognition_worker
        if not isinstance(worker, CameraRecognitionWorker):
            raise TypeError("camera recognition worker is not running")
        self._recognition_frames.append(image_input)
        worker.submit(image_input)
        if final:
            worker.finish_inputs()

    def _cancel_frame_capture(self, *, restore_buttons: bool = True) -> None:
        """停止未完成的多帧采集，并按需要恢复操作按钮。"""

        capture_was_running = self._capture_timer is not None
        if self._capture_timer is not None:
            self._capture_timer.stop()
            self._capture_timer.deleteLater()
            self._capture_timer = None
        if self._captured_frames and restore_buttons:
            if capture_was_running and isinstance(
                self._recognition_worker, CameraRecognitionWorker
            ):
                self._recognition_worker.cancel()
            self._captured_frames = []
            self._busy = False
            self._set_task_buttons_enabled(self._library_valid)

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
        self._render_frame(self._display_frame, None, target=self.result_preview_label)
        self._start_recognition((image_input,))

    def _start_recognition(self, image_inputs: tuple[ImageInput, ...]) -> None:
        """创建并启动识别线程，避免模型推理阻塞主界面。"""
        if self._recognition_worker is not None and self._recognition_worker.isRunning():
            self.status_label.setText("上一张图片仍在比对，请稍候。")
            return
        self._prepare_recognition_ui(list(image_inputs))
        self._recognition_worker = RecognitionWorker(
            database_path=self._settings.database_path,
            settings=self._settings,
            face_engine=self._face_engine,
            image_inputs=image_inputs,
            library_snapshot=self._face_library.snapshot(),
        )
        self._connect_recognition_worker(self._recognition_worker)
        self._recognition_worker.start()

    def _prepare_recognition_ui(self, image_inputs: list[ImageInput]) -> None:
        """保存结果画面来源并把界面切换为识别中状态。"""

        self._busy = True
        self._recognition_frames = image_inputs
        self._set_task_buttons_enabled(False)
        self._recognition_page.clear_result_metrics()
        self._recognition_page.set_result_state("working")
        self.result_label.setText("正在识别")
        self._recognition_page.result_name_label.setText("正在检测并提取人脸特征")
        self.status_label.setText("正在进行人脸检测与开放集比对…")

    def _connect_recognition_worker(
        self,
        worker: RecognitionWorker | CameraRecognitionWorker,
    ) -> None:
        """连接单帧和流式识别工作线程共用的结果信号。"""

        worker.result_ready.connect(self.on_recognition_result)
        worker.worker_error.connect(self.on_recognition_error)
        worker.finished.connect(self.on_recognition_finished)

    def on_recognition_result(self, result: RecognitionResult) -> None:
        """把服务结果转换为识别标签、抓拍画面和检测框展示。"""
        if self._recognition_frames:
            frame_index = result.selected_frame_index
            if frame_index is None or frame_index >= len(self._recognition_frames):
                frame_index = 0
            self._display_frame = self._recognition_frames[frame_index].frame.copy()
        self._last_bbox = result.bbox
        self._recognition_page.set_result_state(result.status)
        if result.status == "matched":
            self.result_label.setText("识别成功")
            self._recognition_page.result_name_label.setText(result.display_name or "已登记人员")
            self.status_label.setText("已在标准库中找到符合条件的人员。")
        elif result.status == "unknown":
            self.result_label.setText("未知人员")
            self._recognition_page.result_name_label.setText("未达到登记人员接收条件")
            self.status_label.setText(_format_reason(result.reason))
        else:
            self.result_label.setText("无法识别")
            self._recognition_page.result_name_label.setText(_format_reason(result.reason))
            self.status_label.setText("请调整输入后重试。")
        warning_text = _format_quality_warnings(result.quality_warnings)
        self._recognition_page.top_score_label.setText(_format_score(result.top_score))
        self._recognition_page.score_gap_label.setText(_format_score(result.score_gap))
        self._recognition_page.frame_label.setText(
            f"{result.valid_frame_count} / {result.frame_count}"
        )
        self._recognition_page.latency_label.setText(f"{result.latency_ms:.0f} ms")
        self._recognition_page.decision_label.setText(_format_decision(result))
        self._recognition_page.quality_label.setText(
            "画面建议：" + (warning_text or "当前输入无需额外调整")
        )
        self._recognition_page.result_source_label.setText(
            "摄像头五帧择优" if result.frame_count > 1 else "本地图片"
        )
        if self._display_frame is not None:
            self._render_frame(
                self._display_frame,
                self._last_bbox,
                target=self.result_preview_label,
            )

    def on_recognition_error(self, message: str) -> None:
        """展示识别工作线程抛出的异常信息。"""
        self._recognition_page.clear_result_metrics()
        self._recognition_page.set_result_state("invalid")
        self.result_label.setText("比对失败")
        self._recognition_page.result_name_label.setText(message)
        self.status_label.setText("比对任务异常结束。")

    def on_recognition_finished(self) -> None:
        """识别线程结束后恢复可用操作按钮。"""
        self._busy = False
        self._captured_frames = []
        self._recognition_frames = []
        self._set_task_buttons_enabled(self._library_valid)
        self.compare_button.setEnabled(self._library_valid and self._camera_worker is not None)

    def on_camera_error(self, message: str) -> None:
        """展示摄像头异常并停止当前预览。"""
        self.status_label.setText(f"摄像头错误：{message}")
        self.stop_camera()

    def add_person_from_files(self) -> None:
        """通过文件选择器创建一个至少含一张有效单人脸样本的新人员。"""
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if not accepted:
            return
        inputs = self._select_local_inputs()
        if not inputs:
            return
        self._start_enrollment(
            operation="create",
            image_inputs=tuple(inputs),
            display_name=name,
        )

    def add_person_from_current_frame(self) -> None:
        """使用当前摄像头画面创建一个新人员。"""
        if self._current_frame is None:
            QMessageBox.information(self, "需要摄像头画面", "请先在实时比对页启动摄像头预览。")
            return
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if not accepted:
            return
        self._start_enrollment(
            operation="create",
            image_inputs=(ImageInput.from_camera(self._current_frame),),
            display_name=name,
        )

    def append_local_images_to_selected_person(self) -> None:
        """把用户选中的本地图片追加到当前选中人员。"""
        person_id = self._selected_person_id()
        if person_id is None:
            return
        inputs = self._select_local_inputs()
        if not inputs:
            return
        self._start_enrollment(
            operation="append",
            image_inputs=tuple(inputs),
            person_id=person_id,
        )

    def append_sample_to_selected_person(self) -> None:
        """把当前摄像头画面追加到当前选中人员。"""
        if self._current_frame is None:
            QMessageBox.information(self, "需要摄像头画面", "请先在实时比对页启动摄像头预览。")
            return
        person_id = self._selected_person_id()
        if person_id is None:
            return
        self._start_enrollment(
            operation="append",
            image_inputs=(ImageInput.from_camera(self._current_frame),),
            person_id=person_id,
        )

    def _start_enrollment(
        self,
        *,
        operation: str,
        image_inputs: tuple[ImageInput, ...],
        display_name: str | None = None,
        person_id: str | None = None,
    ) -> None:
        """启动独立录入线程，并与识别任务共享忙碌状态。"""

        if self._busy:
            self.status_label.setText("上一项任务仍在处理，请稍候。")
            return
        self._busy = True
        self._set_task_buttons_enabled(False)
        self.status_label.setText("正在检测图片并写入标准库…")
        self._enrollment_worker = EnrollmentWorker(
            database_path=self._settings.database_path,
            settings=self._settings,
            face_engine=self._face_engine,
            operation=operation,
            image_inputs=image_inputs,
            display_name=display_name,
            person_id=person_id,
        )
        self._enrollment_worker.result_ready.connect(self.on_enrollment_result)
        self._enrollment_worker.worker_error.connect(self.on_enrollment_error)
        self._enrollment_worker.finished.connect(self.on_enrollment_finished)
        self._enrollment_worker.start()

    def on_enrollment_result(self, payload: object) -> None:
        """录入成功后重新检查数据库，并只刷新受影响人员的内存原型。"""

        operation, *values = payload  # type: ignore[misc]
        person_id = values[0].id if operation == "create" else values[0]
        self.recheck_library(refreshed_person_id=person_id)
        if operation == "create":
            person = values[0]
            self.status_label.setText(self._enrollment_message(person))
        else:
            self.status_label.setText(f"已追加 {values[1]} 张有效单人脸图片。")

    def on_enrollment_error(self, message: str) -> None:
        """展示录入线程抛出的异常信息。"""

        self.status_label.setText(f"录入失败：{message}")

    def on_enrollment_finished(self) -> None:
        """录入线程结束后恢复当前可用操作。"""

        self._busy = False
        self._set_task_buttons_enabled(self._library_valid)

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
        """刷新人员及样本数量，不触发完整性检查。"""
        people = self._repository.list_people()
        samples_by_person: dict[str, list[FaceSample]] = {person.id: [] for person in people}
        for sample in self._repository.list_samples():
            samples_by_person.setdefault(sample.person_id, []).append(sample)
        self._library_page.set_people(
            people,
            {person_id: tuple(samples) for person_id, samples in samples_by_person.items()},
        )
        self._recognition_page.library_summary_label.setText(f"标准库：{len(people)} 人")

    def recheck_library(self, *, refreshed_person_id: str | None = None) -> None:
        """执行完整性检查；录入成功时只刷新受影响人员，否则重建整个矩阵。"""

        try:
            report = verify_library(self._repository, self._settings)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._face_library.clear()
            self._library_valid = False
            self._set_integrity_label(f"数据：检查失败（{error}）", warning=True)
            self._set_library_actions_enabled(False)
            self.refresh_people()
            return
        if report.is_valid:
            try:
                if refreshed_person_id is None:
                    self._face_library.rebuild(self._repository)
                else:
                    self._face_library.refresh_person(self._repository, refreshed_person_id)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._face_library.clear()
                self._library_valid = False
                self._set_integrity_label(f"数据：重建失败（{error}）", warning=True)
                self._set_library_actions_enabled(False)
                self.refresh_people()
                return
            self._library_valid = True
            self._set_integrity_label("数据：正常", warning=False)
            self._set_library_actions_enabled(True)
        else:
            self._face_library.clear()
            self._library_valid = False
            first_failure = report.failures[0]
            self._set_integrity_label(
                f"数据：异常（{first_failure.kind}）", warning=True
            )
            self._set_library_actions_enabled(False)
        self.refresh_people()

    def _set_integrity_label(self, text: str, *, warning: bool) -> None:
        """更新完整性状态文本及其样式属性。"""
        self._recognition_page.set_integrity_state(text, warning=warning)

    def _set_library_actions_enabled(self, enabled: bool) -> None:
        """根据标准库可信状态启用或禁用识别和录入操作。"""

        self._set_task_buttons_enabled(enabled)
        self.recheck_library_button.setEnabled(not self._busy)

    def _set_task_buttons_enabled(self, enabled: bool) -> None:
        """按标准库状态和当前忙碌状态更新识别、录入按钮。"""

        enabled = enabled and not self._busy
        # 复查按钮是从完整性异常状态恢复的唯一入口，异常时也必须可用。
        self.recheck_library_button.setEnabled(not self._busy)
        for button in (
            self.import_compare_button,
            self.add_person_from_files_button,
            self.add_person_button,
            self.append_local_button,
            self.append_sample_button,
        ):
            button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled and self._camera_worker is not None)

    def _render_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
        *,
        target: QLabel | None = None,
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
        label = target or self.preview_label
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """关闭窗口前停止采集、等待短任务线程并释放数据库连接。"""
        self.stop_camera()
        for worker in (self._recognition_worker, self._enrollment_worker):
            if worker is not None and worker.isRunning():
                worker.wait(5000)
        self._repository.close()
        event.accept()


def _format_score(value: float | None) -> str:
    """把可空的相似度或候选分差转换为固定三位小数。"""

    return "--" if value is None else f"{value:.3f}"


def _format_decision(result: RecognitionResult) -> str:
    """根据接收规则和结果状态生成简短、完整的判定说明。"""

    if result.status not in {"matched", "unknown"}:
        return "输入无效"
    accepted = result.status == "matched"
    names = {
        "score_threshold": "最高相似度达到阈值" if accepted else "最高相似度低于阈值",
        "score_gap": "候选分差达到阈值" if accepted else "候选分差低于阈值",
        "score_and_gap": "相似度与分差均达标" if accepted else "相似度或分差未达标",
        "nac": "邻域分数达到阈值" if accepted else "邻域分数低于阈值",
    }
    return names[result.acceptance_rule]


def _format_reason(reason: str | None) -> str:
    """把内部失败代码转换为用户可理解的说明。"""

    messages = {
        "no_face_detected": "画面中未检测到人脸。",
        "multiple_faces": "画面中检测到多张人脸，请只保留一人。",
        "invalid_embedding": "人脸特征提取失败。",
        "no_valid_frames": "采集的画面均未得到有效单人脸。",
        "score_below_threshold": "最高相似度低于判定阈值。",
        "score_gap_below_minimum": "前两名候选的分差不足。",
        "insufficient_gallery_identities": "标准库人数不足，无法完成当前判定。",
        "nac_below_threshold": "邻域识别分数低于判定阈值。",
    }
    return messages.get(reason or "", reason or "未能完成识别。")


def _format_quality_warnings(warnings: tuple[str, ...]) -> str:
    """把非阻断质量提示代码转换为简短操作建议。"""

    messages = {
        "low_detection_confidence": "正对镜头",
        "move_closer": "靠近镜头",
        "hold_still": "保持稳定",
        "increase_lighting": "增加光线",
        "reduce_lighting": "避免过曝",
        "improve_contrast": "改善光照对比",
    }
    return "、".join(messages[item] for item in warnings)
