"""主窗口仅协调对话框和页面展示，业务任务与资源由任务协调器管理。"""

from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMainWindow, QMessageBox, QTabWidget

from ..domain import RecognitionResult
from ..image_input import ImageInput
from .controller import TaskController
from .library_page import LibraryPage
from .recognition_page import RecognitionPage
from .theme import APP_STYLE_SHEET


class MainWindow(QMainWindow):
    """将明确分类的设备、识别和标准库事件显示在各自区域。"""

    def __init__(self, *, settings, face_engine, camera):
        """先构造可用窗口，后台检查失败不会阻止窗口存在。"""
        super().__init__()
        self.setWindowTitle("摄像头人脸比对系统")
        self.resize(1180, 760)
        self.setStyleSheet(APP_STYLE_SHEET)
        self._settings = settings
        self._display_frame = self._last_bbox = None
        self._close_timer = None
        self._diagnostic_detail = ""
        self._recognition_page = RecognitionPage()
        self._library_page = LibraryPage(settings.data_dir)
        tabs = QTabWidget()
        tabs.addTab(self._recognition_page, "实时比对")
        tabs.addTab(self._library_page, "标准人脸库")
        self.setCentralWidget(tabs)
        # 这些引用用于窗口的对话框与展示，不再持有数据库连接或工作线程。
        for name in (
            "camera_combo",
            "refresh_button",
            "start_button",
            "stop_button",
            "compare_button",
            "import_compare_button",
            "preview_label",
            "result_preview_label",
            "result_label",
            "status_label",
            "integrity_label",
        ):
            setattr(self, name, getattr(self._recognition_page, name))
        for name in (
            "people_list",
            "add_person_from_files_button",
            "add_person_button",
            "append_local_button",
            "append_sample_button",
            "rename_person_button",
            "delete_person_button",
            "recheck_library_button",
        ):
            setattr(self, name, getattr(self._library_page, name))
        self._recognition_page.threshold_label.setText(
            f"{settings.recognition_policy.minimum_score:.3f}"
        )
        self.controller = TaskController(settings, face_engine, camera, self)
        self.controller.state_changed.connect(self._render_state)
        self.controller.frame_ready.connect(self.on_frame)
        self.controller.devices_ready.connect(self._show_devices)
        self.controller.result_ready.connect(self.on_recognition_result)
        self.controller.result_reset.connect(self._reset_result)
        self.controller.recognition_status.connect(self._recognition_progress)
        self.controller.recognition_error.connect(self._recognition_error)
        self.controller.recognition_cancelled.connect(self._recognition_cancelled)
        self.controller.library_status.connect(self._show_library_status)
        self._library_page.diagnostics_requested.connect(self._show_diagnostics)
        self.controller.device_status.connect(self._recognition_page.device_status_label.setText)
        self.controller.catalog_ready.connect(self._show_catalog)
        self.controller.backend_changed.connect(self._show_backend)
        self.controller.ready_to_close.connect(self._finish_close)
        for signal, handler in (
            (self._recognition_page.refresh_requested, self.refresh_cameras),
            (self._recognition_page.start_requested, self.start_camera),
            (self._recognition_page.stop_requested, self.stop_camera),
            (self._recognition_page.compare_camera_requested, self.compare_current_frame),
            (self._recognition_page.compare_file_requested, self.compare_local_image),
            (self._library_page.create_from_files_requested, self.add_person_from_files),
            (self._library_page.create_from_camera_requested, self.add_person_from_current_frame),
            (
                self._library_page.append_files_requested,
                self.append_local_images_to_selected_person,
            ),
            (self._library_page.append_camera_requested, self.append_sample_to_selected_person),
            (self._library_page.rename_person_requested, self.rename_selected_person),
            (self._library_page.delete_person_requested, self.delete_selected_person),
            (self._library_page.delete_sample_requested, self.delete_selected_sample),
            (self._library_page.recheck_requested, self.recheck_library),
        ):
            signal.connect(handler)
        self._show_backend(getattr(face_engine, "backend", None))
        self._render_state(self.controller.state)
        self.controller.recheck_library()
        self.controller.refresh_devices()

    def _render_state(self, state):
        """根据一个完整状态统一计算按钮，不让单独回调恢复按钮。"""
        idle = state.task == "idle" and not state.closing
        allowed = idle and state.library == "healthy"
        device_idle = state.camera == "stopped" and not state.scanning and not state.closing
        self.camera_combo.setEnabled(device_idle)
        self.refresh_button.setEnabled(device_idle)
        self.start_button.setEnabled(device_idle and self.camera_combo.count() > 0)
        self.stop_button.setEnabled(state.camera in {"starting", "running"} and not state.closing)
        self.compare_button.setEnabled(allowed and state.camera_ready)
        self.import_compare_button.setEnabled(allowed)
        self._library_page.set_management_enabled(allowed, camera_ready=state.camera_ready)
        self.recheck_library_button.setEnabled(idle)
        self._recognition_page.set_camera_running(state.camera == "running")
        labels = {
            "unchecked": "未检查",
            "checking": "检查中",
            "healthy": "正常",
            "invalid": "检查未通过",
            "unavailable": "无法读取",
        }
        self._recognition_page.set_integrity_state(
            "数据：" + labels[state.library], warning=state.library != "healthy"
        )
        self.integrity_label.setToolTip(
            "这是最近一次检查的结果，不是实时监控。"
            if state.library == "healthy"
            else "请在标准人脸库页查看诊断原因并重新检查。"
        )
        if self.controller.catalog is None:
            self._recognition_page.library_summary_label.setText("标准库：--")
        if state.closing:
            self.statusBar().showMessage("正在等待后台任务和设备退出，窗口会在安全释放后关闭。")

    def _show_devices(self, devices):
        """设备列表只接设备扫描结果，不改变识别提示。"""
        self.camera_combo.clear()
        for device in devices:
            self.camera_combo.addItem(device.label, device.index)
        self._render_state(self.controller.state)

    def _show_backend(self, backend):
        """显示当前会话后端，未准备时不猜测为 CUDA 或 CPU。"""
        name = backend.name.upper() if backend is not None else "未准备"
        if backend is not None and backend.name == "mixed":
            roles = dict(backend.model_providers)

            def role_name(role):
                """将会话首选 provider 转为简短中文显示。"""
                return "CUDA" if roles[role][0] == "CUDAExecutionProvider" else "CPU"

            name = "检测 " + role_name("detection") + "／特征 " + role_name("recognition")
        self._recognition_page.model_backend_label.setText("推理：" + name)

    def _show_library_status(self, message):
        """主页面显示完整简短说明，原始诊断放在独立详情中。"""
        summary, _, detail = message.partition("\n")
        self._library_page.status_label.setText(summary)
        self._diagnostic_detail = detail
        self._library_page.diagnostics_button.setEnabled(bool(detail))

    def _show_diagnostics(self):
        """仅在用户主动查看时展示具体记录、路径和底层错误。"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("标准库诊断")
        dialog.setText(self._library_page.status_label.text())
        dialog.setDetailedText(self._diagnostic_detail)
        dialog.exec()

    def _show_catalog(self, report):
        """展示仅含元信息的人员目录，不从主线程重新读数据库。"""
        grouped = {person.id: [] for person in report.people}
        for sample in report.metadata:
            grouped.setdefault(sample.person_id, []).append(sample)
        self._library_page.set_people(
            list(report.people), {pid: tuple(items) for pid, items in grouped.items()}
        )
        self._recognition_page.library_summary_label.setText(f"标准库：{len(report.people)} 人")

    def refresh_cameras(self):
        """请求后台扫描设备，驱动阻塞不会冻结主窗口。"""
        self.controller.refresh_devices()

    def start_camera(self):
        """请求打开选中设备，未选择时只更新设备提示。"""
        if self.camera_combo.currentIndex() >= 0:
            self.controller.start_camera(int(self.camera_combo.currentData()))
        else:
            self._recognition_page.device_status_label.setText("请先刷新并选择摄像头。")

    def stop_camera(self):
        """请求停止并立即清空预览，保留已经完成的历史判定。"""
        self.controller.stop_camera()

    def on_frame(self, frame):
        """只接摄像头画面；停止后的空画面不影响结果照片。"""
        if frame is None:
            self.preview_label.clear()
            self.preview_label.setText("预览已停止")
        else:
            self._render_frame(frame, None, self.preview_label)

    def compare_current_frame(self):
        """提交摄像头识别请求，不直接管理采集定时器。"""
        self.controller.recognize_camera()

    def compare_local_image(self):
        """文件选择后只传路径，解码和推理均在后台完成。"""
        paths = self._select_paths(False)
        if paths:
            self.controller.recognize_file(paths[0])

    def _reset_result(self):
        """完整清空照片、来源、质量建议和动态值，保留固定阈值。"""
        page = self._recognition_page
        self._display_frame = self._last_bbox = None
        page.clear_result_metrics()
        page.set_result_state("idle")
        self.result_label.setText("等待识别")
        page.result_name_label.setText("选择摄像头抓拍或本地图片")
        page.result_name_label.setToolTip("")
        page.result_source_label.setText("等待输入")
        page.quality_label.setText("画面建议：--")
        page.log_warning_label.clear()
        self.status_label.setText("等待输入图片")
        self.result_preview_label.clear()
        self.result_preview_label.setText("等待抓拍画面")

    def _recognition_progress(self, message):
        """识别进度只更新识别区，不显示设备和标准库操作。"""
        self._recognition_page.set_result_state("working")
        self.result_label.setText("正在识别")
        self._recognition_page.result_name_label.setText("正在检测并提取人脸特征")
        self.status_label.setText(message)

    def on_recognition_result(self, outcome):
        """画面、来源、结论和数值来自同一个任务结果。"""
        result, page = outcome.result, self._recognition_page
        self._display_frame, self._last_bbox = outcome.frame, result.bbox
        page.set_result_state(result.status)
        if result.status == "matched":
            self.result_label.setText("识别成功")
            page.result_name_label.setText(result.display_name or "已登记人员")
            self.status_label.setText("已在标准库中找到符合条件的人员。")
        elif result.status == "unknown":
            self.result_label.setText("未知人员")
            page.result_name_label.setText(
                "请先录入标准库人员"
                if result.reason == "empty_face_library"
                else "未达到登记人员接收条件"
            )
            self.status_label.setText(_format_reason(result.reason))
        else:
            self.result_label.setText("无法识别")
            page.result_name_label.setText(_format_reason(result.reason))
            self.status_label.setText("请调整输入后重试。")
        page.top_score_label.setText(_format_score(result.top_score))
        page.score_gap_label.setText(_format_score(result.score_gap))
        page.frame_label.setText(f"{result.valid_frame_count} / {result.frame_count}")
        page.decision_label.setText(_format_decision(result))
        suggestions = (
            _format_quality_warnings(result.quality_warnings)
            if result.status != "invalid"
            else _format_reason(result.reason)
        )
        page.quality_label.setText("画面建议：" + (suggestions or "当前画面没有明显质量异常。"))
        page.log_warning_label.setText(result.log_warning or "")
        page.result_source_label.setText(
            "摄像头五帧择优" if outcome.source == "camera" else "本地图片"
        )
        if outcome.frame is not None:
            self._render_frame(outcome.frame, result.bbox, self.result_preview_label)
        # 到控件及结果图片生成结束才读取请求计时，日志故障不会抹掉已计算的结论。
        started = self.controller.request_started_at
        elapsed = result.latency_ms if started is None else (perf_counter() - started) * 1000
        page.latency_label.setText(f"{elapsed:.0f} ms")

    def _recognition_cancelled(self, message):
        """取消有独立终态，不留下永久的正在识别或错误颜色。"""
        self._reset_result()
        self.result_label.setText("识别已取消")
        self.status_label.setText(message)

    def _recognition_error(self, failure):
        """区别输入失败与系统失败，同时保留诊断详情供排查。"""
        page = self._recognition_page
        page.clear_result_metrics()
        page.set_result_state(
            "invalid"
            if failure.kind in {"business_constraint", "file_error", "input_invalid"}
            else "error"
        )
        self.result_label.setText("比对失败")
        page.result_name_label.setText(failure.message)
        page.result_name_label.setToolTip(failure.detail)
        self.status_label.setText("本次任务未能完成，请检查输入或处理环境。")
        page.quality_label.setText("画面建议：本次未完成质量评估。")
        page.log_warning_label.clear()

    def _selected_person(self):
        """从已发布的展示目录取得选择，不额外读取数据库。"""
        item = self.people_list.currentItem()
        if item is None or self.controller.catalog is None:
            return None
        pid = str(item.data(Qt.UserRole))
        return next((p for p in self.controller.catalog.people if p.id == pid), None)

    def _select_paths(self, multiple=True):
        """只选择路径，不在界面线程解码图片。"""
        filters = "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        if multiple:
            paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", filters)
            return tuple(Path(p) for p in paths)
        path, _ = QFileDialog.getOpenFileName(self, "选择待识别图片", "", filters)
        return (Path(path),) if path else ()

    def add_person_from_files(self):
        """输入姓名和图片路径，提交原子录入任务。"""
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if accepted:
            paths = self._select_paths()
            if paths:
                self.controller.library_operation("create", name=name, paths=paths)

    def add_person_from_current_frame(self):
        """复制当前摄像头画面后提交录入，不受后续停止影响。"""
        if self.controller.current_frame is None:
            QMessageBox.information(self, "需要画面", "请先启动摄像头预览。")
            return
        # 对话框会处理设备事件，因此先冻结点击时的画面，不能确认姓名后再读共享当前帧。
        image_input = ImageInput.from_camera(self.controller.current_frame)
        name, accepted = QInputDialog.getText(self, "新增人员", "人员姓名：")
        if accepted:
            self.controller.library_operation("create", name=name, inputs=(image_input,))

    def append_local_images_to_selected_person(self):
        """为选中人员追加本地图片，解码和写入均在后台。"""
        person = self._selected_person()
        if person is None:
            self._show_library_status("请先选择标准库中的人员。")
            return
        paths = self._select_paths()
        if paths:
            self.controller.library_operation("append", person_id=person.id, paths=paths)

    def append_sample_to_selected_person(self):
        """复制当前画面并追加到已选择人员。"""
        person = self._selected_person()
        if person is None:
            self._show_library_status("请先选择标准库中的人员。")
            return
        if self.controller.current_frame is None:
            self._show_library_status("请先启动摄像头预览并等待画面。")
            return
        self.controller.library_operation(
            "append",
            person_id=person.id,
            inputs=(ImageInput.from_camera(self.controller.current_frame),),
        )

    def rename_selected_person(self):
        """确认新姓名后提交管理任务，保持人员编号不变。"""
        person = self._selected_person()
        if person is not None:
            name, accepted = QInputDialog.getText(
                self, "修改姓名", "人员姓名：", text=person.display_name
            )
            if accepted:
                self.controller.library_operation("rename", person_id=person.id, name=name)

    def delete_selected_person(self):
        """明确确认后删除人员，成功时由协调器清空旧结果。"""
        person = self._selected_person()
        if (
            person is not None
            and QMessageBox.question(
                self,
                "删除人员",
                f"确定删除“{person.display_name}”及其样本吗？此操作无法撤销。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.controller.library_operation("delete_person", person_id=person.id)

    def delete_selected_sample(self, sample_id):
        """删除样本的归属和最后一张限制仍由数据库事务检查。"""
        person = self._selected_person()
        if (
            person is not None
            and QMessageBox.question(
                self,
                "删除样本",
                "确定删除此样本吗？此操作无法撤销。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.controller.library_operation(
                "delete_sample", person_id=person.id, sample_id=sample_id
            )

    def recheck_library(self):
        """重新检查并在成功后恢复标准库功能。"""
        self.controller.recheck_library()

    def _render_frame(self, frame, bbox, target):
        """把当前结果或预览画面转为 Qt 图片，检测框只属于结果画面。"""
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        if bbox is not None:
            left, top, right, bottom = map(int, bbox)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.green, 3))
            painter.drawRect(left, top, right - left, bottom - top)
            painter.end()
        target.setPixmap(pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        """窗口变化后按原始结果画面重新缩放，不拉伸旧缩略画面。"""
        super().resizeEvent(event)
        if self._display_frame is not None:
            self._render_frame(self._display_frame, self._last_bbox, self.result_preview_label)

    def closeEvent(self, event):
        """线程未退出时忽略关闭请求，异步释放完毕后才接受。"""
        if self.controller.can_close and self._library_page.thumbnail_loader.idle:
            self._library_page.thumbnail_loader.cancel()
            if self._close_timer is not None:
                self._close_timer.stop()
            event.accept()
        else:
            event.ignore()
            self._library_page.thumbnail_loader.cancel()
            if self._close_timer is None:
                self._close_timer = QTimer(self)
                self._close_timer.setInterval(30)
                self._close_timer.timeout.connect(self._finish_close)
                self._close_timer.start()
            self.controller.request_close()

    def _finish_close(self):
        """模型、设备与缩略图均已退出后再次关闭，避免递归请求。"""
        if self.controller.can_close and self._library_page.thumbnail_loader.idle:
            self.close()


def _format_score(value: float | None) -> str:
    """把可空的相似度或候选分差转换为固定三位小数。"""

    return "--" if value is None else f"{value:.3f}"


def _format_decision(result: RecognitionResult) -> str:
    """根据接收规则和结果状态生成简短、完整的判定说明。"""

    if result.reason == "empty_face_library":
        return "标准库为空，未进行身份比较"
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
        "empty_face_library": "标准库为空，请先录入人员。",
        "missing_face_landmarks": "人脸关键点无效，请正对镜头并保持面部无遮挡。",
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
        "low_detection_confidence": "请正对镜头并保持面部无遮挡",
        "move_closer": "靠近镜头",
        "hold_still": "请改善识别画质",
        "increase_lighting": "增加光线",
        "reduce_lighting": "避免过曝",
        "improve_contrast": "改善光照对比",
    }
    return "、".join(messages[item] for item in warnings)
