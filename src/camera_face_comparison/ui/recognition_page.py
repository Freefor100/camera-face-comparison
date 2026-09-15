from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class RecognitionPage(QWidget):
    """构造实时识别页面，并通过信号把用户操作交给主窗口。"""

    refresh_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    compare_camera_requested = Signal()
    compare_file_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """创建实时识别页面及其可被主窗口更新的控件。"""
        super().__init__(parent)
        self.setObjectName("recognitionPage")
        self._build()

    def _build(self) -> None:
        """创建页面标题、设备操作、实时预览和识别结果区域。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 30)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel("现场识别")
        title.setObjectName("pageTitle")
        self.subtitle = QLabel("摄像头和本地图片使用同一套开放集 1:N 识别流程。")
        self.subtitle.setObjectName("pageSubtitle")
        title_group.addWidget(title)
        title_group.addWidget(self.subtitle)
        header.addLayout(title_group, 1)
        header.addLayout(self._build_header_status())
        layout.addLayout(header)

        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_workbench(), 1)

    def _build_header_status(self) -> QHBoxLayout:
        """在标题右侧创建紧凑的运行状态信息。"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        self.model_backend_label = QLabel("推理：加载中")
        self.model_backend_label.setObjectName("headerStatus")
        self.library_summary_label = QLabel("标准库：0 人")
        self.library_summary_label.setObjectName("headerStatus")
        self.integrity_label = QLabel("数据：检查中")
        self.integrity_label.setObjectName("integrityText")
        layout.addWidget(self.model_backend_label)
        layout.addWidget(self.library_summary_label)
        layout.addWidget(self.integrity_label)
        return layout

    def _build_controls(self) -> QHBoxLayout:
        """创建摄像头选择和识别操作按钮。"""
        controls = QHBoxLayout()
        controls.setSpacing(8)
        camera_caption = QLabel("输入设备")
        camera_caption.setObjectName("controlCaption")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(180)
        self.refresh_button = QPushButton("刷新设备")
        self.start_button = QPushButton("启动预览")
        self.stop_button = QPushButton("停止预览")
        self.compare_button = QPushButton("抓拍并比对")
        self.import_compare_button = QPushButton("选择本地图片")
        self.stop_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.refresh_button.clicked.connect(self.refresh_requested)
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.compare_button.clicked.connect(self.compare_camera_requested)
        self.import_compare_button.clicked.connect(self.compare_file_requested)
        controls.addWidget(camera_caption)
        controls.addWidget(self.camera_combo)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(self.compare_button)
        controls.addWidget(self.import_compare_button)
        return controls

    def _build_workbench(self) -> QSplitter:
        """创建左侧实时预览和右侧抓拍结果的双栏工作区。"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("recognitionWorkbench")
        splitter.setChildrenCollapsible(False)

        live_card = QFrame()
        live_card.setObjectName("previewCard")
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(16, 14, 16, 16)
        live_layout.setSpacing(8)
        live_header = QHBoxLayout()
        live_title = QLabel("实时画面")
        live_title.setObjectName("cardTitle")
        self.live_state_label = QLabel("未启动")
        self.live_state_label.setObjectName("cardMeta")
        live_header.addWidget(live_title)
        live_header.addStretch(1)
        live_header.addWidget(self.live_state_label)
        live_layout.addLayout(live_header)
        self.preview_label = QLabel("尚未启动摄像头")
        self.preview_label.setObjectName("previewSurface")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(500, 360)
        live_layout.addWidget(self.preview_label, 1)

        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 14, 16, 16)
        result_layout.setSpacing(8)
        result_header = QHBoxLayout()
        result_title = QLabel("本次判定")
        result_title.setObjectName("cardTitle")
        self.result_source_label = QLabel("等待输入")
        self.result_source_label.setObjectName("cardMeta")
        result_header.addWidget(result_title)
        result_header.addStretch(1)
        result_header.addWidget(self.result_source_label)
        result_layout.addLayout(result_header)
        self.result_preview_label = QLabel("等待抓拍画面")
        self.result_preview_label.setObjectName("resultPreviewSurface")
        self.result_preview_label.setAlignment(Qt.AlignCenter)
        self.result_preview_label.setMinimumSize(320, 220)
        result_layout.addWidget(self.result_preview_label, 1)
        self.result_label = QLabel("等待识别")
        self.result_label.setObjectName("resultTitle")
        self.result_name_label = QLabel("选择摄像头抓拍或本地图片")
        self.result_name_label.setObjectName("resultName")
        self.result_name_label.setWordWrap(True)
        self.status_label = QLabel("等待输入图片")
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)
        metrics = QGridLayout()
        metrics.setContentsMargins(0, 4, 0, 4)
        metrics.setHorizontalSpacing(18)
        metrics.setVerticalSpacing(10)
        self.top_score_label = self._add_metric(metrics, "最高相似度", 0, 0)
        self.threshold_label = self._add_metric(metrics, "判定阈值", 0, 1)
        self.score_gap_label = self._add_metric(metrics, "候选分差", 1, 0)
        self.frame_label = self._add_metric(metrics, "有效帧", 1, 1)
        self.latency_label = self._add_metric(metrics, "处理耗时", 2, 0)
        self.decision_label = self._add_metric(metrics, "判定依据", 2, 1)
        self.quality_label = QLabel("画面建议：--")
        self.quality_label.setObjectName("qualityText")
        self.quality_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_name_label)
        result_layout.addWidget(self.status_label)
        result_layout.addLayout(metrics)
        result_layout.addWidget(self.quality_label)

        splitter.addWidget(live_card)
        splitter.addWidget(result_card)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        return splitter

    def _add_metric(
        self,
        layout: QGridLayout,
        caption: str,
        row: int,
        column: int,
    ) -> QLabel:
        """向结果网格加入固定标题和待更新的数值。"""
        item = QWidget()
        item.setObjectName("metricItem")
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(2)
        caption_label = QLabel(caption)
        caption_label.setObjectName("metricCaption")
        value_label = QLabel("--")
        value_label.setObjectName("metricValue")
        value_label.setWordWrap(True)
        item_layout.addWidget(caption_label)
        item_layout.addWidget(value_label)
        layout.addWidget(item, row, column)
        return value_label

    def clear_result_metrics(self) -> None:
        """开始新任务或任务失败时清空上一轮动态结果。"""
        for label in (
            self.top_score_label,
            self.score_gap_label,
            self.frame_label,
            self.latency_label,
            self.decision_label,
        ):
            label.setText("--")

    def set_camera_running(self, running: bool) -> None:
        """更新实时预览区的运行状态文字。"""
        self.live_state_label.setText("预览中" if running else "已停止")

    def set_integrity_state(self, text: str, *, warning: bool) -> None:
        """更新标题栏中的数据完整性状态。"""
        self.integrity_label.setText(text)
        self.integrity_label.setProperty("state", "warning" if warning else "ok")
        self.integrity_label.style().unpolish(self.integrity_label)
        self.integrity_label.style().polish(self.integrity_label)
