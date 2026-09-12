from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
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
        self.page_hint = QLabel("证据不足时明确拒识")
        self.page_hint.setObjectName("pageHint")
        header.addWidget(self.page_hint, 0, Qt.AlignTop)
        layout.addLayout(header)

        layout.addWidget(self._build_health_strip())
        layout.addLayout(self._build_controls())
        layout.addWidget(self._build_workbench(), 1)

    def _build_health_strip(self) -> QWidget:
        """创建显示模型、标准库数量和完整性状态的顶部状态条。"""
        strip = QWidget()
        strip.setObjectName("healthStrip")
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._health_cards: list[QFrame] = []
        self.model_backend_label = self._make_health_card("模型后端", "加载中")
        self.library_summary_label = self._make_health_card("标准库", "0 个身份")
        self.integrity_label = self._make_health_card("数据状态", "检查中")
        self.integrity_label.setObjectName("integrityText")
        layout.addWidget(self.model_backend_label.parentWidget())
        layout.addWidget(self.library_summary_label.parentWidget())
        layout.addWidget(self.integrity_label.parentWidget())
        layout.addStretch(1)
        return strip

    def _make_health_card(self, caption: str, value: str) -> QLabel:
        """创建一张状态卡片并返回其中的值标签。"""
        card = QFrame()
        card.setObjectName("healthCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 18, 10)
        card_layout.setSpacing(2)
        caption_label = QLabel(caption.upper())
        caption_label.setObjectName("healthCaption")
        value_label = QLabel(value)
        value_label.setObjectName("healthValue")
        card_layout.addWidget(caption_label)
        card_layout.addWidget(value_label)
        self._health_cards.append(card)
        return value_label

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
        self.result_label = QLabel("等待输入图片")
        self.result_label.setObjectName("resultTitle")
        self.result_label.setWordWrap(True)
        self.status_label = QLabel("状态：未启动")
        self.status_label.setObjectName("statusText")
        self.status_label.setWordWrap(True)
        self.threshold_label = QLabel("最低相似度 --")
        self.threshold_label.setObjectName("metricText")
        self.frame_label = QLabel("参与帧 --")
        self.frame_label.setObjectName("metricText")
        self.quality_label = QLabel("画面建议：--")
        self.quality_label.setObjectName("qualityText")
        self.quality_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.status_label)
        result_layout.addWidget(self.threshold_label)
        result_layout.addWidget(self.frame_label)
        result_layout.addWidget(self.quality_label)

        splitter.addWidget(live_card)
        splitter.addWidget(result_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def set_camera_running(self, running: bool) -> None:
        """更新实时预览区的运行状态文字。"""
        self.live_state_label.setText("预览中" if running else "已停止")

    def set_integrity_state(self, text: str, *, warning: bool) -> None:
        """更新顶部完整性卡片的文字和状态属性。"""
        self.integrity_label.setText(text)
        self.integrity_label.setProperty("state", "warning" if warning else "ok")
        self.integrity_label.style().unpolish(self.integrity_label)
        self.integrity_label.style().polish(self.integrity_label)
