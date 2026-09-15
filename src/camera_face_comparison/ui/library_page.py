from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..domain import FaceSample, Person

SAMPLE_CARD_TARGET_WIDTH = 176
SAMPLE_GRID_SPACING = 12


class SampleThumbnail(QLabel):
    """按控件可用空间缩放原图，并始终保持图片宽高比。"""

    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None) -> None:
        """保存原始图片，后续尺寸变化时从原图重新缩放。"""
        super().__init__(parent)
        self.setObjectName("sampleThumb")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(112, 88)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._source_pixmap = pixmap
        if pixmap.isNull():
            self.setText("图片不可用")

    def sizeHint(self) -> QSize:
        """返回适合多数肖像图片的初始显示尺寸。"""
        return QSize(SAMPLE_CARD_TARGET_WIDTH - 12, 126)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """缩放缩略图时保留原始比例，避免固定正方形裁切或拉伸。"""
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self.setPixmap(
                self._source_pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


class LibraryPage(QWidget):
    """构造标准库管理页面，并集中处理人员列表与样本缩略图展示。"""

    create_from_files_requested = Signal()
    create_from_camera_requested = Signal()
    append_files_requested = Signal()
    append_camera_requested = Signal()
    recheck_requested = Signal()

    def __init__(self, data_dir: Path, parent: QWidget | None = None) -> None:
        """创建标准库页面。

        参数：
            data_dir：用于解析数据库相对样本路径的应用数据目录。
        """
        super().__init__(parent)
        self.setObjectName("libraryPage")
        self._data_dir = data_dir
        self._people: tuple[Person, ...] = ()
        self._samples_by_person: dict[str, tuple[FaceSample, ...]] = {}
        self._sample_cards: list[QWidget] = []
        self._build()

    def _build(self) -> None:
        """创建搜索栏、人员列表、详情区和标准库操作按钮。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 30)
        layout.setSpacing(16)

        title = QLabel("标准人脸库")
        title.setObjectName("pageTitle")
        subtitle = QLabel("一张有效单人脸图片即可使用；可以继续追加不同来源和角度的样本。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("libraryWorkbench")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_people_panel())
        splitter.addWidget(self._build_details_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

    def _build_people_panel(self) -> QWidget:
        """创建可搜索人员列表及其操作按钮。"""
        panel = QFrame()
        panel.setObjectName("peoplePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        label = QLabel("人员")
        label.setObjectName("cardTitle")
        self.people_count_label = QLabel("0 个身份")
        self.people_count_label.setObjectName("cardMeta")
        heading.addWidget(label)
        heading.addStretch(1)
        heading.addWidget(self.people_count_label)
        layout.addLayout(heading)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索姓名")
        self.search_edit.setObjectName("searchField")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)
        self.people_list = QListWidget()
        self.people_list.setObjectName("peopleList")
        self.people_list.currentItemChanged.connect(self._show_selected_person)
        layout.addWidget(self.people_list, 1)

        self.add_person_from_files_button = QPushButton("从本地图片新增人员")
        self.add_person_button = QPushButton("从当前画面新增人员")
        self.append_local_button = QPushButton("为选中人员导入图片")
        self.append_sample_button = QPushButton("为选中人员添加当前画面")
        self.recheck_library_button = QPushButton("重新检查标准库")
        self.add_person_from_files_button.clicked.connect(self.create_from_files_requested)
        self.add_person_button.clicked.connect(self.create_from_camera_requested)
        self.append_local_button.clicked.connect(self.append_files_requested)
        self.append_sample_button.clicked.connect(self.append_camera_requested)
        self.recheck_library_button.clicked.connect(self.recheck_requested)
        for button in (
            self.add_person_from_files_button,
            self.add_person_button,
            self.append_local_button,
            self.append_sample_button,
            self.recheck_library_button,
        ):
            layout.addWidget(button)
        return panel

    def _build_details_panel(self) -> QWidget:
        """创建选中人员信息和样本缩略图区域。"""
        panel = QFrame()
        panel.setObjectName("personDetails")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        self.selected_name_label = QLabel("未选择人员")
        self.selected_name_label.setObjectName("detailTitle")
        self.selected_count_label = QLabel("样本数 --")
        self.selected_count_label.setObjectName("metricText")
        layout.addWidget(self.selected_name_label)
        layout.addWidget(self.selected_count_label)
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("sectionDivider")
        layout.addWidget(divider)

        self.sample_scroll = QScrollArea()
        self.sample_scroll.setWidgetResizable(True)
        self.sample_scroll.setObjectName("sampleScroll")
        self.sample_container = QWidget()
        self.sample_container.setObjectName("sampleContainer")
        self.sample_grid = QGridLayout(self.sample_container)
        self.sample_grid.setContentsMargins(0, 4, 0, 4)
        self.sample_grid.setSpacing(SAMPLE_GRID_SPACING)
        self.sample_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.sample_scroll.setWidget(self.sample_container)
        self.sample_scroll.viewport().installEventFilter(self)
        layout.addWidget(self.sample_scroll, 1)
        return panel

    def set_people(
        self,
        people: list[Person],
        samples_by_person: dict[str, tuple[FaceSample, ...]],
    ) -> None:
        """更新人员列表和详情数据，并尽量保留当前选中人员。"""
        selected_id = self._selected_person_id()
        self._people = tuple(people)
        self._samples_by_person = samples_by_person
        self.people_count_label.setText(f"{len(self._people)} 个身份")
        self._apply_filter(selected_id=selected_id)

    def _apply_filter(self, _text: str = "", *, selected_id: str | None = None) -> None:
        """按姓名过滤人员列表并恢复可用的当前选择。"""
        wanted = self.search_edit.text().strip().casefold()
        old_id = selected_id or self._selected_person_id()
        self.people_list.blockSignals(True)
        self.people_list.clear()
        for person in self._people:
            if wanted and wanted not in person.display_name.casefold():
                continue
            sample_count = len(self._samples_by_person.get(person.id, ()))
            item = QListWidgetItem(f"{person.display_name}（{sample_count} 张）")
            item.setData(Qt.UserRole, person.id)
            self.people_list.addItem(item)
            if person.id == old_id:
                self.people_list.setCurrentItem(item)
        self.people_list.blockSignals(False)
        if self.people_list.currentItem() is None and self.people_list.count() > 0:
            self.people_list.setCurrentRow(0)
        else:
            # 恢复选中项发生在信号阻断期间，需要主动刷新一次详情。
            self._show_selected_person(self.people_list.currentItem(), None)

    def _selected_person_id(self) -> str | None:
        """返回列表当前选中人员的编号。"""
        item = self.people_list.currentItem()
        return None if item is None else str(item.data(Qt.UserRole))

    def _show_selected_person(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """根据当前列表项刷新人员详情和样本缩略图。"""
        self._clear_sample_grid()
        if current is None:
            self.selected_name_label.setText("未选择人员")
            self.selected_count_label.setText("样本数 --")
            return
        person_id = str(current.data(Qt.UserRole))
        person = next((item for item in self._people if item.id == person_id), None)
        if person is None:
            return
        samples = self._samples_by_person.get(person_id, ())
        self.selected_name_label.setText(person.display_name)
        self.selected_count_label.setText(f"样本数 {len(samples)}")
        for index, sample in enumerate(samples):
            self._sample_cards.append(self._sample_card(sample, index))
        self._relayout_sample_cards()

    def _sample_card(self, sample: FaceSample, index: int) -> QWidget:
        """创建一个样本缩略图卡片；图片缺失时保留明确占位提示。"""
        card = QFrame()
        card.setObjectName("sampleCard")
        card.setMinimumWidth(132)
        card.setMaximumWidth(220)
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        path = Path(sample.image_path)
        if not path.is_absolute():
            path = self._data_dir / path
        pixmap = QPixmap(str(path))
        image_label = SampleThumbnail(pixmap)
        caption = QLabel(f"样本 {index + 1}")
        caption.setObjectName("sampleCaption")
        caption.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(image_label)
        card_layout.addWidget(caption)
        return card

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """详情区宽度变化时重新计算样本列数。"""
        if watched is self.sample_scroll.viewport() and event.type() == QEvent.Resize:
            self._relayout_sample_cards(self.sample_scroll.viewport().width())
        return super().eventFilter(watched, event)

    def _relayout_sample_cards(self, available_width: int | None = None) -> None:
        """按当前可用宽度排列样本卡片，窄窗口至少保留一列。"""
        width = available_width or self.sample_scroll.viewport().width()
        columns = max(
            1,
            (max(1, width) + SAMPLE_GRID_SPACING)
            // (SAMPLE_CARD_TARGET_WIDTH + SAMPLE_GRID_SPACING),
        )
        card_width = min(
            220,
            max(
                132,
                (max(1, width) - SAMPLE_GRID_SPACING * (columns - 1)) // columns,
            ),
        )
        for card in self._sample_cards:
            self.sample_grid.removeWidget(card)
            card.setFixedWidth(card_width)
        for index, card in enumerate(self._sample_cards):
            self.sample_grid.addWidget(card, index // columns, index % columns)

    def _clear_sample_grid(self) -> None:
        """删除详情区中的旧缩略图控件。"""
        for card in self._sample_cards:
            self.sample_grid.removeWidget(card)
            card.hide()
            card.setParent(None)
            card.deleteLater()
        self._sample_cards.clear()
