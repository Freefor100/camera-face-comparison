"""验证标准库页面的选择、管理入口和缩略图布局契约。"""

import os
from datetime import UTC, datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QPushButton

from camera_face_comparison.domain import Person, SampleMetadata
from camera_face_comparison.ui.library_page import LibraryPage


@pytest.fixture(scope="module")
def qapplication():
    """复用离屏 Qt 应用。"""
    return QApplication.instance() or QApplication([])


def test_library_page_exposes_person_and_sample_management_actions(
    tmp_path,
    qapplication,
) -> None:
    """选中人员后应能修改姓名、删除人员或请求删除指定样本。"""

    person = Person(id="alice", display_name="Alice", created_at=datetime.now(UTC))
    sample = SampleMetadata(
        id="sample-1",
        person_id=person.id,
        image_path="missing.jpg",
        source_type="file",
    )
    page = LibraryPage(tmp_path)
    requested_samples: list[str] = []
    page.delete_sample_requested.connect(requested_samples.append)

    page.set_people([person], {person.id: (sample,)})
    page.set_management_enabled(True)

    assert page.rename_person_button.text() == "修改姓名"
    assert page.delete_person_button.text() == "删除人员"
    delete_buttons = page.findChildren(QPushButton, "sampleDeleteButton")
    assert len(delete_buttons) == 1
    delete_buttons[0].click()
    assert requested_samples == ["sample-1"]
    page.close()


def test_switching_people_detaches_previous_sample_cards(tmp_path, qapplication) -> None:
    """切换人员时应立即移除旧样本卡片，避免短暂重叠显示。"""

    created_at = datetime.now(UTC)
    people = [
        Person(id="alice", display_name="Alice", created_at=created_at),
        Person(id="bob", display_name="Bob", created_at=created_at),
    ]
    samples = {
        person.id: (
            SampleMetadata(
                id=f"{person.id}-sample",
                person_id=person.id,
                image_path="missing.jpg",
                source_type="file",
            ),
        )
        for person in people
    }
    page = LibraryPage(tmp_path)
    page.set_people(people, samples)
    previous_cards = tuple(page._sample_cards)

    page.people_list.setCurrentRow(1)

    assert all(card.parentWidget() is None for card in previous_cards)
    page.close()


def test_invalid_path_keeps_its_placeholder_and_other_sample_cards(tmp_path, qapplication):
    """坏路径只能影响自己的缩略图，不能中断人员详情展示。"""
    person = Person(id="alice", display_name="测试", created_at=datetime.now(UTC))
    samples = (
        SampleMetadata("bad", person.id, "faces/\x00.jpg", "file"),
        SampleMetadata("good", person.id, "faces/missing.jpg", "file"),
    )
    page = LibraryPage(tmp_path)
    page.set_people([person], {person.id: samples})
    assert len(page._sample_cards) == 2
    assert page._image_labels["bad"].text() == "图片不可用"
    page.close()


def test_library_sample_cards_reflow_with_available_width(tmp_path, qapplication):
    """样本卡片按可用宽度重排，窄窗口至少保留一列。"""
    person = Person(id="alice", display_name="测试", created_at=datetime.now(UTC))
    samples = tuple(
        SampleMetadata(str(i), person.id, f"faces/missing-{i}.jpg", "file") for i in range(5)
    )
    page = LibraryPage(tmp_path)
    page.set_people([person], {person.id: samples})
    page._relayout_sample_cards(150)
    assert {page.sample_grid.getItemPosition(i)[1] for i in range(5)} == {0}
    page._relayout_sample_cards(720)
    assert max(page.sample_grid.getItemPosition(i)[1] for i in range(5)) >= 2
    page.close()
