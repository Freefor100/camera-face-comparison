"""验证标准库页面的选择、管理入口和缩略图布局契约。"""
import os
from datetime import UTC, datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QPushButton

from camera_face_comparison.domain import FaceSample, Person
from camera_face_comparison.ui.library_page import LibraryPage
from tests.application.helpers import embedding as production_embedding


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
    sample = FaceSample(
        id="sample-1",
        person_id=person.id,
        image_path="missing.jpg",
        embedding=production_embedding([1.0, 0.0]),
        quality_metrics={},
        created_at=datetime.now(UTC),
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
            FaceSample(
                id=f"{person.id}-sample",
                person_id=person.id,
                image_path="missing.jpg",
                embedding=production_embedding([1.0, 0.0]),
                quality_metrics={},
                created_at=created_at,
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
