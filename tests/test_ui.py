from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from camera_face_comparison.camera import CameraDevice
from camera_face_comparison.config import load_settings
from camera_face_comparison.repository import FaceRepository
from camera_face_comparison.ui.main_window import MainWindow


class FakeCamera:
    def discover(self) -> list[CameraDevice]:
        return [CameraDevice(index=3, label="Fake external camera")]

    def open(self, index: int) -> None:
        assert index == 3

    def read_frame(self) -> np.ndarray:
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class FakeFaceEngine:
    pass


@pytest.fixture(scope="module")
def qapplication() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_main_window_shows_library_and_updates_camera_controls(tmp_path, qapplication) -> None:
    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    person = repository.create_person("Alice")
    repository.close()

    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )
    assert window.camera_combo.count() == 1
    assert window.people_list.count() == 1
    assert "草稿" in window.people_list.item(0).text()
    assert window.import_compare_button.text() == "选择本地图片"
    assert window.add_person_from_files_button.text() == "从本地图片新增人员"
    assert "尚无有效样本" in window._enrollment_message(person)
    assert not window.compare_button.isEnabled()

    window.start_camera()
    assert window.stop_button.isEnabled()
    assert window.compare_button.isEnabled()
    window.stop_camera()
    assert window.start_button.isEnabled()
    assert not window.compare_button.isEnabled()
    window.close()


def test_stopping_preview_clears_the_last_camera_frame(tmp_path, qapplication) -> None:
    """A stopped camera must not leave a stale image that looks like a live preview."""

    settings = load_settings(tmp_path)
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )
    window.on_frame(np.full((120, 160, 3), 120, dtype=np.uint8))

    window.stop_camera()

    preview = window.preview_label.pixmap()
    assert window._current_frame is None
    assert window.preview_label.text() == "预览已停止"
    assert preview is None or preview.isNull()
    window.close()
