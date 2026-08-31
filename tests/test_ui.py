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
    repository.create_person("Alice")
    repository.close()

    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )
    assert window.camera_combo.count() == 1
    assert window.people_list.count() == 1
    assert not window.compare_button.isEnabled()

    window.start_camera()
    assert window.stop_button.isEnabled()
    assert window.compare_button.isEnabled()
    window.stop_camera()
    assert window.start_button.isEnabled()
    assert not window.compare_button.isEnabled()
    window.close()
