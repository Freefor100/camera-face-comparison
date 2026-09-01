from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from camera_face_comparison.camera import CameraDevice
from camera_face_comparison.config import load_settings
from camera_face_comparison.domain import RecognitionResult
from camera_face_comparison.repository import FaceRepository, SampleInput
from camera_face_comparison.ui.main_window import MainWindow


class FakeCamera:
    def __init__(self) -> None:
        self.discover_calls = 0

    def discover(self) -> list[CameraDevice]:
        self.discover_calls += 1
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
    person = repository.create_person_with_samples(
        person_id=str(uuid4()),
        display_name="Alice",
        samples=[
            SampleInput(
                image_path="faces/alice/sample.jpg",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                pose="sample_001",
                quality={"quality_score": 0.9, "tier": "high"},
            )
        ],
    )
    repository.close()

    camera = FakeCamera()
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=camera,  # type: ignore[arg-type]
    )
    assert window.camera_combo.count() == 1
    assert window.people_list.count() == 1
    assert "1 张样本" in window.people_list.item(0).text()
    assert window.import_compare_button.text() == "选择本地图片"
    assert window.add_person_from_files_button.text() == "从本地图片新增人员"
    assert window._enrollment_message(person) == "Alice 已录入，可以参与识别。"
    assert not window.compare_button.isEnabled()

    window.start_camera()
    assert window.stop_button.isEnabled()
    assert window.compare_button.isEnabled()
    assert not window.refresh_button.isEnabled()
    assert not window.camera_combo.isEnabled()

    window.refresh_cameras()
    assert camera.discover_calls == 1
    assert window.camera_combo.count() == 1

    window.stop_camera()
    assert window.start_button.isEnabled()
    assert not window.compare_button.isEnabled()
    assert window.refresh_button.isEnabled()
    assert window.camera_combo.isEnabled()
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


def test_recognition_result_shows_candidate_gap(tmp_path, qapplication) -> None:
    settings = load_settings(tmp_path)
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )

    window.on_recognition_result(
        RecognitionResult(
            status="matched",
            person_id="alice",
            display_name="Alice",
            top_score=0.72,
            runner_up_score=0.61,
            latency_ms=18.0,
            reason=None,
            bbox=None,
        )
    )

    assert "相似度 0.720" in window.result_label.text()
    assert "候选差距 0.110" in window.result_label.text()
    assert "18 ms" in window.status_label.text()
    window.close()
