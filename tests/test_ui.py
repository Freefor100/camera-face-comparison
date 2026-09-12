from __future__ import annotations

import hashlib
import os
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from camera_face_comparison.camera import CameraDevice
from camera_face_comparison.config import load_settings
from camera_face_comparison.domain import RecognitionResult
from camera_face_comparison.integrity import IntegrityFailure, LibraryVerificationReport
from camera_face_comparison.repository import FaceRepository, SampleInput
from camera_face_comparison.ui import main_window as main_window_module
from camera_face_comparison.ui.main_window import MainWindow


class FakeCamera:
    """提供固定设备和帧的界面测试摄像头替身。"""

    def __init__(self) -> None:
        """初始化设备扫描调用计数器。"""
        self.discover_calls = 0

    def discover(self) -> list[CameraDevice]:
        """返回一个固定的外置摄像头设备。"""
        self.discover_calls += 1
        return [CameraDevice(index=3, label="Fake external camera")]

    def open(self, index: int) -> None:
        """确认主窗口使用了预期的摄像头索引。"""
        assert index == 3

    def read_frame(self) -> np.ndarray:
        """返回一帧固定尺寸的黑色测试图像。"""
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        """模拟释放测试摄像头。"""


class FakeFaceEngine:
    """仅用于窗口装配测试的空人脸引擎替身。"""


@pytest.fixture(scope="module")
def qapplication() -> QApplication:
    """返回模块级 Qt 应用实例，供离屏界面测试复用。"""
    return QApplication.instance() or QApplication([])


def test_main_window_shows_library_and_updates_camera_controls(tmp_path, qapplication) -> None:
    """主窗口应显示标准库，并在启动/停止预览时更新控件状态。"""
    settings = load_settings(tmp_path)
    image_path = settings.faces_dir / "alice" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"valid sample")
    repository = FaceRepository(settings.database_path)
    person = repository.create_person_with_samples(
        person_id=str(uuid4()),
        display_name="Alice",
        samples=[
            SampleInput(
                    image_path="faces/alice/sample.jpg",
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                    quality_metrics={"face_size_px": 160.0},
                    image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
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
    """停止摄像头后不能留下看起来仍在实时更新的旧画面。"""

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


def test_recognition_result_shows_score_gap(tmp_path, qapplication) -> None:
    """主窗口应显示识别相似度、候选分差和耗时。"""
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
            second_score=0.61,
            score_gap=0.11,
            acceptance_score=0.72,
            acceptance_rule="score_threshold",
            latency_ms=18.0,
            reason=None,
            bbox=None,
            quality_metrics={},
            quality_warnings=(),
        )
    )

    assert "相似度 0.720" in window.result_label.text()
    assert "候选分差 0.110" in window.result_label.text()
    assert "18 ms" in window.status_label.text()
    window.close()


def test_integrity_failure_disables_library_actions_until_manual_recheck_succeeds(
    tmp_path, qapplication, monkeypatch
) -> None:
    """标准库异常时应清空检索矩阵，手动复查恢复后再启用操作。"""

    settings = load_settings(tmp_path)
    image_path = settings.faces_dir / "alice" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"valid sample")
    repository = FaceRepository(settings.database_path)
    repository.create_person_with_samples(
        person_id="alice",
        display_name="Alice",
        samples=[
            SampleInput(
                image_path="faces/alice/sample.jpg",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                quality_metrics={},
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
        ],
    )
    repository.close()

    failed = LibraryVerificationReport(
        (IntegrityFailure("image_hash_mismatch", "sample"),)
    )
    state = {"valid": False}

    def fake_verify(repository, settings):
        """按测试状态返回异常或正常的完整性报告。"""

        return LibraryVerificationReport(()) if state["valid"] else failed

    monkeypatch.setattr(main_window_module, "verify_library", fake_verify)
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )

    assert window._face_library.snapshot().person_ids == ()
    assert not window.add_person_from_files_button.isEnabled()
    assert not window.append_local_button.isEnabled()
    assert "异常" in window.integrity_label.text()

    state["valid"] = True
    window.recheck_library()

    assert window._face_library.snapshot().person_ids == ("alice",)
    assert window.add_person_from_files_button.isEnabled()
    assert window.append_local_button.isEnabled()
    assert "正常" in window.integrity_label.text()
    window.close()
