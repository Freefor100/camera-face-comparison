from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from camera_face_comparison.camera import CameraDevice
from camera_face_comparison.config import load_settings
from camera_face_comparison.domain import FaceSample, Person, RecognitionResult
from camera_face_comparison.integrity import IntegrityFailure, LibraryVerificationReport
from camera_face_comparison.repository import FaceRepository, SampleInput
from camera_face_comparison.ui import main_window as main_window_module
from camera_face_comparison.ui.library_page import LibraryPage
from camera_face_comparison.ui.main_window import (
    CAMERA_FRAME_COUNT,
    CAMERA_FRAME_INTERVAL_MS,
    MainWindow,
)


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
    assert window.people_list.item(0).text() == "Alice（1 张）"
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


def test_recognition_result_uses_separate_fixed_fields(tmp_path, qapplication) -> None:
    """识别结论与各项数值应分栏显示，不能拼成一条长句。"""
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

    page = window._recognition_page
    assert window.result_label.text() == "识别成功"
    assert page.result_name_label.text() == "Alice"
    assert window.result_label.property("state") == "matched"
    assert page.top_score_label.text() == "0.720"
    assert page.threshold_label.text() == "0.556"
    assert page.threshold_label.property("role") == "fixed"
    assert page.score_gap_label.text() == "0.110"
    assert page.frame_label.text() == "1 / 1"
    assert page.latency_label.text() == "18 ms"
    assert page.decision_label.text() == "最高相似度达到阈值"
    metric_captions = {
        label.text() for label in page.findChildren(QLabel, "metricCaption")
    }
    assert "固定阈值" in metric_captions
    assert "可选帧" in metric_captions
    window.close()


@pytest.mark.parametrize(
    ("status", "expected_state"),
    [("matched", "matched"), ("unknown", "unknown"), ("invalid", "invalid")],
)
def test_recognition_status_exposes_visual_state(
    tmp_path, qapplication, status, expected_state
) -> None:
    """识别成功、未知人员和无效输入应提供不同的视觉状态。"""

    settings = load_settings(tmp_path)
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )
    window.on_recognition_result(
        RecognitionResult(
            status=status,
            person_id="alice" if status == "matched" else None,
            display_name="Alice" if status == "matched" else None,
            top_score=0.72 if status != "invalid" else None,
            second_score=0.61 if status != "invalid" else None,
            score_gap=0.11 if status != "invalid" else None,
            acceptance_score=0.72 if status == "matched" else None,
            acceptance_rule="score_threshold",
            latency_ms=18.0,
            reason=None if status == "matched" else "score_below_threshold",
            bbox=None,
            quality_metrics={},
            quality_warnings=(),
            frame_count=5,
            valid_frame_count=4,
        )
    )

    assert window.result_label.property("state") == expected_state
    assert window._recognition_page.result_source_label.text() == "摄像头五帧择优"
    assert window._recognition_page.frame_label.toolTip() == (
        "通过人脸检测并参与清晰度选择的帧数"
    )
    window.close()


def test_library_sample_cards_reflow_and_hide_internal_source(tmp_path, qapplication) -> None:
    """样本卡片应按宽度重排，并只显示用户可理解的样本编号。"""

    person = Person(id="alice", display_name="Alice", created_at=datetime.now(UTC))
    samples = tuple(
        FaceSample(
            id=f"sample-{index}",
            person_id=person.id,
            image_path=f"missing-{index}.jpg",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            quality_metrics={},
            created_at=datetime.now(UTC),
            source_type="file",
        )
        for index in range(5)
    )
    page = LibraryPage(tmp_path)
    page.set_people([person], {person.id: samples})

    page._relayout_sample_cards(150)
    narrow_columns = [
        page.sample_grid.getItemPosition(index)[1]
        for index in range(page.sample_grid.count())
    ]
    assert narrow_columns == [0, 0, 0, 0, 0]

    page._relayout_sample_cards(720)
    wide_columns = [
        page.sample_grid.getItemPosition(index)[1]
        for index in range(page.sample_grid.count())
    ]
    assert max(wide_columns) >= 2
    captions = page.findChildren(QLabel, "sampleCaption")
    assert [caption.text() for caption in captions] == [
        "样本 1",
        "样本 2",
        "样本 3",
        "样本 4",
        "样本 5",
    ]
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
                embedding=np.array([1.0, 0.0], dtype=np.float32),
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
    assert window.recheck_library_button.isEnabled()
    assert "异常" in window.integrity_label.text()

    state["valid"] = True
    window.recheck_library()

    assert window._face_library.snapshot().person_ids == ("alice",)
    assert window.add_person_from_files_button.isEnabled()
    assert window.append_local_button.isEnabled()
    assert "正常" in window.integrity_label.text()
    window.close()


def test_camera_recognition_collects_fixed_five_frame_window(tmp_path, qapplication, monkeypatch) -> None:
    """摄像头识别应立即启动流水线，并按80毫秒间隔提交五帧。"""

    settings = load_settings(tmp_path)
    window = MainWindow(
        settings=settings,
        face_engine=FakeFaceEngine(),  # type: ignore[arg-type]
        camera=FakeCamera(),  # type: ignore[arg-type]
    )
    window.on_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    started: list[tuple[object, float]] = []
    submitted: list[object] = []
    monkeypatch.setattr(
        window,
        "_start_camera_recognition_stream",
        lambda first_input, *, started_at: started.append((first_input, started_at)),
    )
    monkeypatch.setattr(
        window,
        "_submit_camera_recognition_frame",
        lambda image_input, final=False: submitted.append((image_input, final)),
    )

    window.compare_current_frame()
    assert len(started) == 1
    assert started[0][1] > 0.0
    assert window._capture_timer is not None
    assert window._capture_timer.interval() == CAMERA_FRAME_INTERVAL_MS
    for value in range(CAMERA_FRAME_COUNT - 1):
        window.on_frame(np.full((120, 160, 3), value + 1, dtype=np.uint8))
        window._collect_next_frame()

    assert len(submitted) == CAMERA_FRAME_COUNT - 1
    assert submitted[-1][1] is True
    assert window._capture_timer is None
    window.close()
