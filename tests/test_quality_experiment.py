from __future__ import annotations

import sqlite3

import numpy as np

from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from camera_face_comparison.quality_degradation import (
    DegradationSpec,
    apply_degradation,
    degradation_specs,
    select_primary_face,
)
from camera_face_comparison.quality_experiment import (
    build_quality_experiment_protocol,
    read_quality_experiment_protocol,
    write_quality_experiment_protocol,
)
from camera_face_comparison.quality_experiment_runner import measure_degraded_frame
from camera_face_comparison.quality_experiment_store import QualityExperimentStore


def _write_scores(path) -> None:
    """建立只包含协议构建所需字段的小型无阈值分数库。"""

    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE decision_scores (
            split TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            expected_person_id TEXT,
            method TEXT NOT NULL,
            top_k INTEGER NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO decision_scores VALUES (?, ?, ?, ?, ?, ?)",
        (
            ("calibration", "Alice/Alice_0003.jpg", "Alice", "Alice", "single", 0),
            ("calibration", "Alice/Alice_0002.jpg", "Alice", "Alice", "single", 0),
            ("calibration", "U1/U1_0001.jpg", "U1", None, "single", 0),
            ("calibration", "U2/U2_0001.jpg", "U2", None, "single", 0),
            ("calibration", "U3/U3_0001.jpg", "U3", None, "single", 0),
            ("evaluation", "Bob/Bob_0002.jpg", "Bob", "Bob", "single", 0),
            ("evaluation", "U4/U4_0001.jpg", "U4", None, "single", 0),
        ),
    )
    connection.commit()
    connection.close()


def _split_protocol() -> LfwSplitProtocol:
    """返回同时含标定和独立评估记录的小型固定协议。"""

    return LfwSplitProtocol(
        enrollment={
            "Alice": ("Alice/Alice_0001.jpg",),
            "Bob": ("Bob/Bob_0001.jpg",),
        },
        probes=(
            LfwSplitProbe("Alice/Alice_0002.jpg", "Alice", "Alice", "calibration"),
            LfwSplitProbe("Bob/Bob_0002.jpg", "Bob", "Bob", "evaluation"),
            LfwSplitProbe("U1/U1_0001.jpg", None, "U1", "calibration"),
            LfwSplitProbe("U2/U2_0001.jpg", None, "U2", "calibration"),
            LfwSplitProbe("U3/U3_0001.jpg", None, "U3", "calibration"),
            LfwSplitProbe("U4/U4_0001.jpg", None, "U4", "evaluation"),
        ),
        seed=2026,
        calibration_fraction=0.5,
        source_protocol_sha256="source-hash",
    )


def test_quality_protocol_uses_only_calibration_and_is_reproducible(tmp_path) -> None:
    """评估身份不能泄漏到质量实验，固定种子必须固定 Unknown 子集。"""

    scores_path = tmp_path / "scores.sqlite"
    _write_scores(scores_path)

    first = build_quality_experiment_protocol(
        _split_protocol(), scores_path, unknown_count=2, seed=17
    )
    second = build_quality_experiment_protocol(
        _split_protocol(), scores_path, unknown_count=2, seed=17
    )

    assert first == second
    assert [(case.person_id, case.probe_path) for case in first.known] == [
        ("Alice", "Alice/Alice_0002.jpg")
    ]
    assert first.known[0].gallery_candidates == ("Alice/Alice_0001.jpg",)
    assert len(first.unknown) == 2
    assert {case.source_identity for case in first.unknown} <= {"U1", "U2", "U3"}
    assert "Bob" not in {case.person_id for case in first.known}
    assert "U4" not in {case.source_identity for case in first.unknown}


def test_quality_protocol_json_round_trip_preserves_cases(tmp_path) -> None:
    """写入本地协议后必须能恢复完全相同的实验身份和图片路径。"""

    scores_path = tmp_path / "scores.sqlite"
    output_path = tmp_path / "protocol.json"
    _write_scores(scores_path)
    expected = build_quality_experiment_protocol(
        _split_protocol(), scores_path, unknown_count=2, seed=17
    )

    write_quality_experiment_protocol(expected, output_path)

    assert read_quality_experiment_protocol(output_path) == expected


def test_degradation_specs_are_19_unique_single_factor_conditions() -> None:
    """重复基线会浪费真实推理，条件表必须只有一个 baseline。"""

    specs = degradation_specs()

    assert len(specs) == 19
    assert len({spec.key for spec in specs}) == 19
    assert [spec.key for spec in specs].count("baseline:1") == 1


def test_degradations_apply_hand_checked_brightness_contrast_and_face_scale() -> None:
    """三个数值变换必须符合手算结果，避免实验标签和实际图片不一致。"""

    gray = np.full((100, 100, 3), 100, dtype=np.uint8)
    darker = apply_degradation(gray, DegradationSpec("brightness", 0.5), None)
    assert np.all(darker == 50)

    two_tone = np.zeros((2, 2, 3), dtype=np.uint8)
    two_tone[:, 1] = 200
    lower_contrast = apply_degradation(
        two_tone,
        DegradationSpec("contrast", 0.5),
        None,
    )
    assert np.all(lower_contrast[:, 0] == 50)
    assert np.all(lower_contrast[:, 1] == 150)

    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    scaled = apply_degradation(
        white,
        DegradationSpec("face_size", 100.0),
        (25.0, 25.0, 75.0, 75.0),
    )
    assert scaled.shape == (640, 640, 3)
    assert np.count_nonzero(np.all(scaled == 255, axis=2)) == 200 * 200


def test_select_primary_face_prefers_larger_face_and_normalizes_embedding() -> None:
    """数据集中的背景误检不能替代面积更大的标注主体。"""

    small = FaceObservation(
        bbox=(0.0, 0.0, 40.0, 40.0),
        detection_score=0.99,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        blur_variance=100.0,
        landmarks=None,
    )
    large = FaceObservation(
        bbox=(10.0, 10.0, 110.0, 110.0),
        detection_score=0.80,
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        blur_variance=120.0,
        landmarks=None,
    )

    selected = select_primary_face([small, large])

    assert selected.bbox == large.bbox
    np.testing.assert_allclose(selected.embedding, [0.6, 0.8])


def test_quality_experiment_store_recovers_measurement_and_invalidates_keys(tmp_path) -> None:
    """中断恢复必须复用同一图片和模型结果，变化的输入不能命中。"""

    path = tmp_path / "measurements.sqlite"
    first = QualityExperimentStore(path, "model-a")
    first.put_observed(
        relative_path="Alice/Alice_0001.jpg",
        file_sha256="hash-a",
        spec=DegradationSpec("gaussian_blur", 2.0),
        embedding=np.array([0.6, 0.8], dtype=np.float32),
        metrics={"brightness": 100.0, "face_size_px": 120.0},
        bbox=(1.0, 2.0, 121.0, 122.0),
        face_count=1,
        latency_ms=4.5,
    )
    first.commit()
    first.close()

    second = QualityExperimentStore(path, "model-a")
    entry = second.get(
        "Alice/Alice_0001.jpg",
        "hash-a",
        DegradationSpec("gaussian_blur", 2.0),
    )
    assert entry is not None
    assert entry.status == "observed"
    assert entry.metrics == {"brightness": 100.0, "face_size_px": 120.0}
    assert entry.bbox == (1.0, 2.0, 121.0, 122.0)
    assert entry.face_count == 1
    assert entry.embedding is not None
    np.testing.assert_allclose(entry.embedding, [0.6, 0.8])
    assert second.get(
        "Alice/Alice_0001.jpg",
        "hash-b",
        DegradationSpec("gaussian_blur", 2.0),
    ) is None
    second.close()

    changed_model = QualityExperimentStore(path, "model-b")
    assert changed_model.get(
        "Alice/Alice_0001.jpg",
        "hash-a",
        DegradationSpec("gaussian_blur", 2.0),
    ) is None
    changed_model.close()


def test_measure_degraded_frame_reuses_cache_without_calling_engine_twice(tmp_path) -> None:
    """恢复实验时已完成条件不能再次执行模型推理。"""

    class FakeEngine:
        """返回固定主脸并记录真实提取次数。"""

        def __init__(self) -> None:
            self.calls = 0

        def extract_faces(self, frame):
            """为每次未命中缓存的测量返回一个固定观察对象。"""

            self.calls += 1
            return [
                FaceObservation(
                    bbox=(0.0, 0.0, 80.0, 80.0),
                    detection_score=0.9,
                    embedding=np.array([3.0, 4.0], dtype=np.float32),
                    blur_variance=120.0,
                    landmarks=None,
                )
            ]

    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    spec = DegradationSpec("baseline", 1.0)
    engine = FakeEngine()
    with QualityExperimentStore(tmp_path / "measurements.sqlite", "model-a") as store:
        first, first_hit = measure_degraded_frame(
            relative_path="Alice/Alice_0001.jpg",
            file_sha256="hash-a",
            frame=frame,
            spec=spec,
            baseline_bbox=None,
            face_engine=engine,
            store=store,
        )
        store.commit()
        second, second_hit = measure_degraded_frame(
            relative_path="Alice/Alice_0001.jpg",
            file_sha256="hash-a",
            frame=frame,
            spec=spec,
            baseline_bbox=None,
            face_engine=engine,
            store=store,
        )

    assert first.status == "observed"
    assert second.status == "observed"
    assert first_hit is False
    assert second_hit is True
    assert engine.calls == 1
