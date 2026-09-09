from __future__ import annotations

import sqlite3

from camera_face_comparison.lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from camera_face_comparison.quality_experiment import (
    build_quality_experiment_protocol,
    read_quality_experiment_protocol,
    write_quality_experiment_protocol,
)


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
