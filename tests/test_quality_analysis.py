from __future__ import annotations

from dataclasses import replace

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.quality_analysis import (
    QualityUtilityRecord,
    analyze_probe_quality,
    error_vs_reject,
)
from camera_face_comparison.quality_experiment_store import QualityExperimentEntry


def _entry(embedding: tuple[float, float]) -> QualityExperimentEntry:
    """构造通过当前质量门的手工 embedding 记录。"""

    return QualityExperimentEntry(
        status="observed",
        embedding=np.asarray(embedding, dtype=np.float32),
        metrics={
            "detection_score": 0.99,
            "face_size_px": 160.0,
            "blur_variance": 200.0,
            "brightness": 100.0,
            "contrast": 50.0,
        },
        bbox=(0.0, 0.0, 160.0, 160.0),
        face_count=1,
        latency_ms=10.0,
        reason=None,
    )


def test_probe_quality_analysis_reports_rank1_and_unknown_score_distribution(tmp_path) -> None:
    """Probe 退化分析必须区分 Known 排名正确性和 Unknown 连续分数。"""

    settings = replace(
        load_settings(tmp_path),
        min_contrast=0.0,
        medium_quality_score=0.0,
    )

    summary = analyze_probe_quality(
        gallery_entries={"A": _entry((1.0, 0.0)), "B": _entry((0.0, 1.0))},
        known_entries={"A": _entry((1.0, 0.0)), "B": _entry((1.0, 0.0))},
        unknown_entries=(_entry((0.6, 0.8)),),
        settings=settings,
    )

    assert summary.known.rank1_correct == 1
    assert summary.known.rank1_rate == 0.5
    assert summary.known.genuine_similarity.mean == 0.5
    assert summary.unknown.top_score.p50 == 0.8
    assert summary.unknown.top_score.maximum == 0.8


def test_error_vs_reject_uses_clean_retention_threshold_and_counts_tradeoff() -> None:
    """质量门候选必须同时报告移除错误和误拒正确输入，不能只看错误下降。"""

    records = (
        QualityUtilityRecord(0.10, False),
        QualityUtilityRecord(0.20, True),
        QualityUtilityRecord(0.80, True),
        QualityUtilityRecord(0.90, True),
    )

    point = error_vs_reject(
        records,
        clean_values=(0.20, 0.80, 0.90),
        retention_targets=(2 / 3,),
    )[0]

    assert point.threshold == 0.80
    assert point.accepted_total == 2
    assert point.rejected_total == 2
    assert point.errors_before == 1
    assert point.errors_after == 0
    assert point.errors_removed == 1
    assert point.correct_rejected == 1
