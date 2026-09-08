from __future__ import annotations

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment import (
    EmbeddingExperimentRecord,
    ExperimentRecord,
    aggregate_embedding_person_scores,
    evaluate_algorithm_methods,
    evaluate_experiments,
)


def test_optimized_experiment_reduces_misidentifications_on_the_same_records() -> None:
    """报告必须能够量化 Top-2 聚合和候选分差规则的改善。"""

    baseline, optimized = evaluate_experiments(
        records=[
            ExperimentRecord("alice", {"alice": [0.92, 0.90], "bob": [0.30, 0.20]}, 90),
            ExperimentRecord("bob", {"alice": [0.91, 0.10], "bob": [0.83, 0.81]}, 120),
            ExperimentRecord(None, {"alice": [0.65, 0.10], "bob": [0.64, 0.10]}, 150),
        ],
        match_threshold=0.60,
        min_score_gap=0.05,
    )

    assert baseline.known_correct == 1
    assert baseline.unknown_rejected == 0
    assert baseline.misidentifications == 2
    assert optimized.known_correct == 2
    assert optimized.unknown_rejected == 1
    assert optimized.misidentifications == 0
    assert optimized.average_latency_ms == 120.0
    assert baseline.false_positive_identification_rate == 1.0
    assert optimized.false_positive_identification_rate == 0.0
    assert baseline.false_negative_identification_rate == 0.5
    assert optimized.rank_one_identification_rate == 1.0


def test_optimized_evaluation_applies_the_probe_quality_tier_policy(tmp_path) -> None:
    """中等质量探针必须使用更严格的接收阈值进行评测。"""

    settings = load_settings(tmp_path)
    baseline, optimized = evaluate_experiments(
        records=[
            ExperimentRecord(
                "alice",
                {"alice": [0.55, 0.55], "bob": [0.20, 0.20]},
                100,
                sample_quality_scores={"alice": [0.95, 0.95], "bob": [0.95, 0.95]},
                probe_quality_tier="medium",
            )
        ],
        match_threshold=0.50,
        min_score_gap=0.05,
        top_k=2,
        quality_tiers=settings.quality_tiers,
    )

    assert baseline.known_correct == 1
    assert optimized.known_correct == 0


def test_mean_prototype_uses_normalized_person_centroid() -> None:
    """Mean Prototype 必须先平均身份样本，再计算归一化中心与探针的相似度。"""

    scores = aggregate_embedding_person_scores(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        gallery_embeddings={
            "alice": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        },
        method="mean_prototype",
        top_k=2,
    )

    assert scores["alice"] == np.float32(2**-0.5)


def test_algorithm_methods_return_four_results_on_the_same_records() -> None:
    """四种聚合方法必须复用同一批探针并分别输出可比较指标。"""

    records = [
        EmbeddingExperimentRecord(
            expected_person_id="alice",
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            gallery_embeddings={
                "alice": [
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.8, 0.6], dtype=np.float32),
                ],
                "bob": [np.array([0.0, 1.0], dtype=np.float32)],
            },
            latency_ms=12.0,
        ),
        EmbeddingExperimentRecord(
            expected_person_id=None,
            query_embedding=np.array([0.7, 0.7], dtype=np.float32),
            gallery_embeddings={
                "alice": [np.array([1.0, 0.0], dtype=np.float32)],
                "bob": [np.array([0.0, 1.0], dtype=np.float32)],
            },
            latency_ms=18.0,
        ),
    ]

    results = evaluate_algorithm_methods(
        records=records,
        methods=("single", "max", "mean_prototype", "top_k_mean"),
        match_threshold=0.75,
        min_score_gap=0.05,
        top_k=2,
    )

    assert tuple(results) == ("single", "max", "mean_prototype", "top_k_mean")
    assert all(result.total == 2 for result in results.values())
    assert results["max"].known_correct == 1
    assert results["max"].unknown_rejected == 1


def test_top_k_mean_does_not_use_quality_weights() -> None:
    """Phase 2 的 Top-K Mean 必须是普通平均，质量分只记录不参与主候选。"""

    scores = aggregate_embedding_person_scores(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        gallery_embeddings={
            "alice": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
        },
        method="top_k_mean",
        top_k=2,
        sample_quality_by_person={"alice": [0.1, 1.0]},
    )

    assert scores["alice"] == 0.5
