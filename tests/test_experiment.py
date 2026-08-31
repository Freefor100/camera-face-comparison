from __future__ import annotations

from camera_face_comparison.experiment import ExperimentRecord, evaluate_experiments


def test_optimized_experiment_reduces_misidentifications_on_the_same_records() -> None:
    """The report must make the Top-2 and margin improvement measurable."""

    baseline, optimized = evaluate_experiments(
        records=[
            ExperimentRecord("alice", {"alice": [0.92, 0.90], "bob": [0.30, 0.20]}, 90),
            ExperimentRecord("bob", {"alice": [0.91, 0.10], "bob": [0.83, 0.81]}, 120),
            ExperimentRecord(None, {"alice": [0.65, 0.10], "bob": [0.64, 0.10]}, 150),
        ],
        match_threshold=0.60,
        min_margin=0.05,
    )

    assert baseline.known_correct == 1
    assert baseline.unknown_rejected == 0
    assert baseline.misidentifications == 2
    assert optimized.known_correct == 2
    assert optimized.unknown_rejected == 1
    assert optimized.misidentifications == 0
    assert optimized.average_latency_ms == 120.0
