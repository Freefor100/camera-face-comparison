from __future__ import annotations

import pytest

from camera_face_comparison.joint_calibration import (
    JointScoreRow,
    calibrate_joint_rule,
    evaluate_joint_operating_point,
    select_joint_operating_point,
)


def test_joint_calibration_models_threshold_gap_and_both_as_distinct_rules() -> None:
    """三个简单接收规则必须分别选参，不能再用负一阈值模拟仅分差。"""

    rows = (
        JointScoreRow("natural", "alice", "alice", True, (0.70, 0.40)),
        JointScoreRow("natural", "unknown-a", None, False, (0.80, 0.75)),
        JointScoreRow("cross", "bob", "bob", True, (0.65, 0.30)),
        JointScoreRow("cross", "unknown-b", None, False, (0.75, 0.70)),
    )
    protocol_known = {"natural": 1, "cross": 1}

    threshold = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_threshold",
        target_fpir=0.0,
        protocol_known_totals=protocol_known,
    )
    gap = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_gap",
        target_fpir=0.0,
        protocol_known_totals=protocol_known,
    )
    combined = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_and_gap",
        target_fpir=0.0,
        protocol_known_totals=protocol_known,
    )

    assert threshold.rule == "score_threshold"
    assert threshold.minimum_score is not None
    assert threshold.minimum_gap is None
    assert gap.rule == "score_gap"
    assert gap.minimum_score is None
    assert gap.minimum_gap == pytest.approx(0.30)
    assert combined.rule == "score_and_gap"
    assert combined.minimum_score is not None
    assert combined.minimum_gap is not None


def test_joint_calibration_uses_nac_probability_and_all_scenarios() -> None:
    """NAC 必须共同约束每个场景，而不是把多域 Unknown 合并后掩盖超标场景。"""

    rows = (
        JointScoreRow("natural", "alice", "alice", True, (0.9, 0.1, 0.0)),
        JointScoreRow("natural", "unknown-a", None, False, (0.8, 0.79, 0.0)),
        JointScoreRow("cross", "bob", "bob", True, (0.7, 0.0, -0.1)),
        JointScoreRow("cross", "unknown-b", None, False, (0.6, 0.59, 0.58)),
    )

    point = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="nac",
        nac_neighbors=3,
        target_fpir=0.0,
        protocol_known_totals={"natural": 1, "cross": 1},
    )

    assert point.rule == "nac"
    assert point.nac_neighbors == 3
    assert point.minimum_probability is not None
    assert all(metric.fpir_valid == 0.0 for metric in point.scenarios)


def test_joint_selection_requires_one_point_of_worst_domain_gain_for_complex_rule() -> None:
    """复杂规则收益不足一个百分点时必须保留简单 Mean Prototype 阈值基线。"""

    rows = (
        JointScoreRow("natural", "alice", "alice", True, (0.90, 0.10)),
        JointScoreRow("natural", "bob", "bob", True, (0.89, 0.10)),
        JointScoreRow("natural", "unknown", None, False, (0.20, 0.10)),
    )
    baseline = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_threshold",
        target_fpir=0.0,
        protocol_known_totals={"natural": 2},
    )
    complex_point = calibrate_joint_rule(
        rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_gap",
        target_fpir=0.0,
        protocol_known_totals={"natural": 2},
    )

    selected = select_joint_operating_point((baseline, complex_point), minimum_complex_gain=0.01)

    assert selected.rule == "score_threshold"


def test_joint_evaluation_reuses_frozen_parameters_without_recalibration() -> None:
    """Evaluation 必须原样使用 Calibration 参数，即使重新标定能接受更多 Known。"""

    calibration_rows = (
        JointScoreRow("natural", "alice", "alice", True, (0.80, 0.10)),
        JointScoreRow("natural", "unknown", None, False, (0.60, 0.20)),
    )
    selected = calibrate_joint_rule(
        calibration_rows,
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_threshold",
        target_fpir=0.0,
        protocol_known_totals={"natural": 1},
    )
    evaluation = evaluate_joint_operating_point(
        (
            JointScoreRow("natural", "bob", "bob", True, (0.70, 0.20)),
            JointScoreRow("natural", "unknown-b", None, False, (0.65, 0.10)),
        ),
        selected=selected,
        protocol_known_totals={"natural": 1},
    )

    assert evaluation.minimum_score == selected.minimum_score == 0.80
    assert evaluation.scenarios[0].known_true_accepts == 0
    assert evaluation.scenarios[0].unknown_false_accepts == 0
