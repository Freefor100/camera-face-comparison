from __future__ import annotations

import pytest

from camera_face_comparison.calibration import (
    CalibrationReport,
    DecisionScoreRow,
    OperatingPoint,
    VariantCalibration,
    calibrate_method,
    select_best_operating_point,
)
from camera_face_comparison.joint_calibration import (
    JointScoreRow,
    calibrate_joint_rule,
    evaluate_joint_operating_point,
    select_joint_operating_point,
)


def test_exact_threshold_scan_uses_observed_breakpoints() -> None:
    """仅阈值规则应找到满足零误接收且保留全部 Known 的最小实际断点。"""

    rows = (
        DecisionScoreRow("alice", 0.80, 0.20, True),
        DecisionScoreRow("bob", 0.70, 0.10, True),
        DecisionScoreRow(None, 0.60, 0.05, False),
        DecisionScoreRow(None, 0.40, 0.02, False),
    )

    point = calibrate_method(rows, target_fpir=0.0, use_score_gap=False)

    assert point.meets_target
    assert point.match_threshold == 0.70
    assert point.min_score_gap == 0.0
    assert point.unknown_false_accepts == 0
    assert point.known_true_accepts == 2
    assert point.tpir == 1.0


def test_score_gap_can_reject_a_close_unknown_without_raising_match_threshold() -> None:
    """候选分差只有在同一 FPIR 下提高 TPIR 时才算有实际收益。"""

    rows = (
        DecisionScoreRow("alice", 0.70, 0.30, True),
        DecisionScoreRow(None, 0.80, 0.05, False),
    )

    threshold_only = calibrate_method(rows, target_fpir=0.0, use_score_gap=False)
    with_gap = calibrate_method(rows, target_fpir=0.0, use_score_gap=True)

    assert threshold_only.known_true_accepts == 0
    assert with_gap.meets_target
    assert with_gap.match_threshold == -1.0
    assert with_gap.min_score_gap == 0.30
    assert with_gap.known_true_accepts == 1


def test_scan_marks_unsatisfied_target_when_bounded_scores_cannot_reject_unknown() -> None:
    """若分数和分差都达到合法上限，扫描器不能伪造零 FPIR 工作点。"""

    rows = (
        DecisionScoreRow("alice", 0.90, 0.50, True),
        DecisionScoreRow(None, 1.00, 2.00, False),
    )

    point = calibrate_method(rows, target_fpir=0.0, use_score_gap=True)

    assert not point.meets_target
    assert point.unknown_false_accepts == 1
    assert point.fpir == 1.0


def test_scan_accepts_negative_cosine_top_scores() -> None:
    """余弦相似度允许为负数，扫描器不能误当成非法概率值。"""

    rows = (
        DecisionScoreRow("alice", -0.10, 0.30, True),
        DecisionScoreRow(None, -0.40, 0.05, False),
    )

    point = calibrate_method(rows, target_fpir=0.0, use_score_gap=False)

    assert point.match_threshold == -0.10
    assert point.known_true_accepts == 1
    assert point.unknown_false_accepts == 0


def test_global_selection_prefers_tpir_then_simpler_rule() -> None:
    """全局选择应先最大化 TPIR，同分时优先不启用候选分差。"""

    def point(*, tpir: float, use_score_gap: bool) -> OperatingPoint:
        """构造只保留本测试关心字段的合法工作点。"""

        return OperatingPoint(
            target_fpir=0.003,
            use_score_gap=use_score_gap,
            match_threshold=0.6,
            min_score_gap=0.2 if use_score_gap else 0.0,
            meets_target=True,
            known_total=100,
            rank1_correct=99,
            known_true_accepts=int(tpir * 100),
            known_wrong_accepts=0,
            unknown_total=1000,
            unknown_false_accepts=3,
            fpir=0.003,
            tpir=tpir,
            fnir=1.0 - tpir,
        )

    report = CalibrationReport(
        run_id="run-a",
        source_split="calibration",
        targets=(0.003,),
        variants=(
            VariantCalibration("max", 0, 100, 1000, 99, 0.99, (point(tpir=0.8, use_score_gap=False),)),
            VariantCalibration(
                "mean_prototype",
                0,
                100,
                1000,
                99,
                0.99,
                (
                    point(tpir=0.9, use_score_gap=True),
                    point(tpir=0.9, use_score_gap=False),
                ),
            ),
        ),
    )

    selected = select_best_operating_point(report, target_fpir=0.003)

    assert selected.method == "mean_prototype"
    assert selected.top_k == 0
    assert not selected.operating_point.use_score_gap


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
