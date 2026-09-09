from __future__ import annotations

from camera_face_comparison.calibration import (
    CalibrationReport,
    DecisionScoreRow,
    OperatingPoint,
    VariantCalibration,
    calibrate_method,
    select_best_operating_point,
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
