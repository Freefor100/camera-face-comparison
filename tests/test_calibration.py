from __future__ import annotations

from camera_face_comparison.calibration import DecisionScoreRow, calibrate_method


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
    assert with_gap.match_threshold == 0.0
    assert with_gap.min_score_gap == 0.30
    assert with_gap.known_true_accepts == 1


def test_scan_marks_unsatisfied_target_when_bounded_scores_cannot_reject_unknown() -> None:
    """若分数和分差都等于一，合法闭区间内不能伪造零 FPIR 工作点。"""

    rows = (
        DecisionScoreRow("alice", 0.90, 0.50, True),
        DecisionScoreRow(None, 1.00, 1.00, False),
    )

    point = calibrate_method(rows, target_fpir=0.0, use_score_gap=True)

    assert not point.meets_target
    assert point.unknown_false_accepts == 1
    assert point.fpir == 1.0
