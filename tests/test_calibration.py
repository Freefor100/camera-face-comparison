from __future__ import annotations

from camera_face_comparison.calibration import CalibrationRecord, calibrate_thresholds


def test_calibration_prioritizes_rejecting_unknown_people() -> None:
    """如果接受了未知人员，即使已知正确数更多也不能视为更优阈值。"""

    result = calibrate_thresholds(
        records=[
            CalibrationRecord("alice", {"alice": 0.68, "bob": 0.50}),
            CalibrationRecord("bob", {"alice": 0.49, "bob": 0.69}),
            CalibrationRecord(None, {"alice": 0.65, "bob": 0.60}),
        ],
        threshold_candidates=[0.50, 0.60, 0.70],
        margin_candidates=[0.00, 0.10],
    )

    assert result.match_threshold == 0.60
    assert result.min_margin == 0.10
    assert result.unknown_false_accepts == 0
    assert result.known_correct == 2
