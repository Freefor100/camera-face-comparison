from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from camera_face_comparison.config import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_script_updates_only_the_requested_quality_tier(tmp_path) -> None:
    """中等质量探针的标定记录只能更新中等质量接收策略。"""

    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "\n".join(
            (
                json.dumps({"expected_person_id": "alice", "person_scores": {"alice": 0.68}}),
                json.dumps({"expected_person_id": None, "person_scores": {"alice": 0.62}}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    high_before = load_settings(data_dir).quality_tiers["high"]

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_thresholds.py",
            "--data-dir",
            str(data_dir),
            "--scores",
            str(scores_path),
            "--quality-tier",
            "medium",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    reloaded = load_settings(data_dir)
    assert reloaded.quality_tiers["high"] == high_before
    assert reloaded.quality_tiers["medium"].match_threshold == 0.68
