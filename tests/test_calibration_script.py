from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_script_reads_only_calibration_split(tmp_path) -> None:
    """方法选择报告不能在最终评估前读取 Evaluation 标签决定参数。"""

    scores_path = tmp_path / "scores.sqlite"
    connection = sqlite3.connect(scores_path)
    connection.execute(
        """
        CREATE TABLE decision_scores (
            run_id TEXT NOT NULL,
            split TEXT NOT NULL,
            expected_person_id TEXT,
            method TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            top_score REAL NOT NULL,
            score_gap REAL NOT NULL,
            top_is_correct INTEGER NOT NULL
        )
        """
    )
    rows = [
        ("run-a", "calibration", "alice", "single", 0, 0.8, 0.2, 1),
        ("run-a", "calibration", None, "single", 0, 0.4, 0.1, 0),
        ("run-a", "evaluation", None, "single", 0, 1.0, 1.0, 0),
    ]
    connection.executemany("INSERT INTO decision_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    output_path = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_thresholds.py",
            "--scores",
            str(scores_path),
            "--run-id",
            "run-a",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["source_split"] == "calibration"
    assert report["variants"][0]["known_total"] == 1
    assert report["variants"][0]["unknown_total"] == 1
