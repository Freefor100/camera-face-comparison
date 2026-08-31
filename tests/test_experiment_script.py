from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_script_writes_a_same_dataset_comparison(tmp_path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    output_path = tmp_path / "report.json"
    scores_path.write_text(
        json.dumps(
            {
                "expected_person_id": "alice",
                "sample_scores": {"alice": [0.92, 0.90], "bob": [0.30, 0.20]},
                "latency_ms": 90,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_experiment.py",
            "--data-dir",
            str(tmp_path / "data"),
            "--scores",
            str(scores_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["baseline"]["known_correct"] == 1
    assert report["optimized"]["known_correct"] == 1
    assert "optimized" in completed.stdout
