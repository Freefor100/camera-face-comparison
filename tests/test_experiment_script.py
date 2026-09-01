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


def test_experiment_script_replays_quality_aware_records(tmp_path) -> None:
    """Exported LFW records must retain enough information to reproduce the optimized rule."""

    scores_path = tmp_path / "scores.jsonl"
    output_path = tmp_path / "report.json"
    scores_path.write_text(
        json.dumps(
            {
                "expected_person_id": "alice",
                "sample_scores": {"alice": [0.55, 0.55], "bob": [0.20, 0.20]},
                "sample_quality_scores": {"alice": [0.95, 0.95], "bob": [0.95, 0.95]},
                "probe_quality_tier": "medium",
                "latency_ms": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
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
    assert report["optimized"]["known_correct"] == 0
