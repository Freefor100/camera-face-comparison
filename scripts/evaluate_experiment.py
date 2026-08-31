from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment import ExperimentRecord, evaluate_experiments


def _load_records(path: Path) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            expected = payload["expected_person_id"]
            sample_scores = {
                str(person_id): [float(score) for score in scores]
                for person_id, scores in payload["sample_scores"].items()
            }
            latency_ms = float(payload["latency_ms"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid experiment record at line {line_number}") from error
        records.append(
            ExperimentRecord(
                expected_person_id=None if expected is None else str(expected),
                sample_scores=sample_scores,
                latency_ms=latency_ms,
            )
        )
    return records


def _metrics_to_dict(metrics) -> dict[str, float | int | None]:
    return {
        "total": metrics.total,
        "known_total": metrics.known_total,
        "known_correct": metrics.known_correct,
        "known_accuracy": metrics.known_accuracy,
        "unknown_total": metrics.unknown_total,
        "unknown_rejected": metrics.unknown_rejected,
        "unknown_rejection_rate": metrics.unknown_rejection_rate,
        "misidentifications": metrics.misidentifications,
        "average_latency_ms": metrics.average_latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized recognition metrics on the same probes."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.scores.is_file():
        print(f"Experiment input does not exist: {args.scores}", file=sys.stderr)
        return 1

    settings = load_settings(args.data_dir)
    records = _load_records(args.scores)
    if not records:
        print("Experiment input is empty.", file=sys.stderr)
        return 1
    baseline, optimized = evaluate_experiments(
        records=records,
        match_threshold=settings.match_threshold,
        min_margin=settings.min_margin,
    )
    report = {
        "parameters": {
            "match_threshold": settings.match_threshold,
            "min_margin": settings.min_margin,
        },
        "baseline": _metrics_to_dict(baseline),
        "optimized": _metrics_to_dict(optimized),
    }
    output = args.output or settings.logs_dir / "optimization_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
