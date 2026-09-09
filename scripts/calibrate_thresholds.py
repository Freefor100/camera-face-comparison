from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.calibration import available_run_ids, compare_methods
from camera_face_comparison.evaluation_cache import file_sha256, write_json_atomic


def main() -> int:
    """从无阈值分数库精确扫描六种聚合和两类开放集判定规则。"""

    parser = argparse.ArgumentParser(
        description="Calibrate aggregation and open-set rules from fixed Calibration scores."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "phase2" / "decision_scores.sqlite",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "experiments" / "phase4" / "calibration_report.json"
        ),
    )
    args = parser.parse_args()
    try:
        run_id = _select_run_id(args.scores, args.run_id)
        report = compare_methods(args.scores, run_id=run_id)
        payload = asdict(report) | {
            "artifact": "open-set-calibration-report-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_scores": str(args.scores),
            "source_scores_sha256": file_sha256(args.scores),
        }
        write_json_atomic(args.output, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _select_run_id(path: Path, requested: str | None) -> str:
    """选择调用者指定的运行；只有唯一批次时允许省略。"""

    available = available_run_ids(path)
    if requested is not None:
        if requested not in available:
            raise ValueError(f"run id is unavailable: {requested}; available={list(available)}")
        return requested
    if len(available) != 1:
        raise ValueError(
            "--run-id is required unless the score database contains exactly one run; "
            f"available={list(available)}"
        )
    return available[0]


if __name__ == "__main__":
    raise SystemExit(main())
