from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.face_evaluation.cross_quality_experiment import (
    load_joint_score_rows,
)
from experiments.face_evaluation.decision_scores import METHOD_VARIANTS
from experiments.face_evaluation.experiment_artifacts import file_sha256, write_json_atomic
from experiments.face_evaluation.joint_calibration import (
    JointOperatingPoint,
    calibrate_joint_rule,
    select_joint_operating_point,
)


def main() -> int:
    """只读取 Calibration，联合比较聚合、三类简单规则和 NAC。"""

    parser = argparse.ArgumentParser(
        description="Calibrate one open-set policy across natural and cross-quality scenarios."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "phase5b" / "decision_scores.sqlite",
    )
    parser.add_argument("--run-id", default="lfw-xqlfw-open-set-v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "phase5b" / "calibration_report.json",
    )
    parser.add_argument("--targets", type=float, nargs="+", default=(0.01, 0.003, 0.10))
    args = parser.parse_args()
    try:
        scenarios = _selection_scenarios(args.scores, args.run_id)
        points: list[JointOperatingPoint] = []
        for method, top_k in METHOD_VARIANTS:
            rows, known_totals = load_joint_score_rows(
                args.scores,
                run_id=args.run_id,
                split="calibration",
                scenario_names=scenarios,
                method=method,
                top_k=top_k,
            )
            for target in args.targets:
                for rule in ("score_threshold", "score_gap", "score_and_gap"):
                    print(
                        f"calibrate {method}:{top_k} {rule} target={target}",
                        file=sys.stderr,
                        flush=True,
                    )
                    points.append(
                        calibrate_joint_rule(
                            rows,
                            aggregation_method=method,
                            top_k=top_k,
                            rule=rule,
                            target_fpir=target,
                            protocol_known_totals=known_totals,
                        )
                    )
                if method == "mean_prototype":
                    for neighbors in (2, 4, 8, 16, 32):
                        print(
                            f"calibrate NAC:{neighbors} target={target}",
                            file=sys.stderr,
                            flush=True,
                        )
                        points.append(
                            calibrate_joint_rule(
                                rows,
                                aggregation_method=method,
                                top_k=top_k,
                                rule="nac",
                                nac_neighbors=neighbors,
                                target_fpir=target,
                                protocol_known_totals=known_totals,
                            )
                        )
        selected = {
            str(target): asdict(
                select_joint_operating_point(
                    tuple(point for point in points if point.target_fpir == target),
                    minimum_complex_gain=0.01,
                )
            )
            for target in args.targets
        }
        payload = {
            "artifact": "cross-quality-joint-calibration-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_split": "calibration",
            "scores": str(args.scores),
            "scores_sha256": file_sha256(args.scores),
            "run_id": args.run_id,
            "selection_scenarios": scenarios,
            "targets": list(args.targets),
            "selection_priority": [
                "each_scenario_fpir_valid",
                "worst_tpir_e2e",
                "average_tpir_e2e",
                "minimum_complex_gain_0.01",
            ],
            "operating_points": [asdict(point) for point in points],
            "selected_by_target": selected,
            "evaluation_labels_read": False,
        }
        write_json_atomic(args.output, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(selected, ensure_ascii=False))
    return 0


def _selection_scenarios(path: Path, run_id: str) -> tuple[str, ...]:
    """从分数库读取明确标为共同选参的四个场景。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT name
            FROM scenarios
            WHERE run_id = ? AND participates_in_selection = 1
            ORDER BY name
            """,
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    scenarios = tuple(str(row[0]) for row in rows)
    if len(scenarios) != 4:
        raise ValueError(f"expected four selection scenarios, got {len(scenarios)}")
    return scenarios


if __name__ == "__main__":
    raise SystemExit(main())
