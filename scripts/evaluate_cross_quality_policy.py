from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.cross_quality_evaluation import (
    RejectedProbe,
    bootstrap_identity_confidence_intervals,
)
from camera_face_comparison.cross_quality_experiment import (
    CROSS_QUALITY_SCENARIOS,
    load_joint_score_rows,
)
from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.joint_calibration import (
    JointOperatingPoint,
    ScenarioOperatingMetrics,
    evaluate_joint_operating_point,
)


def main() -> int:
    """把 Calibration 冻结的工作点一次性应用到独立 Evaluation。"""

    parser = argparse.ArgumentParser(
        description="Evaluate frozen cross-quality policies once on the Evaluation split."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5b"
        / "decision_scores.sqlite",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5b"
        / "calibration_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "experiments"
        / "phase5b"
        / "evaluation_report.json",
    )
    parser.add_argument("--run-id", default="lfw-xqlfw-open-set-v1")
    parser.add_argument("--targets", type=float, nargs="+", default=(0.01, 0.003, 0.10))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        if calibration.get("artifact") != "cross-quality-joint-calibration-v1":
            raise ValueError("unsupported cross-quality calibration report")
        if calibration.get("source_split") != "calibration":
            raise ValueError("Evaluation requires a Calibration-only source report")
        scenario_names = tuple(item.name for item in CROSS_QUALITY_SCENARIOS)
        evaluated = []
        for target_index, target in enumerate(args.targets):
            selected = _selected_point(calibration, target)
            rows, known_totals = load_joint_score_rows(
                args.scores,
                run_id=args.run_id,
                split="evaluation",
                scenario_names=scenario_names,
                method=selected.aggregation_method,
                top_k=selected.top_k,
            )
            result = evaluate_joint_operating_point(
                rows,
                selected=selected,
                protocol_known_totals=known_totals,
            )
            confidence = {}
            for scenario_index, scenario in enumerate(scenario_names):
                selected_rows = tuple(row for row in rows if row.scenario == scenario)
                rejected = _load_rejected_probes(
                    args.scores,
                    run_id=args.run_id,
                    scenario=scenario,
                    split="evaluation",
                )
                confidence[scenario] = bootstrap_identity_confidence_intervals(
                    selected_rows,
                    rejected_probes=rejected,
                    selected=selected,
                    repeats=args.bootstrap_repeats,
                    seed=args.seed + target_index * 100 + scenario_index,
                )
            evaluated.append(
                {
                    "target_fpir": target,
                    "selected_calibration_point": _point_parameters(selected),
                    "metrics": asdict(result),
                    "identity_bootstrap_95_percent": confidence,
                }
            )
        payload = {
            "artifact": "cross-quality-fixed-policy-evaluation-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_split": "evaluation",
            "selection_performed": False,
            "evaluation_execution_count": 1,
            "scores": str(args.scores),
            "scores_sha256": file_sha256(args.scores),
            "calibration_report": str(args.calibration_report),
            "calibration_report_sha256": file_sha256(args.calibration_report),
            "run_id": args.run_id,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.seed,
            "results": evaluated,
        }
        write_json_atomic(args.output, payload)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "results": evaluated}, ensure_ascii=False))
    return 0


def _selected_point(payload: dict[str, object], target_fpir: float) -> JointOperatingPoint:
    """按数值目标读取一个已冻结工作点。"""

    values = payload["selected_by_target"]
    if not isinstance(values, dict):
        raise TypeError("selected_by_target must be an object")
    raw = next(
        (value for key, value in values.items() if abs(float(key) - target_fpir) <= 1e-12),
        None,
    )
    if raw is None:
        raise ValueError(f"calibration target is unavailable: {target_fpir}")
    if not isinstance(raw, dict):
        raise TypeError("selected calibration point must be an object")
    point = dict(raw)
    scenarios = tuple(ScenarioOperatingMetrics(**item) for item in point.pop("scenarios", []))
    return JointOperatingPoint(**point, scenarios=scenarios)


def _load_rejected_probes(
    path: Path,
    *,
    run_id: str,
    scenario: str,
    split: str,
) -> tuple[RejectedProbe, ...]:
    """读取未产生 embedding 的 Probe，供端到端分母和分组抽样使用。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT source_identity, expected_person_id
            FROM rejections
            WHERE run_id = ? AND scenario = ? AND role = 'probe' AND split = ?
            ORDER BY relative_path
            """,
            (run_id, scenario, split),
        ).fetchall()
    finally:
        connection.close()
    return tuple(
        RejectedProbe(
            source_identity=str(source_identity),
            expected_person_id=None if expected is None else str(expected),
        )
        for source_identity, expected in rows
    )


def _point_parameters(point: JointOperatingPoint) -> dict[str, object]:
    """只保存部署所需参数，不重复嵌入 Calibration 指标。"""

    return {
        "aggregation_method": point.aggregation_method,
        "top_k": point.top_k,
        "rule": point.rule,
        "minimum_score": point.minimum_score,
        "minimum_gap": point.minimum_gap,
        "minimum_probability": point.minimum_probability,
        "nac_neighbors": point.nac_neighbors,
    }


if __name__ == "__main__":
    raise SystemExit(main())
