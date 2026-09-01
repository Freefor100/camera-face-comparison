from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment import ExperimentMetrics, evaluate_experiments
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.lfw_dataset import LfwProbe, LfwProtocol
from camera_face_comparison.lfw_evaluation import evaluate_lfw_protocol


def _load_protocol(path: Path) -> LfwProtocol:
    """读取并校验本地 LFW 开放集评测协议。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "lfw-open-set-v1":
        raise ValueError("unsupported LFW protocol file")
    enrollment = {
        str(person_id): tuple(str(relative_path) for relative_path in paths)
        for person_id, paths in payload["enrollment"].items()
    }
    probes = tuple(
        LfwProbe(
            relative_path=str(item["relative_path"]),
            expected_person_id=(
                None if item["expected_person_id"] is None else str(item["expected_person_id"])
            ),
        )
        for item in payload["probes"]
    )
    return LfwProtocol(enrollment=enrollment, probes=probes)


def _metrics_dict(metrics: ExperimentMetrics) -> dict[str, float | int | None]:
    """将评测统计对象转换为 JSON 可序列化字典。"""
    return {
        "total": metrics.total,
        "known_total": metrics.known_total,
        "known_correct": metrics.known_correct,
        "rank_one_identification_rate": metrics.rank_one_identification_rate,
        "unknown_total": metrics.unknown_total,
        "unknown_rejected": metrics.unknown_rejected,
        "unknown_rejection_rate": metrics.unknown_rejection_rate,
        "false_positive_identification_rate": metrics.false_positive_identification_rate,
        "false_negative_identification_rate": metrics.false_negative_identification_rate,
        "misidentifications": metrics.misidentifications,
        "average_latency_ms": metrics.average_latency_ms,
    }


def main() -> int:
    """加载本地模型和 LFW 协议，执行并保存开放集评测。"""
    parser = argparse.ArgumentParser(
        description="Run the local InsightFace model against a fixed LFW open-set protocol."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="evaluation-only face-size floor for 250px LFW images; does not alter config.toml",
    )
    args = parser.parse_args()
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    settings = load_settings(args.data_dir)
    evaluation_settings = replace(settings, min_face_size_px=args.min_face_size)
    protocol_path = args.protocol or settings.data_dir / "datasets" / "lfw_open_set_protocol.json"
    dataset_dir = settings.data_dir / "datasets" / "lfw_funneled"
    try:
        protocol = _load_protocol(protocol_path)
        face_engine = FaceEngine.from_local_model(evaluation_settings)
        run = evaluate_lfw_protocol(
            dataset_dir=dataset_dir,
            protocol=protocol,
            settings=evaluation_settings,
            face_engine=face_engine,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    baseline, optimized = evaluate_experiments(
        records=run.records,
        match_threshold=settings.match_threshold,
        min_margin=settings.min_margin,
        top_k=settings.top_k,
        quality_tiers=settings.quality_tiers,
    )
    scores_output = args.scores_output or settings.logs_dir / "lfw_scores.jsonl"
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.write_text(
        "".join(
            json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in run.records
        ),
        encoding="utf-8",
    )
    report = {
        "protocol": str(protocol_path),
        "evaluation_min_face_size_px": evaluation_settings.min_face_size_px,
        "gallery_person_ids": list(run.gallery_person_ids),
        "enrollment_rejections": [asdict(item) for item in run.enrollment_rejections],
        "probe_rejections": [asdict(item) for item in run.probe_rejections],
        "baseline": _metrics_dict(baseline),
        "optimized": _metrics_dict(optimized),
    }
    report_output = args.report_output or settings.logs_dir / "lfw_evaluation_report.json"
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
