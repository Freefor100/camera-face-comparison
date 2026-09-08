from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.evaluation_cache import (
    EvaluationEmbeddingCache,
    embedding_extraction_id,
    write_json_atomic,
)
from camera_face_comparison.experiment import (
    AggregationMethod,
    ExperimentMetrics,
    evaluate_algorithm_methods,
    evaluate_experiments,
)
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.lfw_dataset import read_lfw_protocol
from camera_face_comparison.lfw_evaluation import (
    evaluate_lfw_protocol,
    evaluate_lfw_protocol_streaming,
)
from camera_face_comparison.runtime import backend_metadata


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
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="evaluation-only face-size floor for 250px LFW images; does not alter config.toml",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream records and retain only aggregate metrics for full-dataset protocols",
    )
    args = parser.parse_args()
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    settings = load_settings(args.data_dir)
    evaluation_settings = replace(settings, min_face_size_px=args.min_face_size)
    protocol_path = args.protocol or settings.data_dir / "datasets" / "lfw_open_set_protocol.json"
    dataset_dir = settings.data_dir / "datasets" / "lfw_funneled"
    try:
        protocol = read_lfw_protocol(protocol_path)
        cache_path = args.cache_path or settings.logs_dir / "cache" / "lfw.sqlite"
        with EvaluationEmbeddingCache(
            cache_path,
            "lfw-full-open-set-v1",
            embedding_extraction_id(evaluation_settings),
        ) as cache:
            face_engine = FaceEngine.from_local_model(evaluation_settings)
            if args.stream:
                run = evaluate_lfw_protocol_streaming(
                    dataset_dir=dataset_dir,
                    protocol=protocol,
                    settings=evaluation_settings,
                    face_engine=face_engine,
                    methods=("single", "max", "mean_prototype", "top_k_mean"),
                    on_progress=_print_progress,
                    cache=cache,
                )
            else:
                run = evaluate_lfw_protocol(
                    dataset_dir=dataset_dir,
                    protocol=protocol,
                    settings=evaluation_settings,
                    face_engine=face_engine,
                    cache=cache,
                )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    scores_output = args.scores_output or settings.logs_dir / "lfw_scores.jsonl"
    if args.stream:
        scores_output = None
        report = {
            "protocol": str(protocol_path),
            "cache_path": str(cache_path),
            "execution_backend": backend_metadata(face_engine.backend),
            "coverage": "streaming",
            "evaluation_min_face_size_px": evaluation_settings.min_face_size_px,
            "gallery_person_count": len(run.gallery_person_ids),
            "gallery_image_total": run.gallery_image_total,
            "gallery_valid_image_total": run.gallery_valid_image_total,
            "enrollment_rejections": [asdict(item) for item in run.enrollment_rejections],
            "probe_total": run.probe_total,
            "probe_valid_image_total": run.probe_valid_image_total,
            "probe_rejections": [asdict(item) for item in run.probe_rejections],
            "algorithms": {
                method: _metrics_dict(metrics) for method, metrics in run.methods.items()
            },
        }
    else:
        baseline, optimized = evaluate_experiments(
            records=run.records,
            match_threshold=settings.match_threshold,
            min_score_gap=settings.min_score_gap,
            top_k=settings.top_k,
            quality_tiers=settings.quality_tiers,
        )
        methods: tuple[AggregationMethod, ...] = (
            "single",
            "max",
            "mean_prototype",
            "top_k_mean",
        )
        algorithm_results = evaluate_algorithm_methods(
            records=run.embedding_records,
            methods=methods,
            match_threshold=settings.match_threshold,
            min_score_gap=settings.min_score_gap,
            top_k=settings.top_k,
            quality_tiers=settings.quality_tiers,
        )
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        scores_output.write_text(
            "".join(
                json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in run.records
            ),
            encoding="utf-8",
        )
        report = {
            "protocol": str(protocol_path),
            "cache_path": str(cache_path),
            "execution_backend": backend_metadata(face_engine.backend),
            "evaluation_min_face_size_px": evaluation_settings.min_face_size_px,
            "gallery_person_ids": list(run.gallery_person_ids),
            "enrollment_rejections": [asdict(item) for item in run.enrollment_rejections],
            "probe_rejections": [asdict(item) for item in run.probe_rejections],
            "baseline": _metrics_dict(baseline),
            "optimized": _metrics_dict(optimized),
            "algorithms": {
                method: _metrics_dict(algorithm_results[method]) for method in methods
            },
        }
    report_output = args.report_output or settings.logs_dir / "lfw_evaluation_report.json"
    report_output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _print_progress(stage: str, done: int, total: int) -> None:
    """按固定间隔输出全量 LFW 图片处理进度。"""

    if done == 1 or done == total or done % 100 == 0:
        print(f"LFW {stage}: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
