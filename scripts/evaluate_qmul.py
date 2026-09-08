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
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.lfw_evaluation import evaluate_lfw_protocol_streaming
from camera_face_comparison.qmul_survface import build_qmul_protocol
from camera_face_comparison.runtime import backend_metadata


def main() -> int:
    """按 QMUL 官方开放集识别协议执行全量流式评测并保存汇总。"""

    parser = argparse.ArgumentParser(description="Evaluate the QMUL-SurvFace open-set protocol.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "qmul-survface" / "QMUL-SurvFace",
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=112,
        help="QMUL 评测专用人脸尺寸下限，不修改应用 config.toml",
    )
    args = parser.parse_args()
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    settings = load_settings(args.data_dir)
    try:
        protocol = build_qmul_protocol(args.dataset_root)
        evaluation_settings = replace(settings, min_face_size_px=args.min_face_size)
        cache_path = args.cache_path or settings.logs_dir / "cache" / "qmul_survface.sqlite"
        with EvaluationEmbeddingCache(
            cache_path,
            "qmul-survface-identification-v1",
            embedding_extraction_id(evaluation_settings),
        ) as cache:
            face_engine = FaceEngine.from_local_model(evaluation_settings)
            run = evaluate_lfw_protocol_streaming(
                dataset_dir=args.dataset_root / "Face_Identification_Test_Set",
                protocol=protocol,
                settings=evaluation_settings,
                face_engine=face_engine,
                methods=("single", "max", "mean_prototype", "top_k_mean"),
                on_progress=_print_progress,
                cache=cache,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    report = {
        "protocol": "qmul-survface-open-set-v1",
        "dataset_root": str(args.dataset_root),
        "evaluation_min_face_size_px": args.min_face_size,
        "cache_path": str(cache_path),
        "execution_backend": backend_metadata(face_engine.backend),
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
    report_output = args.report_output or settings.logs_dir / "qmul_survface_evaluation_report.json"
    write_json_atomic(report_output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _metrics_dict(metrics) -> dict[str, float | int | None]:
    """把开放集评测指标转换为 JSON 字段。"""

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


def _print_progress(stage: str, done: int, total: int) -> None:
    """按固定间隔输出 QMUL 全量处理进度。"""

    if done == 1 or done == total or done % 1000 == 0:
        print(f"QMUL {stage}: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
