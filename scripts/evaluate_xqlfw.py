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
    quality_policy_id,
    write_json_atomic,
)
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.runtime import backend_metadata
from camera_face_comparison.xqlfw import load_xqlfw_protocol
from camera_face_comparison.xqlfw_evaluation import evaluate_xqlfw_protocol


def main() -> int:
    """使用本地模型执行 XQLFW 全部官方验证对并保存报告。"""

    parser = argparse.ArgumentParser(description="Evaluate all official XQLFW verification pairs.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "datasets"
        / "xqlfw"
        / "lfw_original_imgs_min_qual0.85variant11",
    )
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="XQLFW 评测专用人脸尺寸下限，不修改应用 config.toml",
    )
    args = parser.parse_args()
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    settings = load_settings(args.data_dir)
    pairs_path = args.pairs or args.data_dir / "datasets" / "xqlfw" / "xqlfw_pairs.txt"
    try:
        protocol = load_xqlfw_protocol(pairs_path, args.dataset_dir)
        threshold = settings.match_threshold if args.threshold is None else args.threshold
        evaluation_settings = replace(settings, min_face_size_px=args.min_face_size)
        cache_path = args.cache_path or settings.logs_dir / "cache" / "xqlfw.sqlite"
        with EvaluationEmbeddingCache(
            cache_path,
            "xqlfw-verification-v1",
            embedding_extraction_id(evaluation_settings),
            quality_policy_id(evaluation_settings),
        ) as cache:
            face_engine = FaceEngine.from_local_model(evaluation_settings)
            result = evaluate_xqlfw_protocol(
                dataset_dir=args.dataset_dir,
                protocol=protocol,
                settings=evaluation_settings,
                face_engine=face_engine,
                threshold=threshold,
                on_image=_print_progress,
                cache=cache,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    report = asdict(result)
    report["accuracy"] = result.accuracy
    report["rejected_images"] = [asdict(item) for item in result.rejected_images]
    report["protocol"] = str(pairs_path)
    report["dataset_dir"] = str(args.dataset_dir)
    report["cache_path"] = str(cache_path)
    report["execution_backend"] = backend_metadata(face_engine.backend)
    report["evaluation_min_face_size_px"] = args.min_face_size
    report_output = args.report_output or settings.logs_dir / "xqlfw_evaluation_report.json"
    write_json_atomic(report_output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _print_progress(done: int, total: int) -> None:
    """按固定间隔输出全量 XQLFW 图片处理进度。"""

    if done == 1 or done == total or done % 100 == 0:
        print(f"XQLFW images: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
