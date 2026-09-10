from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment_artifacts import (
    embedding_extraction_id,
    file_sha256,
    write_json_atomic,
)
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.xqlfw import load_xqlfw_protocol, load_xqlfw_quality_scores
from camera_face_comparison.xqlfw_evaluation import evaluate_xqlfw_protocol


def main() -> int:
    """从原始缓存执行 XQLFW 官方按折验证和跨质量分析。"""

    parser = argparse.ArgumentParser(
        description="Evaluate official XQLFW folds from policy-independent raw embeddings."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--quality-scores", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-dataset-id", default="xqlfw-official-pairs-v1")
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.data_dir)
    dataset_dir = args.dataset_dir or (
        settings.data_dir
        / "datasets"
        / "xqlfw"
        / "lfw_original_imgs_min_qual0.85variant11"
    )
    pairs_path = args.pairs or settings.data_dir / "datasets" / "xqlfw" / "xqlfw_pairs.txt"
    quality_scores_path = args.quality_scores or (
        settings.data_dir / "datasets" / "xqlfw" / "xqlfw_scores.txt"
    )
    cache_path = args.cache_path or settings.logs_dir / "cache" / "xqlfw_raw.sqlite"
    report_output = args.report_output or (
        settings.data_dir / "experiments" / "phase5" / "xqlfw_evaluation_report.json"
    )
    try:
        protocol = load_xqlfw_protocol(pairs_path, dataset_dir)
        quality_scores = load_xqlfw_quality_scores(quality_scores_path)
        extraction_id = embedding_extraction_id()
        with RawEmbeddingCache(
            cache_path,
            args.cache_dataset_id,
            extraction_id,
        ) as cache:
            result = evaluate_xqlfw_protocol(
                dataset_dir=dataset_dir,
                protocol=protocol,
                quality_scores=quality_scores,
                cache=cache,
            )
        report = {
            "artifact": "xqlfw-official-cross-quality-evaluation-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "protocol": str(pairs_path),
            "protocol_sha256": file_sha256(pairs_path),
            "quality_scores": str(quality_scores_path),
            "quality_scores_sha256": file_sha256(quality_scores_path),
            "dataset_dir": str(dataset_dir),
            "cache_path": str(cache_path),
            "cache_dataset_id": args.cache_dataset_id,
            "embedding_extraction_id": extraction_id,
            "quality_policy": None,
            "threshold_protocol": "nine-fold calibration, one-fold evaluation",
            "result": asdict(result),
        }
        write_json_atomic(report_output, report)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    summary = {
        "report": str(report_output),
        "image_total": result.image_total,
        "valid_image_total": result.valid_image_total,
        "pair_total": result.pair_total,
        "valid_pair_total": result.valid_pair_total,
        "accuracy": result.accuracy,
        "fold_accuracy_mean": result.fold_accuracy_mean,
        "fold_accuracy_std": result.fold_accuracy_std,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
