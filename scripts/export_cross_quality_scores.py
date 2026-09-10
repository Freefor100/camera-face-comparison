from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.cross_quality_experiment import export_cross_quality_scores
from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.lfw_dataset import read_lfw_split_protocol
from camera_face_comparison.raw_embedding_cache import (
    RawEmbeddingCache,
    available_raw_extraction_ids,
)


def main() -> int:
    """从自然 LFW 与 XQLFW 原始缓存导出六场景无阈值候选排序。"""

    parser = argparse.ArgumentParser(
        description="Export threshold-free open-set rankings for six LFW/XQLFW scenarios."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--natural-cache", type=Path)
    parser.add_argument("--xqlfw-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extraction-id")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    protocol_path = args.protocol or data_dir / "experiments" / "phase4" / "protocol.json"
    natural_cache_path = args.natural_cache or data_dir / "logs" / "cache" / "lfw_raw.sqlite"
    xqlfw_cache_path = (
        args.xqlfw_cache or data_dir / "logs" / "cache" / "xqlfw_full_cuda.sqlite"
    )
    output_dir = args.output_dir or data_dir / "experiments" / "phase5b"
    output_path = output_dir / "decision_scores.sqlite"
    manifest_path = output_dir / "manifest.json"
    natural_dir = data_dir / "datasets" / "lfw_funneled"
    xqlfw_dir = (
        data_dir
        / "datasets"
        / "xqlfw"
        / "lfw_original_imgs_min_qual0.85variant11"
    )
    try:
        protocol = read_lfw_split_protocol(protocol_path)
        extraction_id = args.extraction_id or _shared_extraction_id(
            natural_cache_path, xqlfw_cache_path
        )
        run_id = "lfw-xqlfw-open-set-v1"
        base_manifest = {
            "artifact": "lfw-xqlfw-threshold-free-rankings-v1",
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "natural_cache": str(natural_cache_path),
            "xqlfw_cache": str(xqlfw_cache_path),
            "embedding_extraction_id": extraction_id,
            "run_id": run_id,
            "decision_policy": None,
        }
        write_json_atomic(manifest_path, base_manifest)
        with (
            RawEmbeddingCache(natural_cache_path, "lfw-natural-v1", extraction_id) as natural_cache,
            RawEmbeddingCache(
                xqlfw_cache_path, "xqlfw-official-pairs-v1", extraction_id
            ) as xqlfw_cache,
        ):
            summary = export_cross_quality_scores(
                natural_dataset_dir=natural_dir,
                xqlfw_dataset_dir=xqlfw_dir,
                protocol=protocol,
                natural_cache=natural_cache,
                xqlfw_cache=xqlfw_cache,
                output_path=output_path,
                run_id=run_id,
                protocol_sha256=base_manifest["protocol_sha256"],
                embedding_extraction_id=extraction_id,
                batch_size=args.batch_size,
                on_progress=_print_progress,
            )
        manifest = base_manifest | {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "summary": asdict(summary),
        }
        write_json_atomic(manifest_path, manifest)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _shared_extraction_id(natural_cache: Path, xqlfw_cache: Path) -> str:
    """要求两个域各自只有一个且相同的模型提取标识。"""

    natural_ids = available_raw_extraction_ids(natural_cache, "lfw-natural-v1")
    xqlfw_ids = available_raw_extraction_ids(xqlfw_cache, "xqlfw-official-pairs-v1")
    shared = sorted(set(natural_ids) & set(xqlfw_ids))
    if len(shared) != 1:
        raise ValueError(
            "--extraction-id is required unless caches share exactly one extraction id; "
            f"natural={list(natural_ids)}, xqlfw={list(xqlfw_ids)}"
        )
    return shared[0]


def _print_progress(stage: str, done: int, total: int) -> None:
    """按批次显示场景分数导出进度。"""

    if done == total or done % 320 == 0 or stage == "scenario":
        print(f"{stage}: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
