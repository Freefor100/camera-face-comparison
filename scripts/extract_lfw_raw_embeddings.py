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
from camera_face_comparison.evaluation_cache import (
    embedding_extraction_id,
    file_sha256,
    write_json_atomic,
)
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.lfw_dataset import read_lfw_protocol
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.raw_lfw_extraction import extract_lfw_protocol_raw_embeddings
from camera_face_comparison.runtime import backend_metadata


def main() -> int:
    """用本地 GPU/CPU 模型建立与质量和判定策略无关的完整 LFW 缓存。"""

    parser = argparse.ArgumentParser(
        description="Extract all natural LFW embeddings into a policy-independent raw cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()
    if args.commit_every < 1:
        parser.error("--commit-every must be at least one")

    settings = load_settings(args.data_dir)
    protocol_path = (
        args.protocol
        or settings.data_dir / "datasets" / "lfw_full_open_set_protocol.json"
    )
    dataset_dir = settings.data_dir / "datasets" / "lfw_funneled"
    cache_path = args.cache_path or settings.logs_dir / "cache" / "lfw_raw.sqlite"
    manifest_path = (
        args.manifest
        or settings.data_dir / "experiments" / "phase4" / "raw_extraction_manifest.json"
    )
    try:
        protocol = read_lfw_protocol(protocol_path)
        engine = FaceEngine.from_local_model(settings)
        extraction_id = embedding_extraction_id(settings)
        base_manifest = {
            "artifact": "lfw-policy-independent-raw-embeddings-v1",
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "dataset_dir": str(dataset_dir),
            "cache_path": str(cache_path),
            "cache_dataset_id": "lfw-natural-v1",
            "embedding_extraction_id": extraction_id,
            "primary_face_rule": "largest-detected-face",
            "quality_policy": None,
            "execution_backend": backend_metadata(engine.backend),
        }
        write_json_atomic(manifest_path, base_manifest)
        with RawEmbeddingCache(cache_path, "lfw-natural-v1", extraction_id) as cache:
            summary = extract_lfw_protocol_raw_embeddings(
                dataset_dir=dataset_dir,
                protocol=protocol,
                face_engine=engine,
                cache=cache,
                commit_every=args.commit_every,
                on_progress=_print_progress,
            )
        manifest = base_manifest | {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "summary": asdict(summary),
        }
        write_json_atomic(manifest_path, manifest)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _print_progress(done: int, total: int, relative_path: str, cache_hit: bool) -> None:
    """按固定间隔显示全量提取进度和最近缓存状态。"""

    if done == 1 or done == total or done % 100 == 0:
        state = "cache" if cache_hit else "inference"
        print(f"LFW raw: {done}/{total} {relative_path} {state}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
