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
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.lfw_dataset import read_lfw_split_protocol
from camera_face_comparison.raw_dataset_extraction import extract_dataset_raw_embeddings
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.runtime import backend_metadata
from camera_face_comparison.xqlfw import load_xqlfw_protocol


def main() -> int:
    """用本地 GPU/CPU 模型建立 XQLFW 官方验证图片的无质量门原始缓存。"""

    parser = argparse.ArgumentParser(
        description="Extract official XQLFW pair images into a policy-independent cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="可选的 LFW 身份识别分区协议；提供后提取协议引用的全部 13,233 张变体。",
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()
    if args.commit_every < 1:
        parser.error("--commit-every must be at least one")

    settings = load_settings(args.data_dir)
    dataset_dir = args.dataset_dir or (
        settings.data_dir
        / "datasets"
        / "xqlfw"
        / "lfw_original_imgs_min_qual0.85variant11"
    )
    pairs_path = args.pairs or settings.data_dir / "datasets" / "xqlfw" / "xqlfw_pairs.txt"
    cache_path = args.cache_path or settings.logs_dir / "cache" / "xqlfw_raw_optimized.sqlite"
    manifest_path = args.manifest or (
        settings.data_dir / "experiments" / "phase5" / "xqlfw_raw_extraction_manifest.json"
    )
    try:
        if args.protocol is None:
            pair_protocol = load_xqlfw_protocol(pairs_path, dataset_dir)
            image_paths = pair_protocol.image_paths
            protocol_source = pairs_path
            artifact = "xqlfw-policy-independent-raw-embeddings-v1"
        else:
            identification_protocol = read_lfw_split_protocol(args.protocol)
            image_paths = tuple(
                sorted(
                    {
                        *(
                            path
                            for paths in identification_protocol.enrollment.values()
                            for path in paths
                        ),
                        *(probe.relative_path for probe in identification_protocol.probes),
                    }
                )
            )
            protocol_source = args.protocol
            artifact = "xqlfw-full-open-set-raw-embeddings-v1"
        engine = FaceEngine.from_local_model(settings)
        extraction_id = embedding_extraction_id()
        base_manifest = {
            "artifact": artifact,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "protocol_source": str(protocol_source),
            "protocol_sha256": file_sha256(protocol_source),
            "dataset_dir": str(dataset_dir),
            "cache_path": str(cache_path),
            "cache_dataset_id": "xqlfw-official-pairs-v1",
            "embedding_extraction_id": extraction_id,
            "primary_face_rule": "largest-detected-face",
            "quality_policy": None,
            "execution_backend": backend_metadata(engine.backend),
        }
        write_json_atomic(manifest_path, base_manifest)
        with RawEmbeddingCache(
            cache_path,
            "xqlfw-official-pairs-v1",
            extraction_id,
        ) as cache:
            summary = extract_dataset_raw_embeddings(
                dataset_dir=dataset_dir,
                relative_paths=image_paths,
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
    """按固定间隔显示 XQLFW 原始提取进度和缓存状态。"""

    if done == 1 or done == total or done % 100 == 0:
        state = "cache" if cache_hit else "inference"
        print(f"XQLFW raw: {done}/{total} {relative_path} {state}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
