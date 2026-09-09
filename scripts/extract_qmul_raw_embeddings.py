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
from camera_face_comparison.evaluation_cache import embedding_extraction_id, write_json_atomic
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.qmul_survface import build_qmul_protocol
from camera_face_comparison.raw_dataset_extraction import (
    extract_identification_protocol_raw_embeddings,
)
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache
from camera_face_comparison.runtime import backend_metadata


def main() -> int:
    """建立 QMUL-SurvFace 官方识别协议的无质量门原始 embedding 缓存。"""

    parser = argparse.ArgumentParser(
        description="Extract the full QMUL-SurvFace identification protocol into a raw cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "qmul-survface" / "QMUL-SurvFace",
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()
    if args.commit_every < 1:
        parser.error("--commit-every must be at least one")

    settings = load_settings(args.data_dir)
    cache_path = args.cache_path or settings.logs_dir / "cache" / "qmul_survface_raw.sqlite"
    manifest_path = args.manifest or (
        settings.data_dir / "experiments" / "phase5" / "qmul_raw_extraction_manifest.json"
    )
    dataset_dir = args.dataset_root / "Face_Identification_Test_Set"
    try:
        protocol = build_qmul_protocol(args.dataset_root)
        engine = FaceEngine.from_local_model(settings)
        extraction_id = embedding_extraction_id(settings)
        gallery_total = sum(len(paths) for paths in protocol.enrollment.values())
        known_probe_total = sum(
            probe.expected_person_id is not None for probe in protocol.probes
        )
        unknown_probe_total = len(protocol.probes) - known_probe_total
        base_manifest = {
            "artifact": "qmul-survface-policy-independent-raw-embeddings-v1",
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(args.dataset_root),
            "dataset_dir": str(dataset_dir),
            "cache_path": str(cache_path),
            "cache_dataset_id": "qmul-survface-identification-raw-v1",
            "embedding_extraction_id": extraction_id,
            "primary_face_rule": "largest-detected-face",
            "quality_policy": None,
            "execution_backend": backend_metadata(engine.backend),
            "protocol_counts": {
                "gallery_identity_total": len(protocol.enrollment),
                "gallery_image_total": gallery_total,
                "mated_probe_total": known_probe_total,
                "unmated_probe_total": unknown_probe_total,
            },
        }
        write_json_atomic(manifest_path, base_manifest)
        with RawEmbeddingCache(
            cache_path,
            "qmul-survface-identification-raw-v1",
            extraction_id,
        ) as cache:
            summary = extract_identification_protocol_raw_embeddings(
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
    """低频输出长任务的全量进度和当前缓存状态。"""

    if done == 1 or done == total or done % 1000 == 0:
        state = "cache" if cache_hit else "inference"
        print(f"QMUL raw: {done}/{total} {relative_path} {state}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
