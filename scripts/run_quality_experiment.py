from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.evaluation_cache import (
    embedding_extraction_id,
    file_sha256,
    quality_policy_id,
    write_json_atomic,
)
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.quality_degradation import degradation_specs
from camera_face_comparison.quality_experiment import read_quality_experiment_protocol
from camera_face_comparison.quality_experiment_runner import run_quality_experiment
from camera_face_comparison.quality_experiment_store import QualityExperimentStore
from camera_face_comparison.runtime import backend_metadata


def main() -> int:
    """在本地 CUDA/CPU 模型上运行可恢复的 Phase 3 单因素质量实验。"""

    parser = argparse.ArgumentParser(
        description="Run the recoverable Calibration-only face-quality experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--commit-every", type=int, default=20)
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="LFW 实验专用人脸尺寸下限，不修改应用 config.toml",
    )
    args = parser.parse_args()

    settings = replace(load_settings(args.data_dir), min_face_size_px=args.min_face_size)
    phase3_dir = settings.data_dir / "experiments" / "phase3"
    protocol_path = args.protocol or phase3_dir / "protocol.json"
    dataset_dir = args.dataset_dir or settings.data_dir / "datasets" / "lfw_funneled"
    store_path = args.store or phase3_dir / "measurements.sqlite"
    manifest_path = args.manifest or phase3_dir / "manifest.json"
    if args.commit_every < 1:
        parser.error("--commit-every must be at least one")
    if args.min_face_size < 1:
        parser.error("--min-face-size must be at least one")

    try:
        protocol = read_quality_experiment_protocol(protocol_path)
        engine = FaceEngine.from_local_model(settings)
        extraction_id = embedding_extraction_id(settings)
        base_manifest = {
            "artifact": "lfw-quality-experiment-v1",
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "dataset_dir": str(dataset_dir),
            "store_path": str(store_path),
            "embedding_extraction_id": extraction_id,
            "quality_policy_id": quality_policy_id(settings),
            "quality_configuration": {
                "min_detection_score": settings.min_detection_score,
                "min_face_size_px": settings.min_face_size_px,
                "min_blur_variance": settings.min_blur_variance,
                "min_brightness": settings.min_brightness,
                "max_brightness": settings.max_brightness,
                "min_contrast": settings.min_contrast,
                "high_quality_score": settings.high_quality_score,
                "medium_quality_score": settings.medium_quality_score,
            },
            "execution_backend": backend_metadata(engine.backend),
            "known_identity_total": len(protocol.known),
            "unknown_identity_total": len(protocol.unknown),
            "conditions": [asdict(spec) | {"key": spec.key} for spec in degradation_specs()],
        }
        write_json_atomic(manifest_path, base_manifest)
        with QualityExperimentStore(store_path, extraction_id) as store:
            summary = run_quality_experiment(
                protocol=protocol,
                dataset_dir=dataset_dir,
                settings=settings,
                face_engine=engine,
                store=store,
                commit_every=args.commit_every,
                on_progress=_print_progress,
            )
        write_json_atomic(
            manifest_path,
            base_manifest
            | {
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "summary": asdict(summary),
            },
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"completed quality experiment: {manifest_path}")
    return 0


def _print_progress(completed, total, source, spec, cache_hit) -> None:
    """每 25 个条件输出一次角色、退化条件和缓存状态。"""

    if completed == total or completed % 25 == 0:
        cache_state = "cache" if cache_hit else "inference"
        print(
            f"[{completed}/{total}] {source.role} {source.identity} "
            f"{spec.key} {cache_state}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
