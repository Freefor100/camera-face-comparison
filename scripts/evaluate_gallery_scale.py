from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.calibration import OperatingPoint
from camera_face_comparison.config import load_settings
from camera_face_comparison.evaluation_cache import file_sha256, write_json_atomic
from camera_face_comparison.gallery_scale_experiment import (
    load_gallery_scale_inputs,
    run_gallery_scale_experiment,
)
from camera_face_comparison.lfw_dataset import read_lfw_split_protocol
from camera_face_comparison.raw_embedding_cache import (
    RawEmbeddingCache,
    available_raw_extraction_ids,
)


def main() -> int:
    """使用自然 LFW 缓存量化 Gallery 身份规模对开放集工作点的影响。"""

    parser = argparse.ArgumentParser(
        description="Measure small-gallery score-gap and threshold behavior from raw LFW cache."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--cache-extraction-id")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gallery-sizes", type=int, nargs="+", default=(3, 5, 10, 25, 50, 100))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-fpir", type=float, default=0.003)
    args = parser.parse_args()

    settings = load_settings(args.data_dir)
    protocol_path = args.protocol or settings.data_dir / "experiments" / "phase4" / "protocol.json"
    cache_path = args.cache_path or settings.logs_dir / "cache" / "lfw_raw.sqlite"
    policy_path = args.policy or settings.data_dir / "experiments" / "phase4" / "final_evaluation.json"
    output_path = args.output or (
        settings.data_dir / "experiments" / "phase5" / "gallery_scale_report.json"
    )
    try:
        protocol = read_lfw_split_protocol(protocol_path)
        extraction_id = _select_cache_extraction_id(cache_path, args.cache_extraction_id)
        transferred_policy = _read_operating_point(policy_path)
        with RawEmbeddingCache(cache_path, "lfw-natural-v1", extraction_id) as cache:
            inputs = load_gallery_scale_inputs(
                dataset_dir=settings.data_dir / "datasets" / "lfw_funneled",
                protocol=protocol,
                cache=cache,
                on_progress=_print_cache_progress,
            )
        experiment = run_gallery_scale_experiment(
            inputs,
            gallery_sizes=args.gallery_sizes,
            repeats=args.repeats,
            seed=args.seed,
            target_fpir=args.target_fpir,
            transferred_policy=transferred_policy,
            on_progress=_print_run_progress,
        )
        report = {
            "artifact": "lfw-small-gallery-scale-experiment-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_protocol": str(protocol_path),
            "source_protocol_sha256": file_sha256(protocol_path),
            "source_cache": str(cache_path),
            "embedding_extraction_id": extraction_id,
            "quality_policy": None,
            "aggregation_method": "mean_prototype",
            "transferred_full_gallery_policy": asdict(transferred_policy),
            "experiment": experiment,
            "notes": [
                "Calibration 和 Evaluation 的 Known/Unknown 来源身份保持互斥。",
                "每个规模只在 Calibration 选择参数，再应用到不同身份的 Evaluation。",
                "本实验只隔离 Gallery 规模效应，不代表摄像头域复核。",
            ],
        }
        # 原型和向量不能写入 JSON，只保留计数型输入覆盖。
        report["input_coverage"] = {
            "gallery_prototype_total": len(inputs.prototypes),
            "gallery_image_total": inputs.gallery_image_total,
            "gallery_valid_image_total": inputs.gallery_valid_image_total,
            "gallery_fte_total": inputs.gallery_fte_total,
            "probe_total": inputs.probe_total,
            "probe_valid_total": inputs.probe_valid_total,
            "probe_fte_total": inputs.probe_fte_total,
        }
        write_json_atomic(output_path, report)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output_path), "summary": experiment["summary_by_gallery_size"]}, ensure_ascii=False))
    return 0


def _select_cache_extraction_id(path: Path, requested: str | None) -> str:
    """选择唯一自然 LFW 提取批次，多批次并存时要求显式指定。"""

    available = available_raw_extraction_ids(path, "lfw-natural-v1")
    if requested is not None:
        if requested not in available:
            raise ValueError(f"cache extraction id is unavailable: {requested}")
        return requested
    if len(available) != 1:
        raise ValueError(
            "--cache-extraction-id is required unless the cache contains exactly one id; "
            f"available={list(available)}"
        )
    return available[0]


def _read_operating_point(path: Path) -> OperatingPoint:
    """读取 Phase 4 在完整 LFW Gallery 上选择的 Mean Prototype 工作点。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected"]
    if selected["method"] != "mean_prototype":
        raise ValueError("gallery scale experiment requires the selected Mean Prototype policy")
    return OperatingPoint(**selected["operating_point"])


def _print_cache_progress(stage: str, done: int, total: int) -> None:
    """低频显示原始缓存恢复进度。"""

    if done == 1 or done == total or done % 1000 == 0:
        print(f"Gallery scale {stage}: {done}/{total}", file=sys.stderr, flush=True)


def _print_run_progress(done: int, total: int) -> None:
    """显示规模和重复实验进度。"""

    print(f"Gallery scale runs: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
