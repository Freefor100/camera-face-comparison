from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.gallery_scale_experiment import (
    load_gallery_scale_inputs,
    run_gallery_scale_experiment,
)
from camera_face_comparison.joint_calibration import (
    JointOperatingPoint,
    ScenarioOperatingMetrics,
)
from camera_face_comparison.lfw_dataset import read_lfw_split_protocol
from camera_face_comparison.raw_embedding_cache import (
    RawEmbeddingCache,
    available_raw_extraction_ids,
)


def main() -> int:
    """只用 Calibration 身份检查冻结策略在小 Gallery 中的稳定性。"""

    parser = argparse.ArgumentParser(
        description="Replay the frozen cross-quality policy on Calibration gallery sizes."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--natural-cache", type=Path)
    parser.add_argument("--xqlfw-cache", type=Path)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-fpir", type=float, default=0.01)
    parser.add_argument(
        "--gallery-sizes", type=int, nargs="+", default=(3, 5, 10, 25, 50, 100)
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    protocol_path = args.protocol or data_dir / "experiments" / "phase4" / "protocol.json"
    natural_cache_path = (
        args.natural_cache or data_dir / "logs" / "cache" / "lfw_raw.sqlite"
    )
    xqlfw_cache_path = (
        args.xqlfw_cache or data_dir / "logs" / "cache" / "xqlfw_full_cuda.sqlite"
    )
    calibration_path = (
        args.calibration_report
        or data_dir / "experiments" / "phase5b" / "calibration_report.json"
    )
    output_path = (
        args.output or data_dir / "experiments" / "phase5b" / "gallery_scale_report.json"
    )
    try:
        protocol = read_lfw_split_protocol(protocol_path)
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        selected = _selected_point(calibration_payload, args.target_fpir)
        extraction_id = _shared_extraction_id(natural_cache_path, xqlfw_cache_path)
        natural_dir = data_dir / "datasets" / "lfw_funneled"
        xqlfw_dir = (
            data_dir
            / "datasets"
            / "xqlfw"
            / "lfw_original_imgs_min_qual0.85variant11"
        )
        with (
            RawEmbeddingCache(
                natural_cache_path, "lfw-natural-v1", extraction_id
            ) as natural_cache,
            RawEmbeddingCache(
                xqlfw_cache_path, "xqlfw-official-pairs-v1", extraction_id
            ) as xqlfw_cache,
        ):
            inputs = load_gallery_scale_inputs(
                natural_dataset_dir=natural_dir,
                xqlfw_dataset_dir=xqlfw_dir,
                protocol=protocol,
                natural_cache=natural_cache,
                xqlfw_cache=xqlfw_cache,
                seed=args.seed,
                on_progress=_print_cache_progress,
            )
        experiment = run_gallery_scale_experiment(
            inputs,
            selected=selected,
            gallery_sizes=args.gallery_sizes,
            repeats=args.repeats,
            seed=args.seed,
            on_progress=_print_run_progress,
        )
        report = {
            "artifact": "cross-quality-calibration-gallery-scale-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source_split": "calibration",
            "evaluation_labels_read": False,
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "calibration_report": str(calibration_path),
            "calibration_report_sha256": file_sha256(calibration_path),
            "embedding_extraction_id": extraction_id,
            "input_coverage": inputs.coverage,
            "experiment": experiment,
        }
        write_json_atomic(output_path, report)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output_path), "summary": experiment["summary"]}))
    return 0


def _selected_point(payload: dict[str, object], target_fpir: float) -> JointOperatingPoint:
    """读取主标定报告中的目标工作点，不接受其他报告结构。"""

    if payload.get("artifact") != "cross-quality-joint-calibration-v1":
        raise ValueError("unsupported cross-quality calibration report")
    if payload.get("source_split") != "calibration" or payload.get("evaluation_labels_read"):
        raise ValueError("gallery scale gate requires a Calibration-only report")
    selected_by_target = payload["selected_by_target"]
    if not isinstance(selected_by_target, dict):
        raise TypeError("selected_by_target must be an object")
    selected_payload = next(
        (
            value
            for key, value in selected_by_target.items()
            if abs(float(key) - target_fpir) <= 1e-12
        ),
        None,
    )
    if selected_payload is None:
        raise ValueError(f"calibration target is unavailable: {target_fpir}")
    if not isinstance(selected_payload, dict):
        raise TypeError("selected calibration point must be an object")
    point_payload = dict(selected_payload)
    scenarios = tuple(
        ScenarioOperatingMetrics(**item) for item in point_payload.pop("scenarios", [])
    )
    return JointOperatingPoint(**point_payload, scenarios=scenarios)


def _shared_extraction_id(natural_cache: Path, xqlfw_cache: Path) -> str:
    """选择两个图像域唯一共享的 embedding 提取标识。"""

    natural_ids = available_raw_extraction_ids(natural_cache, "lfw-natural-v1")
    xqlfw_ids = available_raw_extraction_ids(xqlfw_cache, "xqlfw-official-pairs-v1")
    shared = sorted(set(natural_ids) & set(xqlfw_ids))
    if len(shared) != 1:
        raise ValueError(
            "natural and XQLFW caches must share exactly one extraction id; "
            f"natural={list(natural_ids)}, xqlfw={list(xqlfw_ids)}"
        )
    return shared[0]


def _print_cache_progress(stage: str, done: int, total: int) -> None:
    """低频显示缓存恢复进度。"""

    if done == total or done % 1000 == 0:
        print(f"Gallery scale {stage}: {done}/{total}", file=sys.stderr, flush=True)


def _print_run_progress(done: int, total: int) -> None:
    """低频显示小 Gallery 重放进度。"""

    if done == total or done % 20 == 0:
        print(f"Gallery scale replay: {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
