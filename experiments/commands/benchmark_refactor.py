"""以固定 LFW 工作负载对照重构前后的正式识别链路。"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.face_library import FaceLibrarySnapshot
from camera_face_comparison.runtime import ExecutionBackend, backend_metadata
from experiments.commands.benchmark_production_runtime import (
    _build_gallery_matrix,
    _build_sequences,
    _open_log_repository,
    _read_cached_embeddings,
    _run_case,
    _select_cache_partition,
)
from experiments.face_evaluation.experiment_artifacts import write_json_atomic


def summarize(values):
    """返回毫秒测量的平均值、中位数和 95% 分位。"""
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main():
    """读取既有特征缓存并在临时日志库中运行固定规模对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "cpu"), default="auto")
    args = parser.parse_args()
    protocol_path = Path("data/experiments/phase4/protocol.json")
    cache = Path("data/logs/cache/lfw_raw.sqlite")
    protocol = json.loads(protocol_path.read_text())
    partition = _select_cache_partition(cache)
    cached = _read_cached_embeddings(cache, *partition)
    ids, matrix = _build_gallery_matrix(protocol["enrollment"], cached, 4588)
    sequences = _build_sequences(
        protocol["probes"], ids, cached, Path("data/datasets/lfw_funneled"), 5, 5
    )
    settings = load_settings(Path("data"))
    started = perf_counter()
    if args.backend == "cpu":
        # 仅实验显式指定真实 CPU provider；不会替换模型或推理结果。
        with patch(
            "camera_face_comparison.face_engine.detect_execution_backend",
            return_value=ExecutionBackend("cpu", ("CPUExecutionProvider",), -1),
        ):
            engine = FaceEngine.from_local_model(settings)
    else:
        engine = FaceEngine.from_local_model(settings)
    report = {
        "model_load_ms": (perf_counter() - started) * 1000,
        "backend": backend_metadata(engine.backend),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "workload": [[str(p) for p in seq] for seq in sequences],
        "cases": {},
    }
    for count in (3, 100, 4588):
        selected = ids[:count]
        snapshot = FaceLibrarySnapshot(selected, selected, matrix[:count].copy(), 0)
        with tempfile.TemporaryDirectory(prefix="face-refactor-benchmark-") as root:
            repository = _open_log_repository(Path(root) / "log.sqlite", selected)
            try:
                rows = _run_case(
                    [[seq[0]] for seq in sequences],
                    2,
                    engine,
                    settings,
                    snapshot,
                    repository,
                    1,
                    strategy="single_image",
                )
                multi = _run_case(
                    sequences,
                    2,
                    engine,
                    settings,
                    snapshot,
                    repository,
                    5,
                    strategy="stream_capture_and_detect",
                )
            finally:
                repository.close()
        report["cases"][str(count)] = {"single": rows, "camera": multi}
        report["cases"][str(count)]["summary"] = {
            "single": summarize([row["click_to_result_ms"] for row in rows]),
            "camera": summarize([row["click_to_result_ms"] for row in multi]),
        }
    write_json_atomic(args.output, report)
    print(f"已保存固定工作负载结果：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
