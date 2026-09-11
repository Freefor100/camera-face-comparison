from __future__ import annotations

import argparse
import json
import platform
import random
import sqlite3
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy
from camera_face_comparison.recognition import recognize_embedding
from camera_face_comparison.runtime import detect_execution_backend

FROZEN_MINIMUM_SCORE = 0.5557855367660522


def parse_args() -> argparse.Namespace:
    """读取真实 LFW Gallery 检索基准参数。"""

    parser = argparse.ArgumentParser(
        description="Benchmark exact CPU and CUDA Gallery retrieval on cached LFW embeddings."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("data/experiments/phase4/protocol.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/logs/cache/lfw_raw.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/gallery-index/benchmark_report.json"),
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("data/experiments/gallery-index/lfw_mean_prototypes.npy"),
    )
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--rebuild-query-count", type=int, default=30)
    parser.add_argument("--warmup-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scales", default="3,10,100,1000,full")
    parser.add_argument("--skip-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    """运行真实向量检索基准并原子写入 JSON 报告。"""

    args = parse_args()
    if args.query_count < 1 or args.rebuild_query_count < 1 or args.warmup_count < 0:
        raise ValueError("query counts must be positive and warmup count must be non-negative")

    protocol_path = args.protocol.resolve()
    cache_path = args.cache.resolve()
    output_path = args.output.resolve()
    matrix_output = args.matrix_output.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    enrollment = _read_enrollment(protocol)
    probe_paths = _read_probe_paths(protocol)

    dataset_id, extraction_id = _select_cache_partition(cache_path)
    gallery_paths = [path for paths in enrollment.values() for path in paths]
    gallery_started = perf_counter_ns()
    gallery_entries = _load_embeddings(
        cache_path,
        dataset_id,
        extraction_id,
        gallery_paths,
    )
    gallery_load_ms = _elapsed_ms(gallery_started)

    probe_started = perf_counter_ns()
    probe_entries = _load_embeddings(
        cache_path,
        dataset_id,
        extraction_id,
        probe_paths,
    )
    probe_load_ms = _elapsed_ms(probe_started)

    gallery = {
        person_id: [gallery_entries[path] for path in paths if path in gallery_entries]
        for person_id, paths in enrollment.items()
    }
    gallery = {person_id: rows for person_id, rows in gallery.items() if rows}
    if len(gallery) < 3:
        raise RuntimeError("valid LFW Gallery has fewer than three identities")

    prototype_started = perf_counter_ns()
    person_ids, full_matrix = _build_prototype_matrix(gallery)
    prototype_build_ms = _elapsed_ms(prototype_started)
    queries = _select_queries(probe_entries, args.query_count, args.seed)

    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    save_started = perf_counter_ns()
    np.save(matrix_output, full_matrix, allow_pickle=False)
    matrix_save_ms = _elapsed_ms(save_started)
    load_started = perf_counter_ns()
    persisted_matrix = np.load(matrix_output, allow_pickle=False)
    persisted_matrix_load_ms = _elapsed_ms(load_started)
    if not np.array_equal(persisted_matrix, full_matrix):
        raise RuntimeError("persisted prototype matrix changed during round trip")

    identity_order = list(person_ids)
    random.Random(args.seed).shuffle(identity_order)
    scales = _parse_scales(args.scales, len(identity_order))
    results: list[dict[str, Any]] = []
    cuda_status: dict[str, Any] = {"requested": not args.skip_cuda, "available": False}

    for scale in scales:
        selected_ids = tuple(identity_order[:scale])
        selected_gallery = {person_id: gallery[person_id] for person_id in selected_ids}
        row_indices = np.asarray([person_ids.index(person_id) for person_id in selected_ids])
        matrix = np.ascontiguousarray(full_matrix[row_indices], dtype=np.float32)

        current = _benchmark(
            lambda query, current_gallery=selected_gallery: _production_search(
                query, current_gallery
            ),
            queries[: min(args.rebuild_query_count, len(queries))],
            args.warmup_count,
        )
        cached_loop = _benchmark(
            lambda query, current_matrix=matrix: _cached_python_search(query, current_matrix),
            queries,
            args.warmup_count,
        )
        cpu_default = _benchmark(
            lambda query, current_matrix=matrix: _matrix_search(query, current_matrix),
            queries,
            args.warmup_count,
        )
        with threadpool_limits(limits=1, user_api="blas"):
            cpu_one_thread = _benchmark(
                lambda query, current_matrix=matrix: _matrix_search(query, current_matrix),
                queries,
                args.warmup_count,
            )

        correctness = _check_cpu_equivalence(selected_gallery, matrix, queries[:10])
        cpu_variants = {
            "default_threads": cpu_default,
            "one_blas_thread": cpu_one_thread,
        }
        selected_cpu_name, selected_cpu = min(
            cpu_variants.items(), key=lambda item: float(item[1]["p50_ms"])
        )
        result: dict[str, Any] = {
            "identity_count": scale,
            "sample_count": sum(len(rows) for rows in selected_gallery.values()),
            "matrix_bytes": int(matrix.nbytes),
            "current_rebuild_python": current,
            "cached_python_loop": cached_loop,
            "cpu_matrix_default_threads": cpu_default,
            "cpu_matrix_one_blas_thread": cpu_one_thread,
            "selected_cpu_matrix_variant": selected_cpu_name,
            "cpu_equivalence": correctness,
            "speedup_current_to_selected_cpu_matrix": current["p50_ms"]
            / selected_cpu["p50_ms"],
        }
        results.append(result)
        print(
            f"P={scale}: current={current['p50_ms']:.4f} ms, "
            f"cpu-matrix={cpu_one_thread['p50_ms']:.4f} ms"
        )

    if not args.skip_cuda:
        cuda_status = _add_cuda_results(
            results=results,
            identity_order=identity_order,
            person_ids=person_ids,
            full_matrix=full_matrix,
            queries=queries,
            warmup_count=args.warmup_count,
        )

    report = {
        "experiment": "gallery-index-benchmark-v1",
        "scope": "retrieval-only; excludes image decode, face inference, integrity scan and log write",
        "seed": args.seed,
        "protocol": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
        },
        "embedding_cache": {
            "path": str(cache_path),
            "sha256": file_sha256(cache_path),
            "dataset_id": dataset_id,
            "embedding_extraction_id": extraction_id,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu": _cpu_description(),
            "numpy_version": np.__version__,
            "blas": threadpool_info(),
            "nvidia_smi": _nvidia_smi(),
        },
        "data": {
            "protocol_gallery_identity_count": len(enrollment),
            "valid_gallery_identity_count": len(gallery),
            "protocol_gallery_sample_count": len(gallery_paths),
            "valid_gallery_sample_count": sum(len(rows) for rows in gallery.values()),
            "valid_probe_embedding_count": len(probe_entries),
            "measured_query_count": len(queries),
            "rebuild_query_count": min(args.rebuild_query_count, len(queries)),
            "embedding_dimension": int(full_matrix.shape[1]),
        },
        "startup": {
            "gallery_embedding_sqlite_load_ms": gallery_load_ms,
            "probe_embedding_sqlite_load_ms": probe_load_ms,
            "prototype_matrix_build_ms": prototype_build_ms,
            "prototype_matrix_save_ms": matrix_save_ms,
            "persisted_matrix_same_process_load_ms": persisted_matrix_load_ms,
            "persisted_matrix_path": str(matrix_output),
            "persisted_matrix_sha256": file_sha256(matrix_output),
            "note": "same-process load benefits from OS page cache and is not a cold-start claim",
        },
        "cuda": cuda_status,
        "results": results,
    }
    write_json_atomic(output_path, report)
    print(f"report: {output_path}")
    return 0


def _read_enrollment(protocol: Mapping[str, object]) -> dict[str, list[str]]:
    """从固定协议读取人员到 Gallery 路径的映射。"""

    raw = protocol.get("enrollment")
    if not isinstance(raw, dict):
        raise TypeError("protocol.enrollment must be an object")
    result: dict[str, list[str]] = {}
    for person_id, paths in raw.items():
        if not isinstance(person_id, str) or not isinstance(paths, list):
            raise TypeError("invalid enrollment entry")
        result[person_id] = [str(path) for path in paths]
    return result


def _read_probe_paths(protocol: Mapping[str, object]) -> list[str]:
    """从固定协议读取全部 Probe 相对路径。"""

    raw = protocol.get("probes")
    if not isinstance(raw, list):
        raise TypeError("protocol.probes must be an array")
    paths: list[str] = []
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise TypeError("invalid probe entry")
        paths.append(str(row["relative_path"]))
    return paths


def _select_cache_partition(cache_path: Path) -> tuple[str, str]:
    """选择有效向量最多的 LFW 原始缓存分区。"""

    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            """
            SELECT dataset_id, embedding_extraction_id, COUNT(*) AS observed_count
            FROM raw_embeddings
            WHERE status = 'observed'
            GROUP BY dataset_id, embedding_extraction_id
            ORDER BY observed_count DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("raw embedding cache has no observed rows")
    return str(row[0]), str(row[1])


def _load_embeddings(
    cache_path: Path,
    dataset_id: str,
    extraction_id: str,
    paths: Sequence[str],
) -> dict[str, np.ndarray]:
    """按路径分块读取有效 float32 embedding，避免 SQLite 参数上限。"""

    entries: dict[str, np.ndarray] = {}
    with sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True) as connection:
        for start in range(0, len(paths), 800):
            chunk = paths[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT relative_path, embedding_blob, embedding_dim
                FROM raw_embeddings
                WHERE dataset_id = ? AND embedding_extraction_id = ?
                  AND status = 'observed' AND relative_path IN ({placeholders})
                """,
                (dataset_id, extraction_id, *chunk),
            ).fetchall()
            for relative_path, embedding_blob, embedding_dim in rows:
                vector = np.frombuffer(embedding_blob, dtype=np.float32).copy()
                if vector.size != int(embedding_dim):
                    raise RuntimeError(f"invalid embedding dimension: {relative_path}")
                entries[str(relative_path)] = vector
    return entries


def _build_prototype_matrix(
    gallery: Mapping[str, Sequence[np.ndarray]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """按稳定人员顺序构造归一化 Mean Prototype 矩阵。"""

    person_ids = tuple(sorted(gallery))
    prototypes: list[np.ndarray] = []
    for person_id in person_ids:
        samples = np.stack([_normalize(row) for row in gallery[person_id]])
        prototypes.append(_normalize(np.mean(samples, axis=0)))
    return person_ids, np.ascontiguousarray(np.stack(prototypes), dtype=np.float32)


def _select_queries(
    probes: Mapping[str, np.ndarray], count: int, seed: int
) -> list[np.ndarray]:
    """固定种子选择不同真实 Probe embedding。"""

    paths = sorted(probes)
    random.Random(seed).shuffle(paths)
    if len(paths) < count:
        raise ValueError(f"requested {count} queries but only {len(paths)} are valid")
    return [_normalize(probes[path]) for path in paths[:count]]


def _parse_scales(value: str, full_count: int) -> list[int]:
    """解析递增 Gallery 规模，并把 full 替换为真实人数。"""

    result: list[int] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        scale = full_count if normalized == "full" else int(normalized)
        if scale < 3 or scale > full_count:
            raise ValueError(f"invalid Gallery scale: {item}")
        if scale not in result:
            result.append(scale)
    return result


def _benchmark(
    operation: Callable[[np.ndarray], object],
    queries: Sequence[np.ndarray],
    warmup_count: int,
) -> dict[str, float | int]:
    """预热后逐次记录同步单 Query 延迟分布。"""

    for index in range(warmup_count):
        operation(queries[index % len(queries)])
    durations_ms: list[float] = []
    for query in queries:
        started = perf_counter_ns()
        operation(query)
        durations_ms.append((perf_counter_ns() - started) / 1_000_000)
    values = np.asarray(durations_ms, dtype=np.float64)
    return {
        "query_count": len(queries),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def _production_search(
    query: np.ndarray,
    gallery: Mapping[str, Sequence[np.ndarray]],
) -> tuple[str | None, float | None, float | None]:
    """执行当前生产代码的重建原型、完整排序和阈值判定。"""

    decision = recognize_embedding(
        query_embedding=query,
        embeddings_by_person=gallery,
        policy=ScoreThresholdPolicy(FROZEN_MINIMUM_SCORE),
    )
    return decision.top_person_id, decision.top_score, decision.second_score


def _cached_python_search(query: np.ndarray, matrix: np.ndarray) -> tuple[int, float, float]:
    """使用缓存原型但仍逐行执行 Python 点积，作为中间对照。"""

    normalized = _normalize(query)
    scores = np.asarray([float(normalized @ row) for row in matrix], dtype=np.float32)
    return _top_two(scores)


def _matrix_search(query: np.ndarray, matrix: np.ndarray) -> tuple[int, float, float]:
    """使用连续 CPU 矩阵一次计算全部精确余弦分数。"""

    return _top_two(matrix @ _normalize(query))


def _top_two(scores: np.ndarray) -> tuple[int, float, float]:
    """无需完整排序地返回第一名索引和前两名分数。"""

    candidates = np.argpartition(scores, -2)[-2:]
    ordered = candidates[np.argsort(scores[candidates])[::-1]]
    return int(ordered[0]), float(scores[ordered[0]]), float(scores[ordered[1]])


def _check_cpu_equivalence(
    gallery: Mapping[str, Sequence[np.ndarray]],
    matrix: np.ndarray,
    queries: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """确认 CPU 矩阵优化没有改变当前 top-1、top-2 和分数。"""

    ids = tuple(gallery)
    top1_matches = 0
    top2_score_error = 0.0
    for query in queries:
        current_id, current_top, current_second = _production_search(query, gallery)
        matrix_index, matrix_top, matrix_second = _matrix_search(query, matrix)
        top1_matches += int(current_id == ids[matrix_index])
        top2_score_error = max(
            top2_score_error,
            abs(float(current_top) - matrix_top),
            abs(float(current_second) - matrix_second),
        )
    return {
        "query_count": len(queries),
        "top1_match_count": top1_matches,
        "maximum_top2_score_absolute_error": top2_score_error,
    }


def _add_cuda_results(
    *,
    results: list[dict[str, Any]],
    identity_order: Sequence[str],
    person_ids: Sequence[str],
    full_matrix: np.ndarray,
    queries: Sequence[np.ndarray],
    warmup_count: int,
) -> dict[str, Any]:
    """通过 ONNX Runtime CUDA MatMul+TopK 测试原型常驻显存检索。"""

    try:
        backend = detect_execution_backend()
        if backend.name != "cuda":
            return {"requested": True, "available": False, "reason": "CUDA provider unavailable"}
        for result in results:
            scale = int(result["identity_count"])
            selected_ids = tuple(identity_order[:scale])
            row_indices = np.asarray([person_ids.index(person_id) for person_id in selected_ids])
            matrix = np.ascontiguousarray(full_matrix[row_indices], dtype=np.float32)
            session, build_ms = _build_cuda_session(matrix)
            cuda_result = _benchmark(
                lambda query, current_session=session: current_session.run(
                    None, {"query": _normalize(query)[None, :]}
                ),
                queries,
                warmup_count,
            )
            values, indices = session.run(None, {"query": queries[0][None, :]})
            cpu_index, cpu_top, cpu_second = _matrix_search(queries[0], matrix)
            node_providers = _finish_cuda_profile(session)
            result["cuda_onnx_matmul_top2"] = cuda_result
            result["cuda_session_build_ms"] = build_ms
            result["cuda_node_providers"] = node_providers
            result["cuda_equivalence_first_query"] = {
                "top1_index_match": bool(int(indices[0, 0]) == cpu_index),
                "maximum_top2_score_absolute_error": float(
                    max(abs(float(values[0, 0]) - cpu_top), abs(float(values[0, 1]) - cpu_second))
                ),
            }
            selected_cpu = result[
                "cpu_matrix_default_threads"
                if result["selected_cpu_matrix_variant"] == "default_threads"
                else "cpu_matrix_one_blas_thread"
            ]
            result["cuda_to_selected_cpu_p50_latency_ratio"] = (
                cuda_result["p50_ms"] / selected_cpu["p50_ms"]
            )
            result["speedup_current_to_cuda"] = (
                result["current_rebuild_python"]["p50_ms"] / cuda_result["p50_ms"]
            )
        return {
            "requested": True,
            "available": True,
            "providers": list(backend.providers),
            "input_location": "CPU NumPy; copied to CUDA by ONNX Runtime for each query",
            "prototype_location": "CUDA initializer after session construction",
            "output": "top-2 values and indices copied to CPU",
        }
    except Exception as error:  # noqa: BLE001 - 实验报告需要保留任意 CUDA 初始化失败
        return {
            "requested": True,
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }


def _build_cuda_session(matrix: np.ndarray):
    """创建只含 MatMul 与 TopK 的 CUDA ONNX Runtime session。"""

    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    query = helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, matrix.shape[1]])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2])
    weights = numpy_helper.from_array(matrix.T.copy(), name="prototype_weights")
    k = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="top_k")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["query", "prototype_weights"], ["scores"]),
            helper.make_node(
                "TopK", ["scores", "top_k"], ["values", "indices"], axis=1, largest=1, sorted=1
            ),
        ],
        "gallery-index-benchmark",
        [query],
        [values, indices],
        initializer=[weights, k],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="camera-face-comparison-gallery-benchmark",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = "/tmp/camera-face-comparison-gallery-cuda"
    started = perf_counter_ns()
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    build_ms = _elapsed_ms(started)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA session fell back to {session.get_providers()}")
    return session, build_ms


def _finish_cuda_profile(session: object) -> dict[str, str]:
    """结束 ORT profile 并返回计算节点实际使用的 provider。"""

    profile_path = Path(session.end_profiling())
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    providers: dict[str, str] = {}
    for event in payload:
        args = event.get("args", {})
        provider = args.get("provider")
        operation = args.get("op_name")
        if isinstance(provider, str) and isinstance(operation, str):
            providers[operation] = provider
    profile_path.unlink(missing_ok=True)
    return providers


def _nvidia_smi() -> str | None:
    """记录可见 GPU 名称、显存和驱动；不可见时返回空值。"""

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _cpu_description() -> str | None:
    """读取宿主 CPU 型号，命令不可用时返回空值。"""

    try:
        completed = subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return None


def _normalize(vector: np.ndarray) -> np.ndarray:
    """返回有限非零 float32 单位向量。"""

    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("embedding must have a finite positive norm")
    return value / norm


def _elapsed_ms(started_ns: int) -> float:
    """把给定起点到当前时刻转换为毫秒。"""

    return (perf_counter_ns() - started_ns) / 1_000_000


if __name__ == "__main__":
    raise SystemExit(main())
