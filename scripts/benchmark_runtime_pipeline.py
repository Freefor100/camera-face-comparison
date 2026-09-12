from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.face_engine import FaceEngine, FaceInputError
from camera_face_comparison.image_input import ImageInput, measure_quality, quality_warnings
from camera_face_comparison.open_set_policy import (
    OpenSetDecision,
    ScoreThresholdPolicy,
    apply_open_set_policy,
    apply_ranked_open_set_policy,
)
from camera_face_comparison.runtime import backend_metadata

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FROZEN_MINIMUM_SCORE = 0.5557855367660522
DEFAULT_SCALES = (3, 10, 100, 1_000, 4_588)


@dataclass(frozen=True)
class PreparedProbe:
    """一次固定待识别人脸输入及其可复用的阶段耗时。"""

    relative_path: str
    frame: np.ndarray
    embedding: np.ndarray
    quality_metrics: dict[str, float]
    image_read_ms: float
    face_inference_ms: float
    quality_measurement_ms: float


@dataclass(frozen=True)
class GalleryScale:
    """一个固定身份规模下的标准库图片和特征映射。"""

    identity_count: int
    person_ids: tuple[str, ...]
    samples_by_person: dict[str, tuple[tuple[str, np.ndarray], ...]]


def parse_args() -> argparse.Namespace:
    """读取完整链路性能实验参数。"""

    parser = argparse.ArgumentParser(
        description="比较逐次 SQLite/逐人计算与启动时 CPU 矩阵标准库的完整识别链路"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("data/experiments/phase4/protocol.json")
    )
    parser.add_argument(
        "--lfw-root", type=Path, default=Path("data/datasets/lfw_funneled")
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("data/logs/cache/lfw_raw.sqlite")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/phase5c/runtime-performance"),
    )
    parser.add_argument("--query-count", type=int, default=30)
    parser.add_argument("--warmup-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scales", default=",".join(str(value) for value in DEFAULT_SCALES))
    parser.add_argument(
        "--cached-inputs",
        action="store_true",
        help="只使用已有 embedding 缓存，跳过真实模型输入阶段；用于故障排查",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="跳过 Qt 工作线程和界面更新阶段，仅比较计算链路",
    )
    return parser.parse_args()


def main() -> int:
    """运行性能对照并原子写入四份阶段 5 实验产物。"""

    args = parse_args()
    if args.query_count < 1 or args.warmup_count < 0:
        raise ValueError("query-count must be positive and warmup-count must be non-negative")

    data_dir = args.data_dir.resolve()
    protocol_path = args.protocol.resolve()
    lfw_root = args.lfw_root.resolve()
    cache_path = args.cache.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_id, extraction_id = _select_cache_partition(cache_path)
    enrollment = _read_enrollment(protocol)
    gallery_paths = [path for paths in enrollment.values() for path in paths]
    gallery_embeddings = _load_embeddings(
        cache_path, dataset_id, extraction_id, gallery_paths
    )
    gallery_by_person = {
        person_id: tuple(
            (path, gallery_embeddings[path])
            for path in paths
            if path in gallery_embeddings
        )
        for person_id, paths in enrollment.items()
    }
    gallery_by_person = {
        person_id: rows for person_id, rows in gallery_by_person.items() if rows
    }
    if len(gallery_by_person) < max(DEFAULT_SCALES):
        raise RuntimeError(
            f"valid LFW identities {len(gallery_by_person)} are fewer than the largest scale"
        )

    probes = _read_probe_paths(protocol)
    probe_embeddings = _load_embeddings(cache_path, dataset_id, extraction_id, probes)
    probe_paths = _select_probe_paths(probe_embeddings, args.query_count, args.seed)
    prepared_probes, input_report, backend = _prepare_probes(
        paths=probe_paths,
        probe_embeddings=probe_embeddings,
        data_dir=data_dir,
        lfw_root=lfw_root,
        cached_inputs=args.cached_inputs,
    )
    if len(prepared_probes) != args.query_count:
        raise RuntimeError(
            f"only {len(prepared_probes)} valid probes prepared; requested {args.query_count}"
        )

    scales = _parse_scales(args.scales, len(gallery_by_person))
    identity_order = sorted(gallery_by_person)
    random.Random(args.seed).shuffle(identity_order)
    selected_scales = [
        GalleryScale(
            identity_count=scale,
            person_ids=tuple(sorted(identity_order[:scale])),
            samples_by_person={
                person_id: gallery_by_person[person_id]
                for person_id in identity_order[:scale]
            },
        )
        for scale in scales
    ]
    workload = {
        "artifact": "runtime-performance-workload-v1",
        "seed": args.seed,
        "query_count_per_scale": args.query_count,
        "warmup_count": args.warmup_count,
        "scales": [
            {
                "identity_count": item.identity_count,
                "person_ids": list(item.person_ids),
                "gallery_image_count": sum(
                    len(item.samples_by_person[person_id]) for person_id in item.person_ids
                ),
            }
            for item in selected_scales
        ],
        "probe_paths": [probe.relative_path for probe in prepared_probes],
        "protocol": {"path": str(protocol_path), "sha256": file_sha256(protocol_path)},
        "embedding_cache": {
            "path": str(cache_path),
            "sha256": file_sha256(cache_path),
            "dataset_id": dataset_id,
            "embedding_extraction_id": extraction_id,
        },
        "input_mode": "cached_embedding" if args.cached_inputs else "real_face_engine",
    }
    write_json_atomic(output_dir / "workload.json", workload)

    ui_timings = _measure_ui_stages(prepared_probes) if not args.skip_ui else None
    baseline_scales: list[dict[str, Any]] = []
    optimized_scales: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for gallery in selected_scales:
        print(f"P={gallery.identity_count}: preparing optimized startup", flush=True)
        baseline, optimized, exactness = _run_scale(
            gallery=gallery,
            probes=prepared_probes,
            lfw_root=lfw_root,
            cache_path=cache_path,
            dataset_id=dataset_id,
            extraction_id=extraction_id,
            minimum_score=FROZEN_MINIMUM_SCORE,
            warmup_count=args.warmup_count,
            ui_timings=ui_timings,
        )
        baseline_scales.append(baseline)
        optimized_scales.append(optimized)
        equivalence.append(exactness)
        print(
            f"P={gallery.identity_count}: baseline p50={baseline['e2e']['p50_ms']:.3f} ms, "
            f"optimized p50={optimized['e2e']['p50_ms']:.3f} ms",
            flush=True,
        )

    metadata = {
        "artifact": "runtime-performance-report-v1",
        "threshold": FROZEN_MINIMUM_SCORE,
        "protocol": workload["protocol"],
        "embedding_cache": workload["embedding_cache"],
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "nvidia_smi": _nvidia_smi(),
            "model_backend": backend,
        },
        "input": input_report,
        "ui_measurement": {
            "enabled": not args.skip_ui,
            "note": "工作线程往返使用 Qt QThread；界面更新测量 QImage 转换和标签刷新，不包含屏幕合成",
        },
    }
    baseline_report = metadata | {
        "implementation": "per_query_integrity_sqlite_prepare_python_loop",
        "scales": baseline_scales,
    }
    optimized_report = metadata | {
        "implementation": "startup_integrity_cpu_prototype_matrix_numpy_search",
        "scales": optimized_scales,
    }
    comparison_report = {
        "artifact": "runtime-performance-comparison-v1",
        "threshold": FROZEN_MINIMUM_SCORE,
        "equivalence": equivalence,
        "scales": [
            {
                "identity_count": baseline["identity_count"],
                "baseline_e2e_p50_ms": baseline["e2e"]["p50_ms"],
                "optimized_e2e_p50_ms": optimized["e2e"]["p50_ms"],
                "baseline_e2e_p95_ms": baseline["e2e"]["p95_ms"],
                "optimized_e2e_p95_ms": optimized["e2e"]["p95_ms"],
                "baseline_amortized_e2e_p50_ms": baseline["amortized_e2e"]["p50_ms"],
                "optimized_amortized_e2e_p50_ms": optimized["amortized_e2e"]["p50_ms"],
                "baseline_workload_total_ms": baseline["workload_total_ms"],
                "optimized_workload_total_ms": optimized["workload_total_ms"],
                "p50_speedup": baseline["e2e"]["p50_ms"] / optimized["e2e"]["p50_ms"],
                "per_query_e2e_latency_decreased": optimized["e2e"]["p50_ms"]
                < baseline["e2e"]["p50_ms"],
                "amortized_e2e_latency_decreased": optimized["amortized_e2e"]["p50_ms"]
                < baseline["amortized_e2e"]["p50_ms"],
                "workload_total_decreased": optimized["workload_total_ms"]
                < baseline["workload_total_ms"],
            }
            for baseline, optimized in zip(baseline_scales, optimized_scales, strict=True)
        ],
        "conclusion_rule": "只有完整端到端 p50 和 p95 都下降时，才把优化写成端到端收益",
    }
    write_json_atomic(output_dir / "baseline_report.json", baseline_report)
    write_json_atomic(output_dir / "optimized_report.json", optimized_report)
    write_json_atomic(output_dir / "comparison_report.json", comparison_report)
    print(f"reports: {output_dir}")
    return 0


def _run_scale(
    *,
    gallery: GalleryScale,
    probes: list[PreparedProbe],
    lfw_root: Path,
    cache_path: Path,
    dataset_id: str,
    extraction_id: str,
    minimum_score: float,
    warmup_count: int,
    ui_timings: dict[str, list[float]] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """在一个身份规模上运行基线、矩阵优化和逐样本等价性检查。"""

    policy = ScoreThresholdPolicy(minimum_score=minimum_score)
    optimized_integrity_started = perf_counter_ns()
    _verify_protocol_gallery(gallery, lfw_root)
    optimized_integrity_ms = _elapsed_ms(optimized_integrity_started)
    optimized_prepare_started = perf_counter_ns()
    optimized_person_ids, optimized_matrix = _build_prototype_matrix(gallery.samples_by_person)
    optimized_prepare_ms = _elapsed_ms(optimized_prepare_started)
    if optimized_person_ids != gallery.person_ids:
        raise RuntimeError("prototype identity order changed during startup preparation")

    optimized_log = _new_log_connection()
    baseline_log = _new_log_connection()
    baseline_samples: dict[str, list[float]] = {}
    optimized_samples: dict[str, list[float]] = {}
    baseline_e2e: list[float] = []
    optimized_e2e: list[float] = []
    signatures: list[dict[str, Any]] = []

    for index in range(warmup_count):
        probe = probes[index % len(probes)]
        _baseline_query(
            gallery,
            probe,
            lfw_root,
            cache_path,
            dataset_id,
            extraction_id,
            policy,
            baseline_log,
        )
        _optimized_query(
            gallery.person_ids,
            optimized_matrix,
            probe,
            policy,
            optimized_log,
        )

    for probe in probes:
        baseline_result, baseline_timing, baseline_second_id = _baseline_query(
            gallery,
            probe,
            lfw_root,
            cache_path,
            dataset_id,
            extraction_id,
            policy,
            baseline_log,
        )
        optimized_result, optimized_timing, optimized_second_id = _optimized_query(
            gallery.person_ids,
            optimized_matrix,
            probe,
            policy,
            optimized_log,
        )
        _append_timing(baseline_samples, baseline_timing)
        _append_timing(optimized_samples, optimized_timing)
        baseline_e2e.append(_sum_timing(baseline_timing))
        optimized_e2e.append(_sum_timing(optimized_timing))
        signatures.append(
            _compare_decisions(
                baseline_result,
                optimized_result,
                baseline_second_id,
                optimized_second_id,
                probe.relative_path,
            )
        )

    baseline_log.close()
    optimized_log.close()
    for name, values in (ui_timings or {}).items():
        baseline_samples[name] = list(values)
        optimized_samples[name] = list(values)
    baseline_report = {
        "identity_count": gallery.identity_count,
        "gallery_image_count": sum(len(rows) for rows in gallery.samples_by_person.values()),
        "query_count": len(probes),
        "warmup_count": warmup_count,
        "stage_timings": _summarize_timings(baseline_samples),
        "e2e": _summarize_values(baseline_e2e),
        "amortized_e2e": _summarize_values(baseline_e2e),
        "workload_total_ms": float(sum(baseline_e2e)),
        "startup": {"integrity_check_ms": 0.0, "standard_library_prepare_ms": 0.0},
    }
    optimized_report = {
        "identity_count": gallery.identity_count,
        "gallery_image_count": sum(len(rows) for rows in gallery.samples_by_person.values()),
        "query_count": len(probes),
        "warmup_count": warmup_count,
        "stage_timings": _summarize_timings(optimized_samples),
        "e2e": _summarize_values(optimized_e2e),
        "amortized_e2e": _summarize_values(
            [
                value + (optimized_integrity_ms + optimized_prepare_ms) / len(probes)
                for value in optimized_e2e
            ]
        ),
        "workload_total_ms": optimized_integrity_ms
        + optimized_prepare_ms
        + float(sum(optimized_e2e)),
        "startup": {
            "integrity_check_ms": optimized_integrity_ms,
            "standard_library_prepare_ms": optimized_prepare_ms,
            "total_ms": optimized_integrity_ms + optimized_prepare_ms,
            "matrix_bytes": int(optimized_matrix.nbytes),
            "embedding_dimension": int(optimized_matrix.shape[1]),
        },
    }
    exactness = {
        "identity_count": gallery.identity_count,
        "query_count": len(signatures),
        "all_top1_equal": all(row["top1_equal"] for row in signatures),
        "all_second_person_equal": all(row["second_person_equal"] for row in signatures),
        "all_acceptance_equal": all(row["acceptance_equal"] for row in signatures),
        "maximum_top_score_absolute_error": max(
            (row["top_score_absolute_error"] for row in signatures), default=0.0
        ),
        "maximum_second_score_absolute_error": max(
            (row["second_score_absolute_error"] for row in signatures), default=0.0
        ),
        "details": signatures,
    }
    return baseline_report, optimized_report, exactness


def _baseline_query(
    gallery: GalleryScale,
    probe: PreparedProbe,
    lfw_root: Path,
    cache_path: Path,
    dataset_id: str,
    extraction_id: str,
    policy: ScoreThresholdPolicy,
    log_connection: sqlite3.Connection,
) -> tuple[OpenSetDecision, dict[str, float], str | None]:
    """执行逐次完整性检查、SQLite 样本加载、逐人原型和判定基线。"""

    timings = _input_timings(probe)
    started = perf_counter_ns()
    _verify_protocol_gallery(gallery, lfw_root)
    timings["integrity_check_ms"] = _elapsed_ms(started)

    started = perf_counter_ns()
    selected_paths = [path for rows in gallery.samples_by_person.values() for path, _ in rows]
    loaded = _load_embeddings(cache_path, dataset_id, extraction_id, selected_paths)
    samples_by_person = {
        person_id: tuple(
            (path, loaded[path]) for path, _ in rows if path in loaded
        )
        for person_id, rows in gallery.samples_by_person.items()
    }
    _, person_vectors = _build_prototype_matrix(samples_by_person)
    timings["standard_library_prepare_ms"] = _elapsed_ms(started)

    started = perf_counter_ns()
    person_scores = {
        person_id: float(probe.embedding @ person_vectors[index])
        for index, person_id in enumerate(gallery.person_ids)
    }
    timings["candidate_search_ms"] = _elapsed_ms(started)
    started = perf_counter_ns()
    decision = apply_open_set_policy(person_scores, policy)
    timings["decision_ms"] = _elapsed_ms(started)
    timings["log_write_ms"] = _record_benchmark_log(log_connection, decision)
    ranked = sorted(person_scores.items(), key=lambda item: (-item[1], item[0]))
    return decision, timings, ranked[1][0] if len(ranked) > 1 else None


def _optimized_query(
    person_ids: tuple[str, ...],
    matrix: np.ndarray,
    probe: PreparedProbe,
    policy: ScoreThresholdPolicy,
    log_connection: sqlite3.Connection,
) -> tuple[OpenSetDecision, dict[str, float], str | None]:
    """执行内存矩阵检索、前两名选择和判定优化路径。"""

    timings = _input_timings(probe)
    started = perf_counter_ns()
    scores = matrix @ probe.embedding
    top_index = int(np.argmax(scores))
    ranked = [(person_ids[top_index], float(scores[top_index]))]
    if len(person_ids) > 1:
        second_scores = scores.copy()
        second_scores[top_index] = -np.inf
        second_index = int(np.argmax(second_scores))
        ranked.append((person_ids[second_index], float(scores[second_index])))
    timings["candidate_search_ms"] = _elapsed_ms(started)
    started = perf_counter_ns()
    decision = apply_ranked_open_set_policy(ranked, policy)
    timings["decision_ms"] = _elapsed_ms(started)
    timings["log_write_ms"] = _record_benchmark_log(log_connection, decision)
    return decision, timings, ranked[1][0] if len(ranked) > 1 else None


def _input_timings(probe: PreparedProbe) -> dict[str, float]:
    """把预先固定的输入阶段耗时复制到一条链路记录。"""

    return {
        "image_read_ms": probe.image_read_ms,
        "face_inference_ms": probe.face_inference_ms,
        "quality_measurement_ms": probe.quality_measurement_ms,
    }


def _verify_protocol_gallery(gallery: GalleryScale, lfw_root: Path) -> None:
    """扫描当前规模全部标准库图片的 SHA-256，模拟逐次完整性检查。"""

    for rows in gallery.samples_by_person.values():
        for relative_path, _ in rows:
            path = lfw_root / relative_path
            if not path.is_file():
                raise RuntimeError(f"LFW image is missing: {path}")
            file_sha256(path)


def _build_prototype_matrix(
    samples_by_person: dict[str, tuple[tuple[str, np.ndarray], ...]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """从样本向量计算人员平均原型，返回稳定身份顺序和连续矩阵。"""

    person_ids = tuple(sorted(samples_by_person))
    prototypes: list[np.ndarray] = []
    for person_id in person_ids:
        normalized = np.stack([_normalize(vector) for _, vector in samples_by_person[person_id]])
        prototypes.append(_normalize(np.mean(normalized, axis=0)))
    if not prototypes:
        raise ValueError("gallery must contain at least one valid identity")
    return person_ids, np.ascontiguousarray(np.stack(prototypes), dtype=np.float32)


def _record_benchmark_log(connection: sqlite3.Connection, decision: OpenSetDecision) -> float:
    """提交一条内存 SQLite 识别日志并返回写入耗时。"""

    started = perf_counter_ns()
    connection.execute(
        "INSERT INTO recognition_logs (decision, person_id, top_score) VALUES (?, ?, ?)",
        ("matched" if decision.accepted else "unknown", decision.accepted_person_id, decision.top_score),
    )
    connection.commit()
    return _elapsed_ms(started)


def _new_log_connection() -> sqlite3.Connection:
    """创建只用于测量提交开销的内存 SQLite 日志表。"""

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE recognition_logs (decision TEXT, person_id TEXT, top_score REAL)"
    )
    connection.commit()
    return connection


def _prepare_probes(
    *,
    paths: list[str],
    probe_embeddings: dict[str, np.ndarray],
    data_dir: Path,
    lfw_root: Path,
    cached_inputs: bool,
) -> tuple[list[PreparedProbe], dict[str, Any], dict[str, Any]]:
    """准备固定待识别图片，默认真实执行 FaceEngine 并记录实际后端。"""

    settings = load_settings(data_dir)
    engine: FaceEngine | None = None
    backend: dict[str, Any] = {"name": "cached", "providers": [], "context_id": -1}
    if not cached_inputs:
        engine = FaceEngine.from_local_model(settings)
        backend = backend_metadata(engine.backend)

    prepared: list[PreparedProbe] = []
    rejected: list[dict[str, str]] = []
    for relative_path in paths:
        path = lfw_root / relative_path
        try:
            read_started = perf_counter_ns()
            image_input = ImageInput.from_file(path, source_type="dataset")
            image_read_ms = _elapsed_ms(read_started)
            if cached_inputs:
                embedding = _normalize(probe_embeddings[relative_path])
                metrics = {"source": 1.0}
                inference_ms = 0.0
                quality_ms = 0.0
            else:
                assert engine is not None
                inference_started = perf_counter_ns()
                observation = engine.extract_single_face(image_input.frame)
                inference_ms = _elapsed_ms(inference_started)
                quality_started = perf_counter_ns()
                metrics = measure_quality(image_input.frame, observation)
                quality_warnings(metrics, settings.quality_warnings)
                quality_ms = _elapsed_ms(quality_started)
                embedding = observation.embedding
            prepared.append(
                PreparedProbe(
                    relative_path=relative_path,
                    frame=image_input.frame,
                    embedding=_normalize(embedding),
                    quality_metrics=metrics,
                    image_read_ms=image_read_ms,
                    face_inference_ms=inference_ms,
                    quality_measurement_ms=quality_ms,
                )
            )
        except (FaceInputError, OSError, RuntimeError, ValueError, KeyError) as error:
            rejected.append({"relative_path": relative_path, "reason": str(error)})
    return (
        prepared,
        {
            "requested_count": len(paths),
            "prepared_count": len(prepared),
            "rejected": rejected,
            "mean_image_read_ms": _mean([row.image_read_ms for row in prepared]),
            "mean_face_inference_ms": _mean([row.face_inference_ms for row in prepared]),
            "mean_quality_measurement_ms": _mean(
                [row.quality_measurement_ms for row in prepared]
            ),
        },
        backend,
    )


def _measure_ui_stages(probes: list[PreparedProbe]) -> dict[str, list[float]]:
    """测量轻量 Qt 工作线程往返和结果画面更新开销。"""

    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QLabel
    except ImportError:
        return {"worker_thread_round_trip_ms": [0.0] * len(probes), "ui_update_ms": [0.0] * len(probes)}

    class EchoThread(QThread):
        """只用于测量线程启动、信号回到主线程和结束的最小开销。"""

        completed = Signal()

        def run(self) -> None:
            """发出一次完成信号。"""

            self.completed.emit()

    application = QApplication.instance() or QApplication([])
    label = QLabel()
    worker_values: list[float] = []
    ui_values: list[float] = []
    for probe in probes:
        worker = EchoThread()
        worker.completed.connect(application.quit)
        started = perf_counter_ns()
        worker.start()
        application.exec()
        worker.wait()
        worker_values.append(_elapsed_ms(started))

        rgb = np.ascontiguousarray(probe.frame[:, :, ::-1])
        started = perf_counter_ns()
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 3,
            QImage.Format_RGB888,
        ).copy()
        label.setPixmap(QPixmap.fromImage(image))
        application.processEvents()
        ui_values.append(_elapsed_ms(started))
    label.deleteLater()
    return {
        "worker_thread_round_trip_ms": worker_values,
        "ui_update_ms": ui_values,
    }


def _compare_decisions(
    baseline: OpenSetDecision,
    optimized: OpenSetDecision,
    baseline_second_id: str | None,
    optimized_second_id: str | None,
    relative_path: str,
) -> dict[str, Any]:
    """比较两条路径的候选身份、接收结果和分数误差。"""

    return {
        "relative_path": relative_path,
        "top1_equal": baseline.top_person_id == optimized.top_person_id,
        "second_person_equal": baseline_second_id == optimized_second_id,
        "acceptance_equal": baseline.accepted_person_id == optimized.accepted_person_id,
        "top_score_absolute_error": _score_error(baseline.top_score, optimized.top_score),
        "second_score_absolute_error": _score_error(
            baseline.second_score, optimized.second_score
        ),
    }

def _score_error(left: float | None, right: float | None) -> float:
    """计算两个可空浮点分数的绝对误差。"""

    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    return abs(float(left) - float(right))


def _append_timing(target: dict[str, list[float]], values: dict[str, float]) -> None:
    """把一条阶段耗时追加到对应数组。"""

    for name, value in values.items():
        target.setdefault(name, []).append(float(value))


def _sum_timing(values: dict[str, float]) -> float:
    """求一条完整链路的阶段耗时总和。"""

    return float(sum(values.values()))


def _summarize_timings(values: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    """为每个阶段计算平均值、中位数和 95% 分位延迟。"""

    return {name: _summarize_values(rows) for name, rows in values.items()}


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    """计算一组耗时的平均值、p50 和 p95。"""

    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _select_cache_partition(path: Path) -> tuple[str, str]:
    """选择自然 LFW 缓存中有效向量最多的模型提取分区。"""

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            """
            SELECT dataset_id, embedding_extraction_id, COUNT(*)
            FROM raw_embeddings
            WHERE status = 'observed'
            GROUP BY dataset_id, embedding_extraction_id
            ORDER BY COUNT(*) DESC
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
    paths: list[str],
) -> dict[str, np.ndarray]:
    """分块读取缓存中的有效向量。"""

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


def _read_enrollment(protocol: dict[str, Any]) -> dict[str, list[str]]:
    """读取固定协议中的人员标准库图片。"""

    raw = protocol.get("enrollment")
    if not isinstance(raw, dict):
        raise TypeError("protocol.enrollment must be an object")
    return {
        str(person_id): [str(path) for path in paths]
        for person_id, paths in raw.items()
        if isinstance(paths, list)
    }


def _read_probe_paths(protocol: dict[str, Any]) -> list[str]:
    """读取固定协议中的待识别图片路径。"""

    raw = protocol.get("probes")
    if not isinstance(raw, list):
        raise TypeError("protocol.probes must be an array")
    return [str(row["relative_path"]) for row in raw if isinstance(row, dict)]


def _select_probe_paths(
    embeddings: dict[str, np.ndarray], count: int, seed: int
) -> list[str]:
    """用固定种子选择一组可复现的待识别图片。"""

    paths = sorted(embeddings)
    random.Random(seed).shuffle(paths)
    if len(paths) < count:
        raise ValueError(f"requested {count} probes but only {len(paths)} are cached")
    return paths[:count]


def _parse_scales(value: str, maximum: int) -> tuple[int, ...]:
    """解析身份规模并拒绝超过有效标准库数量的值。"""

    scales: list[int] = []
    for item in value.split(","):
        scale = int(item.strip())
        if scale < 1 or scale > maximum:
            raise ValueError(f"invalid gallery scale: {item}")
        if scale not in scales:
            scales.append(scale)
    return tuple(scales)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """返回有限非零 float32 单位向量。"""

    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("embedding must have a finite positive norm")
    return value / norm


def _elapsed_ms(started_ns: int) -> float:
    """把计时起点转换为当前经过的毫秒数。"""

    return (perf_counter_ns() - started_ns) / 1_000_000


def _mean(values: list[float]) -> float:
    """计算可能为空的浮点列表平均值。"""

    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _nvidia_smi() -> str | None:
    """记录 NVIDIA 驱动信息；当前环境不可用时返回空值。"""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
