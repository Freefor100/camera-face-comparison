from __future__ import annotations

import argparse
import json
import platform
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep
from typing import Any

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.experiment_artifacts import write_json_atomic
from camera_face_comparison.face_engine import (
    DetectedFace,
    FaceEngine,
    FaceObservation,
    normalize_embedding,
)
from camera_face_comparison.face_library import FaceLibrarySnapshot
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.recognition import RecognitionService
from camera_face_comparison.repository import FaceRepository
from camera_face_comparison.runtime import backend_metadata
from camera_face_comparison.ui.workers import RecognitionInputStream

FRAME_COUNT = 5
FRAME_INTERVAL_MS = 80.0
DEFAULT_IDENTITY_COUNT = 4_588
DEFAULT_SEQUENCE_COUNT = 10


class MeasuredSplitEngine:
    """记录检测与身份特征模型的实际调用次数和耗时。"""

    def __init__(self, engine: FaceEngine) -> None:
        """包装同一个真实人脸引擎，不改变其推理结果。"""

        self._engine = engine
        self.detection_count = 0
        self.embedding_count = 0
        self.detection_ms = 0.0
        self.embedding_ms = 0.0

    def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
        """执行并记录一次单脸检测。"""

        started_at = perf_counter()
        self.detection_count += 1
        try:
            return self._engine.detect_single_face(frame)
        finally:
            self.detection_ms += _elapsed_ms(started_at)

    def extract_detected_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> FaceObservation:
        """执行并记录一次五点对齐和身份特征提取。"""

        started_at = perf_counter()
        self.embedding_count += 1
        try:
            return self._engine.extract_detected_face(frame, detected_face)
        finally:
            self.embedding_ms += _elapsed_ms(started_at)


class LegacyFullFrameEngine(MeasuredSplitEngine):
    """模拟旧多帧路径：每张检测成功的帧都立即提取身份特征。"""

    def __init__(self, engine: FaceEngine) -> None:
        """创建一条只用于对照实验的旧路径适配器。"""

        super().__init__(engine)
        self._observations: dict[int, FaceObservation] = {}

    def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
        """连续执行检测与特征提取，并暂存完整结果供服务读取。"""

        detected = super().detect_single_face(frame)
        observation = super().extract_detected_face(frame, detected)
        self._observations[id(frame)] = observation
        return detected

    def extract_detected_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> FaceObservation:
        """返回旧路径已经为该帧算好的身份特征，不再次运行模型。"""

        del detected_face
        return self._observations.pop(id(frame))


class TimingCollector:
    """收集一次真实识别服务调用中各阶段的耗时。"""

    def __init__(self) -> None:
        """创建空的阶段耗时收集器。"""
        self.values: dict[str, list[float]] = defaultdict(list)

    def __call__(self, stage: str, elapsed_ms: float) -> None:
        """接收识别服务报告的一个阶段耗时。"""
        self.values[stage].append(float(elapsed_ms))

    def reset(self) -> None:
        """清空上一条识别记录，准备收集下一条记录。"""
        self.values.clear()

    def total(self, stage: str) -> float:
        """返回当前识别中某阶段的累计耗时。"""
        return float(sum(self.values.get(stage, ())))


def parse_args() -> argparse.Namespace:
    """读取生产识别链路回放实验参数。"""
    parser = argparse.ArgumentParser(
        description="使用真实 RecognitionService 分解单帧和五帧识别耗时"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--lfw-root", type=Path, default=Path("data/datasets/lfw_funneled"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("data/experiments/phase4/protocol.json")
    )
    parser.add_argument("--cache", type=Path, default=Path("data/logs/cache/lfw_raw.sqlite"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/phase5d/split-inference-pipeline"),
    )
    parser.add_argument("--identity-count", type=int, default=DEFAULT_IDENTITY_COUNT)
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--frame-count", type=int, default=FRAME_COUNT)
    return parser.parse_args()


def main() -> int:
    """运行真实模型回放，并原子写入阶段耗时报告。"""
    args = parse_args()
    if args.identity_count < 1 or args.sequence_count < 1:
        raise ValueError("identity-count and sequence-count must be positive")
    if args.warmup_count < 0 or args.frame_count < 1:
        raise ValueError("warmup-count must be non-negative and frame-count must be positive")

    data_dir = args.data_dir.resolve()
    lfw_root = args.lfw_root.resolve()
    protocol_path = args.protocol.resolve()
    cache_path = args.cache.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_id, extraction_id = _select_cache_partition(cache_path)
    enrollment = protocol["enrollment"]
    cached = _read_cached_embeddings(cache_path, dataset_id, extraction_id)
    snapshot_started = perf_counter()
    gallery_ids, gallery_matrix = _build_gallery_matrix(
        enrollment, cached, args.identity_count
    )
    snapshot_build_ms = _elapsed_ms(snapshot_started)
    snapshot = FaceLibrarySnapshot(
        person_ids=gallery_ids,
        display_names=gallery_ids,
        prototype_matrix=gallery_matrix,
        revision=0,
    )
    sequences = _build_sequences(
        protocol["probes"],
        gallery_ids,
        cached,
        lfw_root,
        args.sequence_count,
        args.frame_count,
    )

    settings = load_settings(data_dir)
    model_started = perf_counter()
    engine = FaceEngine.from_local_model(settings)
    model_load_ms = _elapsed_ms(model_started)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_repository = _open_log_repository(output_dir / "recognition_log.sqlite", gallery_ids)
    try:
        single_records = _run_single_frame(
            sequences,
            args.warmup_count,
            engine,
            settings,
            snapshot,
            log_repository,
        )
        legacy_multi_records = _run_multi_frame(
            sequences,
            args.warmup_count,
            engine,
            settings,
            snapshot,
            log_repository,
            args.frame_count,
            strategy="legacy_collect_then_full_inference",
        )
        streaming_multi_records = _run_multi_frame(
            sequences,
            args.warmup_count,
            engine,
            settings,
            snapshot,
            log_repository,
            args.frame_count,
            strategy="stream_capture_and_detect",
        )
    finally:
        log_repository.close()

    report = {
        "artifact": "production-runtime-replay-v2",
        "dataset": "LFW natural images",
        "protocol": str(protocol_path),
        "cache": {"path": str(cache_path), "dataset_id": dataset_id, "extraction_id": extraction_id},
        "gallery": {"identity_count": len(gallery_ids), "embedding_dimension": int(gallery_matrix.shape[1])},
        "sequences": {
            "count": len(sequences),
            "frame_count": args.frame_count,
            "frame_interval_ms": FRAME_INTERVAL_MS,
            "paths": [[str(path) for path in sequence] for sequence in sequences],
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "model_backend": backend_metadata(engine.backend),
        },
        "startup": {
            "model_load_ms": model_load_ms,
            "standard_library_snapshot_ms": snapshot_build_ms,
            "integrity_check_ms": None,
            "note": "标准库快照由已缓存 embedding 构造；应用启动完整性检查沿用阶段 5C 独立报告，未在本回放重复计时。",
        },
        "cases": {
            "single_local_image": _summarize_case(single_records),
            "camera_legacy_collect_then_full_inference": _summarize_case(
                legacy_multi_records
            ),
            "camera_stream_capture_and_detect": _summarize_case(
                streaming_multi_records
            ),
        },
        "equivalence": _compare_camera_results(
            legacy_multi_records,
            streaming_multi_records,
        ),
        "scope": {
            "measured": [
                "本地图片解码",
                "真实 FaceEngine 人脸检测",
                "真实 FaceEngine 五点对齐与身份特征提取",
                "质量指标计算",
                "五帧清晰度选择",
                "五帧采集时间与检测计算重叠",
                "CPU 标准库矩阵检索",
                "开放集判定",
                "SQLite 识别日志写入",
                "RecognitionService 调用总耗时",
            ],
            "not_measured": [
                "真实摄像头驱动取帧；实验用线程按 80 毫秒间隔提交 LFW 图片",
                "真实 QThread 往返和主窗口屏幕合成",
                "摄像头预览与 CUDA 推理同时运行时的设备资源竞争",
            ],
        },
    }
    write_json_atomic(output_dir / "report.json", report)
    write_json_atomic(
        output_dir / "records.json",
        {
            "single": single_records,
            "camera_legacy": legacy_multi_records,
            "camera_streaming": streaming_multi_records,
        },
    )
    print(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    print(f"reports: {output_dir}")
    return 0


def _run_single_frame(
    sequences: list[list[Path]],
    warmup_count: int,
    engine: FaceEngine,
    settings: Any,
    snapshot: FaceLibrarySnapshot,
    repository: FaceRepository,
) -> list[dict[str, Any]]:
    """用每个序列的第一张图片回放本地单帧识别。"""
    inputs = [[sequence[0]] for sequence in sequences]
    return _run_case(
        inputs,
        warmup_count,
        engine,
        settings,
        snapshot,
        repository,
        1,
        strategy="single_image",
    )


def _run_multi_frame(
    sequences: list[list[Path]],
    warmup_count: int,
    engine: FaceEngine,
    settings: Any,
    snapshot: FaceLibrarySnapshot,
    repository: FaceRepository,
    frame_count: int,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    """用同身份五张 LFW 图片回放一种摄像头识别执行策略。"""
    return _run_case(
        sequences,
        warmup_count,
        engine,
        settings,
        snapshot,
        repository,
        frame_count,
        strategy=strategy,
    )


def _run_case(
    sequences: list[list[Path]],
    warmup_count: int,
    engine: FaceEngine,
    settings: Any,
    snapshot: FaceLibrarySnapshot,
    repository: FaceRepository,
    frame_count: int,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    """执行一组真实 RecognitionService 调用并保留每条阶段数据。"""
    collector = TimingCollector()
    for index in range(warmup_count):
        _run_once(
            sequences[index % len(sequences)],
            engine,
            settings,
            snapshot,
            repository,
            collector,
            frame_count,
            strategy=strategy,
            keep=False,
        )
    records: list[dict[str, Any]] = []
    for sequence in sequences:
        records.append(
            _run_once(
                sequence,
                engine,
                settings,
                snapshot,
                repository,
                collector,
                frame_count,
                strategy=strategy,
                keep=True,
            )
        )
    return records


def _run_once(
    paths: list[Path],
    engine: FaceEngine,
    settings: Any,
    snapshot: FaceLibrarySnapshot,
    repository: FaceRepository,
    collector: TimingCollector,
    frame_count: int,
    *,
    strategy: str,
    keep: bool,
) -> dict[str, Any]:
    """读取一条输入并调用生产识别服务，返回实际阶段计时。"""
    collector.reset()
    read_started = perf_counter()
    inputs = tuple(ImageInput.from_file(path, source_type="dataset") for path in paths)
    input_decode_ms = _elapsed_ms(read_started)
    measured_engine: MeasuredSplitEngine
    if strategy == "legacy_collect_then_full_inference":
        measured_engine = LegacyFullFrameEngine(engine)
    else:
        measured_engine = MeasuredSplitEngine(engine)
    service = RecognitionService(
        repository,
        settings,
        measured_engine,
        snapshot,
        timing_sink=collector,
    )
    request_started_at = perf_counter()
    if strategy == "single_image":
        result = service.compare_input(inputs[0], started_at=request_started_at)
    elif strategy == "legacy_collect_then_full_inference":
        _wait_until_last_camera_frame(request_started_at, frame_count)
        result = service.compare_inputs(
            inputs,
            frame_count=frame_count,
            started_at=request_started_at,
        )
    elif strategy == "stream_capture_and_detect":
        input_stream = RecognitionInputStream()
        producer = Thread(
            target=_submit_camera_frames,
            args=(input_stream, inputs, request_started_at),
            daemon=True,
        )
        producer.start()
        result = service.compare_inputs(
            input_stream,
            frame_count=frame_count,
            started_at=request_started_at,
        )
        producer.join()
    else:
        raise ValueError(f"unknown replay strategy: {strategy}")
    click_to_result_ms = _elapsed_ms(request_started_at)
    if not keep:
        return {}
    record = {
        "strategy": strategy,
        "input_decode_ms": input_decode_ms,
        "face_detection_ms": measured_engine.detection_ms,
        "embedding_extraction_ms": measured_engine.embedding_ms,
        "detection_call_count": measured_engine.detection_count,
        "embedding_call_count": measured_engine.embedding_count,
        "quality_measurement_ms": collector.total("quality_measurement_ms"),
        "frame_selection_ms": collector.total("frame_selection_ms"),
        "candidate_search_ms": collector.total("candidate_search_ms"),
        "decision_ms": collector.total("decision_ms"),
        "log_write_ms": collector.total("log_write_ms"),
        "click_to_result_ms": click_to_result_ms,
        "service_reported_latency_ms": result.latency_ms,
        "replay_total_ms": input_decode_ms + click_to_result_ms,
        "capture_window_ms": 0.0 if frame_count == 1 else (frame_count - 1) * FRAME_INTERVAL_MS,
        "status": result.status,
        "person_id": result.person_id,
        "top_score": result.top_score,
        "second_score": result.second_score,
        "valid_frame_count": result.valid_frame_count,
        "selected_frame_index": result.selected_frame_index,
    }
    return record


def _wait_until_last_camera_frame(started_at: float, frame_count: int) -> None:
    """模拟旧路径先收齐摄像头帧，再开始模型处理。"""

    deadline = started_at + ((frame_count - 1) * FRAME_INTERVAL_MS / 1000.0)
    remaining = deadline - perf_counter()
    if remaining > 0.0:
        sleep(remaining)


def _submit_camera_frames(
    input_stream: RecognitionInputStream,
    inputs: tuple[ImageInput, ...],
    started_at: float,
) -> None:
    """按摄像头采集时刻提交帧，使检测能与后续等待重叠。"""

    for index, image_input in enumerate(inputs):
        deadline = started_at + (index * FRAME_INTERVAL_MS / 1000.0)
        remaining = deadline - perf_counter()
        if remaining > 0.0:
            sleep(remaining)
        input_stream.submit(image_input)
    input_stream.finish()


def _build_sequences(
    probe_rows: list[dict[str, Any]],
    gallery_ids: tuple[str, ...],
    cached: dict[str, np.ndarray],
    lfw_root: Path,
    sequence_count: int,
    frame_count: int,
) -> list[list[Path]]:
    """按身份选择固定多帧图片，确保每条回放序列不会混入不同身份。"""
    by_identity: dict[str, list[Path]] = defaultdict(list)
    for row in probe_rows:
        person_id = row.get("expected_person_id")
        path = str(row["relative_path"])
        if person_id in gallery_ids and path in cached and (lfw_root / path).is_file():
            by_identity[str(person_id)].append(lfw_root / path)
    sequences: list[list[Path]] = []
    for person_id in sorted(by_identity):
        paths = by_identity[person_id]
        if len(paths) >= frame_count:
            sequences.append(paths[:frame_count])
        if len(sequences) == sequence_count:
            break
    if len(sequences) < sequence_count:
        raise RuntimeError(
            f"only {len(sequences)} same-identity sequences are available; "
            f"requested {sequence_count}"
        )
    return sequences


def _build_gallery_matrix(
    enrollment: dict[str, list[str]],
    cached: dict[str, np.ndarray],
    identity_count: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """从缓存样本重建与应用一致的人员平均特征矩阵。"""
    entries: list[tuple[str, np.ndarray]] = []
    for person_id in sorted(enrollment):
        vectors = [cached[path] for path in enrollment[person_id] if path in cached]
        if not vectors:
            continue
        normalized = np.stack([normalize_embedding(vector) for vector in vectors])
        entries.append((person_id, normalize_embedding(np.mean(normalized, axis=0))))
    entries = entries[:identity_count]
    if len(entries) < identity_count:
        raise RuntimeError(f"only {len(entries)} valid identities are available")
    matrix = np.ascontiguousarray(np.stack([vector for _, vector in entries]), dtype=np.float32)
    matrix.setflags(write=False)
    return tuple(person_id for person_id, _ in entries), matrix


def _read_cached_embeddings(
    path: Path,
    dataset_id: str,
    extraction_id: str,
) -> dict[str, np.ndarray]:
    """读取原始 embedding 缓存中的成功记录，不启动人脸模型。"""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT relative_path, embedding_blob, embedding_dim
            FROM raw_embeddings
            WHERE dataset_id = ? AND embedding_extraction_id = ? AND status = 'observed'
            """,
            (dataset_id, extraction_id),
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, np.ndarray] = {}
    for relative_path, blob, dimension in rows:
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size != int(dimension):
            raise RuntimeError(f"invalid cached embedding dimension: {relative_path}")
        result[str(relative_path)] = vector
    if not result:
        raise RuntimeError("no observed embedding exists in the selected cache")
    return result


def _open_log_repository(path: Path, person_ids: tuple[str, ...]) -> FaceRepository:
    """创建只用于实验日志的仓库，并预置矩阵中的身份外键。"""
    repository = FaceRepository(path)
    repository.close()
    connection = sqlite3.connect(path)
    created_at = datetime.now(UTC).isoformat()
    connection.executemany(
        "INSERT OR IGNORE INTO persons (id, display_name, created_at) VALUES (?, ?, ?)",
        ((person_id, person_id, created_at) for person_id in person_ids),
    )
    connection.commit()
    connection.close()
    return FaceRepository(path)


def _select_cache_partition(path: Path) -> tuple[str, str]:
    """选择 LFW 原始缓存中唯一的数据集和提取版本。"""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT dataset_id, embedding_extraction_id FROM raw_embeddings"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise RuntimeError(f"expected one cache partition, found {len(rows)}")
    return str(rows[0][0]), str(rows[0][1])


def _summarize_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    """为一类实际回放记录计算平均值、中位数和 95% 分位数。"""
    if not records:
        return {"record_count": 0, "stages": {}}
    names = tuple(records[0].keys())
    summary: dict[str, Any] = {"record_count": len(records), "stages": {}}
    for name in names:
        values = [row[name] for row in records if isinstance(row.get(name), (int, float))]
        if not values:
            continue
        summary["stages"][name] = {
            "mean": float(np.mean(values)),
            "median": float(np.percentile(values, 50)),
            "percentile_95": float(np.percentile(values, 95)),
        }
    summary["status_counts"] = {
        status: sum(row["status"] == status for row in records)
        for status in sorted({row["status"] for row in records})
    }
    return summary


def _compare_camera_results(
    legacy_records: list[dict[str, Any]],
    streaming_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查新旧路径的候选、判定和分数是否保持一致。"""

    if len(legacy_records) != len(streaming_records):
        raise ValueError("camera replay record counts do not match")
    mismatches: list[dict[str, Any]] = []
    maximum_score_error = 0.0
    for index, (legacy, streaming) in enumerate(
        zip(legacy_records, streaming_records, strict=True)
    ):
        score_errors = []
        for field in ("top_score", "second_score"):
            left = legacy[field]
            right = streaming[field]
            if left is not None and right is not None:
                score_errors.append(abs(float(left) - float(right)))
        maximum_score_error = max(maximum_score_error, *score_errors, 0.0)
        same_result = all(
            legacy[field] == streaming[field]
            for field in (
                "status",
                "person_id",
                "valid_frame_count",
                "selected_frame_index",
            )
        )
        if not same_result or any(error > 1e-6 for error in score_errors):
            mismatches.append(
                {
                    "record_index": index,
                    "legacy": {
                        field: legacy[field]
                        for field in (
                            "status",
                            "person_id",
                            "top_score",
                            "second_score",
                            "valid_frame_count",
                            "selected_frame_index",
                        )
                    },
                    "streaming": {
                        field: streaming[field]
                        for field in (
                            "status",
                            "person_id",
                            "top_score",
                            "second_score",
                            "valid_frame_count",
                            "selected_frame_index",
                        )
                    },
                }
            )
    return {
        "record_count": len(legacy_records),
        "mismatch_count": len(mismatches),
        "maximum_score_absolute_error": maximum_score_error,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    """提取适合终端查看的简短结果。"""
    result: dict[str, Any] = {
        "model_backend": report["hardware"]["model_backend"],
        "model_load_ms": report["startup"]["model_load_ms"],
    }
    for name, case in report["cases"].items():
        stages = case["stages"]
        result[name] = {
            "records": case["record_count"],
            "replay_total_median_ms": stages["replay_total_ms"]["median"],
            "click_to_result_median_ms": stages["click_to_result_ms"]["median"],
            "face_detection_median_ms": stages["face_detection_ms"]["median"],
            "embedding_extraction_median_ms": stages["embedding_extraction_ms"]["median"],
            "detection_calls_median": stages["detection_call_count"]["median"],
            "embedding_calls_median": stages["embedding_call_count"]["median"],
            "candidate_search_median_ms": stages["candidate_search_ms"]["median"],
        }
    result["equivalence"] = report["equivalence"]
    return result


def _elapsed_ms(started_at: float) -> float:
    """返回从指定单调时钟起点开始经过的毫秒数。"""
    return (perf_counter() - started_at) * 1000


if __name__ == "__main__":
    raise SystemExit(main())
