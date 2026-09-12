from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from camera_face_comparison.experiment_artifacts import file_sha256, write_json_atomic
from camera_face_comparison.multi_frame import (
    MultiFrameObservation,
    select_consistent_frame,
    select_mean_embedding,
    select_sharpest_frame,
)
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy, apply_open_set_policy

FROZEN_MINIMUM_SCORE = 0.5557855367660522
SCENARIOS: dict[str, tuple[str, ...]] = {
    "mild_comprehensive_change": (
        "baseline:1",
        "brightness:0.75",
        "contrast:0.75",
        "gaussian_blur:1",
        "face_size:112",
    ),
    "transient_blur": (
        "baseline:1",
        "gaussian_blur:1",
        "gaussian_blur:2",
        "gaussian_blur:3",
        "baseline:1",
    ),
    "brightness_fluctuation": (
        "baseline:1",
        "brightness:0.75",
        "brightness:0.5",
        "brightness:1.25",
        "baseline:1",
    ),
    "face_size_change": (
        "baseline:1",
        "face_size:160",
        "face_size:112",
        "face_size:80",
        "face_size:96",
    ),
    "mixed_change": (
        "baseline:1",
        "gaussian_blur:2",
        "brightness:0.5",
        "contrast:0.5",
        "face_size:80",
    ),
}
SELECTORS: dict[str, Callable[[Sequence[MultiFrameObservation]], MultiFrameObservation]] = {
    "sharpest_frame": select_sharpest_frame,
    "consistent_frame": select_consistent_frame,
    "mean_embedding": select_mean_embedding,
}
SINGLE_STRATEGY = "single_frame"


@dataclass(frozen=True)
class ProtocolEntry:
    """阶段 3 协议中的一条已知或未知身份待识别记录。"""

    probe_path: str
    identity: str


def parse_args() -> argparse.Namespace:
    """读取多帧策略实验参数。"""

    parser = argparse.ArgumentParser(description="比较阶段 3 数据上的多帧人脸识别策略")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--phase3-protocol",
        type=Path,
        default=Path("data/experiments/phase3/protocol.json"),
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/experiments/phase3/measurements.sqlite"),
    )
    parser.add_argument(
        "--raw-cache", type=Path, default=Path("data/logs/cache/lfw_raw.sqlite")
    )
    parser.add_argument(
        "--phase4-protocol",
        type=Path,
        default=Path("data/experiments/phase4/protocol.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/phase5c/multiframe"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-known-fraction", type=float, default=0.7)
    parser.add_argument("--calibration-unknown-fraction", type=float, default=2 / 3)
    return parser.parse_args()


def main() -> int:
    """运行多帧策略选择集和身份互斥验证集实验。"""

    args = parse_args()
    phase3_protocol_path = args.phase3_protocol.resolve()
    measurements_path = args.measurements.resolve()
    raw_cache_path = args.raw_cache.resolve()
    phase4_protocol_path = args.phase4_protocol.resolve()
    output_dir = args.output_dir.resolve()
    if not 0.0 < args.calibration_known_fraction < 1.0:
        raise ValueError("calibration-known-fraction must be between 0 and 1")
    if not 0.0 < args.calibration_unknown_fraction < 1.0:
        raise ValueError("calibration-unknown-fraction must be between 0 and 1")

    phase3_protocol = _load_json(phase3_protocol_path)
    phase4_protocol = _load_json(phase4_protocol_path)
    known_entries = tuple(
        ProtocolEntry(str(row["probe_path"]), str(row["person_id"]))
        for row in phase3_protocol["known"]
    )
    unknown_entries = tuple(
        ProtocolEntry(str(row["probe_path"]), str(row["source_identity"]))
        for row in phase3_protocol["unknown"]
    )
    conditions = sorted({condition for sequence in SCENARIOS.values() for condition in sequence})
    measurements = _load_measurements(measurements_path, conditions, known_entries, unknown_entries)
    gallery = _load_gallery(raw_cache_path, phase4_protocol)
    if not gallery:
        raise RuntimeError("no valid LFW gallery identities were loaded")

    known_calibration, known_evaluation = _split_by_identity(
        known_entries, args.calibration_known_fraction, args.seed
    )
    unknown_calibration, unknown_evaluation = _split_by_identity(
        unknown_entries, args.calibration_unknown_fraction, args.seed + 1
    )
    policy = ScoreThresholdPolicy(minimum_score=FROZEN_MINIMUM_SCORE)
    calibration = _evaluate_split(
        known_calibration,
        unknown_calibration,
        measurements,
        gallery,
        policy,
    )
    selected_strategy = _select_candidate_strategy(calibration)
    evaluation = _evaluate_split(
        known_evaluation,
        unknown_evaluation,
        measurements,
        gallery,
        policy,
    )
    selection = _selection_summary(calibration, evaluation, selected_strategy)

    manifest = {
        "artifact": "multi-frame-recognition-experiment-v1",
        "status": "completed",
        "git_revision": _git_revision(),
        "threshold": FROZEN_MINIMUM_SCORE,
        "seed": args.seed,
        "phase3_protocol": {
            "path": str(phase3_protocol_path),
            "sha256": file_sha256(phase3_protocol_path),
        },
        "measurements": {"path": str(measurements_path), "sha256": file_sha256(measurements_path)},
        "raw_gallery_cache": {"path": str(raw_cache_path), "sha256": file_sha256(raw_cache_path)},
        "phase4_protocol": {
            "path": str(phase4_protocol_path),
            "sha256": file_sha256(phase4_protocol_path),
        },
        "gallery_identity_count": len(gallery),
        "known_identity_count": len({entry.identity for entry in known_entries}),
        "unknown_identity_count": len({entry.identity for entry in unknown_entries}),
        "scenario_conditions": SCENARIOS,
        "selection_methods": [SINGLE_STRATEGY, *SELECTORS],
        "input_note": "使用阶段 3 已缓存的模型 embedding 和五项原始质量指标，不重复运行人脸模型；模拟序列不包含原始画面",
        "runtime_note": "本实验不改变阈值，也不代表摄像头最终画面；摄像头运行时只有实验通过独立身份门槛后才接入",
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_json_atomic(output_dir / "calibration_report.json", calibration | {"split": "calibration"})
    write_json_atomic(output_dir / "evaluation_report.json", evaluation | {"split": "identity_disjoint_evaluation"})
    write_json_atomic(output_dir / "comparison_report.json", selection)
    print(f"calibration selected: {selected_strategy or 'none'}")
    print(f"reports: {output_dir}")
    return 0


def _evaluate_split(
    known_entries: Sequence[ProtocolEntry],
    unknown_entries: Sequence[ProtocolEntry],
    measurements: Mapping[tuple[str, str], MultiFrameObservation],
    gallery: Mapping[str, np.ndarray],
    policy: ScoreThresholdPolicy,
) -> dict[str, Any]:
    """在一个身份分区上统计单帧和三种多帧策略。"""

    scenario_results: dict[str, Any] = {}
    for scenario_name, condition_sequence in SCENARIOS.items():
        scenario_results[scenario_name] = _evaluate_scenario(
            known_entries,
            unknown_entries,
            condition_sequence,
            measurements,
            gallery,
            policy,
        )

    summary: dict[str, Any] = {
        "known_identity_count": len({entry.identity for entry in known_entries}),
        "known_probe_count": len(known_entries),
        "unknown_identity_count": len({entry.identity for entry in unknown_entries}),
        "unknown_probe_count": len(unknown_entries),
        "scenario_results": scenario_results,
        "strategy_summary": {},
    }
    for strategy in [SINGLE_STRATEGY, *SELECTORS]:
        summary["strategy_summary"][strategy] = _summarize_strategy(scenario_results, strategy)
    return summary


def _evaluate_scenario(
    known_entries: Sequence[ProtocolEntry],
    unknown_entries: Sequence[ProtocolEntry],
    condition_sequence: Sequence[str],
    measurements: Mapping[tuple[str, str], MultiFrameObservation],
    gallery: Mapping[str, np.ndarray],
    policy: ScoreThresholdPolicy,
) -> dict[str, Any]:
    """统计一个五帧场景的所有策略和单帧位置。"""

    known_stats = {
        strategy: _new_known_stats()
        for strategy in [SINGLE_STRATEGY, *SELECTORS]
    }
    unknown_stats = {
        strategy: _new_unknown_stats()
        for strategy in [SINGLE_STRATEGY, *SELECTORS]
    }
    for position in range(len(condition_sequence)):
        known_stats[f"single_frame_position_{position + 1}"] = _new_known_stats()
        unknown_stats[f"single_frame_position_{position + 1}"] = _new_unknown_stats()

    for entry in known_entries:
        frames = _get_frames(entry.probe_path, condition_sequence, measurements)
        if frames is None:
            continue
        for position, frame in enumerate(frames):
            _record_known(known_stats[f"single_frame_position_{position + 1}"], _decide(frame.embedding, gallery, policy), entry.identity)
        for strategy, selected in _selected_embeddings(frames).items():
            _record_known(known_stats[strategy], _decide(selected.embedding, gallery, policy), entry.identity)

    for entry in unknown_entries:
        frames = _get_frames(entry.probe_path, condition_sequence, measurements)
        if frames is None:
            continue
        for position, frame in enumerate(frames):
            _record_unknown(unknown_stats[f"single_frame_position_{position + 1}"], _decide(frame.embedding, gallery, policy))
        for strategy, selected in _selected_embeddings(frames).items():
            _record_unknown(unknown_stats[strategy], _decide(selected.embedding, gallery, policy))

    return {
        "conditions": list(condition_sequence),
        "known": known_stats,
        "unknown": unknown_stats,
    }


def _selected_embeddings(
    frames: Sequence[MultiFrameObservation],
) -> dict[str, MultiFrameObservation]:
    """返回三种多帧策略选中的观察。"""

    return {name: selector(frames) for name, selector in SELECTORS.items()}


def _summarize_strategy(scenarios: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    """汇总某策略跨场景的最差已知接收率和最大未知误接收率。"""

    known_values: list[float] = []
    unknown_values: list[float] = []
    cells: dict[str, Any] = {}
    for scenario_name, scenario in scenarios.items():
        actual_strategy = strategy
        if strategy == SINGLE_STRATEGY:
            positions = [
                f"single_frame_position_{index}"
                for index in range(1, len(scenario["conditions"]) + 1)
            ]
            position_cells = {
                position: _cell_metrics(scenario, position) for position in positions
            }
            known_values.extend(cell["tpir_e2e"] for cell in position_cells.values())
            unknown_values.extend(cell["fpir"] for cell in position_cells.values())
            cells[scenario_name] = {
                "positions": position_cells,
                "worst_tpir_e2e": min(cell["tpir_e2e"] for cell in position_cells.values()),
                "maximum_fpir": max(cell["fpir"] for cell in position_cells.values()),
            }
            continue
        cell = _cell_metrics(scenario, actual_strategy)
        known_values.append(cell["tpir_e2e"])
        unknown_values.append(cell["fpir"])
        cells[scenario_name] = cell
    return {
        "worst_tpir_e2e": min(known_values, default=0.0),
        "maximum_fpir": max(unknown_values, default=0.0),
        "scenario_cells": cells,
    }


def _cell_metrics(scenario: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    """提取一个场景和策略的已知、未知统计指标。"""

    known = scenario["known"][strategy]
    unknown = scenario["unknown"][strategy]
    return {
        "known_total": known["total"],
        "known_correct_accept": known["correct_accept"],
        "known_wrong_identity_accept": known["wrong_identity_accept"],
        "tpir_e2e": known["correct_accept"] / known["total"] if known["total"] else 0.0,
        "unknown_total": unknown["total"],
        "unknown_false_accept": unknown["false_accept"],
        "fpir": unknown["false_accept"] / unknown["total"] if unknown["total"] else 0.0,
    }


def _select_candidate_strategy(calibration: Mapping[str, Any]) -> str | None:
    """按未知误接收率和最差已知接收率选择是否存在可部署多帧策略。"""

    baseline = calibration["strategy_summary"][SINGLE_STRATEGY]
    candidates = []
    for order, strategy in enumerate(SELECTORS):
        summary = calibration["strategy_summary"][strategy]
        gain = summary["worst_tpir_e2e"] - baseline["worst_tpir_e2e"]
        if summary["maximum_fpir"] <= 0.01 and gain >= 0.01:
            candidates.append((summary["worst_tpir_e2e"], -order, strategy))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _selection_summary(
    calibration: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    selected_strategy: str | None,
) -> dict[str, Any]:
    """生成多帧实验的门槛判断、独立验证和运行时决策。"""

    baseline_calibration = calibration["strategy_summary"][SINGLE_STRATEGY]
    baseline_evaluation = evaluation["strategy_summary"][SINGLE_STRATEGY]
    candidate_calibration = (
        calibration["strategy_summary"].get(selected_strategy) if selected_strategy else None
    )
    candidate_evaluation = (
        evaluation["strategy_summary"].get(selected_strategy) if selected_strategy else None
    )
    calibration_gain = _gain(candidate_calibration, baseline_calibration)
    evaluation_gain = _gain(candidate_evaluation, baseline_evaluation)
    adopted = bool(
        selected_strategy
        and candidate_evaluation
        and evaluation_gain >= 0.01
        and candidate_evaluation["maximum_fpir"] <= baseline_evaluation["maximum_fpir"]
    )
    return {
        "artifact": "multi-frame-strategy-selection-v1",
        "threshold": FROZEN_MINIMUM_SCORE,
        "calibration": {
            "baseline": baseline_calibration,
            "selected_strategy": selected_strategy,
            "selected": candidate_calibration,
            "worst_tpir_gain": calibration_gain,
        },
        "identity_disjoint_evaluation": {
            "baseline": baseline_evaluation,
            "selected_strategy": selected_strategy,
            "selected": candidate_evaluation,
            "worst_tpir_gain": evaluation_gain,
        },
        "adopted_for_runtime": adopted,
        "runtime_decision": (
            "collect_5_frames_80ms_interval"
            if adopted
            else "keep_single_frame_runtime"
        ),
        "decision_reason": (
            "独立身份验证集最差场景 TPIR 提升至少 1 个百分点，且未知误接收率未上升"
            if adopted
            else "未同时满足独立身份验证集提升至少 1 个百分点和未知误接收率不升高"
        ),
    }


def _gain(candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any]) -> float | None:
    """计算候选策略相对单帧基线的最差场景 TPIR 增益。"""

    if candidate is None:
        return None
    return float(candidate["worst_tpir_e2e"] - baseline["worst_tpir_e2e"])


def _decide(
    embedding: np.ndarray,
    gallery: Mapping[str, np.ndarray],
    policy: ScoreThresholdPolicy,
):
    """在固定人员平均原型标准库上执行一次阈值判定。"""

    query = _normalize(embedding)
    scores = {person_id: float(query @ prototype) for person_id, prototype in gallery.items()}
    return apply_open_set_policy(scores, policy)


def _record_known(stats: dict[str, int], decision: Any, identity: str) -> None:
    """记录一条已知身份判定。"""

    stats["total"] += 1
    if decision.accepted_person_id == identity:
        stats["correct_accept"] += 1
    elif decision.accepted_person_id is not None:
        stats["wrong_identity_accept"] += 1
    else:
        stats["reject"] += 1


def _record_unknown(stats: dict[str, int], decision: Any) -> None:
    """记录一条未知身份判定。"""

    stats["total"] += 1
    if decision.accepted_person_id is None:
        stats["reject"] += 1
    else:
        stats["false_accept"] += 1


def _new_known_stats() -> dict[str, int]:
    """创建已知身份统计计数器。"""

    return {"total": 0, "correct_accept": 0, "wrong_identity_accept": 0, "reject": 0}


def _new_unknown_stats() -> dict[str, int]:
    """创建未知身份统计计数器。"""

    return {"total": 0, "false_accept": 0, "reject": 0}


def _get_frames(
    probe_path: str,
    conditions: Sequence[str],
    measurements: Mapping[tuple[str, str], MultiFrameObservation],
) -> tuple[MultiFrameObservation, ...] | None:
    """按场景顺序取出五帧；任一数据缺失则跳过该序列。"""

    frames = tuple(measurements.get((probe_path, condition)) for condition in conditions)
    if any(frame is None for frame in frames):
        return None
    return tuple(
        replace(frame, frame_index=index)
        for index, frame in enumerate(frames)
        if frame is not None
    )


def _load_measurements(
    path: Path,
    conditions: Sequence[str],
    known_entries: Sequence[ProtocolEntry],
    unknown_entries: Sequence[ProtocolEntry],
) -> dict[tuple[str, str], MultiFrameObservation]:
    """从阶段 3 数据库读取所有已知和未知序列的 embedding 与质量指标。"""

    paths = sorted({entry.probe_path for entry in [*known_entries, *unknown_entries]})
    result: dict[tuple[str, str], MultiFrameObservation] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        for condition in conditions:
            for start in range(0, len(paths), 400):
                chunk = paths[start : start + 400]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT relative_path, embedding_blob, embedding_dim, metrics_json
                    FROM quality_measurements
                    WHERE degradation_key = ? AND status = 'observed'
                      AND relative_path IN ({placeholders})
                    """,
                    (condition, *chunk),
                ).fetchall()
                for relative_path, blob, dimension, metrics_json in rows:
                    embedding = np.frombuffer(blob, dtype=np.float32).copy()
                    if embedding.size != int(dimension):
                        raise RuntimeError(f"invalid phase3 embedding dimension: {relative_path}")
                    metrics = {
                        str(key): float(value) for key, value in json.loads(metrics_json).items()
                    }
                    result[(str(relative_path), condition)] = MultiFrameObservation(
                        frame=np.empty((1, 1, 3), dtype=np.uint8),
                        embedding=_normalize(embedding),
                        bbox=(0.0, 0.0, 1.0, 1.0),
                        quality_metrics=metrics,
                        quality_warnings=(),
                        frame_index=0,
                    )
    return result


def _load_gallery(raw_cache_path: Path, protocol: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """从自然 LFW 原始缓存构造阶段 4 使用的人员平均原型。"""

    enrollment = protocol.get("enrollment")
    if not isinstance(enrollment, dict):
        raise TypeError("phase4 protocol enrollment must be an object")
    all_paths = [str(path) for paths in enrollment.values() if isinstance(paths, list) for path in paths]
    dataset_id, extraction_id = _select_cache_partition(raw_cache_path)
    embeddings = _load_raw_embeddings(raw_cache_path, dataset_id, extraction_id, all_paths)
    gallery: dict[str, np.ndarray] = {}
    for person_id, paths in enrollment.items():
        if not isinstance(paths, list):
            continue
        samples = [_normalize(embeddings[path]) for path in paths if path in embeddings]
        if samples:
            gallery[str(person_id)] = _normalize(np.mean(np.stack(samples), axis=0))
    return gallery


def _select_cache_partition(path: Path) -> tuple[str, str]:
    """选择自然 LFW 原始缓存中观测数量最多的分区。"""

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            """
            SELECT dataset_id, embedding_extraction_id
            FROM raw_embeddings
            WHERE status = 'observed'
            GROUP BY dataset_id, embedding_extraction_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("raw embedding cache is empty")
    return str(row[0]), str(row[1])


def _load_raw_embeddings(
    path: Path,
    dataset_id: str,
    extraction_id: str,
    paths: Sequence[str],
) -> dict[str, np.ndarray]:
    """分块读取自然 LFW 标准库向量。"""

    result: dict[str, np.ndarray] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        for start in range(0, len(paths), 800):
            chunk = list(paths[start : start + 800])
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
            for relative_path, blob, dimension in rows:
                vector = np.frombuffer(blob, dtype=np.float32).copy()
                if vector.size != int(dimension):
                    raise RuntimeError(f"invalid gallery embedding dimension: {relative_path}")
                result[str(relative_path)] = vector
    return result


def _split_by_identity(
    entries: Sequence[ProtocolEntry], fraction: float, seed: int
) -> tuple[tuple[ProtocolEntry, ...], tuple[ProtocolEntry, ...]]:
    """按身份而不是按图片划分校准和独立验证记录。"""

    identities = sorted({entry.identity for entry in entries})
    random.Random(seed).shuffle(identities)
    cutoff = max(1, min(len(identities) - 1, round(len(identities) * fraction)))
    calibration_ids = set(identities[:cutoff])
    calibration = tuple(entry for entry in entries if entry.identity in calibration_ids)
    evaluation = tuple(entry for entry in entries if entry.identity not in calibration_ids)
    return calibration, evaluation


def _load_json(path: Path) -> dict[str, Any]:
    """读取一个 JSON 协议文件。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _normalize(vector: np.ndarray) -> np.ndarray:
    """返回有限非零 float32 单位向量。"""

    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("embedding must have a finite positive norm")
    return value / norm


def _git_revision() -> str | None:
    """读取当前代码版本，命令不可用时返回空值。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
