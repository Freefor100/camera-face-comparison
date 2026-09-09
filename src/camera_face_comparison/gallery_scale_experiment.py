from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .calibration import DecisionScoreRow, OperatingPoint, calibrate_method
from .evaluation_cache import file_sha256
from .lfw_dataset import LfwSplitProtocol
from .raw_embedding_cache import RawEmbeddingCache


@dataclass(frozen=True)
class GalleryScaleInputs:
    """小 Gallery 规模实验所需的原型、Probe 和缓存覆盖。"""

    prototypes: dict[str, np.ndarray]
    known_probes: dict[str, dict[str, tuple[np.ndarray, ...]]]
    unknown_probes: dict[str, tuple[np.ndarray, ...]]
    gallery_image_total: int
    gallery_valid_image_total: int
    gallery_fte_total: int
    probe_total: int
    probe_valid_total: int
    probe_fte_total: int


def load_gallery_scale_inputs(
    *,
    dataset_dir: Path,
    protocol: LfwSplitProtocol,
    cache: RawEmbeddingCache,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> GalleryScaleInputs:
    """从自然 LFW 原始缓存恢复身份均值原型和身份互斥 Probe。

    本函数不应用质量门或判定参数；模型 FTE 只进入覆盖统计。只有同时拥有有效
    Gallery 原型和有效 Probe 的 Known 身份才进入后续规模抽样。
    """

    prototypes: dict[str, np.ndarray] = {}
    gallery_total = sum(len(paths) for paths in protocol.enrollment.values())
    gallery_valid = gallery_fte = done = 0
    for person_id in sorted(protocol.enrollment):
        embeddings: list[np.ndarray] = []
        for relative_path in protocol.enrollment[person_id]:
            entry = _required_cache_entry(dataset_dir, relative_path, cache)
            if entry.status == "observed":
                if entry.embedding is None:
                    raise RuntimeError(f"cached embedding is missing: {relative_path}")
                embeddings.append(_normalize(entry.embedding))
                gallery_valid += 1
            else:
                gallery_fte += 1
            done += 1
            if on_progress is not None:
                on_progress("gallery-cache", done, gallery_total)
        if embeddings:
            prototypes[person_id] = _normalize(np.mean(embeddings, axis=0))

    known: dict[str, dict[str, list[np.ndarray]]] = {
        "calibration": {},
        "evaluation": {},
    }
    unknown: dict[str, list[np.ndarray]] = {"calibration": [], "evaluation": []}
    probe_valid = probe_fte = 0
    for index, probe in enumerate(protocol.probes, start=1):
        entry = _required_cache_entry(dataset_dir, probe.relative_path, cache)
        if entry.status == "failed":
            probe_fte += 1
        else:
            if entry.embedding is None:
                raise RuntimeError(f"cached embedding is missing: {probe.relative_path}")
            embedding = _normalize(entry.embedding)
            probe_valid += 1
            if probe.expected_person_id is None:
                unknown[probe.split].append(embedding)
            elif probe.expected_person_id in prototypes:
                known[probe.split].setdefault(probe.expected_person_id, []).append(embedding)
        if on_progress is not None:
            on_progress("probe-cache", index, len(protocol.probes))

    return GalleryScaleInputs(
        prototypes=prototypes,
        known_probes={
            split: {
                person_id: tuple(embeddings)
                for person_id, embeddings in sorted(people.items())
                if embeddings
            }
            for split, people in known.items()
        },
        unknown_probes={split: tuple(values) for split, values in unknown.items()},
        gallery_image_total=gallery_total,
        gallery_valid_image_total=gallery_valid,
        gallery_fte_total=gallery_fte,
        probe_total=len(protocol.probes),
        probe_valid_total=probe_valid,
        probe_fte_total=probe_fte,
    )


def run_gallery_scale_experiment(
    inputs: GalleryScaleInputs,
    *,
    gallery_sizes: Sequence[int],
    repeats: int,
    seed: int,
    target_fpir: float,
    transferred_policy: OperatingPoint,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """按 Gallery 规模重复执行 Calibration 参数选择和独立 Evaluation。

    Calibration 与 Evaluation 使用原协议中互斥的 Known/Unknown 身份。每个重复中
    两个分区分别选择相同规模、但互不重叠的 Gallery 身份。LFW 全 Gallery 工作点
    只做迁移对照，不参与当前规模的参数选择。
    """

    sizes = tuple(int(value) for value in gallery_sizes)
    if not sizes or any(value < 2 for value in sizes):
        raise ValueError("gallery sizes must all be at least two")
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    calibration_pool = tuple(sorted(inputs.known_probes["calibration"]))
    evaluation_pool = tuple(sorted(inputs.known_probes["evaluation"]))
    maximum = max(sizes)
    if maximum > min(len(calibration_pool), len(evaluation_pool)):
        raise ValueError(
            "gallery size exceeds available Known identities: "
            f"calibration={len(calibration_pool)}, evaluation={len(evaluation_pool)}"
        )

    runs: list[dict[str, object]] = []
    total = len(sizes) * repeats
    done = 0
    for repeat in range(repeats):
        calibration_order = _shuffled_identities(calibration_pool, seed, repeat, "calibration")
        evaluation_order = _shuffled_identities(evaluation_pool, seed, repeat, "evaluation")
        for gallery_size in sizes:
            calibration_ids = calibration_order[:gallery_size]
            evaluation_ids = evaluation_order[:gallery_size]
            calibration_rows = _score_split(inputs, "calibration", calibration_ids)
            candidates = (
                calibrate_method(
                    calibration_rows,
                    target_fpir=target_fpir,
                    use_score_gap=False,
                ),
                calibrate_method(
                    calibration_rows,
                    target_fpir=target_fpir,
                    use_score_gap=True,
                ),
            )
            selected = _select_point(candidates)
            evaluation_rows = _score_split(inputs, "evaluation", evaluation_ids)
            runs.append(
                {
                    "gallery_size": gallery_size,
                    "repeat": repeat,
                    "calibration_gallery_ids": list(calibration_ids),
                    "evaluation_gallery_ids": list(evaluation_ids),
                    "selected": asdict(selected),
                    "evaluation": _evaluate_rows(evaluation_rows, selected),
                    "full_gallery_policy_transfer": _evaluate_rows(
                        evaluation_rows,
                        transferred_policy,
                    ),
                }
            )
            done += 1
            if on_progress is not None:
                on_progress(done, total)
    return {
        "gallery_sizes": list(sizes),
        "repeats": repeats,
        "seed": seed,
        "target_fpir": target_fpir,
        "calibration_known_identity_pool": len(calibration_pool),
        "evaluation_known_identity_pool": len(evaluation_pool),
        "calibration_unknown_probe_total": len(inputs.unknown_probes["calibration"]),
        "evaluation_unknown_probe_total": len(inputs.unknown_probes["evaluation"]),
        "runs": runs,
        "summary_by_gallery_size": _summarize_runs(runs, sizes),
    }


def _score_split(
    inputs: GalleryScaleInputs,
    split: str,
    selected_ids: Sequence[str],
) -> tuple[DecisionScoreRow, ...]:
    """用指定身份原型为一个分区的 Known 和原始 Unknown 生成连续 Top-2 分数。"""

    person_ids = tuple(selected_ids)
    gallery = np.stack([inputs.prototypes[person_id] for person_id in person_ids])
    embeddings: list[np.ndarray] = []
    expected_ids: list[str | None] = []
    for person_id in person_ids:
        values = inputs.known_probes[split][person_id]
        embeddings.extend(values)
        expected_ids.extend([person_id] * len(values))
    embeddings.extend(inputs.unknown_probes[split])
    expected_ids.extend([None] * len(inputs.unknown_probes[split]))
    query = np.stack(embeddings).astype(np.float32)
    scores = query @ gallery.T
    first_indices = np.argmax(scores, axis=1)
    without_first = scores.copy()
    without_first[np.arange(scores.shape[0]), first_indices] = -np.inf
    second_indices = np.argmax(without_first, axis=1)
    rows = []
    for index, expected_person_id in enumerate(expected_ids):
        first = int(first_indices[index])
        second = int(second_indices[index])
        top_score = float(scores[index, first])
        second_score = float(scores[index, second])
        rows.append(
            DecisionScoreRow(
                expected_person_id=expected_person_id,
                top_score=top_score,
                score_gap=max(0.0, top_score - second_score),
                top_is_correct=(
                    expected_person_id is not None and person_ids[first] == expected_person_id
                ),
            )
        )
    return tuple(rows)


def _select_point(candidates: Sequence[OperatingPoint]) -> OperatingPoint:
    """按满足 FPIR、TPIR 和规则简洁度选择当前规模的唯一工作点。"""

    meeting = [point for point in candidates if point.meets_target]
    pool = meeting or list(candidates)
    return min(
        pool,
        key=lambda point: (
            0.0 if meeting else point.fpir,
            -point.tpir,
            int(point.use_score_gap),
            point.known_wrong_accepts,
            point.unknown_false_accepts,
            point.match_threshold,
            point.min_score_gap,
        ),
    )


def _evaluate_rows(
    rows: Sequence[DecisionScoreRow],
    point: OperatingPoint,
) -> dict[str, float | int | bool]:
    """把一个已选工作点应用到给定连续分数，保留完整 Known/Unknown 分母。"""

    known = [row for row in rows if row.expected_person_id is not None]
    unknown = [row for row in rows if row.expected_person_id is None]

    def accepted(row: DecisionScoreRow) -> bool:
        """判断一条分数是否通过当前外部工作点。"""

        return row.top_score >= point.match_threshold and (
            not point.use_score_gap or row.score_gap >= point.min_score_gap
        )

    rank1_correct = sum(row.top_is_correct for row in known)
    true_accepts = sum(accepted(row) and row.top_is_correct for row in known)
    wrong_accepts = sum(accepted(row) and not row.top_is_correct for row in known)
    false_accepts = sum(accepted(row) for row in unknown)
    return {
        "known_total": len(known),
        "rank1_correct": rank1_correct,
        "rank1_rate": rank1_correct / len(known) if known else 0.0,
        "known_true_accepts": true_accepts,
        "known_wrong_accepts": wrong_accepts,
        "unknown_total": len(unknown),
        "unknown_false_accepts": false_accepts,
        "fpir": false_accepts / len(unknown) if unknown else 0.0,
        "tpir": true_accepts / len(known) if known else 0.0,
        "fnir": 1.0 - true_accepts / len(known) if known else 1.0,
    }


def _summarize_runs(
    runs: Sequence[Mapping[str, object]],
    gallery_sizes: Sequence[int],
) -> list[dict[str, object]]:
    """按 Gallery 规模汇总十次重复的参数和 Evaluation 指标。"""

    summaries = []
    for size in gallery_sizes:
        selected_runs = [run for run in runs if int(run["gallery_size"]) == size]
        selected_points = [run["selected"] for run in selected_runs]
        evaluations = [run["evaluation"] for run in selected_runs]
        transfers = [run["full_gallery_policy_transfer"] for run in selected_runs]
        summaries.append(
            {
                "gallery_size": size,
                "run_total": len(selected_runs),
                "score_gap_selected_runs": sum(
                    bool(point["use_score_gap"]) for point in selected_points
                ),
                "selected_match_threshold": _numeric_summary(
                    [float(point["match_threshold"]) for point in selected_points]
                ),
                "selected_min_score_gap": _numeric_summary(
                    [float(point["min_score_gap"]) for point in selected_points]
                ),
                "evaluation_rank1_rate": _numeric_summary(
                    [float(result["rank1_rate"]) for result in evaluations]
                ),
                "evaluation_tpir": _numeric_summary(
                    [float(result["tpir"]) for result in evaluations]
                ),
                "evaluation_fpir": _numeric_summary(
                    [float(result["fpir"]) for result in evaluations]
                ),
                "full_gallery_policy_transfer_tpir": _numeric_summary(
                    [float(result["tpir"]) for result in transfers]
                ),
                "full_gallery_policy_transfer_fpir": _numeric_summary(
                    [float(result["fpir"]) for result in transfers]
                ),
            }
        )
    return summaries


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    """返回重复实验数值的均值、标准差和范围。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _shuffled_identities(
    identities: Sequence[str],
    seed: int,
    repeat: int,
    split: str,
) -> tuple[str, ...]:
    """使用与 Python 哈希随机化无关的种子生成确定身份顺序。"""

    digest = hashlib.sha256(f"{seed}:{repeat}:{split}".encode()).digest()
    values = list(identities)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    return tuple(values)


def _required_cache_entry(dataset_dir: Path, relative_path: str, cache: RawEmbeddingCache):
    """按图片 SHA-256 读取原始缓存，缺失或过期时立即停止实验。"""

    image_path = dataset_dir / relative_path
    entry = cache.get(relative_path, file_sha256(image_path))
    if entry is None:
        raise RuntimeError(f"embedding cache is incomplete or stale: {relative_path}")
    return entry


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """将 embedding 转为 float32 单位向量。"""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm
