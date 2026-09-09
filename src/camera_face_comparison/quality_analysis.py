from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .image_input import apply_quality_policy, calculate_quality_score
from .quality_experiment_store import QualityExperimentEntry


@dataclass(frozen=True)
class QualityUtilityRecord:
    """一个质量指标值及该输入是否被识别正确。"""

    value: float
    is_correct: bool


@dataclass(frozen=True)
class ErrorVsRejectPoint:
    """在一个干净样本保留率下，质量拒绝带来的错误与误拒变化。"""

    retention_target: float
    threshold: float
    accepted_total: int
    rejected_total: int
    errors_before: int
    errors_after: int
    errors_removed: int
    correct_rejected: int


@dataclass(frozen=True)
class QualityErrorBin:
    """按质量值排序后的一个等频分桶及其原始识别错误率。"""

    index: int
    minimum: float
    maximum: float
    total: int
    error_total: int
    error_rate: float


@dataclass(frozen=True)
class NumericDistribution:
    """一组连续数值的数量、均值和常用分位点。"""

    count: int
    minimum: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None
    mean: float | None


@dataclass(frozen=True)
class KnownQualitySummary:
    """Known Probe 在身份排名和当前质量门下的结果。"""

    total: int
    observed_total: int
    fte_total: int
    current_policy_rejected: int
    rejection_reasons: dict[str, int]
    rank1_correct: int
    rank1_rate: float
    accepted_rank1_correct: int
    accepted_rank1_rate: float
    genuine_similarity: NumericDistribution
    latency_ms: NumericDistribution


@dataclass(frozen=True)
class UnknownQualitySummary:
    """Unknown Probe 的最高候选连续分数和质量门结果。"""

    total: int
    observed_total: int
    fte_total: int
    current_policy_rejected: int
    rejection_reasons: dict[str, int]
    top_score: NumericDistribution
    accepted_top_score: NumericDistribution
    latency_ms: NumericDistribution


@dataclass(frozen=True)
class ProbeQualitySummary:
    """一个退化条件下固定干净 Gallery 的 Probe 分析结果。"""

    gallery_identity_total: int
    gallery_observed_total: int
    known: KnownQualitySummary
    unknown: UnknownQualitySummary


@dataclass(frozen=True)
class GalleryQualitySummary:
    """一个退化条件下固定干净 Probe 的 Gallery 分析结果。"""

    gallery_identity_total: int
    gallery_observed_total: int
    gallery_current_policy_rejected: int
    gallery_rejection_reasons: dict[str, int]
    known: KnownQualitySummary
    unknown: UnknownQualitySummary


def load_quality_measurements(
    path: Path,
    embedding_extraction_id: str,
) -> dict[str, dict[str, QualityExperimentEntry]]:
    """从已完成的 Phase 3 SQLite 中读取指定提取版本的全部原始记录。

    返回值第一层键是退化条件，第二层键是数据集相对路径；该读取只用于分析，
    不承担运行时缓存失效判断。
    """

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT relative_path, degradation_key, status, embedding_blob, embedding_dim,
                   metrics_json, bbox_json, face_count, latency_ms, reason
            FROM quality_measurements
            WHERE embedding_extraction_id = ?
            ORDER BY degradation_key, relative_path
            """,
            (embedding_extraction_id,),
        ).fetchall()
    finally:
        connection.close()
    measurements: dict[str, dict[str, QualityExperimentEntry]] = {}
    for row in rows:
        relative_path, condition_key, status = str(row[0]), str(row[1]), str(row[2])
        if status == "failed":
            entry = QualityExperimentEntry(
                status="failed",
                embedding=None,
                metrics=None,
                bbox=None,
                face_count=int(row[7]),
                latency_ms=float(row[8]),
                reason=str(row[9]),
            )
        elif status == "observed" and row[3] is not None and row[4] is not None:
            embedding = np.frombuffer(row[3], dtype=np.float32).copy()
            if embedding.size != int(row[4]):
                raise RuntimeError(f"invalid embedding dimension: {relative_path}")
            metrics_payload = json.loads(str(row[5]))
            bbox_payload = json.loads(str(row[6]))
            entry = QualityExperimentEntry(
                status="observed",
                embedding=embedding,
                metrics={str(key): float(value) for key, value in metrics_payload.items()},
                bbox=tuple(float(value) for value in bbox_payload),
                face_count=int(row[7]),
                latency_ms=float(row[8]),
                reason=None,
            )
        else:
            raise RuntimeError(f"invalid quality measurement: {relative_path} {condition_key}")
        measurements.setdefault(condition_key, {})[relative_path] = entry
    return measurements


def known_rank1_outcomes(
    *,
    gallery_entries: Mapping[str, QualityExperimentEntry],
    known_entries: Mapping[str, QualityExperimentEntry],
) -> dict[str, bool | None]:
    """返回每个 Known 身份的原始 Rank-1 正确性，FTE 或缺失参考图记为 `None`。"""

    gallery_ids, gallery_matrix = _observed_gallery_matrix(gallery_entries)
    gallery_index = {person_id: index for index, person_id in enumerate(gallery_ids)}
    outcomes: dict[str, bool | None] = {}
    for expected_id, entry in sorted(known_entries.items()):
        expected_index = gallery_index.get(expected_id)
        if (
            expected_index is None
            or gallery_matrix.size == 0
            or entry.status != "observed"
            or entry.embedding is None
        ):
            outcomes[expected_id] = None
            continue
        scores = np.asarray(entry.embedding, dtype=np.float32) @ gallery_matrix.T
        outcomes[expected_id] = int(np.argmax(scores)) == expected_index
    return outcomes


def quality_utility_values(
    entry: QualityExperimentEntry,
    settings: Settings,
) -> dict[str, float]:
    """把原始质量指标转换成统一为“越大越好”的实验效用值。"""

    if entry.status != "observed" or entry.metrics is None:
        raise ValueError("quality utility requires an observed entry")
    metrics = entry.metrics
    return {
        "quality_score": calculate_quality_score(metrics, settings),
        "detection_score": metrics["detection_score"],
        "face_size_px": metrics["face_size_px"],
        "blur_variance": metrics["blur_variance"],
        "brightness": metrics["brightness"],
        "exposure_balance": max(0.0, 1.0 - abs(metrics["brightness"] - 127.5) / 127.5),
        "contrast": metrics["contrast"],
    }


def quality_metric_distributions(
    entries: Iterable[QualityExperimentEntry],
    settings: Settings,
) -> dict[str, NumericDistribution]:
    """统计一批有效输入的五项质量效用和启发式总分分布。"""

    collected: dict[str, list[float]] = {}
    for entry in entries:
        if entry.status != "observed" or entry.metrics is None:
            continue
        for name, value in quality_utility_values(entry, settings).items():
            collected.setdefault(name, []).append(value)
    return {name: numeric_distribution(values) for name, values in sorted(collected.items())}


def analyze_probe_quality(
    *,
    gallery_entries: Mapping[str, QualityExperimentEntry],
    known_entries: Mapping[str, QualityExperimentEntry],
    unknown_entries: Sequence[QualityExperimentEntry],
    settings: Settings,
) -> ProbeQualitySummary:
    """分析固定干净 Gallery 下，一个条件造成的 Probe 质量变化。

    参数：
        gallery_entries：人员 ID 到干净参考图记录的映射。
        known_entries：预期人员 ID 到退化 Known Probe 的映射。
        unknown_entries：不属于 Gallery 的退化 Probe。
        settings：用于复算当前质量门的配置。
    返回：
        Rank-1、连续相似度、质量拒绝与耗时摘要。
    """

    gallery_ids, gallery_matrix = _observed_gallery_matrix(gallery_entries)
    known = _analyze_known(
        gallery_ids=gallery_ids,
        gallery_matrix=gallery_matrix,
        known_entries=known_entries,
        settings=settings,
    )
    unknown = _analyze_unknown(
        gallery_matrix=gallery_matrix,
        unknown_entries=unknown_entries,
        settings=settings,
    )
    return ProbeQualitySummary(
        gallery_identity_total=len(gallery_entries),
        gallery_observed_total=len(gallery_ids),
        known=known,
        unknown=unknown,
    )


def analyze_gallery_quality(
    *,
    gallery_entries: Mapping[str, QualityExperimentEntry],
    known_entries: Mapping[str, QualityExperimentEntry],
    unknown_entries: Sequence[QualityExperimentEntry],
    settings: Settings,
) -> GalleryQualitySummary:
    """分析固定干净 Probe 下，一个条件造成的 Gallery 入库质量变化。

    当前质量门会先移除不合格参考图，再额外计算一次严格的系统级 Rank-1；
    原始 Rank-1 仍保留，用于判断“允许低质量图片入库”的直接影响。
    """

    gallery_ids, gallery_matrix = _observed_gallery_matrix(gallery_entries)
    raw_known = _analyze_known(
        gallery_ids=gallery_ids,
        gallery_matrix=gallery_matrix,
        known_entries=known_entries,
        settings=settings,
        apply_probe_policy=False,
    )
    raw_unknown = _analyze_unknown(
        gallery_matrix=gallery_matrix,
        unknown_entries=unknown_entries,
        settings=settings,
        apply_probe_policy=False,
    )
    accepted_gallery: dict[str, QualityExperimentEntry] = {}
    rejected_reasons: Counter[str] = Counter()
    for person_id, entry in gallery_entries.items():
        if entry.status != "observed" or entry.metrics is None:
            rejected_reasons[entry.reason or "feature_extraction_failed"] += 1
            continue
        profile = apply_quality_policy(entry.metrics, settings)
        if profile.tier == "reject":
            rejected_reasons.update(profile.reasons)
        else:
            accepted_gallery[person_id] = entry

    accepted_ids, accepted_matrix = _observed_gallery_matrix(accepted_gallery)
    gated_known = _analyze_known(
        gallery_ids=accepted_ids,
        gallery_matrix=accepted_matrix,
        known_entries=known_entries,
        settings=settings,
        apply_probe_policy=False,
    )
    gated_unknown = _analyze_unknown(
        gallery_matrix=accepted_matrix,
        unknown_entries=unknown_entries,
        settings=settings,
        apply_probe_policy=False,
    )
    known = KnownQualitySummary(
        **{
            **raw_known.__dict__,
            "accepted_rank1_correct": gated_known.rank1_correct,
            "accepted_rank1_rate": gated_known.rank1_rate,
        }
    )
    unknown = UnknownQualitySummary(
        **{
            **raw_unknown.__dict__,
            "accepted_top_score": gated_unknown.top_score,
        }
    )
    return GalleryQualitySummary(
        gallery_identity_total=len(gallery_entries),
        gallery_observed_total=len(gallery_ids),
        gallery_current_policy_rejected=len(gallery_entries) - len(accepted_gallery),
        gallery_rejection_reasons=dict(sorted(rejected_reasons.items())),
        known=known,
        unknown=unknown,
    )


def numeric_distribution(values: Iterable[float]) -> NumericDistribution:
    """把连续数值压缩为可稳定写入实验报告的描述统计。"""

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return NumericDistribution(0, None, None, None, None, None, None, None)
    return NumericDistribution(
        count=int(array.size),
        minimum=_rounded(float(array.min())),
        p05=_rounded(float(np.quantile(array, 0.05))),
        p50=_rounded(float(np.quantile(array, 0.50))),
        p95=_rounded(float(np.quantile(array, 0.95))),
        p99=_rounded(float(np.quantile(array, 0.99))),
        maximum=_rounded(float(array.max())),
        mean=_rounded(float(array.mean())),
    )


def _observed_gallery_matrix(
    entries: Mapping[str, QualityExperimentEntry],
) -> tuple[tuple[str, ...], np.ndarray]:
    """按人员 ID 排序，构造仅含有效 embedding 的 Gallery 矩阵。"""

    observed = [
        (person_id, entry.embedding)
        for person_id, entry in sorted(entries.items())
        if entry.status == "observed" and entry.embedding is not None
    ]
    if not observed:
        return (), np.empty((0, 0), dtype=np.float32)
    dimensions = {int(vector.size) for _, vector in observed}
    if len(dimensions) != 1:
        raise ValueError("gallery embeddings must have the same dimension")
    return (
        tuple(person_id for person_id, _ in observed),
        np.stack([vector for _, vector in observed]).astype(np.float32, copy=False),
    )


def _analyze_known(
    *,
    gallery_ids: tuple[str, ...],
    gallery_matrix: np.ndarray,
    known_entries: Mapping[str, QualityExperimentEntry],
    settings: Settings,
    apply_probe_policy: bool = True,
) -> KnownQualitySummary:
    """计算 Known 的严格总分母 Rank-1、同人分数与质量拒绝。"""

    gallery_index = {person_id: index for index, person_id in enumerate(gallery_ids)}
    observed_total = 0
    rejected_total = 0
    rank1_correct = 0
    accepted_rank1_correct = 0
    genuine_scores: list[float] = []
    latencies: list[float] = []
    reasons: Counter[str] = Counter()
    for expected_id, entry in sorted(known_entries.items()):
        if entry.status != "observed" or entry.embedding is None:
            reasons[entry.reason or "feature_extraction_failed"] += 1
            continue
        observed_total += 1
        latencies.append(entry.latency_ms)
        accepted = True
        if apply_probe_policy:
            if entry.metrics is None:
                raise RuntimeError("observed quality entry is missing metrics")
            profile = apply_quality_policy(entry.metrics, settings)
            accepted = profile.tier != "reject"
            if not accepted:
                rejected_total += 1
                reasons.update(profile.reasons)
        expected_index = gallery_index.get(expected_id)
        if expected_index is None or gallery_matrix.size == 0:
            continue
        scores = np.asarray(entry.embedding, dtype=np.float32) @ gallery_matrix.T
        genuine_scores.append(float(scores[expected_index]))
        is_correct = int(np.argmax(scores)) == expected_index
        rank1_correct += int(is_correct)
        accepted_rank1_correct += int(accepted and is_correct)
    total = len(known_entries)
    return KnownQualitySummary(
        total=total,
        observed_total=observed_total,
        fte_total=total - observed_total,
        current_policy_rejected=rejected_total,
        rejection_reasons=dict(sorted(reasons.items())),
        rank1_correct=rank1_correct,
        rank1_rate=_rate(rank1_correct, total),
        accepted_rank1_correct=accepted_rank1_correct,
        accepted_rank1_rate=_rate(accepted_rank1_correct, total),
        genuine_similarity=numeric_distribution(genuine_scores),
        latency_ms=numeric_distribution(latencies),
    )


def _analyze_unknown(
    *,
    gallery_matrix: np.ndarray,
    unknown_entries: Sequence[QualityExperimentEntry],
    settings: Settings,
    apply_probe_policy: bool = True,
) -> UnknownQualitySummary:
    """计算 Unknown 对 Gallery 的最高连续分数，不提前应用识别阈值。"""

    observed_total = 0
    rejected_total = 0
    top_scores: list[float] = []
    accepted_top_scores: list[float] = []
    latencies: list[float] = []
    reasons: Counter[str] = Counter()
    for entry in unknown_entries:
        if entry.status != "observed" or entry.embedding is None:
            reasons[entry.reason or "feature_extraction_failed"] += 1
            continue
        observed_total += 1
        latencies.append(entry.latency_ms)
        accepted = True
        if apply_probe_policy:
            if entry.metrics is None:
                raise RuntimeError("observed quality entry is missing metrics")
            profile = apply_quality_policy(entry.metrics, settings)
            accepted = profile.tier != "reject"
            if not accepted:
                rejected_total += 1
                reasons.update(profile.reasons)
        if gallery_matrix.size == 0:
            continue
        top_score = float(np.max(np.asarray(entry.embedding, dtype=np.float32) @ gallery_matrix.T))
        top_scores.append(top_score)
        if accepted:
            accepted_top_scores.append(top_score)
    total = len(unknown_entries)
    return UnknownQualitySummary(
        total=total,
        observed_total=observed_total,
        fte_total=total - observed_total,
        current_policy_rejected=rejected_total,
        rejection_reasons=dict(sorted(reasons.items())),
        top_score=numeric_distribution(top_scores),
        accepted_top_score=numeric_distribution(accepted_top_scores),
        latency_ms=numeric_distribution(latencies),
    )


def _rate(numerator: int, denominator: int) -> float:
    """返回六位小数的比例；空分母按零处理。"""

    return _rounded(numerator / denominator) if denominator else 0.0


def _rounded(value: float) -> float:
    """统一实验报告中的浮点精度，减少无意义平台差异。"""

    return round(value, 6)


def error_vs_reject(
    records: Iterable[QualityUtilityRecord],
    *,
    clean_values: Iterable[float],
    retention_targets: Iterable[float] = (0.99, 0.95, 0.90),
) -> tuple[ErrorVsRejectPoint, ...]:
    """按干净样本保留率生成 Error-versus-Reject 统计点。

    参数：
        records：需要评估的质量值和原始识别结果。
        clean_values：基线干净输入的质量值，用于确定拒绝阈值。
        retention_targets：期望保留的干净输入比例。
    返回：
        每个保留率对应的阈值、错误移除量和正确样本误拒量。
    前置条件：
        输入集合非空，保留率位于开区间 `(0, 1]`。
    """

    materialized_records = tuple(records)
    sorted_clean = sorted(float(value) for value in clean_values)
    targets = tuple(float(target) for target in retention_targets)
    if not materialized_records:
        raise ValueError("records must not be empty")
    if not sorted_clean:
        raise ValueError("clean_values must not be empty")
    if any(not 0.0 < target <= 1.0 for target in targets):
        raise ValueError("retention targets must be within (0, 1]")

    errors_before = sum(not record.is_correct for record in materialized_records)
    points: list[ErrorVsRejectPoint] = []
    for target in targets:
        keep_count = min(len(sorted_clean), math.ceil(target * len(sorted_clean)))
        threshold = sorted_clean[len(sorted_clean) - keep_count]
        accepted = tuple(record for record in materialized_records if record.value >= threshold)
        rejected = tuple(record for record in materialized_records if record.value < threshold)
        errors_after = sum(not record.is_correct for record in accepted)
        points.append(
            ErrorVsRejectPoint(
                retention_target=target,
                threshold=threshold,
                accepted_total=len(accepted),
                rejected_total=len(rejected),
                errors_before=errors_before,
                errors_after=errors_after,
                errors_removed=errors_before - errors_after,
                correct_rejected=sum(record.is_correct for record in rejected),
            )
        )
    return tuple(points)


def quality_error_bins(
    records: Iterable[QualityUtilityRecord],
    *,
    bin_count: int = 10,
) -> tuple[QualityErrorBin, ...]:
    """按质量值从低到高生成等频分桶，用于检查错误率是否单调下降。"""

    if bin_count < 1:
        raise ValueError("bin_count must be at least one")
    ordered = sorted(records, key=lambda record: record.value)
    if not ordered:
        raise ValueError("records must not be empty")
    chunks = np.array_split(np.asarray(ordered, dtype=object), min(bin_count, len(ordered)))
    bins: list[QualityErrorBin] = []
    for index, chunk in enumerate(chunks, start=1):
        materialized = tuple(chunk.tolist())
        error_total = sum(not record.is_correct for record in materialized)
        bins.append(
            QualityErrorBin(
                index=index,
                minimum=_rounded(materialized[0].value),
                maximum=_rounded(materialized[-1].value),
                total=len(materialized),
                error_total=error_total,
                error_rate=_rate(error_total, len(materialized)),
            )
        )
    return tuple(bins)
