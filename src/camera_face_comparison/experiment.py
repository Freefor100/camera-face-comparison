from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import QualityTierPolicy
from .recognition import (
    MatchDecision,
    aggregate_person_scores,
    aggregate_quality_weighted_scores,
    decide_match,
)

AggregationMethod = Literal["single", "max", "mean_prototype", "top_k_mean"]


@dataclass(frozen=True)
class ExperimentRecord:
    """一张带标签探针图片的得分记录，与私人图片本身分离保存。"""

    expected_person_id: str | None
    sample_scores: Mapping[str, Sequence[float]]
    latency_ms: float
    sample_quality_scores: Mapping[str, Sequence[float]] | None = None
    probe_quality_tier: str | None = None


@dataclass(frozen=True)
class EmbeddingExperimentRecord:
    """一张保留原始向量的评测探针，用于比较人员表示方法。"""

    expected_person_id: str | None
    query_embedding: np.ndarray
    gallery_embeddings: Mapping[str, Sequence[np.ndarray]]
    latency_ms: float
    sample_quality_by_person: Mapping[str, Sequence[float]] | None = None
    probe_quality_tier: str | None = None


@dataclass(frozen=True)
class ExperimentMetrics:
    """一组实验记录的开放集识别统计量。"""

    total: int
    known_total: int
    known_correct: int
    unknown_total: int
    unknown_rejected: int
    misidentifications: int
    average_latency_ms: float

    @property
    def known_accuracy(self) -> float | None:
        """返回已知探针被正确识别的比例。"""
        return self.known_correct / self.known_total if self.known_total else None

    @property
    def unknown_rejection_rate(self) -> float | None:
        """返回未知探针被拒识的比例。"""
        return self.unknown_rejected / self.unknown_total if self.unknown_total else None

    @property
    def false_positive_identification_rate(self) -> float | None:
        """返回未知探针被错误分配给库内身份的比例（FPIR）。"""

        if not self.unknown_total:
            return None
        return (self.unknown_total - self.unknown_rejected) / self.unknown_total

    @property
    def false_negative_identification_rate(self) -> float | None:
        """返回被拒识或分配给错误身份的已知探针比例（FNIR）。"""

        if not self.known_total:
            return None
        return (self.known_total - self.known_correct) / self.known_total

    @property
    def rank_one_identification_rate(self) -> float | None:
        """返回与开放集指标并列报告的 Rank-1 正确率。"""

        return self.known_accuracy


def evaluate_experiments(
    *,
    records: Sequence[ExperimentRecord],
    match_threshold: float,
    min_score_gap: float,
    top_k: int = 2,
    quality_tiers: Mapping[str, QualityTierPolicy] | None = None,
) -> tuple[ExperimentMetrics, ExperimentMetrics]:
    """在完全相同的探针上评估简单基线和当前优化规则。

    参数：
        records：已计算样本得分、标签和耗时的实验记录。
        match_threshold：基线使用的最高候选阈值。
        min_score_gap：基线之外的候选分差阈值。
        top_k：优化聚合使用的样本数。
        quality_tiers：按探针质量层级选择阈值的可选策略。
    返回：
        `(baseline_metrics, optimized_metrics)` 两组统计结果。
    前置条件：
        两种方法必须接收同一批记录，避免数据划分差异影响比较。
    """

    baseline_decisions = [
        decide_match(
            _highest_sample_scores(record.sample_scores),
            match_threshold=match_threshold,
            min_score_gap=0.0,
        )
        for record in records
    ]
    optimized_decisions = [
        decide_match(
            _optimized_person_scores(record, top_k=top_k),
            match_threshold=_policy_for(record, quality_tiers, match_threshold, min_score_gap).match_threshold,
            min_score_gap=_policy_for(record, quality_tiers, match_threshold, min_score_gap).min_score_gap,
        )
        for record in records
    ]
    return (
        _metrics(records, baseline_decisions),
        _metrics(records, optimized_decisions),
    )


def aggregate_embedding_person_scores(
    *,
    query_embedding: np.ndarray,
    gallery_embeddings: Mapping[str, Sequence[np.ndarray]],
    method: AggregationMethod,
    top_k: int,
    sample_quality_by_person: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    """用指定人员表示方法计算每个身份的探针得分。

    参数：
        query_embedding：待识别的人脸特征向量。
        gallery_embeddings：身份编号到全部标准样本向量的映射。
        method：单样本、最高样本、平均原型或 Top-K 普通平均。
        top_k：Top-K 方法最多使用的样本数。
        sample_quality_by_person：随记录保留的入库质量分；本阶段主聚合不使用。
    返回：
        身份编号到人员级别相似度的映射。
    前置条件：
        所有向量都必须是非零一维数组；`single` 使用每个身份的第一张样本，
        仅作为多样本算法的控制组。
    """

    if method not in {"single", "max", "mean_prototype", "top_k_mean"}:
        raise ValueError(f"unsupported aggregation method: {method}")
    if top_k < 1:
        raise ValueError("top_k must be at least one")

    query = _normalize_embedding(query_embedding)
    aggregated: dict[str, float] = {}
    for person_id, embeddings in gallery_embeddings.items():
        if not embeddings:
            continue
        normalized_embeddings = [_normalize_embedding(embedding) for embedding in embeddings]
        if method == "mean_prototype":
            prototype = _normalize_embedding(np.mean(normalized_embeddings, axis=0))
            aggregated[person_id] = float(query @ prototype)
            continue

        sample_scores = [float(query @ embedding) for embedding in normalized_embeddings]
        if method == "single":
            aggregated[person_id] = sample_scores[0]
        elif method == "max":
            aggregated[person_id] = max(sample_scores)
        elif method == "top_k_mean":
            aggregated[person_id] = aggregate_person_scores(
                {person_id: sample_scores}, top_k=top_k
            )[person_id]
    return aggregated


def evaluate_algorithm_methods(
    *,
    records: Sequence[EmbeddingExperimentRecord],
    methods: Sequence[AggregationMethod],
    match_threshold: float,
    min_score_gap: float,
    top_k: int = 2,
    quality_tiers: Mapping[str, QualityTierPolicy] | None = None,
) -> dict[str, ExperimentMetrics]:
    """在完全相同的向量探针上比较多种人员聚合方法。

    参数：
        records：包含探针向量、Gallery 向量和真实标签的固定评测记录。
        methods：要运行的人员表示方法，结果顺序与输入顺序保持一致。
        match_threshold：默认的身份匹配阈值。
        min_score_gap：默认的第一、第二候选分差下限。
        top_k：Top-K 方法使用的最大样本数。
        quality_tiers：按探针质量等级覆盖默认阈值的策略。
    返回：
        方法名称到统计指标的映射。
    前置条件：
        所有方法必须使用同一批记录，不能在比较过程中更换数据划分。
    """

    if not records:
        raise ValueError("records must not be empty")
    if not methods:
        raise ValueError("methods must not be empty")

    results: dict[str, ExperimentMetrics] = {}
    for method in methods:
        decisions: list[MatchDecision] = []
        for record in records:
            person_scores = aggregate_embedding_person_scores(
                query_embedding=record.query_embedding,
                gallery_embeddings=record.gallery_embeddings,
                method=method,
                top_k=top_k,
                sample_quality_by_person=record.sample_quality_by_person,
            )
            policy = _policy_for(record, quality_tiers, match_threshold, min_score_gap)
            decisions.append(
                decide_match(
                    person_scores,
                    match_threshold=policy.match_threshold,
                    min_score_gap=policy.min_score_gap,
                )
            )
        results[method] = _metrics(records, decisions)
    return results


def _optimized_person_scores(record: ExperimentRecord, *, top_k: int) -> dict[str, float]:
    """根据记录中的质量分数计算当前优化版的人级别得分。"""
    if record.sample_quality_scores is None:
        return aggregate_person_scores(record.sample_scores, top_k=top_k)
    scores_with_quality = {
        person_id: [
            (score, _quality_at(record.sample_quality_scores.get(person_id, ()), index))
            for index, score in enumerate(sample_scores)
        ]
        for person_id, sample_scores in record.sample_scores.items()
    }
    return aggregate_quality_weighted_scores(scores_with_quality, top_k=top_k)


def _policy_for(
    record: ExperimentRecord,
    quality_tiers: Mapping[str, QualityTierPolicy] | None,
    match_threshold: float,
    min_score_gap: float,
) -> QualityTierPolicy:
    """选择当前探针质量层级对应的识别策略。"""
    if quality_tiers is not None and record.probe_quality_tier in quality_tiers:
        return quality_tiers[record.probe_quality_tier]
    return QualityTierPolicy(match_threshold=match_threshold, min_score_gap=min_score_gap)


def _quality_at(scores: Sequence[float], index: int) -> float:
    """按下标读取质量分数，缺失时返回中性默认值。"""
    return scores[index] if index < len(scores) else 0.6


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """把评测向量转换为单位向量，并拒绝空向量和零向量。"""
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm


def _highest_sample_scores(
    sample_scores: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """为基线方法取每个身份的最高单样本得分。"""
    return {
        person_id: max(scores)
        for person_id, scores in sample_scores.items()
        if scores
    }


def _metrics(
    records: Sequence[ExperimentRecord], decisions: Sequence[MatchDecision]
) -> ExperimentMetrics:
    """根据真实标签和判定结果计算开放集指标。"""
    known_total = 0
    known_correct = 0
    unknown_total = 0
    unknown_rejected = 0
    misidentifications = 0

    for record, decision in zip(records, decisions, strict=True):
        if record.expected_person_id is None:
            unknown_total += 1
            if decision.status != "matched":
                unknown_rejected += 1
            else:
                misidentifications += 1
            continue
        known_total += 1
        if decision.status == "matched" and decision.person_id == record.expected_person_id:
            known_correct += 1
        elif decision.status == "matched":
            misidentifications += 1

    total = len(records)
    average_latency_ms = sum(record.latency_ms for record in records) / total if total else 0.0
    return ExperimentMetrics(
        total=total,
        known_total=known_total,
        known_correct=known_correct,
        unknown_total=unknown_total,
        unknown_rejected=unknown_rejected,
        misidentifications=misidentifications,
        average_latency_ms=average_latency_ms,
    )
