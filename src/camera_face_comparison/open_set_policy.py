from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

AcceptanceRule = Literal["score_threshold", "score_gap", "score_and_gap", "nac"]


@dataclass(frozen=True)
class ScoreThresholdPolicy:
    """只检查最高身份分数的开放集接收策略。"""

    minimum_score: float
    rule: Literal["score_threshold"] = "score_threshold"

    def __post_init__(self) -> None:
        """拒绝超出余弦分数范围的阈值。"""

        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")


@dataclass(frozen=True)
class ScoreGapPolicy:
    """只检查第一、第二候选分差的开放集接收策略。"""

    minimum_gap: float
    rule: Literal["score_gap"] = "score_gap"

    def __post_init__(self) -> None:
        """拒绝超出余弦分差范围的下限。"""

        if not 0.0 <= self.minimum_gap <= 2.0:
            raise ValueError("minimum_gap must be between 0 and 2")


@dataclass(frozen=True)
class ScoreAndGapPolicy:
    """同时检查最高分与候选分差的开放集接收策略。"""

    minimum_score: float
    minimum_gap: float
    rule: Literal["score_and_gap"] = "score_and_gap"

    def __post_init__(self) -> None:
        """分别校验最高分和候选分差下限。"""

        ScoreThresholdPolicy(self.minimum_score)
        ScoreGapPolicy(self.minimum_gap)


@dataclass(frozen=True)
class NacPolicy:
    """使用邻域感知余弦概率的开放集接收策略。"""

    neighbor_count: int
    minimum_probability: float
    rule: Literal["nac"] = "nac"

    def __post_init__(self) -> None:
        """要求至少两个邻居，并校验概率阈值。"""

        if self.neighbor_count < 2:
            raise ValueError("neighbor_count must be at least 2")
        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError("minimum_probability must be between 0 and 1")


RecognitionPolicy: TypeAlias = (
    ScoreThresholdPolicy | ScoreGapPolicy | ScoreAndGapPolicy | NacPolicy
)


@dataclass(frozen=True)
class OpenSetDecision:
    """人员排序经过一种明确接收规则后的完整判定。"""

    accepted_person_id: str | None
    top_person_id: str | None
    top_score: float | None
    second_score: float | None
    score_gap: float | None
    acceptance_score: float | None
    rule: AcceptanceRule
    reason: str | None

    @property
    def accepted(self) -> bool:
        """返回当前判定是否接受第一候选身份。"""

        return self.accepted_person_id is not None


def apply_open_set_policy(
    person_scores: Mapping[str, float],
    policy: RecognitionPolicy,
) -> OpenSetDecision:
    """排序人员分数并应用一种互斥的开放集接收规则。

    参数：
        person_scores：身份编号到连续人员分数的映射。
        policy：最高分、候选分差、联合或 NAC 策略。
    返回：
        包含候选分数、实际接收分数和拒绝原因的判定。
    前置条件：
        人员分数必须是有限数；相对规则至少需要两个身份。
    """

    ranked = sorted(person_scores.items(), key=lambda item: (-float(item[1]), item[0]))
    return apply_ranked_open_set_policy(ranked, policy)


def apply_ranked_open_set_policy(
    ranked: Sequence[tuple[str, float]],
    policy: RecognitionPolicy,
) -> OpenSetDecision:
    """对已按分数降序排列的候选列表应用开放集接收规则。

    参数：
        ranked：候选身份及分数，必须已按分数降序排列；同分时按身份编号排列。
        policy：最高分、候选分差、联合或 NAC 策略。
    返回：
        包含候选分数、实际接收分数和拒绝原因的判定。
    前置条件：
        候选列表中的分数必须是有限数；使用 NAC 时需要至少两个候选。
    """

    ranked = list(ranked)
    if not ranked:
        return OpenSetDecision(
            None, None, None, None, None, None, policy.rule, "empty_face_library"
        )
    if any(not np.isfinite(score) for _, score in ranked):
        raise ValueError("person scores must be finite")

    top_person_id, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    score_gap = top_score - second_score if second_score is not None else None
    if isinstance(policy, ScoreThresholdPolicy):
        accepted = top_score >= policy.minimum_score
        return _decision(
            top_person_id,
            top_score,
            second_score,
            score_gap,
            acceptance_score=top_score,
            rule=policy.rule,
            accepted=accepted,
            reason=None if accepted else "score_below_threshold",
        )

    if second_score is None or score_gap is None:
        return _decision(
            top_person_id,
            top_score,
            second_score,
            score_gap,
            acceptance_score=None,
            rule=policy.rule,
            accepted=False,
            reason="insufficient_gallery_identities",
        )
    if isinstance(policy, ScoreGapPolicy):
        accepted = score_gap >= policy.minimum_gap
        return _decision(
            top_person_id,
            top_score,
            second_score,
            score_gap,
            acceptance_score=score_gap,
            rule=policy.rule,
            accepted=accepted,
            reason=None if accepted else "score_gap_below_minimum",
        )
    if isinstance(policy, ScoreAndGapPolicy):
        if top_score < policy.minimum_score:
            return _decision(
                top_person_id,
                top_score,
                second_score,
                score_gap,
                acceptance_score=top_score,
                rule=policy.rule,
                accepted=False,
                reason="score_below_threshold",
            )
        accepted = score_gap >= policy.minimum_gap
        return _decision(
            top_person_id,
            top_score,
            second_score,
            score_gap,
            acceptance_score=min(top_score, score_gap),
            rule=policy.rule,
            accepted=accepted,
            reason=None if accepted else "score_gap_below_minimum",
        )

    probability = neighborhood_aware_cosine(
        [score for _, score in ranked], neighbor_count=policy.neighbor_count
    )
    accepted = probability >= policy.minimum_probability
    return _decision(
        top_person_id,
        top_score,
        second_score,
        score_gap,
        acceptance_score=probability,
        rule=policy.rule,
        accepted=accepted,
        reason=None if accepted else "nac_below_threshold",
    )


def neighborhood_aware_cosine(scores: list[float], *, neighbor_count: int) -> float:
    """计算第一候选在前 K 个身份中的稳定局部 softmax 概率。"""

    if neighbor_count < 2:
        raise ValueError("neighbor_count must be at least 2")
    if len(scores) < 2:
        raise ValueError("NAC requires at least two Gallery identities")
    values = np.asarray(sorted(scores, reverse=True)[:neighbor_count], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("NAC scores must be finite")
    shifted = values - values[0]
    return float(1.0 / np.exp(shifted).sum())


def _decision(
    top_person_id: str,
    top_score: float,
    second_score: float | None,
    score_gap: float | None,
    *,
    acceptance_score: float | None,
    rule: AcceptanceRule,
    accepted: bool,
    reason: str | None,
) -> OpenSetDecision:
    """用统一字段构造接受或拒绝结果。"""

    return OpenSetDecision(
        accepted_person_id=top_person_id if accepted else None,
        top_person_id=top_person_id,
        top_score=float(top_score),
        second_score=None if second_score is None else float(second_score),
        score_gap=None if score_gap is None else float(score_gap),
        acceptance_score=acceptance_score,
        rule=rule,
        reason=reason,
    )
