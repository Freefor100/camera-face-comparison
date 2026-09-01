from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .recognition import decide_match


@dataclass(frozen=True)
class CalibrationRecord:
    """一张留出探针及其已经计算好的人级别得分。"""

    expected_person_id: str | None
    person_scores: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationResult:
    """候选阈值中安全性优先且已记录测量结果的一组阈值。"""

    match_threshold: float
    min_margin: float
    unknown_false_accepts: int
    known_correct: int


def calibrate_thresholds(
    *,
    records: Sequence[CalibrationRecord],
    threshold_candidates: Iterable[float],
    margin_candidates: Iterable[float],
) -> CalibrationResult:
    """先优先拒绝未知人员，再在候选范围内最大化已知人员正确数。

    参数：
        records：带真实标签的人级得分记录。
        threshold_candidates：待搜索的最高相似度阈值。
        margin_candidates：待搜索的第一、第二候选差距阈值。
    返回：
        未知误接受数最少、已知正确数最多的阈值结果。
    前置条件：
        三个输入序列都不能为空；记录中的分数应已由同一评测协议生成。
    """

    thresholds = sorted({float(value) for value in threshold_candidates})
    margins = sorted({float(value) for value in margin_candidates})
    if not records or not thresholds or not margins:
        raise ValueError("records, threshold_candidates, and margin_candidates must not be empty")

    candidates: list[CalibrationResult] = []
    for threshold in thresholds:
        for margin in margins:
            unknown_false_accepts = 0
            known_correct = 0
            for record in records:
                decision = decide_match(
                    record.person_scores,
                    match_threshold=threshold,
                    min_margin=margin,
                )
                if record.expected_person_id is None:
                    unknown_false_accepts += int(decision.status == "matched")
                else:
                    known_correct += int(decision.person_id == record.expected_person_id)
            candidates.append(
                CalibrationResult(
                    match_threshold=threshold,
                    min_margin=margin,
                    unknown_false_accepts=unknown_false_accepts,
                    known_correct=known_correct,
                )
            )

    return max(
        candidates,
        key=lambda item: (
            -item.unknown_false_accepts,
            item.known_correct,
            item.match_threshold,
            item.min_margin,
        ),
    )
