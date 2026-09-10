from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .joint_calibration import JointOperatingPoint, JointScoreRow, accepts_joint_score


@dataclass(frozen=True)
class RejectedProbe:
    """一张未产生 embedding、只进入端到端分母的 Probe。"""

    source_identity: str
    expected_person_id: str | None


def bootstrap_identity_confidence_intervals(
    rows: Sequence[JointScoreRow],
    *,
    rejected_probes: Sequence[RejectedProbe],
    selected: JointOperatingPoint,
    repeats: int = 2000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    """按身份分组重采样并计算开放集指标的 95% 置信区间。

    同一人的多张图片在每次重采样中一起出现，避免把相关图片错误地当成
    独立样本。Known 与 Unknown 身份分别有放回抽样；FTE 只影响端到端分母。
    """

    if repeats < 1:
        raise ValueError("bootstrap repeats must be at least one")
    known_groups: dict[str, np.ndarray] = {}
    unknown_groups: dict[str, np.ndarray] = {}
    for row in rows:
        accepted = accepts_joint_score(row, selected)
        if row.expected_person_id is None:
            values = unknown_groups.setdefault(row.source_identity, np.zeros(2, dtype=np.int64))
            values += (1, int(accepted))
        else:
            values = known_groups.setdefault(row.source_identity, np.zeros(4, dtype=np.int64))
            values += (
                1,
                1,
                int(accepted and row.top_is_correct),
                int(row.top_is_correct),
            )
    for rejected in rejected_probes:
        if rejected.expected_person_id is None:
            unknown_groups.setdefault(rejected.source_identity, np.zeros(2, dtype=np.int64))
        else:
            values = known_groups.setdefault(
                rejected.source_identity, np.zeros(4, dtype=np.int64)
            )
            values[0] += 1
    if not known_groups or not unknown_groups:
        raise ValueError("bootstrap requires both Known and Unknown identity groups")

    rng = np.random.default_rng(seed)
    known = np.stack(tuple(known_groups.values()))
    unknown = np.stack(tuple(unknown_groups.values()))
    known_samples = known[
        rng.integers(0, known.shape[0], size=(repeats, known.shape[0]))
    ].sum(axis=1)
    unknown_samples = unknown[
        rng.integers(0, unknown.shape[0], size=(repeats, unknown.shape[0]))
    ].sum(axis=1)
    metrics = {
        "fpir_valid": _safe_ratio(unknown_samples[:, 1], unknown_samples[:, 0]),
        "tpir_valid": _safe_ratio(known_samples[:, 2], known_samples[:, 1]),
        "tpir_e2e": _safe_ratio(known_samples[:, 2], known_samples[:, 0]),
        "rank1_valid": _safe_ratio(known_samples[:, 3], known_samples[:, 1]),
    }
    return {
        name: {
            "lower": float(np.nanpercentile(values, 2.5)),
            "upper": float(np.nanpercentile(values, 97.5)),
        }
        for name, values in metrics.items()
    }


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """执行逐项除法，并把零分母写成 NaN 以排除无效重复。"""

    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result
