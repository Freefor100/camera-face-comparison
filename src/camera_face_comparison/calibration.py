from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DecisionScoreRow:
    """一张 Probe 在一种聚合方法下的不含阈值连续结果。"""

    expected_person_id: str | None
    top_score: float
    score_gap: float
    top_is_correct: bool


@dataclass(frozen=True)
class OperatingPoint:
    """一种接收规则在指定 FPIR 目标下的精确工作点。"""

    target_fpir: float
    use_score_gap: bool
    match_threshold: float
    min_score_gap: float
    meets_target: bool
    known_total: int
    rank1_correct: int
    known_true_accepts: int
    known_wrong_accepts: int
    unknown_total: int
    unknown_false_accepts: int
    fpir: float
    tpir: float
    fnir: float


@dataclass(frozen=True)
class VariantCalibration:
    """一种人员聚合变体的覆盖、Rank-1 和六个候选工作点。"""

    method: str
    top_k: int
    known_total: int
    unknown_total: int
    rank1_correct: int
    rank1_rate: float
    operating_points: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class CalibrationReport:
    """只从 Calibration 分区生成的全部聚合与判定规则报告。"""

    run_id: str
    source_split: Literal["calibration"]
    targets: tuple[float, ...]
    variants: tuple[VariantCalibration, ...]


@dataclass(frozen=True)
class FixedPointEvaluation:
    """把 Calibration 选定参数原样应用到独立分区后的结果。"""

    run_id: str
    source_split: Literal["evaluation"]
    method: str
    top_k: int
    use_score_gap: bool
    match_threshold: float
    min_score_gap: float
    known_total: int
    rank1_correct: int
    known_true_accepts: int
    known_wrong_accepts: int
    unknown_total: int
    unknown_false_accepts: int
    fpir: float
    tpir: float
    fnir: float


def calibrate_method(
    rows: Sequence[DecisionScoreRow],
    *,
    target_fpir: float,
    use_score_gap: bool,
) -> OperatingPoint:
    """在实际分数断点上精确选择一个开放集工作点。

    参数：
        rows：同一聚合方法、同一数据分区的连续分数。
        target_fpir：允许的 Unknown 误接收率上限。
        use_score_gap：是否同时检查第一、第二候选分差。
    返回：
        优先满足 FPIR、再最大化 Known TPIR 的工作点。
    前置条件：
        必须同时包含 Known 和 Unknown；分数及分差位于闭区间 `[0, 1]`。

    判定只会在已有 `top_score` 或 `score_gap` 断点处变化，因此无需反向传播、
    固定网格或粗搜后细搜。二维规则通过离散直方图的后缀累计精确计算。
    """

    if not rows:
        raise ValueError("decision score rows must not be empty")
    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be between zero and one")
    _validate_rows(rows)

    top_scores = np.asarray([row.top_score for row in rows], dtype=np.float64)
    score_gaps = np.asarray([row.score_gap for row in rows], dtype=np.float64)
    known_mask = np.asarray([row.expected_person_id is not None for row in rows], dtype=bool)
    correct_mask = np.asarray([row.top_is_correct for row in rows], dtype=bool) & known_mask
    wrong_known_mask = known_mask & ~correct_mask
    unknown_mask = ~known_mask
    known_total = int(np.sum(known_mask))
    unknown_total = int(np.sum(unknown_mask))
    if known_total == 0 or unknown_total == 0:
        raise ValueError("calibration requires both Known and Unknown rows")

    thresholds = np.unique(np.concatenate((np.array([0.0, 1.0]), top_scores)))
    gaps = (
        np.unique(np.concatenate((np.array([0.0, 1.0]), score_gaps)))
        if use_score_gap
        else np.array([0.0])
    )
    correct_counts, wrong_counts, unknown_counts = _acceptance_count_matrices(
        top_scores=top_scores,
        score_gaps=score_gaps,
        correct_mask=correct_mask,
        wrong_known_mask=wrong_known_mask,
        unknown_mask=unknown_mask,
        thresholds=thresholds,
        gaps=gaps,
    )
    allowed_unknown = int(np.floor(target_fpir * unknown_total + 1e-12))
    meets = unknown_counts <= allowed_unknown
    if np.any(meets):
        candidate_indices = np.argwhere(meets)
        meets_target = True
    else:
        candidate_indices = np.argwhere(unknown_counts == np.min(unknown_counts))
        meets_target = False

    selected_threshold_index, selected_gap_index = min(
        (tuple(int(value) for value in index) for index in candidate_indices),
        key=lambda index: (
            -int(correct_counts[index]),
            int(wrong_counts[index]),
            int(unknown_counts[index]),
            float(gaps[index[1]]),
            float(thresholds[index[0]]),
        ),
    )
    selected = (selected_threshold_index, selected_gap_index)
    known_true_accepts = int(correct_counts[selected])
    unknown_false_accepts = int(unknown_counts[selected])
    return OperatingPoint(
        target_fpir=target_fpir,
        use_score_gap=use_score_gap,
        match_threshold=float(thresholds[selected_threshold_index]),
        min_score_gap=float(gaps[selected_gap_index]),
        meets_target=meets_target,
        known_total=known_total,
        rank1_correct=int(np.sum(correct_mask)),
        known_true_accepts=known_true_accepts,
        known_wrong_accepts=int(wrong_counts[selected]),
        unknown_total=unknown_total,
        unknown_false_accepts=unknown_false_accepts,
        fpir=unknown_false_accepts / unknown_total,
        tpir=known_true_accepts / known_total,
        fnir=1.0 - known_true_accepts / known_total,
    )


def compare_methods(
    path: Path,
    *,
    run_id: str,
    targets: Sequence[float] = (0.01, 0.003, 0.0),
) -> CalibrationReport:
    """只读取固定分数库的 Calibration 分区并比较全部聚合变体。"""

    normalized_targets = tuple(float(value) for value in targets)
    if not normalized_targets:
        raise ValueError("calibration targets must not be empty")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        variants = connection.execute(
            """
            SELECT DISTINCT method, top_k
            FROM decision_scores
            WHERE run_id = ? AND split = 'calibration'
            ORDER BY method, top_k
            """,
            (run_id,),
        ).fetchall()
        reports: list[VariantCalibration] = []
        for method, top_k in variants:
            raw_rows = connection.execute(
                """
                SELECT expected_person_id, top_score, score_gap, top_is_correct
                FROM decision_scores
                WHERE run_id = ? AND split = 'calibration' AND method = ? AND top_k = ?
                ORDER BY rowid
                """,
                (run_id, method, top_k),
            ).fetchall()
            rows = tuple(
                DecisionScoreRow(
                    expected_person_id=(None if row[0] is None else str(row[0])),
                    top_score=float(row[1]),
                    score_gap=float(row[2]),
                    top_is_correct=bool(row[3]),
                )
                for row in raw_rows
            )
            points = tuple(
                calibrate_method(rows, target_fpir=target, use_score_gap=use_gap)
                for target in normalized_targets
                for use_gap in (False, True)
            )
            known_total = sum(row.expected_person_id is not None for row in rows)
            rank1_correct = sum(row.top_is_correct for row in rows)
            reports.append(
                VariantCalibration(
                    method=str(method),
                    top_k=int(top_k),
                    known_total=known_total,
                    unknown_total=len(rows) - known_total,
                    rank1_correct=rank1_correct,
                    rank1_rate=rank1_correct / known_total if known_total else 0.0,
                    operating_points=points,
                )
            )
    finally:
        connection.close()
    if not reports:
        raise ValueError(f"no Calibration decision scores found for run: {run_id}")
    return CalibrationReport(
        run_id=run_id,
        source_split="calibration",
        targets=normalized_targets,
        variants=tuple(reports),
    )


def evaluate_operating_point(
    path: Path,
    *,
    run_id: str,
    method: str,
    top_k: int,
    selected: OperatingPoint,
) -> FixedPointEvaluation:
    """只在显式调用时把一个已选工作点应用到独立 Evaluation 分区。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        raw_rows = connection.execute(
            """
            SELECT expected_person_id, top_score, score_gap, top_is_correct
            FROM decision_scores
            WHERE run_id = ? AND split = 'evaluation' AND method = ? AND top_k = ?
            ORDER BY rowid
            """,
            (run_id, method, top_k),
        ).fetchall()
    finally:
        connection.close()
    rows = tuple(
        DecisionScoreRow(
            expected_person_id=None if row[0] is None else str(row[0]),
            top_score=float(row[1]),
            score_gap=float(row[2]),
            top_is_correct=bool(row[3]),
        )
        for row in raw_rows
    )
    if not rows:
        raise ValueError("no Evaluation rows found for selected method")
    _validate_rows(rows)
    known_mask = np.asarray([row.expected_person_id is not None for row in rows], dtype=bool)
    correct_mask = np.asarray([row.top_is_correct for row in rows], dtype=bool) & known_mask
    accepted = np.asarray(
        [
            row.top_score >= selected.match_threshold
            and (not selected.use_score_gap or row.score_gap >= selected.min_score_gap)
            for row in rows
        ],
        dtype=bool,
    )
    unknown_mask = ~known_mask
    known_total = int(np.sum(known_mask))
    unknown_total = int(np.sum(unknown_mask))
    if known_total == 0 or unknown_total == 0:
        raise ValueError("Evaluation requires both Known and Unknown rows")
    known_true_accepts = int(np.sum(accepted & correct_mask))
    unknown_false_accepts = int(np.sum(accepted & unknown_mask))
    return FixedPointEvaluation(
        run_id=run_id,
        source_split="evaluation",
        method=method,
        top_k=top_k,
        use_score_gap=selected.use_score_gap,
        match_threshold=selected.match_threshold,
        min_score_gap=selected.min_score_gap,
        known_total=known_total,
        rank1_correct=int(np.sum(correct_mask)),
        known_true_accepts=known_true_accepts,
        known_wrong_accepts=int(np.sum(accepted & known_mask & ~correct_mask)),
        unknown_total=unknown_total,
        unknown_false_accepts=unknown_false_accepts,
        fpir=unknown_false_accepts / unknown_total,
        tpir=known_true_accepts / known_total,
        fnir=1.0 - known_true_accepts / known_total,
    )


def available_run_ids(path: Path) -> tuple[str, ...]:
    """列出分数库中实际存在的运行编号，不读取任何 Probe 标签。"""

    if not path.is_file():
        return ()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT DISTINCT run_id FROM decision_scores ORDER BY run_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return ()
    finally:
        connection.close()
    return tuple(str(row[0]) for row in rows)


def _acceptance_count_matrices(
    *,
    top_scores: np.ndarray,
    score_gaps: np.ndarray,
    correct_mask: np.ndarray,
    wrong_known_mask: np.ndarray,
    unknown_mask: np.ndarray,
    thresholds: np.ndarray,
    gaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用二维离散直方图后缀和统计每个精确断点的接收数量。"""

    shape = (thresholds.size, gaps.size)
    score_indices = np.searchsorted(thresholds, top_scores)
    gap_indices = (
        np.searchsorted(gaps, score_gaps)
        if gaps.size > 1
        else np.zeros(score_gaps.size, dtype=np.int64)
    )

    def cumulative(mask: np.ndarray) -> np.ndarray:
        """生成当前标签集合在所有 `score>=阈值且gap>=下限` 下的计数。"""

        histogram = np.zeros(shape, dtype=np.int32)
        np.add.at(histogram, (score_indices[mask], gap_indices[mask]), 1)
        return (
            histogram[::-1, ::-1]
            .cumsum(axis=0, dtype=np.int32)
            .cumsum(axis=1, dtype=np.int32)[::-1, ::-1]
        )

    return cumulative(correct_mask), cumulative(wrong_known_mask), cumulative(unknown_mask)


def _validate_rows(rows: Sequence[DecisionScoreRow]) -> None:
    """拒绝超出余弦得分和候选分差合法范围的实验数据。"""

    for row in rows:
        if not 0.0 <= row.top_score <= 1.0:
            raise ValueError("top_score must be between zero and one")
        if not 0.0 <= row.score_gap <= 1.0:
            raise ValueError("score_gap must be between zero and one")
        if row.expected_person_id is None and row.top_is_correct:
            raise ValueError("Unknown row cannot have a correct Gallery identity")
