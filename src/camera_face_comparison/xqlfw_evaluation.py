from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .evaluation_cache import file_sha256
from .raw_embedding_cache import RawEmbeddingCache
from .xqlfw import XqlfwProtocol


@dataclass(frozen=True)
class XqlfwImageRejection:
    """一张 XQLFW 图片的模型特征提取失败记录。"""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class XqlfwPairResult:
    """一对图片在其折次阈值下的连续分数和验证结果。"""

    left_path: str
    right_path: str
    fold: int
    same_identity: bool
    similarity: float
    threshold: float
    predicted_same: bool
    correct: bool
    left_quality: float
    right_quality: float
    min_quality: float
    quality_gap: float


@dataclass(frozen=True)
class XqlfwFoldResult:
    """一折“其余折标定、当前折评测”的验证结果。"""

    fold: int
    training_pair_total: int
    test_pair_total: int
    threshold: float
    positive_total: int
    positive_correct: int
    negative_total: int
    negative_correct: int
    accuracy: float


@dataclass(frozen=True)
class XqlfwQualityBin:
    """按官方质量分或两图质量差分桶后的验证统计。"""

    label: str
    pair_total: int
    positive_total: int
    positive_correct: int
    negative_total: int
    negative_correct: int
    accuracy: float | None


@dataclass(frozen=True)
class XqlfwEvaluationResult:
    """无启发式质量预筛的 XQLFW 官方按折评测结果。"""

    image_total: int
    valid_image_total: int
    pair_total: int
    valid_pair_total: int
    positive_total: int
    positive_correct: int
    negative_total: int
    negative_correct: int
    accuracy: float
    fold_accuracy_mean: float
    fold_accuracy_std: float
    average_extraction_latency_ms: float
    positive_similarity_distribution: dict[str, float]
    negative_similarity_distribution: dict[str, float]
    rejected_images: tuple[XqlfwImageRejection, ...]
    rejected_pairs: tuple[tuple[str, str, str], ...]
    folds: tuple[XqlfwFoldResult, ...]
    min_quality_bins: tuple[XqlfwQualityBin, ...]
    quality_gap_bins: tuple[XqlfwQualityBin, ...]
    pairs: tuple[XqlfwPairResult, ...]


@dataclass(frozen=True)
class _PairInput:
    """尚未应用折次阈值的一对有效图片。"""

    left_path: str
    right_path: str
    fold: int
    same_identity: bool
    similarity: float
    left_quality: float
    right_quality: float


def evaluate_xqlfw_protocol(
    *,
    dataset_dir: Path,
    protocol: XqlfwProtocol,
    quality_scores: Mapping[str, float],
    cache: RawEmbeddingCache,
) -> XqlfwEvaluationResult:
    """从原始缓存执行 XQLFW 官方按折验证，不初始化人脸模型。

    参数：
        dataset_dir：XQLFW 图片根目录。
        protocol：官方 pairs 协议。
        quality_scores：数据集随附的逐图片质量分。
        cache：与质量和判定策略无关的原始 embedding 缓存。
    返回：
        每折阈值、逐对连续分数、FTE 和跨质量分桶结果。
    前置条件：
        协议至少包含两折；缓存和官方质量分必须覆盖协议引用的每张图片。
    """

    if protocol.fold_count < 2:
        raise ValueError("XQLFW cross-validation requires at least two folds")
    embeddings: dict[str, np.ndarray] = {}
    rejected_images: list[XqlfwImageRejection] = []
    latency_total_ms = 0.0
    for relative_path in protocol.image_paths:
        image_path = dataset_dir / relative_path
        if relative_path not in quality_scores:
            raise ValueError(f"XQLFW quality score is missing: {relative_path}")
        entry = cache.get(relative_path, file_sha256(image_path))
        if entry is None:
            raise RuntimeError(f"XQLFW raw cache is incomplete or stale: {relative_path}")
        if entry.status == "failed":
            rejected_images.append(
                XqlfwImageRejection(relative_path, entry.reason or "model_extraction_failed")
            )
            continue
        if entry.embedding is None:
            raise RuntimeError(f"XQLFW cached embedding is missing: {relative_path}")
        embeddings[relative_path] = _normalize(entry.embedding)
        latency_total_ms += entry.latency_ms

    pair_inputs: list[_PairInput] = []
    rejected_pairs: list[tuple[str, str, str]] = []
    for pair in protocol.pairs:
        left = embeddings.get(pair.left_path)
        right = embeddings.get(pair.right_path)
        if left is None or right is None:
            missing = "left" if left is None else "right"
            rejected_pairs.append((pair.left_path, pair.right_path, f"{missing}_image_fte"))
            continue
        left_quality = _validated_quality(quality_scores[pair.left_path])
        right_quality = _validated_quality(quality_scores[pair.right_path])
        pair_inputs.append(
            _PairInput(
                left_path=pair.left_path,
                right_path=pair.right_path,
                fold=pair.fold,
                same_identity=pair.same_identity,
                similarity=float(np.clip(left @ right, -1.0, 1.0)),
                left_quality=left_quality,
                right_quality=right_quality,
            )
        )

    thresholds: dict[int, float] = {}
    folds: list[XqlfwFoldResult] = []
    for fold in range(1, protocol.fold_count + 1):
        training = [pair for pair in pair_inputs if pair.fold != fold]
        test = [pair for pair in pair_inputs if pair.fold == fold]
        if not training or not test:
            raise ValueError(f"XQLFW fold {fold} has no valid training or test pairs")
        threshold = select_verification_threshold(training)
        thresholds[fold] = threshold
        folds.append(_summarize_fold(fold, training, test, threshold))

    pairs = tuple(_apply_threshold(pair, thresholds[pair.fold]) for pair in pair_inputs)
    positive = [pair for pair in pairs if pair.same_identity]
    negative = [pair for pair in pairs if not pair.same_identity]
    correct_total = sum(pair.correct for pair in pairs)
    fold_accuracies = np.asarray([fold.accuracy for fold in folds], dtype=np.float64)
    return XqlfwEvaluationResult(
        image_total=len(protocol.image_paths),
        valid_image_total=len(embeddings),
        pair_total=len(protocol.pairs),
        valid_pair_total=len(pairs),
        positive_total=len(positive),
        positive_correct=sum(pair.correct for pair in positive),
        negative_total=len(negative),
        negative_correct=sum(pair.correct for pair in negative),
        accuracy=correct_total / len(pairs) if pairs else 0.0,
        fold_accuracy_mean=float(np.mean(fold_accuracies)),
        fold_accuracy_std=float(np.std(fold_accuracies)),
        average_extraction_latency_ms=(
            latency_total_ms / len(embeddings) if embeddings else 0.0
        ),
        positive_similarity_distribution=_distribution(
            [pair.similarity for pair in positive]
        ),
        negative_similarity_distribution=_distribution(
            [pair.similarity for pair in negative]
        ),
        rejected_images=tuple(rejected_images),
        rejected_pairs=tuple(rejected_pairs),
        folds=tuple(folds),
        min_quality_bins=_quality_bins(
            pairs,
            values=[pair.min_quality for pair in pairs],
            boundaries=(0.0, 0.4, 0.6, 0.8, 1.0),
        ),
        quality_gap_bins=_quality_bins(
            pairs,
            values=[pair.quality_gap for pair in pairs],
            boundaries=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
        ),
        pairs=pairs,
    )


def select_verification_threshold(rows: Sequence[_PairInput]) -> float:
    """在训练折的实际相似度断点上选择准确率最高、误接收更少的阈值。"""

    if not rows:
        raise ValueError("verification threshold calibration requires pairs")
    ordered = sorted(rows, key=lambda row: row.similarity, reverse=True)
    negative_total = sum(not row.same_identity for row in ordered)
    threshold = float(np.nextafter(ordered[0].similarity, np.inf))
    best_key = (negative_total, 0, threshold)
    best_threshold = threshold
    true_positive = false_positive = 0
    index = 0
    while index < len(ordered):
        score = ordered[index].similarity
        while index < len(ordered) and ordered[index].similarity == score:
            if ordered[index].same_identity:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        correct = true_positive + negative_total - false_positive
        candidate_key = (correct, -false_positive, score)
        if candidate_key > best_key:
            best_key = candidate_key
            best_threshold = score
    return float(best_threshold)


def _summarize_fold(
    fold: int,
    training: Sequence[_PairInput],
    test: Sequence[_PairInput],
    threshold: float,
) -> XqlfwFoldResult:
    """把固定训练折阈值应用到一折测试对并统计准确率。"""

    predicted = [_apply_threshold(pair, threshold) for pair in test]
    positive = [pair for pair in predicted if pair.same_identity]
    negative = [pair for pair in predicted if not pair.same_identity]
    correct = sum(pair.correct for pair in predicted)
    return XqlfwFoldResult(
        fold=fold,
        training_pair_total=len(training),
        test_pair_total=len(test),
        threshold=threshold,
        positive_total=len(positive),
        positive_correct=sum(pair.correct for pair in positive),
        negative_total=len(negative),
        negative_correct=sum(pair.correct for pair in negative),
        accuracy=correct / len(predicted),
    )


def _apply_threshold(pair: _PairInput, threshold: float) -> XqlfwPairResult:
    """把一折已冻结阈值应用到一条连续相似度记录。"""

    predicted_same = pair.similarity >= threshold
    return XqlfwPairResult(
        left_path=pair.left_path,
        right_path=pair.right_path,
        fold=pair.fold,
        same_identity=pair.same_identity,
        similarity=pair.similarity,
        threshold=threshold,
        predicted_same=predicted_same,
        correct=predicted_same == pair.same_identity,
        left_quality=pair.left_quality,
        right_quality=pair.right_quality,
        min_quality=min(pair.left_quality, pair.right_quality),
        quality_gap=abs(pair.left_quality - pair.right_quality),
    )


def _quality_bins(
    pairs: Sequence[XqlfwPairResult],
    *,
    values: Sequence[float],
    boundaries: Sequence[float],
) -> tuple[XqlfwQualityBin, ...]:
    """用预先固定的边界统计不同图片质量条件下的验证结果。"""

    bins: list[XqlfwQualityBin] = []
    for index, (lower, upper) in enumerate(pairwise(boundaries)):
        is_last = index == len(boundaries) - 2
        if is_last:
            selected = [
                pair
                for pair, value in zip(pairs, values, strict=True)
                if lower <= value <= upper
            ]
        else:
            selected = [
                pair
                for pair, value in zip(pairs, values, strict=True)
                if lower <= value < upper
            ]
        positive = [pair for pair in selected if pair.same_identity]
        negative = [pair for pair in selected if not pair.same_identity]
        correct = sum(pair.correct for pair in selected)
        closing = "]" if is_last else ")"
        bins.append(
            XqlfwQualityBin(
                label=f"[{lower:.2f}, {upper:.2f}{closing}",
                pair_total=len(selected),
                positive_total=len(positive),
                positive_correct=sum(pair.correct for pair in positive),
                negative_total=len(negative),
                negative_correct=sum(pair.correct for pair in negative),
                accuracy=correct / len(selected) if selected else None,
            )
        )
    return tuple(bins)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    """返回相似度的最小值、常用分位点和最大值。"""

    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _validated_quality(value: float) -> float:
    """检查官方质量分范围并返回统一浮点数。"""

    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"XQLFW quality score must be between zero and one: {normalized}")
    return normalized


def _normalize(vector: np.ndarray) -> np.ndarray:
    """恢复缓存向量时再次保证 L2 归一化。"""

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("XQLFW embedding must have a finite positive norm")
    return array / norm
