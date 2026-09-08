from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import Settings
from .evaluation_cache import EvaluationEmbeddingCache
from .face_engine import FaceInputError
from .lfw_evaluation import DatasetRejection, EvaluationFaceEngine, extract_or_load_embedding
from .xqlfw import XqlfwProtocol


@dataclass(frozen=True)
class XqlfwEvaluationResult:
    """一次 XQLFW 全协议验证评测的汇总结果。"""

    image_total: int
    valid_image_total: int
    pair_total: int
    valid_pair_total: int
    positive_total: int
    positive_correct: int
    negative_total: int
    negative_correct: int
    threshold: float
    average_latency_ms: float
    rejected_images: tuple[DatasetRejection, ...]
    rejected_pairs: tuple[tuple[str, str, str], ...]

    @property
    def accuracy(self) -> float | None:
        """返回通过特征提取的验证对准确率。"""

        return (
            (self.positive_correct + self.negative_correct) / self.valid_pair_total
            if self.valid_pair_total
            else None
        )


def evaluate_xqlfw_protocol(
    *,
    dataset_dir: Path,
    protocol: XqlfwProtocol,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
    threshold: float,
    on_image: Callable[[int, int], None] | None = None,
    cache: EvaluationEmbeddingCache | None = None,
    cache_commit_every: int = 100,
) -> XqlfwEvaluationResult:
    """提取 XQLFW 协议涉及的全部图片并统计正负验证对。

    参数：
        dataset_dir：XQLFW 身份图片根目录。
        protocol：由官方 pairs 文件解析出的完整协议。
        settings：评测专用质量门控配置。
        face_engine：提供人脸检测和特征提取的模型适配器。
        threshold：把余弦相似度转换为同人/不同人的验证阈值。
        on_image：可选的进度回调，参数为“已处理图片数、总图片数”。
        cache：可选的可恢复 embedding 缓存。
        cache_commit_every：缓存每处理多少张图片提交一次。
    返回：
        全部有效验证对的准确率、数量、耗时和明确拒绝原因。
    前置条件：
        threshold 位于 `[0, 1]`；协议中的图片路径相对于 dataset_dir。
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("XQLFW verification threshold must be between 0 and 1")
    if cache_commit_every < 1:
        raise ValueError("cache_commit_every must be at least one")
    embeddings = {}
    rejections: list[DatasetRejection] = []
    latency_total_ms = 0.0
    images_since_commit = 0
    for index, relative_path in enumerate(protocol.image_paths, start=1):
        started_at = perf_counter()
        try:
            embedding, _ = extract_or_load_embedding(
                dataset_dir / relative_path,
                relative_path,
                settings,
                face_engine,
                cache,
            )
        except (FaceInputError, ValueError) as error:
            rejections.append(DatasetRejection(relative_path, str(error)))
        else:
            embeddings[relative_path] = embedding
            latency_total_ms += (perf_counter() - started_at) * 1000
        images_since_commit += 1
        if cache is not None and images_since_commit >= cache_commit_every:
            cache.commit()
            images_since_commit = 0
        if on_image is not None:
            on_image(index, len(protocol.image_paths))

    if cache is not None:
        cache.commit()

    positive_total = positive_correct = negative_total = negative_correct = 0
    valid_pair_total = 0
    rejected_pairs: list[tuple[str, str, str]] = []
    for pair in protocol.pairs:
        left = embeddings.get(pair.left_path)
        right = embeddings.get(pair.right_path)
        if left is None or right is None:
            missing = "left" if left is None else "right"
            rejected_pairs.append((pair.left_path, pair.right_path, f"{missing}_image_rejected"))
            continue
        valid_pair_total += 1
        score = float(left @ right)
        predicted_same = score >= threshold
        if pair.same_identity:
            positive_total += 1
            positive_correct += int(predicted_same)
        else:
            negative_total += 1
            negative_correct += int(not predicted_same)

    return XqlfwEvaluationResult(
        image_total=len(protocol.image_paths),
        valid_image_total=len(embeddings),
        pair_total=len(protocol.pairs),
        valid_pair_total=valid_pair_total,
        positive_total=positive_total,
        positive_correct=positive_correct,
        negative_total=negative_total,
        negative_correct=negative_correct,
        threshold=threshold,
        average_latency_ms=latency_total_ms / len(embeddings) if embeddings else 0.0,
        rejected_images=tuple(rejections),
        rejected_pairs=tuple(rejected_pairs),
    )
