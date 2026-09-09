from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from .evaluation_cache import file_sha256
from .face_engine import FaceInputError, FaceObservation
from .image_input import ImageInput, measure_quality
from .lfw_dataset import LfwProtocol
from .quality_degradation import select_primary_face
from .raw_embedding_cache import RawEmbeddingCache, RawEmbeddingEntry


class RawDatasetFaceEngine(Protocol):
    """有标签图片数据集原始提取所需的模型接口。"""

    def extract_faces(self, frame: np.ndarray) -> list[FaceObservation]:
        """返回一张图片中的全部模型检测结果。"""


@dataclass(frozen=True)
class RawExtractionSummary:
    """一次完整原始数据集提取的覆盖和缓存摘要。"""

    image_total: int
    observed_total: int
    failed_total: int
    cache_hit_total: int
    inference_total: int


def extract_or_load_raw_embedding(
    *,
    image_path: Path,
    relative_path: str,
    face_engine: RawDatasetFaceEngine,
    cache: RawEmbeddingCache,
) -> RawEmbeddingEntry:
    """从原始缓存读取图片，未命中时提取面积最大的主体脸。

    数据集已有身份标签，因此背景次要人脸不会触发桌面应用的多人脸拒绝；
    无人脸仍记录为 FTE。任何数值质量门都不会在这里生效。
    """

    digest = file_sha256(image_path)
    cached = cache.get(relative_path, digest)
    if cached is not None:
        return cached
    image_input = ImageInput.from_file(image_path, source_type="dataset")
    started_at = perf_counter()
    faces = face_engine.extract_faces(image_input.frame)
    latency_ms = (perf_counter() - started_at) * 1000
    try:
        primary = select_primary_face(faces)
    except FaceInputError as error:
        cache.put_failed(
            relative_path=relative_path,
            file_sha256=digest,
            face_count=len(faces),
            latency_ms=latency_ms,
            reason=str(error),
        )
        return RawEmbeddingEntry("failed", None, None, len(faces), latency_ms, str(error))
    metrics = measure_quality(image_input.frame, primary)
    cache.put_observed(
        relative_path=relative_path,
        file_sha256=digest,
        embedding=primary.embedding,
        metrics=metrics,
        face_count=len(faces),
        latency_ms=latency_ms,
    )
    return RawEmbeddingEntry(
        "observed",
        primary.embedding,
        metrics,
        len(faces),
        latency_ms,
        None,
    )


def extract_dataset_raw_embeddings(
    *,
    dataset_dir: Path,
    relative_paths: Sequence[str],
    face_engine: RawDatasetFaceEngine,
    cache: RawEmbeddingCache,
    commit_every: int = 100,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
) -> RawExtractionSummary:
    """对一组去重相对路径执行可恢复、无质量门的原始特征提取。"""

    if commit_every < 1:
        raise ValueError("commit_every must be at least one")
    paths = tuple(sorted(set(relative_paths)))
    observed = failed = cache_hits = inferences = pending = 0
    for index, relative_path in enumerate(paths, start=1):
        image_path = dataset_dir / relative_path
        digest = file_sha256(image_path)
        was_cached = cache.get(relative_path, digest) is not None
        entry = extract_or_load_raw_embedding(
            image_path=image_path,
            relative_path=relative_path,
            face_engine=face_engine,
            cache=cache,
        )
        observed += int(entry.status == "observed")
        failed += int(entry.status == "failed")
        cache_hits += int(was_cached)
        inferences += int(not was_cached)
        pending += int(not was_cached)
        if pending >= commit_every:
            cache.commit()
            pending = 0
        if on_progress is not None:
            on_progress(index, len(paths), relative_path, was_cached)
    cache.commit()
    return RawExtractionSummary(
        image_total=len(paths),
        observed_total=observed,
        failed_total=failed,
        cache_hit_total=cache_hits,
        inference_total=inferences,
    )


def extract_lfw_protocol_raw_embeddings(
    *,
    dataset_dir: Path,
    protocol: LfwProtocol,
    face_engine: RawDatasetFaceEngine,
    cache: RawEmbeddingCache,
    commit_every: int = 100,
    on_progress: Callable[[int, int, str, bool], None] | None = None,
) -> RawExtractionSummary:
    """提取 LFW Gallery 与 Probe 引用的全部去重图片。"""

    relative_paths = (
        *(path for paths in protocol.enrollment.values() for path in paths),
        *(probe.relative_path for probe in protocol.probes),
    )
    return extract_dataset_raw_embeddings(
        dataset_dir=dataset_dir,
        relative_paths=relative_paths,
        face_engine=face_engine,
        cache=cache,
        commit_every=commit_every,
        on_progress=on_progress,
    )
