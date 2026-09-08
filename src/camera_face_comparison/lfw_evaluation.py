from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .evaluation_cache import EvaluationEmbeddingCache, file_sha256
from .experiment import (
    AggregationMethod,
    EmbeddingExperimentRecord,
    ExperimentMetrics,
    ExperimentRecord,
)
from .face_engine import FaceInputError, FaceObservation, validate_single_face
from .image_input import ImageInput, QualityProfile, assess_quality
from .lfw_dataset import LfwProtocol
from .recognition import MatchDecision, decide_match


class EvaluationFaceEngine(Protocol):
    """评测流程所需的最小人脸特征提取接口。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """从一张 BGR 图片提取通过质量门控的人脸。"""
        ...


@dataclass(frozen=True)
class DatasetRejection:
    """一张在打分前被排除的 LFW 图片及可复现的原因。"""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class LfwEvaluationRun:
    """一次固定 LFW 协议产生的可用得分和明确排除项。"""

    gallery_person_ids: tuple[str, ...]
    records: tuple[ExperimentRecord, ...]
    embedding_records: tuple[EmbeddingExperimentRecord, ...]
    enrollment_rejections: tuple[DatasetRejection, ...]
    probe_rejections: tuple[DatasetRejection, ...]


@dataclass(frozen=True)
class StreamingLfwEvaluationRun:
    """全量 LFW 评测的汇总结果，不保留所有探针分数。"""

    gallery_person_ids: tuple[str, ...]
    gallery_image_total: int
    gallery_valid_image_total: int
    methods: Mapping[str, ExperimentMetrics]
    enrollment_rejections: tuple[DatasetRejection, ...]
    probe_total: int
    probe_valid_image_total: int
    probe_rejections: tuple[DatasetRejection, ...]


@dataclass
class _MetricAccumulator:
    """以常数内存累计开放集评测指标。"""

    known_total: int = 0
    known_correct: int = 0
    unknown_total: int = 0
    unknown_rejected: int = 0
    misidentifications: int = 0
    latency_total_ms: float = 0.0
    valid_total: int = 0

    def add(
        self,
        *,
        expected_person_id: str | None,
        decision: MatchDecision,
        latency_ms: float,
    ) -> None:
        """加入一条已通过检测和质量门控的探针结果。"""
        self.valid_total += 1
        self.latency_total_ms += latency_ms
        if expected_person_id is None:
            self.unknown_total += 1
            if decision.status != "matched":
                self.unknown_rejected += 1
            else:
                self.misidentifications += 1
            return
        self.known_total += 1
        if decision.status == "matched" and decision.person_id == expected_person_id:
            self.known_correct += 1
        elif decision.status == "matched":
            self.misidentifications += 1

    def finish(self) -> ExperimentMetrics:
        """把累计状态转换为与小规模评测一致的统计对象。"""
        return ExperimentMetrics(
            total=self.valid_total,
            known_total=self.known_total,
            known_correct=self.known_correct,
            unknown_total=self.unknown_total,
            unknown_rejected=self.unknown_rejected,
            misidentifications=self.misidentifications,
            average_latency_ms=(
                self.latency_total_ms / self.valid_total if self.valid_total else 0.0
            ),
        )


@dataclass(frozen=True)
class _GalleryIndex:
    """将有效 Gallery 向量整理为全量评测可复用的矩阵索引。"""

    person_ids: tuple[str, ...]
    sample_matrix: np.ndarray
    sample_positions: Mapping[str, tuple[int, ...]]
    sample_quality: Mapping[str, tuple[float, ...]]
    prototypes: np.ndarray

    def person_scores(
        self,
        query_embedding: np.ndarray,
        *,
        method: AggregationMethod,
        top_k: int,
    ) -> dict[str, float]:
        """使用矩阵点积和指定聚合方法生成身份得分。"""
        query = _normalize_embedding(query_embedding)
        sample_scores = self.sample_matrix @ query
        if method == "mean_prototype":
            prototype_scores = self.prototypes @ query
            return {
                person_id: float(prototype_scores[index])
                for index, person_id in enumerate(self.person_ids)
            }

        scores_by_person: dict[str, float] = {}
        for person_id in self.person_ids:
            positions = self.sample_positions[person_id]
            scores = [float(sample_scores[position]) for position in positions]
            if method == "single":
                scores_by_person[person_id] = scores[0]
            elif method == "max":
                scores_by_person[person_id] = max(scores)
            elif method == "top_k_mean":
                strongest = sorted(scores, reverse=True)[:top_k]
                scores_by_person[person_id] = sum(strongest) / len(strongest)
            else:
                raise ValueError(f"unsupported aggregation method: {method}")
        return scores_by_person


def evaluate_lfw_protocol(
    *,
    dataset_dir,
    protocol: LfwProtocol,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
    cache: EvaluationEmbeddingCache | None = None,
    cache_commit_every: int = 100,
) -> LfwEvaluationRun:
    """提取 LFW 特征，并复用应用的开放集打分代码生成评测记录。

    参数：
        dataset_dir：LFW 图片根目录。
        protocol：固定的已知/未知身份划分。
        settings：评测时使用的质量和识别参数。
        face_engine：提供人脸检测和特征提取的模型适配器。
        cache：可选的可恢复 embedding 缓存。
        cache_commit_every：缓存每处理多少张图片提交一次。
    返回：
        可用于基础版与优化版同条件比较的评测运行结果。
    前置条件：
        协议中的相对路径必须存在且能被图片读取器解码。
    """

    _validate_cache_commit_interval(cache_commit_every)
    gallery_embeddings: dict[str, list[np.ndarray]] = {}
    gallery_quality: dict[str, list[float]] = {}
    enrollment_rejections: list[DatasetRejection] = []
    for person_id, paths in protocol.enrollment.items():
        person_embeddings: list[np.ndarray] = []
        person_quality: list[float] = []
        for relative_path in paths:
            try:
                embedding, profile = extract_or_load_embedding(
                    dataset_dir / relative_path,
                    relative_path,
                    settings,
                    face_engine,
                    cache,
                )
            except (FaceInputError, ValueError) as error:
                enrollment_rejections.append(DatasetRejection(relative_path, str(error)))
                continue
            person_embeddings.append(embedding)
            person_quality.append(profile.score)
        if person_embeddings:
            gallery_embeddings[person_id] = person_embeddings
            gallery_quality[person_id] = person_quality

    records: list[ExperimentRecord] = []
    embedding_records: list[EmbeddingExperimentRecord] = []
    probe_rejections: list[DatasetRejection] = []
    for probe in protocol.probes:
        started_at = perf_counter()
        try:
            query, profile = extract_or_load_embedding(
                dataset_dir / probe.relative_path,
                probe.relative_path,
                settings,
                face_engine,
                cache,
            )
        except (FaceInputError, ValueError) as error:
            probe_rejections.append(DatasetRejection(probe.relative_path, str(error)))
            continue
        latency_ms = (perf_counter() - started_at) * 1000
        sample_scores = {
            person_id: [float(query @ embedding) for embedding in embeddings]
            for person_id, embeddings in gallery_embeddings.items()
        }
        records.append(
            ExperimentRecord(
                expected_person_id=probe.expected_person_id,
                sample_scores=sample_scores,
                latency_ms=latency_ms,
                sample_quality_scores=gallery_quality,
                probe_quality_tier=profile.tier,
            )
        )
        embedding_records.append(
            EmbeddingExperimentRecord(
                expected_person_id=probe.expected_person_id,
                query_embedding=query,
                gallery_embeddings=gallery_embeddings,
                latency_ms=latency_ms,
                sample_quality_by_person=gallery_quality,
                probe_quality_tier=profile.tier,
            )
        )
    return LfwEvaluationRun(
        gallery_person_ids=tuple(gallery_embeddings),
        records=tuple(records),
        embedding_records=tuple(embedding_records),
        enrollment_rejections=tuple(enrollment_rejections),
        probe_rejections=tuple(probe_rejections),
    )


def evaluate_lfw_protocol_streaming(
    *,
    dataset_dir,
    protocol: LfwProtocol,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
    methods: Sequence[AggregationMethod],
    on_progress: Callable[[str, int, int], None] | None = None,
    cache: EvaluationEmbeddingCache | None = None,
    cache_commit_every: int = 100,
) -> StreamingLfwEvaluationRun:
    """逐张处理 LFW 全量协议，并以常数级评测状态汇总结果。

    参数：
        dataset_dir：LFW 图片根目录。
        protocol：应覆盖全部图片的 Gallery/Probe 协议。
        settings：评测时使用的质量和识别参数。
        face_engine：提供人脸检测和特征提取的模型适配器。
        methods：在同一 Gallery 和 Probe 上比较的聚合方法。
        on_progress：可选进度回调，参数为阶段名、已处理数量和总数量。
        cache：可选的可恢复 embedding 缓存。
        cache_commit_every：缓存每处理多少张图片提交一次。
    返回：
        包含 Gallery/Probe 覆盖数、拒绝项和各方法指标的汇总结果。
    前置条件：
        `protocol` 中每个相对路径最多出现一次，且至少有一个方法。
        该接口不保留所有探针分数，适合全量数据集运行。
    """

    supported_methods = {"single", "max", "mean_prototype", "top_k_mean"}
    if not methods:
        raise ValueError("methods must not be empty")
    if any(method not in supported_methods for method in methods):
        raise ValueError("unsupported aggregation method")
    _validate_cache_commit_interval(cache_commit_every)

    gallery_embeddings: dict[str, list[np.ndarray]] = {}
    gallery_quality: dict[str, list[float]] = {}
    enrollment_rejections: list[DatasetRejection] = []
    gallery_image_total = sum(len(paths) for paths in protocol.enrollment.values())
    gallery_processed = 0
    images_since_commit = 0
    for person_id, paths in protocol.enrollment.items():
        for relative_path in paths:
            try:
                embedding, profile = extract_or_load_embedding(
                    dataset_dir / relative_path,
                    relative_path,
                    settings,
                    face_engine,
                    cache,
                )
            except (FaceInputError, ValueError) as error:
                enrollment_rejections.append(DatasetRejection(relative_path, str(error)))
                images_since_commit += 1
                if cache is not None and images_since_commit >= cache_commit_every:
                    cache.commit()
                    images_since_commit = 0
                continue
            gallery_embeddings.setdefault(person_id, []).append(embedding)
            gallery_quality.setdefault(person_id, []).append(profile.score)
            gallery_processed += 1
            images_since_commit += 1
            if cache is not None and images_since_commit >= cache_commit_every:
                cache.commit()
                images_since_commit = 0
            if on_progress is not None:
                on_progress("gallery", gallery_processed, gallery_image_total)

    if not gallery_embeddings:
        raise ValueError("full LFW protocol produced no valid Gallery embeddings")
    gallery_index = _build_gallery_index(gallery_embeddings, gallery_quality)
    accumulators = {method: _MetricAccumulator() for method in methods}
    probe_rejections: list[DatasetRejection] = []
    probe_valid_image_total = 0
    for probe_index, probe in enumerate(protocol.probes, start=1):
        started_at = perf_counter()
        try:
            query, profile = extract_or_load_embedding(
                dataset_dir / probe.relative_path,
                probe.relative_path,
                settings,
                face_engine,
                cache,
            )
        except (FaceInputError, ValueError) as error:
            probe_rejections.append(DatasetRejection(probe.relative_path, str(error)))
            images_since_commit += 1
            if cache is not None and images_since_commit >= cache_commit_every:
                cache.commit()
                images_since_commit = 0
            if on_progress is not None:
                on_progress("probe", probe_index, len(protocol.probes))
            continue
        latency_ms = (perf_counter() - started_at) * 1000
        policy = settings.quality_tiers[profile.tier]
        for method, accumulator in accumulators.items():
            person_scores = gallery_index.person_scores(
                query,
                method=method,
                top_k=settings.top_k,
            )
            decision = decide_match(
                person_scores,
                match_threshold=policy.match_threshold,
                min_score_gap=policy.min_score_gap,
            )
            accumulator.add(
                expected_person_id=probe.expected_person_id,
                decision=decision,
                latency_ms=latency_ms,
            )
        probe_valid_image_total += 1
        images_since_commit += 1
        if cache is not None and images_since_commit >= cache_commit_every:
            cache.commit()
            images_since_commit = 0
        if on_progress is not None:
            on_progress("probe", probe_index, len(protocol.probes))

    if cache is not None:
        cache.commit()

    return StreamingLfwEvaluationRun(
        gallery_person_ids=gallery_index.person_ids,
        gallery_image_total=gallery_image_total,
        gallery_valid_image_total=sum(len(values) for values in gallery_embeddings.values()),
        methods={method: accumulator.finish() for method, accumulator in accumulators.items()},
        enrollment_rejections=tuple(enrollment_rejections),
        probe_total=len(protocol.probes),
        probe_valid_image_total=probe_valid_image_total,
        probe_rejections=tuple(probe_rejections),
    )


def _build_gallery_index(
    gallery_embeddings: Mapping[str, Sequence[np.ndarray]],
    gallery_quality: Mapping[str, Sequence[float]],
) -> _GalleryIndex:
    """把有效 Gallery 样本转换为矩阵、人员分组和平均原型。"""
    person_ids = tuple(gallery_embeddings)
    rows: list[np.ndarray] = []
    sample_positions: dict[str, tuple[int, ...]] = {}
    sample_quality: dict[str, tuple[float, ...]] = {}
    prototypes: list[np.ndarray] = []
    for person_id in person_ids:
        normalized = [_normalize_embedding(embedding) for embedding in gallery_embeddings[person_id]]
        start = len(rows)
        rows.extend(normalized)
        sample_positions[person_id] = tuple(range(start, len(rows)))
        quality = tuple(float(value) for value in gallery_quality.get(person_id, ()))
        if len(quality) != len(normalized):
            raise ValueError("Gallery quality and embedding counts must match")
        sample_quality[person_id] = quality
        prototypes.append(_normalize_embedding(np.mean(normalized, axis=0)))
    return _GalleryIndex(
        person_ids=person_ids,
        sample_matrix=np.stack(rows).astype(np.float32),
        sample_positions=sample_positions,
        sample_quality=sample_quality,
        prototypes=np.stack(prototypes).astype(np.float32),
    )


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """把全量评测向量转换为单位向量，并拒绝零向量。"""
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm


def extract_valid_embedding(
    image_path,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
) -> tuple[np.ndarray, QualityProfile]:
    """读取一张 LFW 图片并返回有效特征，失败时抛出可记录的原因。"""
    image_input = ImageInput.from_file(image_path, source_type="dataset")
    observed_face = face_engine.extract_single_face(image_input.frame)
    face = validate_single_face([observed_face], settings)
    profile = assess_quality(image_input.frame, face, settings)
    if profile.tier == "reject":
        raise FaceInputError("quality_rejected:" + ",".join(profile.reasons or ("low_score",)))
    return face.embedding, profile


def extract_or_load_embedding(
    image_path,
    relative_path: str,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
    cache: EvaluationEmbeddingCache | None,
) -> tuple[np.ndarray, QualityProfile]:
    """优先读取缓存，否则提取图片特征并保存成功或拒绝结果。"""

    if cache is None:
        return extract_valid_embedding(image_path, settings, face_engine)
    digest = file_sha256(image_path)
    cached = cache.get(relative_path, digest)
    if cached is not None:
        if cached.status == "rejected":
            raise FaceInputError(cached.reason or "cached_rejection")
        if cached.embedding is None or cached.quality is None:
            raise RuntimeError(f"invalid cached valid entry: {relative_path}")
        return cached.embedding, cached.quality
    try:
        embedding, profile = extract_valid_embedding(image_path, settings, face_engine)
    except (FaceInputError, ValueError) as error:
        cache.put_rejected(relative_path, digest, str(error))
        raise
    cache.put_valid(relative_path, digest, embedding, profile)
    return embedding, profile


def _validate_cache_commit_interval(value: int) -> None:
    """确认缓存提交间隔为正整数。"""

    if value < 1:
        raise ValueError("cache_commit_every must be at least one")
