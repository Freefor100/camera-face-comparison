from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .experiment import ExperimentRecord
from .face_engine import FaceInputError, FaceObservation, validate_single_face
from .image_input import ImageInput, QualityProfile, assess_quality
from .lfw_dataset import LfwProtocol


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
    enrollment_rejections: tuple[DatasetRejection, ...]
    probe_rejections: tuple[DatasetRejection, ...]


def evaluate_lfw_protocol(
    *,
    dataset_dir,
    protocol: LfwProtocol,
    settings: Settings,
    face_engine: EvaluationFaceEngine,
) -> LfwEvaluationRun:
    """提取 LFW 特征，并复用应用的开放集打分代码生成评测记录。

    参数：
        dataset_dir：LFW 图片根目录。
        protocol：固定的已知/未知身份划分。
        settings：评测时使用的质量和识别参数。
        face_engine：提供人脸检测和特征提取的模型适配器。
    返回：
        可用于基础版与优化版同条件比较的评测运行结果。
    前置条件：
        协议中的相对路径必须存在且能被图片读取器解码。
    """

    gallery_embeddings: dict[str, list[np.ndarray]] = {}
    gallery_quality: dict[str, list[float]] = {}
    enrollment_rejections: list[DatasetRejection] = []
    for person_id, paths in protocol.enrollment.items():
        person_embeddings: list[np.ndarray] = []
        person_quality: list[float] = []
        for relative_path in paths:
            try:
                embedding, profile = _extract_valid_embedding(
                    dataset_dir / relative_path, settings, face_engine
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
    probe_rejections: list[DatasetRejection] = []
    for probe in protocol.probes:
        started_at = perf_counter()
        try:
            query, profile = _extract_valid_embedding(
                dataset_dir / probe.relative_path, settings, face_engine
            )
        except (FaceInputError, ValueError) as error:
            probe_rejections.append(DatasetRejection(probe.relative_path, str(error)))
            continue
        sample_scores = {
            person_id: [float(query @ embedding) for embedding in embeddings]
            for person_id, embeddings in gallery_embeddings.items()
        }
        records.append(
            ExperimentRecord(
                expected_person_id=probe.expected_person_id,
                sample_scores=sample_scores,
                latency_ms=(perf_counter() - started_at) * 1000,
                sample_quality_scores=gallery_quality,
                probe_quality_tier=profile.tier,
            )
        )
    return LfwEvaluationRun(
        gallery_person_ids=tuple(gallery_embeddings),
        records=tuple(records),
        enrollment_rejections=tuple(enrollment_rejections),
        probe_rejections=tuple(probe_rejections),
    )


def _extract_valid_embedding(
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
