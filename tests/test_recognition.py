from __future__ import annotations

import hashlib

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy
from camera_face_comparison.recognition import RecognitionService, recognize_embedding
from camera_face_comparison.repository import FaceRepository, SampleInput


def test_mean_prototype_uses_all_reference_embeddings_before_scoring() -> None:
    """人员分数必须来自归一化平均原型，而不是最高样本或 Top-K 分数。"""

    decision = recognize_embedding(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_person={
            "alice": (
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.10, 0.995], dtype=np.float32),
            ),
            "bob": (
                np.array([0.91, 0.414], dtype=np.float32),
                np.array([0.90, 0.436], dtype=np.float32),
            ),
        },
        policy=ScoreThresholdPolicy(minimum_score=0.80),
    )

    assert decision.accepted_person_id == "bob"
    assert decision.rule == "score_threshold"
    assert decision.acceptance_score == decision.top_score


def test_frozen_score_rule_does_not_reject_a_close_second_candidate() -> None:
    """部署规则只检查最高分，候选分差只用于解释结果。"""

    decision = recognize_embedding(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        embeddings_by_person={
            "alice": (np.array([0.90, 0.43589], dtype=np.float32),),
            "bob": (np.array([0.89, 0.45596], dtype=np.float32),),
        },
        policy=ScoreThresholdPolicy(minimum_score=0.80),
    )

    assert decision.accepted_person_id == "alice"
    assert decision.score_gap is not None and decision.score_gap < 0.02


def test_recognition_service_returns_calibrated_fields_and_quality_warnings(tmp_path) -> None:
    """低质量数值不阻断单脸，结果需同时返回策略字段和调整提示。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    alice = _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    _create_person(
        repository,
        settings,
        "Bob",
        [np.array([0.2, 0.98], dtype=np.float32)],
    )

    class ProbeEngine:
        """返回低质量指标但 embedding 有效的 Alice 特征。"""

        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            """构造可继续识别的单脸观察。"""

            return FaceObservation(
                bbox=(0.0, 0.0, 32.0, 32.0),
                detection_score=0.30,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                blur_variance=1.0,
                landmarks=None,
            )

    result = RecognitionService(repository, settings, ProbeEngine()).compare(
        np.full((80, 80, 3), 5, dtype=np.uint8)
    )
    repository.close()

    assert result.status == "matched"
    assert result.person_id == alice.id
    assert result.display_name == "Alice"
    assert result.acceptance_rule == "score_threshold"
    assert result.acceptance_score == result.top_score
    assert result.score_gap is not None
    assert "move_closer" in result.quality_warnings
    assert "hold_still" in result.quality_warnings
    assert "increase_lighting" in result.quality_warnings


def _create_person(
    repository: FaceRepository,
    settings,
    name: str,
    embeddings: list[np.ndarray],
):
    """为识别服务测试创建一个带指定特征和真实文件的人员。"""

    person_id = name.lower()
    samples = []
    for index, embedding in enumerate(embeddings, start=1):
        relative_path = f"faces/{person_id}/sample_{index:03d}.jpg"
        image_path = settings.data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"{name}-{index}".encode())
        samples.append(
            SampleInput(
                image_path=relative_path,
                embedding=embedding,
                quality_metrics={
                    "detection_score": 0.95,
                    "face_size_px": 160.0,
                    "blur_variance": 150.0,
                    "brightness": 120.0,
                    "contrast": 20.0,
                },
                source_type="file",
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
        )
    return repository.create_person_with_samples(
        person_id=person_id,
        display_name=name,
        samples=samples,
    )
