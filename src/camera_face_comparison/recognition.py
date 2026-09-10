from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import FaceInputError, FaceObservation, normalize_embedding
from .image_input import ImageInput, measure_quality, quality_warnings
from .integrity import verify_library
from .open_set_policy import OpenSetDecision, RecognitionPolicy, apply_open_set_policy
from .repository import FaceRepository


class ProbeFaceEngine(Protocol):
    """识别流程所需的最小单人脸提取接口。"""

    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        """从一张 BGR 图片返回唯一人脸和有效 embedding。"""
        ...


class RecognitionService:
    """连接图片输入、标准库、Mean Prototype 和开放集判定。"""

    def __init__(
        self,
        repository: FaceRepository,
        settings: Settings,
        face_engine: ProbeFaceEngine,
    ) -> None:
        """保存仓库、冻结策略和人脸引擎依赖。"""

        self._repository = repository
        self._settings = settings
        self._face_engine = face_engine

    def compare(self, frame: np.ndarray) -> RecognitionResult:
        """复制摄像头当前帧并执行与本地图片相同的识别流程。"""

        return self.compare_input(ImageInput.from_camera(frame))

    def compare_input(self, image_input: ImageInput) -> RecognitionResult:
        """对一张图片执行开放集 1:N 身份识别。

        数值质量指标只产生提示，不会阻断有效单脸。损坏图片在构造 `ImageInput`
        时失败；无人脸、多人脸和无效 embedding 由人脸引擎阻断。
        """

        started_at = perf_counter()
        policy_rule = self._settings.recognition_policy.rule
        try:
            verification = verify_library(self._repository, self._settings)
            if not verification.is_valid:
                first_failure = verification.failures[0]
                return self._record_and_return(
                    RecognitionResult(
                        status="invalid",
                        person_id=None,
                        display_name=None,
                        top_score=None,
                        second_score=None,
                        score_gap=None,
                        acceptance_score=None,
                        acceptance_rule=policy_rule,
                        latency_ms=(perf_counter() - started_at) * 1000,
                        reason=f"library_integrity_failed:{first_failure.kind}",
                        bbox=None,
                        quality_metrics={},
                        quality_warnings=(),
                    )
                )

            probe = self._face_engine.extract_single_face(image_input.frame)
            metrics = measure_quality(image_input.frame, probe)
            warnings = quality_warnings(metrics, self._settings.quality_warnings)
            samples = self._repository.list_samples()
            embeddings_by_person: dict[str, list[np.ndarray]] = {}
            for sample in samples:
                embeddings_by_person.setdefault(sample.person_id, []).append(sample.embedding)
            decision = recognize_embedding(
                query_embedding=probe.embedding,
                embeddings_by_person=embeddings_by_person,
                policy=self._settings.recognition_policy,
            )
            names = {
                person.id: person.display_name for person in self._repository.list_people()
            }
            result = RecognitionResult(
                status="matched" if decision.accepted else "unknown",
                person_id=decision.accepted_person_id,
                display_name=names.get(decision.accepted_person_id),
                top_score=decision.top_score,
                second_score=decision.second_score,
                score_gap=decision.score_gap,
                acceptance_score=decision.acceptance_score,
                acceptance_rule=decision.rule,
                latency_ms=(perf_counter() - started_at) * 1000,
                reason=decision.reason,
                bbox=probe.bbox,
                quality_metrics=metrics,
                quality_warnings=warnings,
            )
        except (FaceInputError, TypeError, ValueError) as error:
            result = RecognitionResult(
                status="invalid",
                person_id=None,
                display_name=None,
                top_score=None,
                second_score=None,
                score_gap=None,
                acceptance_score=None,
                acceptance_rule=policy_rule,
                latency_ms=(perf_counter() - started_at) * 1000,
                reason=str(error),
                bbox=None,
                quality_metrics={},
                quality_warnings=(),
            )
        return self._record_and_return(result)

    def _record_and_return(self, result: RecognitionResult) -> RecognitionResult:
        """记录识别结果后原样返回，保证 UI 和日志使用同一事实。"""

        self._repository.record_recognition(
            decision=result.status,
            person_id=result.person_id,
            top_score=result.top_score,
            second_score=result.second_score,
            score_gap=result.score_gap,
            acceptance_score=result.acceptance_score,
            acceptance_rule=result.acceptance_rule,
            latency_ms=result.latency_ms,
            reason=result.reason,
        )
        return result


def mean_prototype_scores(
    query_embedding: np.ndarray,
    embeddings_by_person: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, float]:
    """用每个身份的归一化平均原型计算人员级余弦相似度。

    每张参考 embedding 先归一化，随后按身份求平均并再次归一化。空样本身份
    不进入结果；Query 与每个身份原型只计算一次点积。
    """

    query = normalize_embedding(query_embedding)
    scores: dict[str, float] = {}
    for person_id, embeddings in embeddings_by_person.items():
        normalized = [normalize_embedding(embedding) for embedding in embeddings]
        if not normalized:
            continue
        prototype = normalize_embedding(np.mean(normalized, axis=0))
        scores[person_id] = float(query @ prototype)
    return scores


def recognize_embedding(
    *,
    query_embedding: np.ndarray,
    embeddings_by_person: Mapping[str, Sequence[np.ndarray]],
    policy: RecognitionPolicy,
) -> OpenSetDecision:
    """计算 Mean Prototype 人员分数并应用一套明确开放集规则。"""

    person_scores = mean_prototype_scores(query_embedding, embeddings_by_person)
    return apply_open_set_policy(person_scores, policy)
