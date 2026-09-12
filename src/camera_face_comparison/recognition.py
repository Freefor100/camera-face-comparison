from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import FaceInputError, FaceObservation
from .face_library import FaceLibrarySnapshot
from .image_input import ImageInput, measure_quality, quality_warnings
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
        face_library: FaceLibrarySnapshot,
    ) -> None:
        """保存日志仓库、冻结策略、引擎和本次查询使用的快照。"""

        self._repository = repository
        self._settings = settings
        self._face_engine = face_engine
        self._face_library = face_library

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
            probe = self._face_engine.extract_single_face(image_input.frame)
            metrics = measure_quality(image_input.frame, probe)
            warnings = quality_warnings(metrics, self._settings.quality_warnings)
            decision = self._face_library.search(
                probe.embedding,
                self._settings.recognition_policy,
            )
            names = dict(zip(self._face_library.person_ids, self._face_library.display_names))
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
