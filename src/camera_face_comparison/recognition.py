from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import FaceInputError, FaceObservation
from .face_library import FaceLibrarySnapshot
from .image_input import ImageInput, measure_quality, quality_warnings
from .multi_frame import MultiFrameObservation, select_sharpest_frame
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
        try:
            observation = self._extract_observation(image_input, frame_index=0)
            result = self._result_from_observation(
                observation,
                started_at=started_at,
                frame_count=1,
                valid_frame_count=1,
                selected_method="single_frame",
            )
        except (FaceInputError, TypeError, ValueError) as error:
            result = self._invalid_result(
                started_at=started_at,
                reason=str(error),
                frame_count=1,
                valid_frame_count=0,
            )
        return self._record_and_return(result)

    def compare_inputs(self, image_inputs: Sequence[ImageInput]) -> RecognitionResult:
        """对短时间采集的多张图片提取特征并用清晰度最高帧完成判定。

        参数：
            image_inputs：按采集顺序排列的图片输入，允许其中部分图片检测失败。
        返回：
            使用有效帧完成的识别结果；全部失败时返回 `no_valid_frames`。
        前置条件：
            当前运行策略已验证采用清晰度最高帧，且调用方已限制采集窗口大小。
        """

        started_at = perf_counter()
        valid_observations: list[MultiFrameObservation] = []
        for frame_index, image_input in enumerate(image_inputs):
            try:
                valid_observations.append(
                    self._extract_observation(image_input, frame_index=frame_index)
                )
            except (FaceInputError, TypeError, ValueError):
                continue

        if not valid_observations:
            result = self._invalid_result(
                started_at=started_at,
                reason="no_valid_frames",
                frame_count=len(image_inputs),
                valid_frame_count=0,
            )
        else:
            selected = select_sharpest_frame(valid_observations)
            result = self._result_from_observation(
                selected,
                started_at=started_at,
                frame_count=len(image_inputs),
                valid_frame_count=len(valid_observations),
                selected_method="sharpest_frame",
            )
        return self._record_and_return(result)

    def _extract_observation(
        self,
        image_input: ImageInput,
        *,
        frame_index: int,
    ) -> MultiFrameObservation:
        """从一张输入提取可用于多帧选择的完整观察。"""

        probe = self._face_engine.extract_single_face(image_input.frame)
        metrics = measure_quality(image_input.frame, probe)
        warnings = quality_warnings(metrics, self._settings.quality_warnings)
        return MultiFrameObservation(
            frame=image_input.frame,
            embedding=probe.embedding,
            bbox=probe.bbox,
            quality_metrics=metrics,
            quality_warnings=warnings,
            frame_index=frame_index,
        )

    def _result_from_observation(
        self,
        observation: MultiFrameObservation,
        *,
        started_at: float,
        frame_count: int,
        valid_frame_count: int,
        selected_method: str,
    ) -> RecognitionResult:
        """把选中的人脸观察转换成统一识别结果。"""

        decision = self._face_library.search(
            observation.embedding,
            self._settings.recognition_policy,
        )
        names = dict(zip(self._face_library.person_ids, self._face_library.display_names))
        return RecognitionResult(
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
            bbox=observation.bbox,
            quality_metrics=observation.quality_metrics,
            quality_warnings=observation.quality_warnings,
            frame_count=frame_count,
            valid_frame_count=valid_frame_count,
            selected_frame_index=observation.frame_index,
            selected_method=selected_method,
        )

    def _invalid_result(
        self,
        *,
        started_at: float,
        reason: str,
        frame_count: int,
        valid_frame_count: int,
    ) -> RecognitionResult:
        """构造无效输入结果并保留本次采集帧数。"""

        return RecognitionResult(
            status="invalid",
            person_id=None,
            display_name=None,
            top_score=None,
            second_score=None,
            score_gap=None,
            acceptance_score=None,
            acceptance_rule=self._settings.recognition_policy.rule,
            latency_ms=(perf_counter() - started_at) * 1000,
            reason=reason,
            bbox=None,
            quality_metrics={},
            quality_warnings=(),
            frame_count=frame_count,
            valid_frame_count=valid_frame_count,
            selected_method=None,
        )

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
