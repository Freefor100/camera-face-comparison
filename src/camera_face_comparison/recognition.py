from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import RecognitionResult
from .face_engine import DetectedFace, FaceInputError, FaceObservation
from .face_library import FaceLibrarySnapshot
from .image_input import ImageInput, measure_quality, quality_warnings
from .multi_frame import MultiFrameObservation
from .open_set_policy import apply_ranked_open_set_policy
from .repository import FaceRepository


class ProbeFaceEngine(Protocol):
    """识别流程所需的分阶段人脸推理接口。"""

    def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
        """只检测一张 BGR 图片中的唯一人脸。"""
        ...

    def extract_detected_face(
        self,
        frame: np.ndarray,
        detected_face: DetectedFace,
    ) -> FaceObservation:
        """对已检测人脸执行对齐和身份特征提取。"""
        ...


TimingSink = Callable[[str, float], None]


@dataclass(frozen=True)
class _DetectedInput:
    """一张待识别图片及其尚未提取身份特征的检测结果。"""

    image_input: ImageInput
    detected_face: DetectedFace
    frame_index: int


class RecognitionService:
    """连接图片输入、标准库、Mean Prototype 和开放集判定。"""

    def __init__(
        self,
        repository: FaceRepository,
        settings: Settings,
        face_engine: ProbeFaceEngine,
        face_library: FaceLibrarySnapshot,
        timing_sink: TimingSink | None = None,
    ) -> None:
        """保存识别依赖，并可选记录本次流程的各阶段耗时。

        参数：
            repository：用于写入识别日志的仓库。
            settings：当前运行配置和开放集接收策略。
            face_engine：提供单人脸特征提取能力的模型适配器。
            face_library：本次任务使用的不可变标准库快照。
            timing_sink：可选的阶段耗时回调，供性能实验使用；业务运行可省略。
        """

        self._repository = repository
        self._settings = settings
        self._face_engine = face_engine
        self._face_library = face_library
        self._timing_sink = timing_sink

    def compare(self, frame: np.ndarray) -> RecognitionResult:
        """复制摄像头当前帧并执行与本地图片相同的识别流程。"""

        return self.compare_input(ImageInput.from_camera(frame))

    def compare_input(
        self,
        image_input: ImageInput,
        *,
        started_at: float | None = None,
    ) -> RecognitionResult:
        """对一张图片执行开放集 1:N 身份识别。

        数值质量指标只产生提示，不会阻断有效单脸。损坏图片在构造 `ImageInput`
        时失败；无人脸、多人脸和无效 embedding 由人脸引擎阻断。
        """

        request_started_at = perf_counter() if started_at is None else started_at
        try:
            detected = self._detect_input(image_input, frame_index=0)
            observation = self._extract_detected_observation(detected)
            result = self._result_from_observation(
                observation,
                started_at=request_started_at,
                frame_count=1,
                valid_frame_count=1,
                selected_method="single_frame",
            )
        except (FaceInputError, TypeError, ValueError) as error:
            result = self._invalid_result(
                started_at=request_started_at,
                reason=str(error),
                frame_count=1,
                valid_frame_count=0,
            )
        return self._record_and_return(result)

    def compare_inputs(
        self,
        image_inputs: Iterable[ImageInput],
        *,
        frame_count: int | None = None,
        started_at: float | None = None,
    ) -> RecognitionResult:
        """对短时间采集的多张图片提取特征并用清晰度最高帧完成判定。

        参数：
            image_inputs：按采集顺序排列的图片输入，允许其中部分图片检测失败。
        返回：
            使用有效帧完成的识别结果；全部失败时返回 `no_valid_frames`。
        前置条件：
            当前运行策略已验证采用清晰度最高帧，且调用方已限制采集窗口大小。
        """

        request_started_at = perf_counter() if started_at is None else started_at
        detected_inputs: list[_DetectedInput] = []
        received_count = 0
        for frame_index, image_input in enumerate(image_inputs):
            received_count += 1
            try:
                detected_inputs.append(self._detect_input(image_input, frame_index=frame_index))
            except (FaceInputError, TypeError, ValueError):
                continue

        total_frame_count = received_count if frame_count is None else frame_count
        observation = self._extract_sharpest_available(detected_inputs)
        if observation is None:
            result = self._invalid_result(
                started_at=request_started_at,
                reason="no_valid_frames",
                frame_count=total_frame_count,
                valid_frame_count=0,
            )
        else:
            result = self._result_from_observation(
                observation,
                started_at=request_started_at,
                frame_count=total_frame_count,
                valid_frame_count=len(detected_inputs),
                selected_method="sharpest_frame",
            )
        return self._record_and_return(result)

    def _detect_input(
        self,
        image_input: ImageInput,
        *,
        frame_index: int,
    ) -> _DetectedInput:
        """只检测一张输入，供到达即处理的多帧流水线使用。"""

        detection_started = perf_counter()
        try:
            detected_face = self._face_engine.detect_single_face(image_input.frame)
        finally:
            self._record_timing("face_detection_ms", detection_started)
        return _DetectedInput(
            image_input=image_input,
            detected_face=detected_face,
            frame_index=frame_index,
        )

    def _extract_detected_observation(
        self,
        detected_input: _DetectedInput,
    ) -> MultiFrameObservation:
        """只为已经选中的检测结果提取身份特征和质量指标。"""

        extraction_started = perf_counter()
        try:
            probe = self._face_engine.extract_detected_face(
                detected_input.image_input.frame,
                detected_input.detected_face,
            )
        finally:
            self._record_timing("embedding_extraction_ms", extraction_started)
        quality_started = perf_counter()
        metrics = measure_quality(detected_input.image_input.frame, probe)
        warnings = quality_warnings(metrics, self._settings.quality_warnings)
        self._record_timing("quality_measurement_ms", quality_started)
        return MultiFrameObservation(
            frame=detected_input.image_input.frame,
            embedding=probe.embedding,
            bbox=probe.bbox,
            quality_metrics=metrics,
            quality_warnings=warnings,
            frame_index=detected_input.frame_index,
        )

    def _extract_sharpest_available(
        self,
        detected_inputs: Sequence[_DetectedInput],
    ) -> MultiFrameObservation | None:
        """按清晰度尝试身份特征提取，失败时回退到下一帧。"""

        selection_started = perf_counter()
        ranked = sorted(
            detected_inputs,
            key=lambda item: (
                item.detected_face.blur_variance,
                -item.frame_index,
            ),
            reverse=True,
        )
        self._record_timing("frame_selection_ms", selection_started)
        for detected_input in ranked:
            try:
                return self._extract_detected_observation(detected_input)
            except (FaceInputError, TypeError, ValueError):
                continue
        return None

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

        search_started = perf_counter()
        ranked = self._face_library.rank_candidates(observation.embedding)
        self._record_timing("candidate_search_ms", search_started)
        decision_started = perf_counter()
        decision = apply_ranked_open_set_policy(
            ranked,
            self._settings.recognition_policy,
        )
        self._record_timing("decision_ms", decision_started)
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

        log_started = perf_counter()
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
        self._record_timing("log_write_ms", log_started)
        return result

    def _record_timing(self, stage: str, started_at: float) -> None:
        """把一个已完成阶段的耗时交给可选实验记录器。"""

        if self._timing_sink is not None:
            self._timing_sink(stage, (perf_counter() - started_at) * 1000)
