from __future__ import annotations

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import DetectedFace, FaceObservation
from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.recognition import RecognitionService
from camera_face_comparison.repository import FaceRepository
from camera_face_comparison.ui.workers import RecognitionInputStream


def test_recognition_input_stream_yields_frames_before_producer_finishes() -> None:
    """识别线程必须能在采集端关闭输入前取得已经到达的第一帧。"""

    stream = RecognitionInputStream()
    first = ImageInput.from_camera(np.zeros((20, 20, 3), dtype=np.uint8))
    iterator = iter(stream)

    stream.submit(first)

    assert next(iterator) is first
    stream.finish()
    assert list(iterator) == []


def test_cancelled_input_stream_does_not_write_a_recognition_log(tmp_path) -> None:
    """取消尚未收齐的摄像头任务时，不得把取消动作记录成识别结果。"""

    class FakeEngine:
        """返回一张有效检测，保证服务会继续等待后续帧。"""

        def detect_single_face(self, frame: np.ndarray) -> DetectedFace:
            """返回与输入画面对应的固定单脸检测。"""

            return DetectedFace(
                bbox=(0.0, 0.0, 20.0, 20.0),
                detection_score=0.9,
                blur_variance=10.0,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )

        def extract_detected_face(
            self,
            frame: np.ndarray,
            detected_face: DetectedFace,
        ) -> FaceObservation:
            """该测试在取消后不应到达身份特征提取阶段。"""

            raise AssertionError("cancelled recognition must not extract an embedding")

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    stream = RecognitionInputStream()
    stream.submit(ImageInput.from_camera(np.zeros((20, 20, 3), dtype=np.uint8)))
    stream.cancel()
    service = RecognitionService(
        repository,
        settings,
        FakeEngine(),
        InMemoryFaceLibrary.empty().snapshot(),
    )

    with pytest.raises(RuntimeError, match="recognition was cancelled"):
        service.compare_inputs(stream, frame_count=5)

    log_count = repository._connection.execute(
        "SELECT COUNT(*) FROM recognition_logs"
    ).fetchone()[0]
    assert log_count == 0
    repository.close()
