from __future__ import annotations

import numpy as np

from camera_face_comparison.image_input import ImageInput
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
