from __future__ import annotations

import numpy as np

from camera_face_comparison.image_input import ImageInput


def test_file_input_decodes_bgr_and_exposes_only_safe_name(tmp_path) -> None:
    """本地图片识别输入必须可用，同时不能保留原始绝对路径。"""

    import cv2

    path = tmp_path / "nested" / "probe.jpg"
    path.parent.mkdir()
    frame = np.full((32, 48, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)

    image = ImageInput.from_file(path)

    assert image.source_type == "file"
    assert image.safe_name == "probe.jpg"
    assert image.frame.shape == (32, 48, 3)
    assert image.frame.dtype == np.uint8


def test_camera_input_copies_frame_before_background_processing() -> None:
    """后台任务不能在抓拍后继续观察可变的摄像头缓冲区。"""

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    image = ImageInput.from_camera(frame)
    frame[:, :, :] = 255

    assert image.source_type == "camera"
    assert image.safe_name is None
    assert int(image.frame.max()) == 0
