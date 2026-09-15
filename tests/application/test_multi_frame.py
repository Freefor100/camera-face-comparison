from __future__ import annotations

import numpy as np
import pytest

from camera_face_comparison.multi_frame import (
    MultiFrameObservation,
    select_consistent_frame,
    select_mean_embedding,
    select_sharpest_frame,
)


def _observation(index: int, embedding: list[float], blur: float) -> MultiFrameObservation:
    """构造多帧选择测试用的有效观察。"""

    return MultiFrameObservation(
        frame=np.full((4, 4, 3), index, dtype=np.uint8),
        embedding=np.asarray(embedding, dtype=np.float32),
        bbox=(0.0, 0.0, 4.0, 4.0),
        quality_metrics={"blur_variance": blur, "face_size_px": 4.0},
        quality_warnings=(),
        frame_index=index,
    )


def test_sharpest_frame_uses_blur_variance_and_keeps_frame_metadata() -> None:
    """清晰度策略应选择人脸区域拉普拉斯方差最高的帧。"""

    frames = [
        _observation(0, [1.0, 0.0], 10.0),
        _observation(1, [0.0, 1.0], 30.0),
        _observation(2, [1.0, 0.0], 20.0),
    ]

    selected = select_sharpest_frame(frames)

    assert selected.frame_index == 1
    assert np.array_equal(selected.frame, frames[1].frame)


def test_consistent_frame_uses_highest_mean_similarity_and_sharpness_as_tie_break() -> None:
    """一致性策略应选择与其他有效帧平均余弦相似度最高的帧。"""

    frames = [
        _observation(0, [1.0, 0.0], 10.0),
        _observation(1, [0.99, 0.1], 20.0),
        _observation(2, [0.995, 0.05], 30.0),
        _observation(3, [0.0, 1.0], 100.0),
    ]

    selected = select_consistent_frame(frames)

    assert selected.frame_index == 1


def test_mean_embedding_renormalizes_fused_vector_and_displays_sharpest_frame() -> None:
    """特征融合应平均单位向量后重新归一化，并用最清晰帧作为代表画面。"""

    frames = [
        _observation(0, [1.0, 0.0], 10.0),
        _observation(1, [0.0, 1.0], 50.0),
    ]

    selected = select_mean_embedding(frames)

    assert selected.frame_index == 1
    assert np.allclose(selected.embedding, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])


def test_all_selection_methods_reject_an_empty_valid_frame_list() -> None:
    """全部帧无效时必须返回统一的无有效帧原因。"""

    for selector in (select_sharpest_frame, select_consistent_frame, select_mean_embedding):
        with pytest.raises(ValueError, match="no_valid_frames"):
            selector([])
