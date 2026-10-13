"""验证单脸约束、特征归一化和非阻断式质量提示。"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import (
    FaceEngine,
    FaceInputError,
    FaceObservation,
    validate_single_face,
)
from camera_face_comparison.image_input import measure_quality, quality_warnings
from camera_face_comparison.runtime import ExecutionBackend
from tests.application.helpers import embedding as production_embedding


def _face(*, score: float = 0.95, size: int = 160, blur: float = 140.0) -> FaceObservation:
    """构造可调检测指标的人脸观察对象。"""

    return FaceObservation(
        bbox=(0.0, 0.0, float(size), float(size)),
        detection_score=score,
        embedding=np.array([0.3, 0.4, 0.5], dtype=np.float32),
        blur_variance=blur,
        landmarks=None,
    )


def test_validate_single_face_only_blocks_zero_or_multiple_faces() -> None:
    """检测分数较低也不能隐藏第二张脸；多人脸始终属于无效输入。"""

    with pytest.raises(FaceInputError, match="multiple_faces"):
        validate_single_face([_face(), _face(score=0.20)])


def test_low_numeric_quality_still_returns_a_normalized_single_face() -> None:
    """低分辨率、低检测分或模糊只提示，不能阻断有效 embedding。"""

    accepted = validate_single_face([_face(score=0.20, size=32, blur=1.0)])

    assert np.allclose(accepted.embedding, [0.424264, 0.565685, 0.707107])


def test_quality_measurements_produce_non_blocking_adjustment_warnings(tmp_path) -> None:
    """原始指标应保留，并把异常转换为可操作提示而非拒绝结果。"""

    settings = load_settings(tmp_path)
    frame = np.full((240, 320, 3), 5, dtype=np.uint8)
    metrics = measure_quality(frame, _face(score=0.2, size=32, blur=1.0))

    warnings = quality_warnings(metrics, settings.quality_warnings)

    assert set(metrics) == {
        "detection_score",
        "face_size_px",
        "blur_variance",
        "brightness",
        "contrast",
    }
    assert "low_detection_confidence" in warnings
    assert "move_closer" in warnings
    assert "hold_still" in warnings
    assert "increase_lighting" in warnings


def test_face_engine_adapts_model_output_and_normalizes_embedding(tmp_path) -> None:
    """模型适配器只返回通用观察对象，并在单脸出口完成归一化。"""

    class Detector:
        """返回一张固定人脸。"""

        def detect(self, frame, max_num=0, metric="default"):
            """返回固定人脸框和关键点。"""

            return (
                np.array([[10.0, 20.0, 190.0, 200.0, 0.96]], dtype=np.float32),
                np.array([[[20.0, 30.0]]], dtype=np.float32),
            )

    class Recognizer:
        """返回固定身份特征。"""

        def get(self, frame, face):
            """模拟识别模型输出。"""

            return production_embedding([3.0, 4.0])

    detector = Detector()
    analyzer = SimpleNamespace(
        det_model=detector,
        models={"detection": detector, "recognition": Recognizer()},
    )
    blur_inputs: list[tuple[int, int, int]] = []

    def blur_metric(face_crop: np.ndarray) -> float:
        """记录清晰度裁剪尺寸并返回测量值。"""

        blur_inputs.append(face_crop.shape)
        return 1.0

    engine = FaceEngine(analyzer=analyzer, blur_metric=blur_metric)
    observation = engine.extract_single_face(np.zeros((240, 320, 3), dtype=np.uint8))

    assert observation.bbox == (10.0, 20.0, 190.0, 200.0)
    assert np.allclose(observation.embedding, production_embedding([0.6, 0.8]))
    assert observation.blur_variance == 1.0
    assert blur_inputs == [(180, 180, 3)]


def test_face_engine_detection_does_not_run_identity_model() -> None:
    """仅检测人脸时不得提前执行身份特征模型。"""

    class Detector:
        """返回一张带五点关键点的人脸。"""

        def detect(self, frame, max_num=0, metric="default"):
            """返回测试人脸框和关键点。"""

            assert max_num == 0
            assert metric == "default"
            return (
                np.array([[10.0, 20.0, 190.0, 200.0, 0.96]], dtype=np.float32),
                np.array(
                    [[[40.0, 70.0], [150.0, 70.0], [95.0, 110.0], [55.0, 155.0], [135.0, 155.0]]],
                    dtype=np.float32,
                ),
            )

    class Recognizer:
        """记录身份特征模型的调用次数。"""

        def __init__(self) -> None:
            """创建调用次数为零的测试识别器。"""

            self.calls = 0

        def get(self, frame, face):
            """返回固定的未归一化身份特征。"""

            self.calls += 1
            return production_embedding([3.0, 4.0])

    recognizer = Recognizer()
    analyzer = SimpleNamespace(
        det_model=Detector(),
        models={"detection": Detector(), "recognition": recognizer},
    )
    engine = FaceEngine(analyzer=analyzer, blur_metric=lambda crop: 25.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detected = engine.detect_single_face(frame)

    assert recognizer.calls == 0
    assert detected.bbox == (10.0, 20.0, 190.0, 200.0)
    observation = engine.extract_detected_face(frame, detected)
    assert recognizer.calls == 1
    assert np.allclose(observation.embedding, production_embedding([0.6, 0.8]))


def test_local_model_loading_disables_updates_and_loads_only_two_required_files(
    tmp_path, monkeypatch
):
    """离线加载只调用检测与特征模型，并保留尺寸、后端和更新禁用约束。"""
    settings = load_settings(tmp_path)
    model_dir = settings.models_dir / "buffalo_l"
    model_dir.mkdir()
    for name in ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"):
        (model_dir / name).touch()
    loaded = []

    class Model:
        """在厂商加载边界替换昂贵模型会话。"""

        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        def prepare(self, **kwargs):
            """记录有效检测尺寸，不创建额外会话。"""
            self.size = kwargs["input_size"]

    def get_model(path, **kwargs):
        """记录真实加载请求的文件名，不能加载额外关键点文件。"""
        loaded.append(Path(path).name)
        return Model()

    from pathlib import Path

    module = ModuleType("insightface.model_zoo")
    module.get_model = get_model
    root = ModuleType("insightface")
    root.model_zoo = module
    monkeypatch.setitem(sys.modules, "insightface", root)
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", module)
    monkeypatch.setattr(
        "camera_face_comparison.face_engine.detect_execution_backend",
        lambda: ExecutionBackend("cpu", ("CPUExecutionProvider",), -1),
    )
    engine = FaceEngine.from_local_model(settings)
    assert loaded == ["det_10g.onnx", "w600k_r50.onnx"]
    assert engine.backend.name == "cpu"
    assert os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert engine._analyzer.det_model.size == (640, 640)
