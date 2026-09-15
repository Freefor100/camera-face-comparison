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

            return np.array([3.0, 4.0], dtype=np.float32)

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
    assert np.allclose(observation.embedding, [0.6, 0.8])
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
            return np.array([3.0, 4.0], dtype=np.float32)

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
    assert np.allclose(observation.embedding, [0.6, 0.8])


def test_local_model_loading_disables_dependency_update_checks(tmp_path, monkeypatch) -> None:
    """离线启动必须在导入 InsightFace 前设置依赖更新防护变量。"""

    settings = load_settings(tmp_path)
    (settings.models_dir / "buffalo_l").mkdir()
    monkeypatch.delenv("NO_ALBUMENTATIONS_UPDATE", raising=False)
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    class FakeAnalysis:
        """模拟 InsightFace 分析器构造和 prepare 调用。"""

        def __init__(self, **kwargs) -> None:
            """保存模型构造参数。"""

            type(self).last_kwargs = kwargs

        def prepare(self, **kwargs) -> None:
            """保存模型准备参数。"""

            type(self).last_prepare_kwargs = kwargs

    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeAnalysis
    insightface_module = ModuleType("insightface")
    insightface_module.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)
    monkeypatch.setattr(
        "camera_face_comparison.face_engine.detect_execution_backend",
        lambda: ExecutionBackend(
            name="cuda",
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            context_id=0,
        ),
    )

    FaceEngine.from_local_model(settings)

    assert os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert os.environ["MPLCONFIGDIR"] == str(settings.logs_dir / "matplotlib")
    assert FakeAnalysis.last_kwargs["allowed_modules"] == ["detection", "recognition"]
    assert FakeAnalysis.last_prepare_kwargs["ctx_id"] == 0
