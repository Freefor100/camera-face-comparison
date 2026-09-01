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
from camera_face_comparison.image_input import assess_quality


def _face(*, score: float = 0.95, size: int = 160, blur: float = 140.0) -> FaceObservation:
    return FaceObservation(
        bbox=(0.0, 0.0, float(size), float(size)),
        detection_score=score,
        embedding=np.array([0.3, 0.4, 0.5], dtype=np.float32),
        blur_variance=blur,
        landmarks=None,
    )


def test_validate_single_face_rejects_multiple_people(tmp_path) -> None:
    """A group photo must never be accidentally enrolled under one identity."""

    settings = load_settings(tmp_path)

    with pytest.raises(FaceInputError, match="multiple_faces"):
        validate_single_face([_face(), _face()], settings)


def test_validate_single_face_ignores_a_low_confidence_extra_detection(tmp_path) -> None:
    """A sub-threshold false detection must not turn a valid single face into a group photo."""

    settings = load_settings(tmp_path)

    accepted_face = validate_single_face([_face(), _face(score=0.55)], settings)

    assert accepted_face.detection_score == 0.95


def test_validate_single_face_rejects_blurry_face(tmp_path) -> None:
    """A low-quality sample must not enter the standard library."""

    settings = load_settings(tmp_path)

    with pytest.raises(FaceInputError, match="blur_below_minimum"):
        validate_single_face([_face(blur=20.0)], settings)


def test_validate_single_face_returns_valid_face(tmp_path) -> None:
    """A valid sample is normalized before it is persisted or compared."""

    settings = load_settings(tmp_path)
    valid_face = _face()

    accepted_face = validate_single_face([valid_face], settings)

    assert accepted_face.bbox == valid_face.bbox
    assert np.allclose(accepted_face.embedding, [0.424264, 0.565685, 0.707107])


def test_face_engine_adapts_model_output_without_importing_vendor_types(tmp_path) -> None:
    """Changing the InsightFace object wrapper must not leak into application logic."""

    settings = load_settings(tmp_path)
    vendor_face = SimpleNamespace(
        bbox=np.array([10.0, 20.0, 190.0, 200.0]),
        det_score=0.96,
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        kps=np.array([[20.0, 30.0]], dtype=np.float32),
    )
    analyzer = SimpleNamespace(get=lambda frame: [vendor_face])
    blur_inputs: list[tuple[int, int, int]] = []

    def blur_metric(face_crop: np.ndarray) -> float:
        blur_inputs.append(face_crop.shape)
        return 150.0

    engine = FaceEngine(settings, analyzer=analyzer, blur_metric=blur_metric)

    observation = engine.extract_single_face(np.zeros((240, 320, 3), dtype=np.uint8))

    assert observation.bbox == (10.0, 20.0, 190.0, 200.0)
    assert np.allclose(observation.embedding, [0.6, 0.8])
    assert observation.blur_variance == 150.0
    assert blur_inputs == [(180, 180, 3)]


def test_local_model_loading_disables_dependency_update_checks(tmp_path, monkeypatch) -> None:
    """Offline startup must set dependency guards before InsightFace is imported."""

    settings = load_settings(tmp_path)
    (settings.models_dir / "buffalo_l").mkdir()
    monkeypatch.delenv("NO_ALBUMENTATIONS_UPDATE", raising=False)
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    class FakeAnalysis:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def prepare(self, **kwargs) -> None:
            self.prepare_kwargs = kwargs

    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeAnalysis
    insightface_module = ModuleType("insightface")
    insightface_module.app = app_module
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)

    FaceEngine.from_local_model(settings)

    assert os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert os.environ["MPLCONFIGDIR"] == str(settings.logs_dir / "matplotlib")


def test_quality_profile_rejects_an_underexposed_face(tmp_path) -> None:
    """A sharp but almost-black crop is not reliable enough for an identity decision."""

    settings = load_settings(tmp_path)
    frame = np.full((240, 320, 3), 5, dtype=np.uint8)
    profile = assess_quality(frame, _face(), settings)

    assert profile.tier == "reject"
    assert "underexposed" in profile.reasons
