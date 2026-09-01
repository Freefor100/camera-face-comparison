from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import EnrollmentService
from camera_face_comparison.face_engine import FaceObservation
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.repository import FaceRepository


class FakeFaceEngine:
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=np.array([0.2, 0.3, 0.4], dtype=np.float32),
            blur_variance=150.0,
            landmarks=None,
        )


class DiverseFakeFaceEngine:
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
        marker = float(frame[0, 0, 0])
        return FaceObservation(
            bbox=(0.0, 0.0, 180.0, 180.0),
            detection_score=0.95,
            embedding=np.array([marker + 1.0, 3.0, 7.0], dtype=np.float32),
            blur_variance=150.0,
            landmarks=None,
        )


def _save_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test image")


def test_enrollment_rolls_back_when_an_image_cannot_be_saved(tmp_path) -> None:
    """A failed multi-image import must not leave an identity or face files behind."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    saved_count = 0

    def failing_saver(path: Path, frame: np.ndarray) -> None:
        nonlocal saved_count
        saved_count += 1
        if saved_count == 2:
            raise OSError("disk write failed")
        _save_image(path, frame)

    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=failing_saver,
    )
    with pytest.raises(OSError, match="disk write failed"):
        service.create_from_inputs(
            "Alice",
            [
                ImageInput.from_camera(_textured_frame(80)),
                ImageInput.from_camera(_textured_frame(120)),
            ],
        )

    assert repository.list_people() == []
    assert repository.list_samples() == []
    assert list(settings.faces_dir.iterdir()) == []
    repository.close()


def test_one_valid_input_activates_a_person(tmp_path) -> None:
    """The first accepted camera or local image makes an identity recognizable."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    person = service.create_from_inputs(
        "Alice", [ImageInput.from_camera(_textured_frame(120))]
    )

    restored = repository.get_person(person.id)
    samples = repository.list_samples(person.id)
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(samples) == 1
    assert all(sample.source_type == "camera" for sample in samples)
    assert all(sample.image_sha256 for sample in samples)
    assert all((settings.data_dir / sample.image_path).is_file() for sample in samples)
    repository.close()


def test_one_valid_local_image_activates_a_person(tmp_path) -> None:
    settings = load_settings(tmp_path / "data")
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    image_path = tmp_path / "alice.png"
    assert cv2.imwrite(str(image_path), _textured_frame(120))

    person = service.create_from_inputs("Alice", [ImageInput.from_file(image_path)])

    restored = repository.get_person(person.id)
    samples = repository.list_samples(person.id)
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(samples) == 1
    assert samples[0].source_type == "file"
    repository.close()


def test_local_inputs_can_activate_an_existing_draft_person(tmp_path) -> None:
    """Adding the first valid sample activates an otherwise empty draft identity."""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    person = repository.create_person("Alice")

    service.append_from_inputs(
        person.id,
        [ImageInput.from_camera(_textured_frame(120))],
    )

    restored = repository.get_person(person.id)
    samples = repository.list_samples(person.id)
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(samples) == 1
    repository.close()


def test_multiple_samples_can_be_appended_after_activation(tmp_path) -> None:
    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=DiverseFakeFaceEngine(),
        image_saver=_save_image,
    )
    person = service.create_from_inputs(
        "Alice", [ImageInput.from_camera(_textured_frame(80))]
    )

    added = service.append_from_inputs(
        person.id,
        [
            ImageInput.from_camera(_textured_frame(120)),
            ImageInput.from_camera(_textured_frame(180)),
        ],
    )

    restored = repository.get_person(person.id)
    assert added == 2
    assert restored is not None
    assert restored.lifecycle == "active"
    assert len(repository.list_samples(person.id)) == 3
    repository.close()


def _textured_frame(value: int) -> np.ndarray:
    frame = np.full((240, 320, 3), value, dtype=np.uint8)
    frame[:, ::2] = value + 30
    return frame
