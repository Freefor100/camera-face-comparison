from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Protocol
from uuid import uuid4

import numpy as np

from .config import Settings
from .domain import Person
from .face_engine import FaceObservation, validate_single_face
from .repository import FaceRepository, SampleInput

REQUIRED_POSES = ("front", "left", "right", "up", "down")


class FaceExtractor(Protocol):
    def extract_single_face(self, frame: np.ndarray) -> FaceObservation: ...


ImageSaver = Callable[[Path, np.ndarray], None]


@dataclass
class EnrollmentSession:
    """An in-memory five-pose collection that is committed only when complete."""

    _service: EnrollmentService
    display_name: str
    _captures: dict[str, tuple[np.ndarray, FaceObservation]] = field(default_factory=dict)

    @property
    def completed_poses(self) -> tuple[str, ...]:
        return tuple(pose for pose in REQUIRED_POSES if pose in self._captures)

    def capture(self, pose: str, frame: np.ndarray) -> FaceObservation:
        if pose not in REQUIRED_POSES:
            raise ValueError(f"unsupported pose: {pose}")
        if pose in self._captures:
            raise ValueError(f"pose already captured: {pose}")
        observation = self._service.face_engine.extract_single_face(frame)
        validated = validate_single_face([observation], self._service.settings)
        self._captures[pose] = (frame.copy(), validated)
        return validated

    def commit(self) -> Person:
        missing = [pose for pose in REQUIRED_POSES if pose not in self._captures]
        if missing:
            raise ValueError("five valid captures are required before enrollment")
        return self._service._commit(self.display_name, self._captures)


class EnrollmentService:
    """Coordinates quality checks, image saving, and durable face-library updates."""

    def __init__(
        self,
        *,
        repository: FaceRepository,
        settings: Settings,
        face_engine: FaceExtractor,
        image_saver: ImageSaver,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.face_engine = face_engine
        self.image_saver = image_saver

    def begin(self, display_name: str) -> EnrollmentSession:
        if not display_name.strip():
            raise ValueError("display_name must not be empty")
        return EnrollmentSession(self, display_name.strip())

    def append_sample(
        self,
        person_id: str,
        pose: str,
        frame: np.ndarray,
    ) -> FaceObservation:
        """Append one quality-approved sample to an already enrolled person."""

        if pose not in REQUIRED_POSES:
            raise ValueError(f"unsupported pose: {pose}")
        if self.repository.get_person(person_id) is None:
            raise ValueError("person does not exist")
        observation = self.face_engine.extract_single_face(frame)
        validated = validate_single_face([observation], self.settings)
        image_path = self.settings.faces_dir / person_id / f"{pose}_extra_{uuid4().hex}.jpg"
        self.image_saver(image_path, frame)
        try:
            relative_path = image_path.relative_to(self.settings.data_dir).as_posix()
            self.repository.add_sample(
                person_id=person_id,
                image_path=relative_path,
                embedding=validated.embedding,
                pose=pose,
                quality={
                    "detection_score": validated.detection_score,
                    "blur_variance": validated.blur_variance,
                },
            )
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
        return validated

    def _commit(
        self,
        display_name: str,
        captures: dict[str, tuple[np.ndarray, FaceObservation]],
    ) -> Person:
        person_id = str(uuid4())
        person_dir = self.settings.faces_dir / person_id
        samples: list[SampleInput] = []
        try:
            for pose in REQUIRED_POSES:
                frame, observation = captures[pose]
                image_path = person_dir / f"{pose}.jpg"
                self.image_saver(image_path, frame)
                relative_path = image_path.relative_to(self.settings.data_dir).as_posix()
                samples.append(
                    SampleInput(
                        image_path=relative_path,
                        embedding=observation.embedding,
                        pose=pose,
                        quality={
                            "detection_score": observation.detection_score,
                            "blur_variance": observation.blur_variance,
                        },
                    )
                )
            return self.repository.create_person_with_samples(
                person_id=person_id,
                display_name=display_name,
                samples=samples,
            )
        except Exception:
            rmtree(person_dir, ignore_errors=True)
            raise
