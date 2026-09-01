from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .repository import FaceRepository


@dataclass(frozen=True)
class IntegrityFailure:
    """One local-library item that should not be trusted for recognition."""

    kind: str
    detail: str
    sample_id: str | None = None


@dataclass(frozen=True)
class LibraryVerificationReport:
    """Read-only verification outcome for the current SQLite library and face files."""

    failures: tuple[IntegrityFailure, ...]

    @property
    def is_valid(self) -> bool:
        return not self.failures


def verify_library(repository: FaceRepository, settings: Settings) -> LibraryVerificationReport:
    """Check SQLite structure and the hashes the application records at write time."""

    failures: list[IntegrityFailure] = []
    for message in repository.sqlite_integrity_messages():
        if message.lower() != "ok":
            failures.append(IntegrityFailure(kind="sqlite_integrity_error", detail=message))
    for violation in repository.foreign_key_violations():
        failures.append(IntegrityFailure(kind="foreign_key_violation", detail=violation))

    face_root = settings.faces_dir.resolve()
    for sample in repository.list_samples():
        image_path = (settings.data_dir / sample.image_path).resolve()
        if not _is_within(image_path, face_root):
            failures.append(
                IntegrityFailure("image_path_outside_library", str(image_path), sample.id)
            )
        elif not image_path.is_file():
            failures.append(IntegrityFailure("image_missing", str(image_path), sample.id))
        elif sample.image_sha256 is None:
            failures.append(IntegrityFailure("missing_image_hash", str(image_path), sample.id))
        elif _sha256_file(image_path) != sample.image_sha256:
            failures.append(IntegrityFailure("image_hash_mismatch", str(image_path), sample.id))

        expected_embedding_hash = sample.embedding_sha256
        current_embedding_hash = hashlib.sha256(
            np.asarray(sample.embedding, dtype=np.float32).tobytes()
        ).hexdigest()
        if expected_embedding_hash is None:
            failures.append(IntegrityFailure("missing_embedding_hash", sample.id, sample.id))
        elif current_embedding_hash != expected_embedding_hash:
            failures.append(IntegrityFailure("embedding_hash_mismatch", sample.id, sample.id))
    return LibraryVerificationReport(tuple(failures))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
