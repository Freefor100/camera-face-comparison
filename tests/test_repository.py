from __future__ import annotations

import hashlib

import numpy as np

from camera_face_comparison.repository import FaceRepository


def test_repository_persists_person_and_embedding_across_reopen(tmp_path) -> None:
    """A dynamic library addition must remain usable after the application restarts."""

    database_path = tmp_path / "face_library.sqlite"
    first_repository = FaceRepository(database_path)
    person = first_repository.create_person("Alice")
    first_repository.add_sample(
        person_id=person.id,
        image_path="faces/alice/front.jpg",
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        pose="front",
        quality={"blur_variance": 125.0},
    )
    first_repository.close()

    reopened_repository = FaceRepository(database_path)
    people = reopened_repository.list_people()
    samples = reopened_repository.list_samples()
    reopened_repository.close()

    assert [(item.id, item.display_name) for item in people] == [
        (person.id, "Alice"),
    ]
    assert len(samples) == 1
    assert samples[0].person_id == person.id
    assert samples[0].pose == "front"
    assert np.allclose(samples[0].embedding, [0.1, 0.2, 0.3])


def test_repository_tracks_lifecycle_source_and_hashes(tmp_path) -> None:
    """Stored template provenance and consistency metadata survive a normal reopen."""

    database_path = tmp_path / "face_library.sqlite"
    repository = FaceRepository(database_path)
    person = repository.create_person("Alice")
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample = repository.add_sample(
        person_id=person.id,
        image_path="faces/alice/imported.jpg",
        embedding=embedding,
        pose="imported",
        quality={"tier": "high"},
        source_type="file",
        image_sha256="a" * 64,
    )
    journal_mode = repository._connection.execute("PRAGMA journal_mode").fetchone()[0]
    repository.close()

    reopened = FaceRepository(database_path)
    restored_person = reopened.get_person(person.id)
    restored_sample = reopened.list_samples(person.id)[0]
    reopened.close()

    assert restored_person is not None
    assert restored_person.lifecycle == "active"
    assert restored_sample.source_type == "file"
    assert restored_sample.image_sha256 == "a" * 64
    assert restored_sample.embedding_sha256 == hashlib.sha256(embedding.tobytes()).hexdigest()
    assert sample.embedding_sha256 == restored_sample.embedding_sha256
    assert journal_mode == "wal"


def test_repository_migrates_a_legacy_draft_that_already_has_a_sample(tmp_path) -> None:
    """Reopening an old library must make every non-empty identity recognizable."""

    database_path = tmp_path / "face_library.sqlite"
    repository = FaceRepository(database_path)
    person = repository.create_person("Alice")
    repository.add_sample(
        person_id=person.id,
        image_path="faces/alice/imported.jpg",
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        pose="sample_001",
        quality={"tier": "high"},
    )
    with repository._connection:
        repository._connection.execute(
            "UPDATE persons SET lifecycle = 'draft' WHERE id = ?", (person.id,)
        )
    repository.close()

    reopened = FaceRepository(database_path)
    restored = reopened.get_person(person.id)
    reopened.close()

    assert restored is not None
    assert restored.lifecycle == "active"
