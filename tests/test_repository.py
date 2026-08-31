from __future__ import annotations

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
