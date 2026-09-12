from __future__ import annotations

import numpy as np

from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy
from camera_face_comparison.repository import FaceRepository, SampleInput


def _sample(embedding: list[float], path: str) -> SampleInput:
    """构造内存标准库测试所需的样本输入。"""

    return SampleInput(
        image_path=path,
        embedding=np.asarray(embedding, dtype=np.float32),
        quality_metrics={},
        source_type="file",
    )


def test_empty_and_single_person_library_search(tmp_path) -> None:
    """空库和单身份库都应返回稳定的开放集候选结果。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    library = InMemoryFaceLibrary.from_repository(repository)
    empty = library.snapshot().search(
        np.array([1.0, 0.0], dtype=np.float32),
        ScoreThresholdPolicy(minimum_score=0.5),
    )
    assert empty.reason == "empty_face_library"
    assert empty.top_person_id is None

    repository.create_person_with_samples(
        person_id="alice",
        display_name="Alice",
        samples=[_sample([1.0, 0.0], "faces/alice/1.jpg")],
    )
    library.rebuild(repository)
    decision = library.snapshot().search(
        np.array([1.0, 0.0], dtype=np.float32),
        ScoreThresholdPolicy(minimum_score=0.5),
    )
    repository.close()

    assert decision.accepted_person_id == "alice"
    assert decision.top_score == 1.0
    assert decision.second_score is None
    assert decision.score_gap is None


def test_matrix_search_returns_top_two_with_deterministic_ties(tmp_path) -> None:
    """矩阵检索应得到前两名，并在同分时按身份编号排序。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    for person_id, name in (("alice", "Alice"), ("bob", "Bob"), ("carol", "Carol")):
        repository.create_person_with_samples(
            person_id=person_id,
            display_name=name,
            samples=[_sample([1.0, 0.0], f"faces/{person_id}/1.jpg")],
        )

    library = InMemoryFaceLibrary.from_repository(repository)
    snapshot = library.snapshot()
    decision = snapshot.search(
        np.array([1.0, 0.0], dtype=np.float32),
        ScoreThresholdPolicy(minimum_score=0.9),
    )
    repository.close()

    assert snapshot.person_ids == ("alice", "bob", "carol")
    assert decision.accepted_person_id == "alice"
    assert decision.top_person_id == "alice"
    assert decision.second_score == 1.0
    assert decision.score_gap == 0.0


def test_refresh_person_replaces_one_row_and_old_snapshot_is_unchanged(tmp_path) -> None:
    """追加样本后只更新目标身份，新快照生效且旧快照保持不变。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice",
        display_name="Alice",
        samples=[_sample([1.0, 0.0], "faces/alice/1.jpg")],
    )
    repository.create_person_with_samples(
        person_id="bob",
        display_name="Bob",
        samples=[_sample([0.0, 1.0], "faces/bob/1.jpg")],
    )
    library = InMemoryFaceLibrary.from_repository(repository)
    old_snapshot = library.snapshot()

    repository.add_samples(
        person_id="alice",
        samples=[_sample([0.0, 1.0], "faces/alice/2.jpg")],
    )
    library.refresh_person(repository, "alice")
    new_snapshot = library.snapshot()
    repository.close()

    assert old_snapshot.revision < new_snapshot.revision
    assert np.allclose(old_snapshot.prototype_matrix[0], [1.0, 0.0])
    assert np.allclose(new_snapshot.prototype_matrix[0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    assert np.allclose(old_snapshot.prototype_matrix[1], new_snapshot.prototype_matrix[1])
