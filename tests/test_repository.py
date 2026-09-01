from __future__ import annotations

import hashlib

import numpy as np

from camera_face_comparison.repository import FaceRepository, SampleInput


def _sample_input(
    embedding: np.ndarray,
    *,
    image_path: str = "faces/alice/imported.jpg",
) -> SampleInput:
    """构造仓库测试使用的样本输入对象。"""
    return SampleInput(
        image_path=image_path,
        embedding=embedding,
        pose="sample_001",
        quality={"quality_score": 0.9, "tier": "high"},
        source_type="file",
        image_sha256="a" * 64,
    )


def test_person_schema_contains_only_current_identity_fields(tmp_path) -> None:
    """当前未发布项目的身份表只保存当前版本的身份字段。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    columns = [
        row["name"]
        for row in repository._connection.execute("PRAGMA table_info(persons)").fetchall()
    ]
    repository.close()

    assert columns == ["id", "display_name", "created_at"]


def test_repository_persists_person_and_embedding_across_reopen(tmp_path) -> None:
    """动态新增的标准库内容在应用重启后仍必须可以使用。"""

    database_path = tmp_path / "face_library.sqlite"
    first_repository = FaceRepository(database_path)
    person = first_repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[
            _sample_input(
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                image_path="faces/alice/front.jpg",
            )
        ],
    )
    first_repository.close()

    reopened_repository = FaceRepository(database_path)
    people = reopened_repository.list_people()
    samples = reopened_repository.list_samples()
    reopened_repository.close()

    assert [(item.id, item.display_name) for item in people] == [(person.id, "Alice")]
    assert len(samples) == 1
    assert samples[0].person_id == person.id
    assert samples[0].pose == "sample_001"
    assert np.allclose(samples[0].embedding, [0.1, 0.2, 0.3])


def test_repository_persists_sample_provenance_hashes_and_wal_mode(tmp_path) -> None:
    """样本来源哈希和完整性元数据在正常重启后必须保留。"""

    database_path = tmp_path / "face_library.sqlite"
    repository = FaceRepository(database_path)
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    person = repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[_sample_input(embedding)],
    )
    journal_mode = repository._connection.execute("PRAGMA journal_mode").fetchone()[0]
    repository.close()

    reopened = FaceRepository(database_path)
    restored_person = reopened.get_person(person.id)
    restored_sample = reopened.list_samples(person.id)[0]
    reopened.close()

    assert restored_person is not None
    assert restored_sample.source_type == "file"
    assert restored_sample.image_sha256 == "a" * 64
    assert restored_sample.embedding_sha256 == hashlib.sha256(embedding.tobytes()).hexdigest()
    assert journal_mode == "wal"
