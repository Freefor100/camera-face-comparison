"""验证当前 SQLite 表结构、事务约束和数据持久化。"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from camera_face_comparison.repository import FaceRepository, SampleInput
from tests.application.helpers import embedding as production_embedding


def _sample_input(
    embedding: np.ndarray,
    *,
    image_path: str = "faces/alice/imported.jpg",
) -> SampleInput:
    """构造仓库测试使用的样本输入对象。"""
    return SampleInput(
        image_path=image_path,
        embedding=embedding,
        quality_metrics={
            "detection_score": 0.9,
            "face_size_px": 160.0,
            "blur_variance": 100.0,
            "brightness": 120.0,
            "contrast": 20.0,
        },
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
                production_embedding([0.1, 0.2, 0.3]),
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
    assert samples[0].quality_metrics["face_size_px"] == 160.0
    assert np.allclose(samples[0].embedding, production_embedding([0.1, 0.2, 0.3]))


def test_repository_persists_sample_provenance_hashes_and_wal_mode(tmp_path) -> None:
    """样本来源哈希和完整性元数据在正常重启后必须保留。"""

    database_path = tmp_path / "face_library.sqlite"
    repository = FaceRepository(database_path)
    embedding = production_embedding([0.1, 0.2, 0.3])
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


def test_repository_rejects_the_same_image_for_one_person(tmp_path) -> None:
    """同一人员重复追加内容完全相同的图片时，不得产生重复样本。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[_sample_input(production_embedding([1.0, 0.0]))],
    )

    with pytest.raises(ValueError, match="该人员已经包含内容相同的图片"):
        repository.add_samples(
            person_id="alice-id",
            samples=[
                _sample_input(
                    production_embedding([0.8, 0.2]),
                    image_path="faces/alice/repeated.jpg",
                )
            ],
        )

    assert len(repository.list_samples("alice-id")) == 1
    repository.close()


def test_repository_renames_existing_person_without_changing_identity(tmp_path) -> None:
    """修改姓名时必须保留人员编号和已有样本。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[_sample_input(production_embedding([1.0, 0.0]))],
    )

    renamed = repository.rename_person("alice-id", "  Alice Zhang  ")

    assert renamed.id == "alice-id"
    assert renamed.display_name == "Alice Zhang"
    assert repository.get_person("alice-id") == renamed
    assert len(repository.list_samples("alice-id")) == 1
    repository.close()


def test_repository_deletes_one_sample_but_refuses_to_delete_the_last_sample(
    tmp_path,
) -> None:
    """删除样本后人员仍须至少保留一张可用于识别的图片。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[
            _sample_input(
                production_embedding([1.0, 0.0]),
                image_path="faces/alice/first.jpg",
            ),
            SampleInput(
                image_path="faces/alice/second.jpg",
                embedding=production_embedding([0.8, 0.2]),
                quality_metrics={"face_size_px": 150.0},
                source_type="file",
                image_sha256="b" * 64,
            ),
        ],
    )
    first, second = repository.list_samples("alice-id")

    removed = repository.delete_sample("alice-id", first.id)

    assert removed.id == first.id
    assert [sample.id for sample in repository.list_samples("alice-id")] == [second.id]
    with pytest.raises(ValueError, match="最后一张样本"):
        repository.delete_sample("alice-id", second.id)
    assert [sample.id for sample in repository.list_samples("alice-id")] == [second.id]
    repository.close()


def test_repository_deletes_person_samples_and_detaches_recognition_history(
    tmp_path,
) -> None:
    """删除人员必须清除其样本，并保留不再指向该人员的历史日志。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice-id",
        display_name="Alice",
        samples=[_sample_input(production_embedding([1.0, 0.0]))],
    )
    repository.record_recognition(
        decision="matched",
        person_id="alice-id",
        top_score=0.8,
        second_score=None,
        score_gap=None,
        acceptance_score=0.8,
        acceptance_rule="score_threshold",
        latency_ms=20.0,
        reason=None,
    )

    removed_samples = repository.delete_person("alice-id")
    log_person_id = repository._connection.execute(
        "SELECT person_id FROM recognition_logs"
    ).fetchone()[0]

    assert len(removed_samples) == 1
    assert repository.get_person("alice-id") is None
    assert repository.list_samples("alice-id") == []
    assert log_person_id is None
    repository.close()
