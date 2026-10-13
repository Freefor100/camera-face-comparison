"""验证实际复现的数据库故障、向量与诊断边界。"""

import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from camera_face_comparison.repository import FaceRepository, SampleInput


def sample(path="faces/person/sample.jpg"):
    """创建无需模型的有效生产维度样本。"""
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1
    return SampleInput(image_path=path, embedding=vector, quality_metrics={})


def test_commit_failure_rolls_back_and_allows_next_transaction(tmp_path):
    """提交失败不能留下待提交记录或阻断下一次写入。"""
    repository = FaceRepository(tmp_path / "library.sqlite")
    repository._connection.set_authorizer(
        lambda action, value, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and value == "COMMIT"
            else sqlite3.SQLITE_OK
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        repository.create_person_with_samples(
            person_id="person", display_name="测试人员", samples=[sample()]
        )
    assert not repository._connection.in_transaction
    assert repository.list_people() == []
    repository._connection.set_authorizer(None)
    repository.create_person_with_samples(
        person_id="person", display_name="测试人员", samples=[sample()]
    )
    assert len(repository.list_samples()) == 1
    repository.close()


@pytest.mark.parametrize("vector", [np.full(512, np.nan), np.zeros(512), np.ones(511)])
def test_invalid_production_embedding_is_never_persisted(tmp_path, vector):
    """非有限、全零或错误维度向量不能形成已入库人员。"""
    repository = FaceRepository(tmp_path / "library.sqlite")
    item = replace(sample(), embedding=vector)
    with pytest.raises(ValueError):
        repository.create_person_with_samples(
            person_id="person", display_name="测试人员", samples=[item]
        )
    assert repository.list_people() == []
    repository.close()


def test_shared_image_path_is_rejected_before_second_identity_is_committed(tmp_path):
    """不同人员不能引用同一个受管文件，防止删除互相破坏。"""
    repository = FaceRepository(tmp_path / "library.sqlite")
    repository.create_person_with_samples(person_id="person", display_name="甲", samples=[sample()])
    with pytest.raises(ValueError):
        repository.create_person_with_samples(
            person_id="other", display_name="乙", samples=[sample()]
        )
    assert len(repository.list_people()) == 1
    repository.close()


def test_failed_rollback_isolates_connection_and_preserves_enrollment_images(tmp_path):
    """事务结果不明时不能继续使用连接，也不能盲目删除录入图片。"""
    from camera_face_comparison.config import load_settings
    from camera_face_comparison.enrollment import EnrollmentService
    from camera_face_comparison.image_input import ImageInput
    from tests.application.test_enrollment import FakeFaceEngine, _save_image

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    repository._connection.set_authorizer(
        lambda action, value, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and value in {"COMMIT", "ROLLBACK"}
            else sqlite3.SQLITE_OK
        )
    )
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=_save_image,
    )
    with pytest.raises(sqlite3.DatabaseError):
        service.create_from_inputs(
            "事务异常", [ImageInput.from_camera(np.zeros((200, 200, 3), dtype=np.uint8))]
        )
    assert repository.transaction_uncertain
    assert len(list(settings.faces_dir.glob("*/sample_001.jpg"))) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        repository.list_people()


def test_corrupt_sample_is_reported_and_deletion_does_not_decode_embedding(tmp_path):
    """坏向量需要形成诊断报告，删除不能提交后才解码报错。"""
    from camera_face_comparison.config import load_settings
    from camera_face_comparison.integrity import verify_library

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    repository.create_person_with_samples(
        person_id="person",
        display_name="测试",
        samples=[sample(), sample("faces/person/second.jpg")],
    )
    bad = repository.list_samples()[0].id
    repository._connection.execute("UPDATE face_samples SET embedding_dim=511 WHERE id=?", (bad,))
    repository._connection.commit()
    report = verify_library(repository, settings)
    assert not report.is_valid
    assert any(item.kind == "invalid_sample" for item in report.failures)
    repository._connection.set_authorizer(
        lambda action, table, column, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ
            and table == "face_samples"
            and column in {"embedding_blob", "embedding_dim", "quality_metrics_json"}
            else sqlite3.SQLITE_OK
        )
    )
    removed = repository.delete_sample("person", bad)
    assert removed.id == bad
    repository._connection.set_authorizer(None)
    assert len(repository.list_samples()) == 1
    repository.close()


def test_missing_database_with_existing_photos_is_not_silently_recreated(tmp_path):
    """数据库丢失但照片仍在时，必须报告缺失而不是建立空库。"""
    photo = tmp_path / "faces/person/sample.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"retained source")
    with pytest.raises(RuntimeError):
        FaceRepository(tmp_path / "library.sqlite")
    assert not (tmp_path / "library.sqlite").exists()


def test_quality_overflow_does_not_abort_other_sample_diagnostics(tmp_path):
    """一条质量记录溢出不能遮蔽另一条坏向量的诊断。"""
    import json

    from camera_face_comparison.config import load_settings
    from camera_face_comparison.integrity import verify_library

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    repository.create_person_with_samples(
        person_id="person",
        display_name="测试",
        samples=[sample(), sample("faces/person/second.jpg")],
    )
    first, second = repository.list_samples()
    repository._connection.execute(
        "UPDATE face_samples SET quality_metrics_json=? WHERE id=?",
        (json.dumps({"brightness": 10**400}), first.id),
    )
    repository._connection.execute(
        "UPDATE face_samples SET embedding_dim=511 WHERE id=?",
        (second.id,),
    )
    repository._connection.commit()
    report = verify_library(repository, settings)
    assert {item.sample_id for item in report.failures if item.kind == "invalid_sample"} == {
        first.id,
        second.id,
    }
    repository.close()


def test_interleaved_commit_cannot_mix_people_and_sample_read_snapshots(tmp_path):
    """另一连接在两次查询间提交，读取结果仍须属于同一时刻。"""
    repository = FaceRepository(tmp_path / "library.sqlite")
    repository.create_person_with_samples(person_id="a", display_name="甲", samples=[sample()])
    other = FaceRepository(tmp_path / "library.sqlite")
    committed = False

    def insert_during_sample_query(query):
        """在人员已读完、样本即将读取时插入一次独立提交。"""
        nonlocal committed
        if "FROM face_samples" in query and not committed:
            committed = True
            other.create_person_with_samples(
                person_id="b",
                display_name="乙",
                samples=[sample("faces/b/sample.jpg")],
            )

    repository._connection.set_trace_callback(insert_during_sample_query)
    people, samples, checks, foreign_keys = repository.read_library_rows()
    repository._connection.set_trace_callback(None)
    assert committed
    assert {row["id"] for row in people} == {"a"}
    assert {row["person_id"] for row in samples} == {"a"}
    assert checks == ("ok",) and foreign_keys == ()
    assert len(repository.list_people()) == 2
    other.close()
    repository.close()


def test_append_cleanup_failure_keeps_original_error_and_existing_files(tmp_path, monkeypatch):
    """一项文件补偿失败不能跳过其他文件或覆盖原业务异常。"""
    from pathlib import Path

    from camera_face_comparison.config import load_settings
    from camera_face_comparison.enrollment import EnrollmentService
    from camera_face_comparison.image_input import ImageInput
    from tests.application.test_enrollment import FakeFaceEngine, _save_image

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    service = EnrollmentService(
        repository=repository,
        settings=settings,
        face_engine=FakeFaceEngine(),
        image_saver=_save_image,
    )
    image = ImageInput.from_camera(np.zeros((200, 200, 3), dtype=np.uint8))
    person = service.create_from_inputs("测试", [image])
    old_path = settings.data_dir / repository.list_samples(person.id)[0].image_path
    unlink = Path.unlink

    def fail_one_unlink(path, *args, **kwargs):
        """仅阻断本次追加的第一张图片清理。"""
        if path != old_path and path.parent.name == person.id and path.name.endswith("_001.jpg"):
            raise PermissionError("补偿清理注入故障")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one_unlink)
    with pytest.raises(ValueError) as caught:
        service.append_from_inputs(person.id, [image, image])
    assert "补偿清理注入故障" not in str(caught.value)
    assert any("补偿清理注入故障" in note for note in caught.value.__notes__)
    assert old_path.is_file()
    assert len(repository.list_samples(person.id)) == 1
    assert len(list(old_path.parent.glob("*.jpg"))) == 2
    assert not repository._connection.in_transaction
    repository.close()


def test_database_lock_is_busy_not_corruption(tmp_path):
    """真实写锁超时应提示忙碌，不误报标准库损坏。"""
    from camera_face_comparison.errors import classify_error

    repository = FaceRepository(tmp_path / "library.sqlite")
    other = FaceRepository(tmp_path / "library.sqlite")
    repository._connection.execute("PRAGMA busy_timeout=1")
    other._connection.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError) as caught:
        repository.create_person_with_samples(person_id="a", display_name="甲", samples=[sample()])
    assert classify_error(caught.value).kind == "storage_busy"
    other._connection.rollback()
    other.close()
    repository.close()
