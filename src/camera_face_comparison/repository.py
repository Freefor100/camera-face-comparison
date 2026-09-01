from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .domain import FaceSample, Person


@dataclass(frozen=True)
class SampleInput:
    """录入流程为一次 SQLite 事务准备的一条样本数据。"""

    image_path: str
    embedding: np.ndarray
    pose: str
    quality: dict[str, float | str]
    source_type: str = "camera"
    image_sha256: str | None = None


class FaceRepository:
    """保存人员、特征向量和识别日志的 SQLite 仓库。"""

    def __init__(self, database_path: Path) -> None:
        """打开数据库并初始化当前版本的表结构。

        参数：
            database_path：SQLite 数据库文件路径。
        前置条件：
            父目录已存在或由 SQLite 创建；本项目不迁移旧数据库结构。
        """
        self._connection = sqlite3.connect(database_path, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def close(self) -> None:
        """关闭数据库连接，释放文件句柄。"""
        self._connection.close()

    def create_person_with_samples(
        self,
        *,
        person_id: str,
        display_name: str,
        samples: Sequence[SampleInput],
    ) -> Person:
        """在一个事务中原子保存新身份及其全部样本。

        参数：
            person_id：新身份的唯一编号。
            display_name：新身份的显示名称。
            samples：至少一条已经准备好的样本。
        返回：
            已持久化的人员对象。
        前置条件：
            样本特征为非空一维向量；图片路径和哈希已由录入服务准备好。
        """

        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("display_name must not be empty")
        if not samples:
            raise ValueError("at least one sample is required")

        person = Person(
            id=person_id,
            display_name=normalized_name,
            created_at=_now(),
        )
        prepared_samples = [
            self._make_sample(
                person_id=person.id,
                image_path=sample.image_path,
                embedding=sample.embedding,
                pose=sample.pose,
                quality=sample.quality,
                source_type=sample.source_type,
                image_sha256=sample.image_sha256,
            )
            for sample in samples
        ]
        with self._write_transaction():
            self._connection.execute(
                "INSERT INTO persons (id, display_name, created_at) VALUES (?, ?, ?)",
                (person.id, person.display_name, person.created_at.isoformat()),
            )
            for sample in prepared_samples:
                self._insert_sample(sample)
        return person

    def add_samples(
        self,
        *,
        person_id: str,
        samples: Sequence[SampleInput],
    ) -> list[FaceSample]:
        """在一个事务中向已有身份追加准备好的样本。

        参数：
            person_id：已存在身份的编号。
            samples：至少一条待写入样本。
        返回：
            实际生成并写入的样本对象列表。
        前置条件：
            调用方应确保身份存在，所有文件已移动到最终位置。
        """

        if not samples:
            raise ValueError("at least one sample is required")
        prepared_samples = [
            self._make_sample(
                person_id=person_id,
                image_path=sample.image_path,
                embedding=sample.embedding,
                pose=sample.pose,
                quality=sample.quality,
                source_type=sample.source_type,
                image_sha256=sample.image_sha256,
            )
            for sample in samples
        ]
        with self._write_transaction():
            for sample in prepared_samples:
                self._insert_sample(sample)
        return prepared_samples

    def sqlite_integrity_messages(self) -> tuple[str, ...]:
        """返回 SQLite 完整性检查结果，不修改数据库内容。"""

        return tuple(
            str(row[0]) for row in self._connection.execute("PRAGMA integrity_check").fetchall()
        )

    def foreign_key_violations(self) -> tuple[str, ...]:
        """返回 SQLite 检测到的全部外键悬挂引用。"""

        return tuple(
            ":".join(str(value) for value in row)
            for row in self._connection.execute("PRAGMA foreign_key_check").fetchall()
        )

    def list_people(self) -> list[Person]:
        """按创建时间和名称返回当前全部人员。"""
        rows = self._connection.execute(
            "SELECT id, display_name, created_at FROM persons ORDER BY created_at, display_name"
        ).fetchall()
        return [self._person_from_row(row) for row in rows]

    def get_person(self, person_id: str) -> Person | None:
        """按身份编号查找人员，找不到时返回 `None`。"""
        row = self._connection.execute(
            "SELECT id, display_name, created_at FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return None
        return self._person_from_row(row)

    def list_samples(self, person_id: str | None = None) -> list[FaceSample]:
        """返回全部样本，或返回指定身份的样本。

        参数：
            person_id：可选的身份编号过滤条件。
        返回：
            按创建时间排序的样本对象列表。
        """
        query = """
            SELECT id, person_id, image_path, embedding_blob, embedding_dim,
                   pose, quality_json, created_at, source_type, image_sha256,
                   embedding_sha256
            FROM face_samples
        """
        parameters: tuple[str, ...] = ()
        if person_id is not None:
            query += " WHERE person_id = ?"
            parameters = (person_id,)
        query += " ORDER BY created_at"
        rows = self._connection.execute(query, parameters).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def record_recognition(
        self,
        *,
        decision: str,
        person_id: str | None,
        top_score: float | None,
        runner_up_score: float | None,
        latency_ms: float,
        reason: str | None,
    ) -> None:
        """把一次识别判定和耗时写入识别日志表。"""
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO recognition_logs (
                    id, captured_at, decision, person_id, top_score,
                    runner_up_score, latency_ms, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    _now().isoformat(),
                    decision,
                    person_id,
                    top_score,
                    runner_up_score,
                    latency_ms,
                    reason,
                ),
            )

    def _initialize_schema(self) -> None:
        """创建当前版本的人员、样本和识别日志表。"""
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_samples (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE RESTRICT,
                    image_path TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    pose TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'camera',
                    image_sha256 TEXT,
                    embedding_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                    top_score REAL,
                    runner_up_score REAL,
                    latency_ms REAL NOT NULL,
                    reason TEXT
                );
                """
            )

    @staticmethod
    def _make_sample(
        *,
        person_id: str,
        image_path: str,
        embedding: np.ndarray,
        pose: str,
        quality: dict[str, float | str],
        source_type: str,
        image_sha256: str | None,
    ) -> FaceSample:
        """校验向量并构造带哈希的样本领域对象。"""
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        return FaceSample(
            id=str(uuid4()),
            person_id=person_id,
            image_path=image_path,
            embedding=vector,
            pose=pose,
            quality=quality,
            created_at=_now(),
            source_type=source_type,
            image_sha256=image_sha256,
            embedding_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
        )

    def _insert_sample(self, sample: FaceSample) -> None:
        """将一个样本对象写入已存在的 SQLite 事务。"""
        self._connection.execute(
            """
            INSERT INTO face_samples (
                id, person_id, image_path, embedding_blob, embedding_dim,
                pose, quality_json, created_at, source_type, image_sha256,
                embedding_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.id,
                sample.person_id,
                sample.image_path,
                sample.embedding.tobytes(),
                sample.embedding.size,
                sample.pose,
                json.dumps(sample.quality, ensure_ascii=False),
                sample.created_at.isoformat(),
                sample.source_type,
                sample.image_sha256,
                sample.embedding_sha256,
            ),
        )

    @staticmethod
    def _sample_from_row(row: sqlite3.Row) -> FaceSample:
        """把 SQLite 行恢复为样本对象并校验特征维度。"""
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).copy()
        if embedding.size != row["embedding_dim"]:
            raise ValueError(f"stored embedding {row['id']} has an invalid dimension")
        return FaceSample(
            id=row["id"],
            person_id=row["person_id"],
            image_path=row["image_path"],
            embedding=embedding,
            pose=row["pose"],
            quality=json.loads(row["quality_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            source_type=row["source_type"],
            image_sha256=row["image_sha256"],
            embedding_sha256=row["embedding_sha256"],
        )

    @staticmethod
    def _person_from_row(row: sqlite3.Row) -> Person:
        """把 SQLite 行恢复为人员对象。"""
        return Person(
            id=row["id"],
            display_name=row["display_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @contextmanager
    def _write_transaction(self) -> Generator[None, None, None]:
        """以 `BEGIN IMMEDIATE` 提供单写事务，并在异常时回滚。"""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()


def _now() -> datetime:
    """返回当前 UTC 时间，供数据库记录使用。"""
    return datetime.now(UTC)
