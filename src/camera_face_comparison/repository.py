"""通过 SQLite 持久化人员、样本特征和识别日志。"""

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

from .domain import FaceSample, Person, SampleMetadata
from .errors import LibraryUnavailableError


@dataclass(frozen=True)
class SampleInput:
    """录入流程为一次 SQLite 事务准备的一条样本数据。"""

    image_path: str
    embedding: np.ndarray
    quality_metrics: dict[str, float]
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
        self.transaction_uncertain = False
        self._faces_root = database_path.resolve().parent / "faces"
        existed = database_path.exists()
        if (
            not existed
            and self._faces_root.exists()
            and any(path.is_file() for path in self._faces_root.rglob("*"))
        ):
            raise LibraryUnavailableError("数据库缺失，但标准库图片仍存在；请恢复数据库后重新检查")
        self._connection = sqlite3.connect(database_path, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        # 外键保护逻辑关系，WAL 允许界面读取与工作线程写日志并行，busy_timeout 吸收短暂写竞争。
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if existed:
                self._validate_current_schema()
            else:
                self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

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
        # 进入事务前先完成全部向量校验和哈希，避免持有 SQLite 写锁时做纯计算。
        prepared_samples = [
            self._make_sample(
                person_id=person.id,
                image_path=sample.image_path,
                embedding=sample.embedding,
                quality_metrics=sample.quality_metrics,
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
        # 先把所有输入转换为领域对象，任一向量无效时数据库不会开始部分写入。
        prepared_samples = [
            self._make_sample(
                person_id=person_id,
                image_path=sample.image_path,
                embedding=sample.embedding,
                quality_metrics=sample.quality_metrics,
                source_type=sample.source_type,
                image_sha256=sample.image_sha256,
            )
            for sample in samples
        ]
        with self._write_transaction():
            for sample in prepared_samples:
                self._insert_sample(sample)
        return prepared_samples

    def rename_person(self, person_id: str, display_name: str) -> Person:
        """修改已有人员的显示姓名，并保留人员编号和创建时间。

        参数：
            person_id：需要修改的人员编号。
            display_name：去除首尾空白后的新显示姓名。
        返回：
            修改后的人员对象。
        前置条件：
            人员必须存在，新姓名不能为空且不能与其他人员重名。
        """

        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("人员姓名不能为空")
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT id, display_name, created_at FROM persons WHERE id = ?",
                (person_id,),
            ).fetchone()
            if row is None:
                raise ValueError("人员不存在")
            try:
                self._connection.execute(
                    "UPDATE persons SET display_name = ? WHERE id = ?",
                    (normalized_name, person_id),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("该姓名已经存在") from error
        return Person(
            id=str(row["id"]),
            display_name=normalized_name,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def delete_sample(self, person_id: str, sample_id: str) -> SampleMetadata:
        """删除指定人员的一张样本，但禁止删除该人员的最后一张样本。

        参数：
            person_id：样本所属人员编号。
            sample_id：待删除样本编号。
        返回：
            已从数据库删除的样本，供调用方继续清理对应图片文件。
        前置条件：
            样本必须属于指定人员，并且该人员至少有两张样本。
        """

        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT id, person_id, image_path, embedding_blob, embedding_dim,
                       quality_metrics_json, created_at, source_type, image_sha256,
                       embedding_sha256
                FROM face_samples
                WHERE person_id = ?
                ORDER BY created_at
                """,
                (person_id,),
            ).fetchall()
            selected = next((row for row in rows if row["id"] == sample_id), None)
            if selected is None:
                raise ValueError("样本不存在或不属于所选人员")
            if len(rows) <= 1:
                raise ValueError("不能删除人员的最后一张样本，请删除整个人员")
            removed = self._metadata_from_row(selected)
            self._validate_image_path(removed.image_path)
            self._connection.execute(
                "DELETE FROM face_samples WHERE id = ? AND person_id = ?",
                (sample_id, person_id),
            )
        return removed

    def delete_person(self, person_id: str) -> list[SampleMetadata]:
        """删除人员及其全部样本，并保留已与人员解绑的识别历史。

        参数：
            person_id：待删除人员编号。
        返回：
            已删除的样本列表，供调用方清理对应图片文件。
        前置条件：
            人员必须存在。
        """

        with self._write_transaction():
            exists = self._connection.execute(
                "SELECT 1 FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            if exists is None:
                raise ValueError("人员不存在")
            rows = self._connection.execute(
                """
                SELECT id, person_id, image_path, embedding_blob, embedding_dim,
                       quality_metrics_json, created_at, source_type, image_sha256,
                       embedding_sha256
                FROM face_samples
                WHERE person_id = ?
                ORDER BY created_at
                """,
                (person_id,),
            ).fetchall()
            removed_samples = [self._metadata_from_row(row) for row in rows]
            for sample in removed_samples:
                self._validate_image_path(sample.image_path)
            # 样本外键使用 RESTRICT，因此必须先删子表；日志外键会在删人员时自动置空。
            self._connection.execute("DELETE FROM face_samples WHERE person_id = ?", (person_id,))
            self._connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return removed_samples

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
                   quality_metrics_json, created_at, source_type, image_sha256,
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

    def list_sample_metadata(self, person_id: str | None = None) -> list[SampleMetadata]:
        """只读取展示和安全删除所需字段，不恢复损坏的向量。"""
        query = "SELECT id, person_id, image_path, source_type FROM face_samples"
        parameters = () if person_id is None else (person_id,)
        if person_id is not None:
            query += " WHERE person_id = ?"
        rows = self._connection.execute(query + " ORDER BY created_at", parameters)
        return [self._metadata_from_row(row) for row in rows]

    def read_library_rows(self):
        """在同一个 SQLite 读事务内取得人员、样本和结构诊断。"""
        self._connection.execute("BEGIN")
        try:
            people = tuple(self._connection.execute("SELECT * FROM persons ORDER BY id"))
            samples = tuple(
                self._connection.execute("SELECT * FROM face_samples ORDER BY created_at")
            )
            return people, samples, self.sqlite_integrity_messages(), self.foreign_key_violations()
        finally:
            self._connection.rollback()

    def image_is_referenced(self, path: Path) -> bool:
        """确认清理目标是否仍被样本引用，防止共享文件误删。"""
        return any(
            (self._faces_root.parent / row[0]).resolve() == path.resolve()
            for row in self._connection.execute("SELECT image_path FROM face_samples")
        )

    def _validate_image_path(self, stored_path: str) -> Path:
        """在提交前解析并限制删除目标的受管范围。"""
        path = (self._faces_root.parent / stored_path).resolve()
        if not path.is_relative_to(self._faces_root) or path == self._faces_root:
            raise ValueError("样本图片必须位于受管图片目录内")
        return path

    @staticmethod
    def _metadata_from_row(row) -> SampleMetadata:
        """仅恢复样本编号、归属、路径和来源。"""
        if any(
            not isinstance(row[key], str) or not row[key].strip()
            for key in ("id", "person_id", "image_path")
        ):
            raise ValueError("样本编号、人员编号或图片路径无效")
        if row["source_type"] not in {"file", "camera", "dataset"}:
            raise ValueError("样本来源无效")
        return SampleMetadata(row["id"], row["person_id"], row["image_path"], row["source_type"])

    def record_recognition(
        self,
        *,
        decision: str,
        person_id: str | None,
        top_score: float | None,
        second_score: float | None,
        score_gap: float | None,
        acceptance_score: float | None,
        acceptance_rule: str,
        latency_ms: float,
        reason: str | None,
    ) -> None:
        """把一次识别判定和耗时写入识别日志表。"""
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO recognition_logs (
                    id, captured_at, decision, person_id, top_score,
                    second_score, score_gap, acceptance_score, acceptance_rule,
                    latency_ms, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    _now().isoformat(),
                    decision,
                    person_id,
                    top_score,
                    second_score,
                    score_gap,
                    acceptance_score,
                    acceptance_rule,
                    latency_ms,
                    reason,
                ),
            )

    def _initialize_schema(self) -> None:
        """创建当前版本的人员、样本和识别日志表。"""
        with self._connection:
            # CREATE IF NOT EXISTS 只负责全新数据库；随后仍需精确核对列，拒绝旧开发库继续运行。
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
                    quality_metrics_json TEXT NOT NULL,
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
                    second_score REAL,
                    score_gap REAL,
                    acceptance_score REAL,
                    acceptance_rule TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    reason TEXT
                );
                """
            )
        self._validate_current_schema()

    def _validate_current_schema(self) -> None:
        """拒绝旧表结构，要求开发者直接重建本地开发数据库。"""

        expected = {
            "persons": ("id", "display_name", "created_at"),
            "face_samples": (
                "id",
                "person_id",
                "image_path",
                "embedding_blob",
                "embedding_dim",
                "quality_metrics_json",
                "created_at",
                "source_type",
                "image_sha256",
                "embedding_sha256",
            ),
            "recognition_logs": (
                "id",
                "captured_at",
                "decision",
                "person_id",
                "top_score",
                "second_score",
                "score_gap",
                "acceptance_score",
                "acceptance_rule",
                "latency_ms",
                "reason",
            ),
        }
        for table, expected_columns in expected.items():
            actual = tuple(
                str(row["name"])
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            if actual != expected_columns:
                raise LibraryUnavailableError(
                    f"数据库表结构无效：{table}；请恢复正确数据后重新检查"
                )

    @staticmethod
    def _make_sample(
        *,
        person_id: str,
        image_path: str,
        embedding: np.ndarray,
        quality_metrics: dict[str, float],
        source_type: str,
        image_sha256: str | None,
    ) -> FaceSample:
        """校验向量并构造带哈希的样本领域对象。"""
        vector = np.asarray(embedding, dtype=np.float32)
        if (
            vector.ndim != 1
            or vector.size != 512
            or not np.isfinite(vector).all()
            or not np.isfinite(np.linalg.norm(vector))
            or np.linalg.norm(vector) == 0
        ):
            raise ValueError("人脸特征必须是有限、非零的 512 维向量")
        # 哈希基于最终入库的 float32 字节，完整性检查可直接复算而不受输入 dtype 影响。
        return FaceSample(
            id=str(uuid4()),
            person_id=person_id,
            image_path=image_path,
            embedding=vector,
            quality_metrics={str(key): float(value) for key, value in quality_metrics.items()},
            created_at=_now(),
            source_type=source_type,
            image_sha256=image_sha256,
            embedding_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
        )

    def _insert_sample(self, sample: FaceSample) -> None:
        """将一个样本对象写入已存在的 SQLite 事务。"""
        path = (self._faces_root.parent / sample.image_path).resolve()
        if not path.is_relative_to(self._faces_root) or path == self._faces_root:
            raise ValueError("样本图片必须位于受管图片目录内")
        normalized_path = path.relative_to(self._faces_root.parent).as_posix()
        if self._connection.execute(
            "SELECT 1 FROM face_samples WHERE image_path = ?", (normalized_path,)
        ).fetchone():
            raise ValueError("样本图片路径已经被引用")
        # 只在同一人员内部排除内容完全相同的图片。同一张图片属于不同人员时不在此层推断业务错误，
        # 相似但字节不同的照片也必须保留，因为它们可能包含有价值的角度和光照变化。
        if sample.image_sha256 is not None:
            repeated = self._connection.execute(
                """
                SELECT 1 FROM face_samples
                WHERE person_id = ? AND image_sha256 = ?
                LIMIT 1
                """,
                (sample.person_id, sample.image_sha256),
            ).fetchone()
            if repeated is not None:
                raise ValueError("该人员已经包含内容相同的图片")
        self._connection.execute(
            """
            INSERT INTO face_samples (
                id, person_id, image_path, embedding_blob, embedding_dim,
                quality_metrics_json, created_at, source_type, image_sha256,
                embedding_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.id,
                sample.person_id,
                normalized_path,
                sample.embedding.tobytes(),
                sample.embedding.size,
                json.dumps(sample.quality_metrics, ensure_ascii=False),
                sample.created_at.isoformat(),
                sample.source_type,
                sample.image_sha256,
                sample.embedding_sha256,
            ),
        )

    @staticmethod
    def _sample_from_row(row: sqlite3.Row) -> FaceSample:
        """把 SQLite 行恢复为样本对象并校验特征维度。"""
        # copy() 解除数组对 SQLite BLOB 缓冲区的只读视图依赖，返回值可安全独立持有。
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).copy()
        FaceRepository._metadata_from_row(row)
        if (
            embedding.size != 512
            or embedding.size != row["embedding_dim"]
            or not np.isfinite(embedding).all()
            or not np.isfinite(np.linalg.norm(embedding))
            or np.linalg.norm(embedding) == 0
        ):
            raise ValueError(f"stored embedding {row['id']} has an invalid dimension")
        return FaceSample(
            id=row["id"],
            person_id=row["person_id"],
            image_path=row["image_path"],
            embedding=embedding,
            quality_metrics={
                str(key): float(value)
                for key, value in json.loads(row["quality_metrics_json"]).items()
            },
            created_at=datetime.fromisoformat(row["created_at"]),
            source_type=row["source_type"],
            image_sha256=row["image_sha256"],
            embedding_sha256=row["embedding_sha256"],
        )

    @staticmethod
    def _person_from_row(row: sqlite3.Row) -> Person:
        """把 SQLite 行恢复为人员对象。"""
        if any(
            not isinstance(row[key], str) or not row[key].strip() for key in ("id", "display_name")
        ):
            raise ValueError("人员编号或姓名无效")
        return Person(
            id=row["id"],
            display_name=row["display_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @contextmanager
    def _write_transaction(self) -> Generator[None, None, None]:
        """以 `BEGIN IMMEDIATE` 提供单写事务，并在异常时回滚。"""
        # 提前取得保留写锁，使并发写冲突发生在事务入口，而不是多条语句执行到一半时。
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._connection.commit()
        except Exception as error:
            try:
                self._connection.rollback()
            except sqlite3.Error as rollback_error:
                # 结果不明时隔离连接，并通知图片补偿保留现场，不能把潜在已提交图片删掉。
                self.transaction_uncertain = True
                error.library_state_uncertain = True
                error.add_note(f"回滚失败，已隔离连接：{rollback_error}")
                try:
                    self._connection.close()
                except sqlite3.Error as close_error:
                    error.add_note(f"隔离连接时也发生异常：{close_error}")
            raise


def _now() -> datetime:
    """返回当前 UTC 时间，供数据库记录使用。"""
    return datetime.now(UTC)
