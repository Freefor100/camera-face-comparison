from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np


@dataclass(frozen=True)
class RawEmbeddingEntry:
    """一张数据集图片的模型原始成功结果或 FTE。"""

    status: Literal["observed", "failed"]
    embedding: np.ndarray | None
    metrics: dict[str, float] | None
    face_count: int
    latency_ms: float
    reason: str | None


class RawEmbeddingCache:
    """保存完全不依赖质量门和识别规则的数据集 embedding。"""

    def __init__(self, path: Path, dataset_id: str, embedding_extraction_id: str) -> None:
        """打开原始缓存，并创建只以数据集、提取版本和图片路径为键的表。"""

        if not dataset_id or not embedding_extraction_id:
            raise ValueError("raw cache identifiers must not be empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_id = dataset_id
        self._embedding_extraction_id = embedding_extraction_id
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_embeddings (
                dataset_id TEXT NOT NULL,
                embedding_extraction_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('observed', 'failed')),
                embedding_blob BLOB,
                embedding_dim INTEGER,
                metrics_json TEXT,
                face_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                reason TEXT,
                PRIMARY KEY (dataset_id, embedding_extraction_id, relative_path)
            )
            """
        )
        self._connection.commit()

    def get(self, relative_path: str, file_sha256: str) -> RawEmbeddingEntry | None:
        """读取内容哈希一致的原始模型结果；图片变化时返回未命中。"""

        row = self._connection.execute(
            """
            SELECT file_sha256, status, embedding_blob, embedding_dim, metrics_json,
                   face_count, latency_ms, reason
            FROM raw_embeddings
            WHERE dataset_id = ? AND embedding_extraction_id = ? AND relative_path = ?
            """,
            (self._dataset_id, self._embedding_extraction_id, relative_path),
        ).fetchone()
        if row is None or row[0] != file_sha256:
            return None
        if row[1] == "failed":
            return RawEmbeddingEntry(
                status="failed",
                embedding=None,
                metrics=None,
                face_count=int(row[5]),
                latency_ms=float(row[6]),
                reason=str(row[7]),
            )
        if row[1] != "observed" or row[2] is None or row[3] is None or row[4] is None:
            raise RuntimeError(f"invalid raw embedding cache entry: {relative_path}")
        embedding = np.frombuffer(row[2], dtype=np.float32).copy()
        if embedding.size != int(row[3]):
            raise RuntimeError(f"invalid raw embedding dimension: {relative_path}")
        metrics_payload = json.loads(str(row[4]))
        return RawEmbeddingEntry(
            status="observed",
            embedding=embedding,
            metrics={str(key): float(value) for key, value in metrics_payload.items()},
            face_count=int(row[5]),
            latency_ms=float(row[6]),
            reason=None,
        )

    def put_observed(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        embedding: np.ndarray,
        metrics: dict[str, float],
        face_count: int,
        latency_ms: float,
    ) -> None:
        """缓存主体脸 embedding 和原始质量测量，不执行任何质量判定。"""

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            status="observed",
            embedding_blob=vector.tobytes(),
            embedding_dim=int(vector.size),
            metrics_json=json.dumps(metrics, sort_keys=True),
            face_count=face_count,
            latency_ms=latency_ms,
            reason=None,
        )

    def put_failed(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        face_count: int,
        latency_ms: float,
        reason: str,
    ) -> None:
        """缓存模型未产生主体脸的 FTE，避免重启后反复推理。"""

        if not reason:
            raise ValueError("raw extraction failure reason must not be empty")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            status="failed",
            embedding_blob=None,
            embedding_dim=None,
            metrics_json=None,
            face_count=face_count,
            latency_ms=latency_ms,
            reason=reason,
        )

    def commit(self) -> None:
        """提交当前批次结果，支持长实验中断恢复。"""

        self._connection.commit()

    def close(self) -> None:
        """提交并关闭缓存连接。"""

        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> Self:
        """返回上下文管理器中的当前缓存。"""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """异常退出时回滚未提交批次，随后关闭连接。"""

        if exc_type is not None:
            self._connection.rollback()
        self.close()

    def _upsert(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        status: str,
        embedding_blob: bytes | None,
        embedding_dim: int | None,
        metrics_json: str | None,
        face_count: int,
        latency_ms: float,
        reason: str | None,
    ) -> None:
        """原子插入或覆盖同一提取版本下的一张图片结果。"""

        self._connection.execute(
            """
            INSERT INTO raw_embeddings (
                dataset_id, embedding_extraction_id, relative_path, file_sha256,
                status, embedding_blob, embedding_dim, metrics_json,
                face_count, latency_ms, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, embedding_extraction_id, relative_path) DO UPDATE SET
                file_sha256 = excluded.file_sha256,
                status = excluded.status,
                embedding_blob = excluded.embedding_blob,
                embedding_dim = excluded.embedding_dim,
                metrics_json = excluded.metrics_json,
                face_count = excluded.face_count,
                latency_ms = excluded.latency_ms,
                reason = excluded.reason
            """,
            (
                self._dataset_id,
                self._embedding_extraction_id,
                relative_path,
                file_sha256,
                status,
                embedding_blob,
                embedding_dim,
                metrics_json,
                face_count,
                latency_ms,
                reason,
            ),
        )


def available_raw_extraction_ids(path: Path, dataset_id: str) -> tuple[str, ...]:
    """列出现有原始缓存中指定数据集的模型提取版本。"""

    if not path.is_file():
        return ()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT DISTINCT embedding_extraction_id
            FROM raw_embeddings
            WHERE dataset_id = ?
            ORDER BY embedding_extraction_id
            """,
            (dataset_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return ()
    finally:
        connection.close()
    return tuple(str(row[0]) for row in rows)
