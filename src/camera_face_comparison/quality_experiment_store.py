from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np

from .quality_degradation import DegradationSpec


@dataclass(frozen=True)
class QualityExperimentEntry:
    """一次图片退化条件下的原始模型测量或失败结果。"""

    status: Literal["observed", "failed"]
    embedding: np.ndarray | None
    metrics: dict[str, float] | None
    bbox: tuple[float, float, float, float] | None
    face_count: int
    latency_ms: float
    reason: str | None


class QualityExperimentStore:
    """保存与质量策略无关的 Phase 3 原始 embedding 和测量值。"""

    def __init__(self, path: Path, embedding_extraction_id: str) -> None:
        """打开当前实验缓存并创建不兼容旧版本的表结构。"""

        if not embedding_extraction_id:
            raise ValueError("embedding_extraction_id must not be empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_extraction_id = embedding_extraction_id
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_measurements (
                embedding_extraction_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                degradation_key TEXT NOT NULL,
                degradation_kind TEXT NOT NULL,
                degradation_level REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('observed', 'failed')),
                embedding_blob BLOB,
                embedding_dim INTEGER,
                metrics_json TEXT,
                bbox_json TEXT,
                face_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                reason TEXT,
                PRIMARY KEY (embedding_extraction_id, relative_path, degradation_key)
            )
            """
        )
        self._connection.commit()

    def get(
        self,
        relative_path: str,
        file_sha256: str,
        spec: DegradationSpec,
    ) -> QualityExperimentEntry | None:
        """读取同一图片内容、模型和退化条件下的可恢复原始结果。"""

        row = self._connection.execute(
            """
            SELECT file_sha256, status, embedding_blob, embedding_dim, metrics_json,
                   bbox_json, face_count, latency_ms, reason
            FROM quality_measurements
            WHERE embedding_extraction_id = ? AND relative_path = ? AND degradation_key = ?
            """,
            (self._embedding_extraction_id, relative_path, spec.key),
        ).fetchone()
        if row is None or row[0] != file_sha256:
            return None
        if row[1] == "failed":
            return QualityExperimentEntry(
                status="failed",
                embedding=None,
                metrics=None,
                bbox=None,
                face_count=int(row[6]),
                latency_ms=float(row[7]),
                reason=str(row[8]),
            )
        if row[1] != "observed" or row[2] is None or row[3] is None:
            raise RuntimeError(f"invalid quality experiment entry: {relative_path} {spec.key}")
        embedding = np.frombuffer(row[2], dtype=np.float32).copy()
        if embedding.size != int(row[3]):
            raise RuntimeError(f"invalid quality experiment embedding: {relative_path}")
        metrics_payload = json.loads(str(row[4]))
        bbox_payload = json.loads(str(row[5]))
        return QualityExperimentEntry(
            status="observed",
            embedding=embedding,
            metrics={str(key): float(value) for key, value in metrics_payload.items()},
            bbox=tuple(float(value) for value in bbox_payload),
            face_count=int(row[6]),
            latency_ms=float(row[7]),
            reason=None,
        )

    def put_observed(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        spec: DegradationSpec,
        embedding: np.ndarray,
        metrics: dict[str, float],
        bbox: tuple[float, float, float, float],
        face_count: int,
        latency_ms: float,
    ) -> None:
        """缓存成功检测到主脸的 embedding 和全部原始质量指标。"""

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            spec=spec,
            status="observed",
            embedding_blob=vector.tobytes(),
            embedding_dim=int(vector.size),
            metrics_json=json.dumps(metrics, sort_keys=True),
            bbox_json=json.dumps(bbox),
            face_count=face_count,
            latency_ms=latency_ms,
            reason=None,
        )

    def put_failed(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        spec: DegradationSpec,
        face_count: int,
        latency_ms: float,
        reason: str,
    ) -> None:
        """缓存模型未产生可用主脸的 FTE 原因。"""

        if not reason:
            raise ValueError("failure reason must not be empty")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            spec=spec,
            status="failed",
            embedding_blob=None,
            embedding_dim=None,
            metrics_json=None,
            bbox_json=None,
            face_count=face_count,
            latency_ms=latency_ms,
            reason=reason,
        )

    def commit(self) -> None:
        """提交当前批次写入，供中断后恢复。"""

        self._connection.commit()

    def close(self) -> None:
        """提交并关闭 SQLite 连接。"""

        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> Self:
        """返回上下文管理器中的当前缓存。"""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """异常时回滚未提交批次，然后关闭连接。"""

        if exc_type is not None:
            self._connection.rollback()
        self.close()

    def _upsert(
        self,
        *,
        relative_path: str,
        file_sha256: str,
        spec: DegradationSpec,
        status: str,
        embedding_blob: bytes | None,
        embedding_dim: int | None,
        metrics_json: str | None,
        bbox_json: str | None,
        face_count: int,
        latency_ms: float,
        reason: str | None,
    ) -> None:
        """插入或覆盖同一模型、图片路径和退化条件的结果。"""

        self._connection.execute(
            """
            INSERT INTO quality_measurements (
                embedding_extraction_id, relative_path, file_sha256, degradation_key,
                degradation_kind, degradation_level, status, embedding_blob,
                embedding_dim, metrics_json, bbox_json, face_count, latency_ms, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(embedding_extraction_id, relative_path, degradation_key) DO UPDATE SET
                file_sha256 = excluded.file_sha256,
                degradation_kind = excluded.degradation_kind,
                degradation_level = excluded.degradation_level,
                status = excluded.status,
                embedding_blob = excluded.embedding_blob,
                embedding_dim = excluded.embedding_dim,
                metrics_json = excluded.metrics_json,
                bbox_json = excluded.bbox_json,
                face_count = excluded.face_count,
                latency_ms = excluded.latency_ms,
                reason = excluded.reason
            """,
            (
                self._embedding_extraction_id,
                relative_path,
                file_sha256,
                spec.key,
                spec.kind,
                spec.level,
                status,
                embedding_blob,
                embedding_dim,
                metrics_json,
                bbox_json,
                face_count,
                latency_ms,
                reason,
            ),
        )
