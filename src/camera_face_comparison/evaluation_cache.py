from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np

from .config import Settings
from .image_input import QualityProfile


@dataclass(frozen=True)
class CacheEntry:
    """一张评测图片在缓存中的有效特征或拒绝结果。"""

    status: Literal["valid", "rejected"]
    embedding: np.ndarray | None
    quality: QualityProfile | None
    reason: str | None


class EvaluationEmbeddingCache:
    """使用 SQLite 保存经过指定质量策略筛选的可恢复评测结果。"""

    def __init__(
        self,
        path: Path,
        dataset_id: str,
        embedding_extraction_id: str,
        quality_policy_id: str,
    ) -> None:
        """打开评测缓存并建立当前版本的表结构。

        参数：
            path：评测缓存数据库路径。
            dataset_id：数据集和协议的稳定标识。
            embedding_extraction_id：模型、检测、对齐和特征提取的稳定标识。
            quality_policy_id：质量硬门和启发式分层规则的稳定标识。
        返回：
            无；实例持有一个打开的 SQLite 连接。
        前置条件：
            父目录可创建，且数据集与特征提取标识非空。
        """

        if not dataset_id or not embedding_extraction_id or not quality_policy_id:
            raise ValueError("cache identifiers must not be empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_id = dataset_id
        self._embedding_extraction_id = embedding_extraction_id
        self._quality_policy_id = quality_policy_id
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_embeddings (
                dataset_id TEXT NOT NULL,
                embedding_extraction_id TEXT NOT NULL,
                quality_policy_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('valid', 'rejected')),
                embedding_blob BLOB,
                embedding_dim INTEGER,
                quality_json TEXT,
                reason TEXT,
                PRIMARY KEY (
                    dataset_id, embedding_extraction_id, quality_policy_id, relative_path
                )
            )
            """
        )
        self._connection.commit()

    def get(self, relative_path: str, file_sha256: str) -> CacheEntry | None:
        """按图片路径和 SHA-256 读取可复用的评测结果。

        参数：
            relative_path：相对于数据集根目录的图片路径。
            file_sha256：当前图片内容摘要。
        返回：
            命中时返回有效 embedding 或拒绝原因；未命中或摘要变化时返回 `None`。
        前置条件：
            当前实例仍处于打开状态。
        """

        row = self._connection.execute(
            """
            SELECT file_sha256, status, embedding_blob, embedding_dim, quality_json, reason
            FROM evaluation_embeddings
            WHERE dataset_id = ? AND embedding_extraction_id = ?
              AND quality_policy_id = ? AND relative_path = ?
            """,
            (
                self._dataset_id,
                self._embedding_extraction_id,
                self._quality_policy_id,
                relative_path,
            ),
        ).fetchone()
        if row is None or row[0] != file_sha256:
            return None
        status = row[1]
        if status == "rejected":
            return CacheEntry("rejected", None, None, row[5])
        if status != "valid" or row[2] is None or row[3] is None or row[4] is None:
            raise RuntimeError(f"invalid evaluation cache entry: {relative_path}")
        embedding = np.frombuffer(row[2], dtype=np.float32).copy()
        if embedding.ndim != 1 or embedding.size != row[3]:
            raise RuntimeError(f"invalid cached embedding dimension: {relative_path}")
        quality = _quality_from_json(row[4])
        return CacheEntry("valid", embedding, quality, None)

    def put_valid(
        self,
        relative_path: str,
        file_sha256: str,
        embedding: np.ndarray,
        quality: QualityProfile,
    ) -> None:
        """写入一张图片的有效 embedding 和质量记录，等待显式提交。"""

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("cached embedding must be a non-empty one-dimensional vector")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            status="valid",
            embedding_blob=vector.tobytes(),
            embedding_dim=int(vector.size),
            quality_json=json.dumps(_quality_to_dict(quality), ensure_ascii=False),
            reason=None,
        )

    def put_rejected(self, relative_path: str, file_sha256: str, reason: str) -> None:
        """写入一张图片的拒绝原因，等待显式提交。"""

        if not reason:
            raise ValueError("cache rejection reason must not be empty")
        self._upsert(
            relative_path=relative_path,
            file_sha256=file_sha256,
            status="rejected",
            embedding_blob=None,
            embedding_dim=None,
            quality_json=None,
            reason=reason,
        )

    def commit(self) -> None:
        """提交当前批次缓存写入，使中断后的进程可以恢复。"""

        self._connection.commit()

    def close(self) -> None:
        """提交并关闭 SQLite 连接。"""

        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> Self:
        """返回可用于 with 语句的缓存实例。"""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """正常退出时提交，异常退出时回滚未提交批次后关闭。"""

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
        quality_json: str | None,
        reason: str | None,
    ) -> None:
        """插入或替换当前数据集/特征提取配置下的一条缓存记录。"""

        self._connection.execute(
            """
            INSERT INTO evaluation_embeddings (
                dataset_id, embedding_extraction_id, quality_policy_id, relative_path,
                file_sha256, status, embedding_blob, embedding_dim, quality_json, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                dataset_id, embedding_extraction_id, quality_policy_id, relative_path
            ) DO UPDATE SET
                file_sha256 = excluded.file_sha256,
                status = excluded.status,
                embedding_blob = excluded.embedding_blob,
                embedding_dim = excluded.embedding_dim,
                quality_json = excluded.quality_json,
                reason = excluded.reason
            """,
            (
                self._dataset_id,
                self._embedding_extraction_id,
                self._quality_policy_id,
                relative_path,
                file_sha256,
                status,
                embedding_blob,
                embedding_dim,
                quality_json,
                reason,
            ),
        )


def file_sha256(path: Path) -> str:
    """分块计算评测图片的 SHA-256，作为缓存内容校验键。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_extraction_id(settings: Settings, base_model: str = "buffalo_l") -> str:
    """生成只描述检测、对齐和特征提取实现的稳定标识。

    数值质量门、匹配阈值、候选分差和人员聚合参数都不会改变模型产生的
    embedding，因此不得进入此标识。
    """

    payload = {
        "base_model": base_model,
        "detector_input_size": [640, 640],
        "alignment": "insightface-default-five-point",
        "embedding_normalization": "l2",
    }
    return _identifier(base_model, payload)


def quality_policy_id(settings: Settings) -> str:
    """生成描述质量硬门和启发式分层规则的稳定标识。"""

    payload = {
        "min_detection_score": settings.min_detection_score,
        "min_face_size_px": settings.min_face_size_px,
        "min_blur_variance": settings.min_blur_variance,
        "min_brightness": settings.min_brightness,
        "max_brightness": settings.max_brightness,
        "min_contrast": settings.min_contrast,
        "high_quality_score": settings.high_quality_score,
        "medium_quality_score": settings.medium_quality_score,
    }
    return _identifier("quality", payload)


def decision_policy_id(
    settings: Settings,
    *,
    aggregation_method: str = "quality_weighted_top_k",
) -> str:
    """生成识别聚合和接收判定参数的稳定标识。"""

    payload = {
        "aggregation_method": aggregation_method,
        "match_threshold": settings.match_threshold,
        "min_score_gap": settings.min_score_gap,
        "top_k": settings.top_k,
        "quality_tiers": {
            name: {
                "match_threshold": policy.match_threshold,
                "min_score_gap": policy.min_score_gap,
            }
            for name, policy in sorted(settings.quality_tiers.items())
        },
    }
    return _identifier("decision", payload)


def available_cache_extraction_ids(
    path: Path,
    dataset_id: str,
    quality_policy_id: str,
) -> tuple[str, ...]:
    """列出一个现有缓存中指定数据集的特征提取标识。

    该函数只查询缓存，不创建数据库，供 cache-only 导出显式选择已经完成的
    embedding 批次。
    """

    if not path.is_file():
        return ()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT DISTINCT embedding_extraction_id FROM evaluation_embeddings "
            "WHERE dataset_id = ? AND quality_policy_id = ? ORDER BY embedding_extraction_id",
            (dataset_id, quality_policy_id),
        ).fetchall()
    except sqlite3.OperationalError:
        return ()
    finally:
        connection.close()
    return tuple(str(row[0]) for row in rows)


def _identifier(prefix: str, payload: object) -> str:
    """把规范化 JSON 内容转换为简短、可读且稳定的标识。"""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()[:16]}"


def write_json_atomic(path: Path, payload: object) -> None:
    """先写临时文件再原子替换评测报告，避免留下半截 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _quality_to_dict(profile: QualityProfile) -> dict[str, object]:
    """把质量对象转换成 JSON 可保存的字典。"""

    return {
        "tier": profile.tier,
        "score": profile.score,
        "metrics": profile.metrics,
        "reasons": list(profile.reasons),
    }


def _quality_from_json(payload: str) -> QualityProfile:
    """从缓存 JSON 恢复质量对象并检查基础字段。"""

    try:
        raw = json.loads(payload)
        tier = raw["tier"]
        score = float(raw["score"])
        metrics = {str(key): float(value) for key, value in raw["metrics"].items()}
        reasons = tuple(str(reason) for reason in raw["reasons"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid cached quality profile") from error
    if tier not in {"high", "medium", "reject"}:
        raise RuntimeError("invalid cached quality tier")
    return QualityProfile(tier, score, metrics, reasons)
