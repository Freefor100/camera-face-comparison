from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .evaluation_cache import file_sha256
from .lfw_dataset import LfwProbe, LfwProtocol
from .raw_embedding_cache import RawEmbeddingCache


@dataclass(frozen=True)
class QmulPressureExportSummary:
    """QMUL 无阈值压力分数导出的完整覆盖统计。"""

    gallery_identity_total: int
    gallery_valid_identity_total: int
    gallery_image_total: int
    gallery_valid_image_total: int
    gallery_fte_total: int
    mated_probe_total: int
    mated_valid_probe_total: int
    mated_model_fte_total: int
    mated_gallery_unavailable_total: int
    unmated_probe_total: int
    unmated_valid_probe_total: int
    unmated_model_fte_total: int
    probe_fte_total: int
    score_record_total: int


@dataclass(frozen=True)
class _ScoringProbe:
    """一张可参与 Mean Prototype 检索的 QMUL Probe。"""

    protocol: LfwProbe
    role: str
    embedding: np.ndarray
    quality_metrics: dict[str, float]


def export_qmul_pressure_scores(
    *,
    dataset_dir: Path,
    protocol: LfwProtocol,
    cache: RawEmbeddingCache,
    output_path: Path,
    run_id: str,
    protocol_sha256: str,
    embedding_extraction_id: str,
    batch_size: int = 256,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> QmulPressureExportSummary:
    """从完整原始缓存导出 QMUL Mean Prototype 无阈值候选分数。

    参数：
        dataset_dir：`Face_Identification_Test_Set` 图片根目录。
        protocol：由官方 MAT 标签建立的 Gallery、Mated 和 Unmated 协议。
        cache：不含质量门和判定参数的完整原始 embedding 缓存。
        output_path：原子生成的无阈值 SQLite 文件。
        run_id：本次运行的稳定编号。
        protocol_sha256：确定性协议内容摘要。
        embedding_extraction_id：缓存对应的模型提取版本。
        batch_size：每次与全部身份原型做矩阵乘法的 Probe 数量。
        on_progress：可选的低频阶段进度回调。
    返回：
        Gallery、两类 Probe、FTE 和分数记录的实际覆盖数量。
    前置条件：
        协议中的每张图片都必须存在并已写入缓存；本函数不会初始化模型，
        也不会使用或搜索匹配阈值。
    """

    if not run_id or not protocol_sha256 or not embedding_extraction_id:
        raise ValueError("run identifiers must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    (
        person_ids,
        prototypes,
        gallery_valid_images,
        gallery_rejections,
    ) = _build_gallery_prototypes(dataset_dir, protocol.enrollment, cache, on_progress)
    if len(person_ids) < 2:
        raise ValueError("QMUL pressure evaluation requires at least two valid identities")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)
    try:
        _create_schema(connection)
        connection.executemany(
            """
            INSERT INTO rejections(role, relative_path, expected_person_id, reason)
            VALUES ('gallery', ?, ?, ?)
            """,
            gallery_rejections,
        )
        probe_counts = _write_probe_scores(
            connection=connection,
            dataset_dir=dataset_dir,
            probes=protocol.probes,
            cache=cache,
            person_ids=person_ids,
            prototypes=prototypes,
            batch_size=batch_size,
            on_progress=on_progress,
        )
        gallery_image_total = sum(len(paths) for paths in protocol.enrollment.values())
        summary = QmulPressureExportSummary(
            gallery_identity_total=len(protocol.enrollment),
            gallery_valid_identity_total=len(person_ids),
            gallery_image_total=gallery_image_total,
            gallery_valid_image_total=gallery_valid_images,
            gallery_fte_total=len(gallery_rejections),
            mated_probe_total=probe_counts["mated_total"],
            mated_valid_probe_total=probe_counts["mated_valid"],
            mated_model_fte_total=probe_counts["mated_fte"],
            mated_gallery_unavailable_total=probe_counts["mated_gallery_unavailable"],
            unmated_probe_total=probe_counts["unmated_total"],
            unmated_valid_probe_total=probe_counts["unmated_valid"],
            unmated_model_fte_total=probe_counts["unmated_fte"],
            probe_fte_total=probe_counts["mated_fte"] + probe_counts["unmated_fte"],
            score_record_total=probe_counts["mated_valid"] + probe_counts["unmated_valid"],
        )
        connection.execute(
            """
            INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                protocol_sha256,
                embedding_extraction_id,
                datetime.now(UTC).isoformat(),
                summary.gallery_identity_total,
                summary.gallery_valid_identity_total,
                summary.gallery_image_total,
                summary.gallery_valid_image_total,
                summary.gallery_fte_total,
                summary.mated_probe_total,
                summary.mated_valid_probe_total,
                summary.mated_model_fte_total,
                summary.mated_gallery_unavailable_total,
                summary.unmated_probe_total,
                summary.unmated_valid_probe_total,
                summary.unmated_model_fte_total,
                summary.score_record_total,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    connection.close()
    temporary_path.replace(output_path)
    return summary


def summarize_qmul_pressure_scores(
    path: Path,
    *,
    run_id: str,
    transferred_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """汇总 QMUL 排序、FTE、分数分布及可选 LFW 工作点迁移表现。"""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"QMUL pressure run does not exist: {run_id}")
        columns = tuple(item[1] for item in connection.execute("PRAGMA table_info(runs)"))
        run_data = dict(zip(columns, run, strict=True))
        mated_rows = connection.execute(
            """
            SELECT top_score, score_gap, top_is_correct, scoring_latency_ms
            FROM probe_scores WHERE role = 'mated' ORDER BY relative_path
            """
        ).fetchall()
        unmated_rows = connection.execute(
            """
            SELECT top_score, score_gap, top_is_correct, scoring_latency_ms
            FROM probe_scores WHERE role = 'unmated' ORDER BY relative_path
            """
        ).fetchall()
        reason_rows = connection.execute(
            "SELECT reason, COUNT(*) FROM rejections GROUP BY reason ORDER BY reason"
        ).fetchall()
    finally:
        connection.close()

    mated_rank1 = sum(int(row[2]) for row in mated_rows)
    mated_valid = int(run_data["mated_valid_probe_total"])
    mated_total = int(run_data["mated_probe_total"])
    unmated_valid = int(run_data["unmated_valid_probe_total"])
    report: dict[str, object] = {
        "artifact": "qmul-survface-mean-prototype-pressure-summary-v1",
        "run_id": run_id,
        "protocol_sha256": str(run_data["protocol_sha256"]),
        "embedding_extraction_id": str(run_data["embedding_extraction_id"]),
        "aggregation_method": "mean_prototype",
        "quality_policy": None,
        "gallery": {
            "identity_total": int(run_data["gallery_identity_total"]),
            "valid_identity_total": int(run_data["gallery_valid_identity_total"]),
            "image_total": int(run_data["gallery_image_total"]),
            "valid_image_total": int(run_data["gallery_valid_image_total"]),
            "model_fte": int(run_data["gallery_fte_total"]),
        },
        "mated": {
            "probe_total": mated_total,
            "valid_scored_probe_total": mated_valid,
            "model_fte": int(run_data["mated_model_fte_total"]),
            "gallery_identity_unavailable": int(
                run_data["mated_gallery_unavailable_total"]
            ),
            "rank1_correct": mated_rank1,
            "valid_rank1_rate": mated_rank1 / mated_valid if mated_valid else None,
            "protocol_rank1_rate": mated_rank1 / mated_total if mated_total else None,
            "top_score_distribution": _distribution([float(row[0]) for row in mated_rows]),
            "score_gap_distribution": _distribution([float(row[1]) for row in mated_rows]),
            "average_scoring_latency_ms": _mean([float(row[3]) for row in mated_rows]),
        },
        "unmated": {
            "probe_total": int(run_data["unmated_probe_total"]),
            "valid_scored_probe_total": unmated_valid,
            "model_fte": int(run_data["unmated_model_fte_total"]),
            "top_score_distribution": _distribution([float(row[0]) for row in unmated_rows]),
            "score_gap_distribution": _distribution([float(row[1]) for row in unmated_rows]),
            "average_scoring_latency_ms": _mean([float(row[3]) for row in unmated_rows]),
        },
        "failure_reasons": {str(reason): int(count) for reason, count in reason_rows},
    }
    if transferred_policy is not None:
        report["transferred_policy"] = _evaluate_transferred_policy(
            mated_rows=mated_rows,
            unmated_rows=unmated_rows,
            mated_protocol_total=mated_total,
            unmated_protocol_total=int(run_data["unmated_probe_total"]),
            policy=transferred_policy,
        )
    return report


def _build_gallery_prototypes(
    dataset_dir: Path,
    enrollment: Mapping[str, Sequence[str]],
    cache: RawEmbeddingCache,
    on_progress: Callable[[str, int, int], None] | None,
) -> tuple[tuple[str, ...], np.ndarray, int, list[tuple[str, str, str]]]:
    """按身份计算归一化均值原型，并记录 Gallery 模型 FTE。"""

    person_ids: list[str] = []
    prototypes: list[np.ndarray] = []
    rejections: list[tuple[str, str, str]] = []
    valid_images = 0
    done = 0
    total = sum(len(paths) for paths in enrollment.values())
    for person_id in sorted(enrollment):
        embeddings: list[np.ndarray] = []
        for relative_path in enrollment[person_id]:
            entry = _required_cache_entry(dataset_dir, relative_path, cache)
            if entry.status == "failed":
                rejections.append((relative_path, person_id, entry.reason or "model_fte"))
            else:
                if entry.embedding is None:
                    raise RuntimeError(f"cached embedding is missing: {relative_path}")
                embeddings.append(_normalize(entry.embedding))
                valid_images += 1
            done += 1
            if on_progress is not None:
                on_progress("gallery-cache", done, total)
        if embeddings:
            person_ids.append(person_id)
            prototypes.append(_normalize(np.mean(embeddings, axis=0)))
    return tuple(person_ids), np.stack(prototypes).astype(np.float32), valid_images, rejections


def _write_probe_scores(
    *,
    connection: sqlite3.Connection,
    dataset_dir: Path,
    probes: Sequence[LfwProbe],
    cache: RawEmbeddingCache,
    person_ids: tuple[str, ...],
    prototypes: np.ndarray,
    batch_size: int,
    on_progress: Callable[[str, int, int], None] | None,
) -> Counter[str]:
    """流式读取两类 Probe，分批打分并写入无阈值第一、第二候选。"""

    counts: Counter[str] = Counter()
    person_id_set = set(person_ids)
    batch: list[_ScoringProbe] = []
    for index, probe in enumerate(probes, start=1):
        role = "mated" if probe.expected_person_id is not None else "unmated"
        counts[f"{role}_total"] += 1
        entry = _required_cache_entry(dataset_dir, probe.relative_path, cache)
        if entry.status == "failed":
            counts[f"{role}_fte"] += 1
            connection.execute(
                "INSERT INTO rejections VALUES (?, ?, ?, ?)",
                (role, probe.relative_path, probe.expected_person_id, entry.reason or "model_fte"),
            )
        elif probe.expected_person_id is not None and probe.expected_person_id not in person_id_set:
            counts["mated_gallery_unavailable"] += 1
            connection.execute(
                "INSERT INTO rejections VALUES (?, ?, ?, ?)",
                (role, probe.relative_path, probe.expected_person_id, "gallery_identity_unavailable"),
            )
        else:
            if entry.embedding is None or entry.metrics is None:
                raise RuntimeError(f"cached probe data is incomplete: {probe.relative_path}")
            batch.append(
                _ScoringProbe(
                    protocol=probe,
                    role=role,
                    embedding=_normalize(entry.embedding),
                    quality_metrics=entry.metrics,
                )
            )
            counts[f"{role}_valid"] += 1
            if len(batch) >= batch_size:
                _flush_score_batch(connection, batch, person_ids, prototypes)
                batch.clear()
        if index % 10000 == 0:
            connection.commit()
        if on_progress is not None:
            on_progress("probe-cache-score", index, len(probes))
    if batch:
        _flush_score_batch(connection, batch, person_ids, prototypes)
    connection.commit()
    return counts


def _flush_score_batch(
    connection: sqlite3.Connection,
    batch: Sequence[_ScoringProbe],
    person_ids: tuple[str, ...],
    prototypes: np.ndarray,
) -> None:
    """把一批 Query 与全部身份均值原型相乘，并保存 Top-2 连续分数。"""

    started_at = perf_counter()
    query_matrix = np.stack([probe.embedding for probe in batch]).astype(np.float32)
    scores = query_matrix @ prototypes.T
    first_indices = np.argmax(scores, axis=1)
    without_first = scores.copy()
    without_first[np.arange(len(batch)), first_indices] = -np.inf
    second_indices = np.argmax(without_first, axis=1)
    latency_ms = (perf_counter() - started_at) * 1000 / len(batch)
    rows = []
    for index, probe in enumerate(batch):
        first = int(first_indices[index])
        second = int(second_indices[index])
        top_score = float(scores[index, first])
        second_score = float(scores[index, second])
        top_person_id = person_ids[first]
        rows.append(
            (
                probe.role,
                probe.protocol.relative_path,
                probe.protocol.expected_person_id,
                top_person_id,
                top_score,
                person_ids[second],
                second_score,
                max(0.0, top_score - second_score),
                int(
                    probe.protocol.expected_person_id is not None
                    and top_person_id == probe.protocol.expected_person_id
                ),
                json.dumps(probe.quality_metrics, ensure_ascii=False, sort_keys=True),
                latency_ms,
            )
        )
    connection.executemany("INSERT INTO probe_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _evaluate_transferred_policy(
    *,
    mated_rows: Sequence[tuple[object, ...]],
    unmated_rows: Sequence[tuple[object, ...]],
    mated_protocol_total: int,
    unmated_protocol_total: int,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """机械应用外部冻结工作点；结果只描述跨域迁移，不在 QMUL 调参。"""

    if policy.get("method") != "mean_prototype":
        raise ValueError("QMUL pressure evaluation only supports a Mean Prototype policy")
    threshold = float(policy["match_threshold"])
    use_score_gap = bool(policy["use_score_gap"])
    minimum_gap = float(policy["min_score_gap"])

    def accepted(row: tuple[object, ...]) -> bool:
        """判断一条连续分数是否通过给定外部规则。"""

        return float(row[0]) >= threshold and (
            not use_score_gap or float(row[1]) >= minimum_gap
        )

    known_true_accepts = sum(accepted(row) and bool(row[2]) for row in mated_rows)
    known_wrong_accepts = sum(accepted(row) and not bool(row[2]) for row in mated_rows)
    unknown_false_accepts = sum(accepted(row) for row in unmated_rows)
    return {
        "transfer_only": True,
        "source": str(policy.get("source", "external")),
        "method": "mean_prototype",
        "match_threshold": threshold,
        "use_score_gap": use_score_gap,
        "min_score_gap": minimum_gap,
        "known_true_accepts": known_true_accepts,
        "known_wrong_accepts": known_wrong_accepts,
        "unknown_false_accepts": unknown_false_accepts,
        "valid_tpir": known_true_accepts / len(mated_rows) if mated_rows else None,
        "protocol_tpir": (
            known_true_accepts / mated_protocol_total if mated_protocol_total else None
        ),
        "valid_fpir": unknown_false_accepts / len(unmated_rows) if unmated_rows else None,
        "protocol_fpir": (
            unknown_false_accepts / unmated_protocol_total if unmated_protocol_total else None
        ),
    }


def _required_cache_entry(dataset_dir: Path, relative_path: str, cache: RawEmbeddingCache):
    """使用图片内容哈希读取缓存；缺失或图片变化时拒绝继续统计。"""

    image_path = dataset_dir / relative_path
    if not image_path.is_file():
        raise FileNotFoundError(f"protocol image does not exist: {image_path}")
    entry = cache.get(relative_path, file_sha256(image_path))
    if entry is None:
        raise RuntimeError(f"embedding cache is incomplete or stale: {relative_path}")
    return entry


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """把缓存向量恢复成单位向量，拒绝空向量和零向量。"""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """用稳定分位点表示一组连续分数分布。"""

    if not values:
        return {"minimum": None, "p05": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _mean(values: Sequence[float]) -> float | None:
    """返回非空序列均值，空序列返回空值。"""

    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _create_schema(connection: sqlite3.Connection) -> None:
    """创建 QMUL 无阈值分数、运行元数据和拒绝记录表。"""

    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            protocol_sha256 TEXT NOT NULL,
            embedding_extraction_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            gallery_identity_total INTEGER NOT NULL,
            gallery_valid_identity_total INTEGER NOT NULL,
            gallery_image_total INTEGER NOT NULL,
            gallery_valid_image_total INTEGER NOT NULL,
            gallery_fte_total INTEGER NOT NULL,
            mated_probe_total INTEGER NOT NULL,
            mated_valid_probe_total INTEGER NOT NULL,
            mated_model_fte_total INTEGER NOT NULL,
            mated_gallery_unavailable_total INTEGER NOT NULL,
            unmated_probe_total INTEGER NOT NULL,
            unmated_valid_probe_total INTEGER NOT NULL,
            unmated_model_fte_total INTEGER NOT NULL,
            score_record_total INTEGER NOT NULL
        );

        CREATE TABLE probe_scores (
            role TEXT NOT NULL CHECK(role IN ('mated', 'unmated')),
            relative_path TEXT PRIMARY KEY,
            expected_person_id TEXT,
            top_person_id TEXT NOT NULL,
            top_score REAL NOT NULL,
            second_person_id TEXT NOT NULL,
            second_score REAL NOT NULL,
            score_gap REAL NOT NULL,
            top_is_correct INTEGER NOT NULL CHECK(top_is_correct IN (0, 1)),
            probe_quality_json TEXT NOT NULL,
            scoring_latency_ms REAL NOT NULL
        );

        CREATE INDEX probe_scores_role ON probe_scores(role);

        CREATE TABLE rejections (
            role TEXT NOT NULL CHECK(role IN ('gallery', 'mated', 'unmated')),
            relative_path TEXT PRIMARY KEY,
            expected_person_id TEXT,
            reason TEXT NOT NULL
        );
        """
    )
