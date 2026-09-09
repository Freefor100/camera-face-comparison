from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .evaluation_cache import file_sha256
from .lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from .raw_embedding_cache import RawEmbeddingCache

METHOD_VARIANTS = (
    ("single", 0),
    ("max", 0),
    ("mean_prototype", 0),
    ("top_k_mean", 2),
    ("top_k_mean", 3),
    ("top_k_mean", 5),
)


@dataclass(frozen=True)
class DecisionScoreExportSummary:
    """一次无阈值 LFW 分数导出的覆盖数量。"""

    gallery_image_total: int
    gallery_valid_image_total: int
    gallery_rejection_total: int
    probe_total: int
    valid_probe_total: int
    probe_rejection_total: int
    score_record_total: int


@dataclass(frozen=True)
class _ValidProbe:
    """一张从缓存恢复的有效探针及其实验元数据。"""

    protocol: LfwSplitProbe
    embedding: np.ndarray
    quality_metrics: dict[str, float]


@dataclass(frozen=True)
class _GalleryMatrices:
    """按身份连续排列的 Gallery 样本矩阵和人员原型。"""

    person_ids: tuple[str, ...]
    sample_matrix: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    prototypes: np.ndarray


def summarize_decision_scores(
    path: Path,
    *,
    run_id: str,
    split: str,
) -> dict[str, object]:
    """汇总一个固定分区的覆盖、Rank-1 与无阈值得分分布。

    此汇总只描述第一候选排序和连续分数，不执行 Known/Unknown 接收判定。
    """

    if split not in {"calibration", "evaluation"}:
        raise ValueError(f"unsupported decision score split: {split}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        probe_counts = connection.execute(
            """
            SELECT
                COUNT(DISTINCT relative_path),
                COUNT(DISTINCT CASE WHEN expected_person_id IS NOT NULL THEN relative_path END),
                COUNT(DISTINCT CASE WHEN expected_person_id IS NULL THEN relative_path END)
            FROM decision_scores
            WHERE run_id = ? AND split = ?
            """,
            (run_id, split),
        ).fetchone()
        rejection_counts = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(expected_person_id IS NOT NULL),
                SUM(expected_person_id IS NULL)
            FROM rejections
            WHERE run_id = ? AND role = 'probe' AND split = ?
            """,
            (run_id, split),
        ).fetchone()
        variants: list[dict[str, object]] = []
        for method, top_k in METHOD_VARIANTS:
            rows = connection.execute(
                """
                SELECT expected_person_id, top_is_correct, top_score, score_gap,
                       scoring_latency_ms
                FROM decision_scores
                WHERE run_id = ? AND split = ? AND method = ? AND top_k = ?
                """,
                (run_id, split, method, top_k),
            ).fetchall()
            known_total = sum(row[0] is not None for row in rows)
            rank_one_correct = sum(int(row[1]) for row in rows)
            variants.append(
                {
                    "method": method,
                    "top_k": top_k,
                    "record_total": len(rows),
                    "known_probe_total": known_total,
                    "rank_one_correct": rank_one_correct,
                    "rank_one_identification_rate": (
                        rank_one_correct / known_total if known_total else None
                    ),
                    "top_score_distribution": _distribution([float(row[2]) for row in rows]),
                    "score_gap_distribution": _distribution([float(row[3]) for row in rows]),
                    "average_scoring_latency_ms": (
                        sum(float(row[4]) for row in rows) / len(rows) if rows else 0.0
                    ),
                }
            )
    finally:
        connection.close()
    return {
        "artifact": "lfw-threshold-free-split-summary-v1",
        "run_id": run_id,
        "split": split,
        "valid_probe_total": int(probe_counts[0]),
        "known_probe_total": int(probe_counts[1]),
        "unknown_probe_total": int(probe_counts[2]),
        "rejected_probe_total": int(rejection_counts[0]),
        "rejected_known_probe_total": int(rejection_counts[1] or 0),
        "rejected_unknown_probe_total": int(rejection_counts[2] or 0),
        "aggregation_variants": variants,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    """以少量稳定分位点描述分数分布，空输入返回空值。"""

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


def export_lfw_decision_scores(
    *,
    dataset_dir: Path,
    protocol: LfwSplitProtocol,
    cache: RawEmbeddingCache,
    output_path: Path,
    run_id: str,
    protocol_sha256: str,
    embedding_extraction_id: str,
    batch_size: int = 32,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> DecisionScoreExportSummary:
    """仅从现有 embedding 缓存导出六种人员聚合的无阈值结果。

    参数：
        dataset_dir：协议相对路径对应的 LFW 图片根目录。
        protocol：身份互斥的 Calibration/Evaluation 固定协议。
        cache：已完成、与质量策略无关的原始特征缓存。
        output_path：实验分数 SQLite 输出路径。
        run_id：本次导出的稳定运行标识。
        protocol_sha256：当前分区协议文件的内容摘要。
        embedding_extraction_id：缓存中实际选中的特征提取批次标识。
        batch_size：矩阵打分时每批探针数量。
        on_progress：可选的阶段进度回调。
    返回：
        Gallery、Probe、拒绝项和分数记录的实际覆盖统计。
    前置条件：
        所有协议图片必须存在且已在给定缓存中有有效或拒绝记录；本函数不会
        初始化人脸模型，也不会接受匹配阈值或候选分差。
    """

    if not run_id or not protocol_sha256 or not embedding_extraction_id:
        raise ValueError("run_id, protocol_sha256 and embedding_extraction_id must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    gallery_embeddings, gallery_rejections, gallery_image_total = _load_gallery(
        dataset_dir, protocol.enrollment, cache, on_progress
    )
    if not gallery_embeddings:
        raise ValueError("cached protocol contains no valid Gallery embeddings")
    gallery = _build_gallery_matrices(gallery_embeddings)
    valid_probes, probe_rejections = _load_probes(
        dataset_dir, protocol.probes, cache, on_progress
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        _create_schema(connection)
        _write_rejections(
            connection,
            run_id=run_id,
            gallery_rejections=gallery_rejections,
            probe_rejections=probe_rejections,
        )
        score_record_total = _write_score_batches(
            connection,
            run_id=run_id,
            probes=valid_probes,
            gallery=gallery,
            batch_size=batch_size,
            on_progress=on_progress,
        )
        summary = DecisionScoreExportSummary(
            gallery_image_total=gallery_image_total,
            gallery_valid_image_total=sum(len(values) for values in gallery_embeddings.values()),
            gallery_rejection_total=len(gallery_rejections),
            probe_total=len(protocol.probes),
            valid_probe_total=len(valid_probes),
            probe_rejection_total=len(probe_rejections),
            score_record_total=score_record_total,
        )
        connection.execute(
            """
            INSERT INTO runs (
                run_id, protocol_sha256, embedding_extraction_id, generated_at,
                gallery_image_total, gallery_valid_image_total, gallery_rejection_total,
                probe_total, valid_probe_total, probe_rejection_total, score_record_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                protocol_sha256,
                embedding_extraction_id,
                datetime.now(UTC).isoformat(),
                summary.gallery_image_total,
                summary.gallery_valid_image_total,
                summary.gallery_rejection_total,
                summary.probe_total,
                summary.valid_probe_total,
                summary.probe_rejection_total,
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


def _load_gallery(
    dataset_dir: Path,
    enrollment: Mapping[str, Sequence[str]],
    cache: RawEmbeddingCache,
    on_progress: Callable[[str, int, int], None] | None,
) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, str, str]], int]:
    """从缓存恢复全部 Gallery 向量，并保留每个拒绝原因。"""

    gallery: dict[str, list[np.ndarray]] = {}
    rejections: list[tuple[str, str, str]] = []
    total = sum(len(paths) for paths in enrollment.values())
    done = 0
    for person_id, relative_paths in enrollment.items():
        for relative_path in relative_paths:
            entry = _required_cache_entry(dataset_dir, relative_path, cache)
            if entry.status == "failed":
                rejections.append((person_id, relative_path, entry.reason or "cached_failure"))
            else:
                if entry.embedding is None:
                    raise RuntimeError(f"cached embedding is missing: {relative_path}")
                gallery.setdefault(person_id, []).append(_normalize(entry.embedding))
            done += 1
            if on_progress is not None:
                on_progress("gallery", done, total)
    return gallery, rejections, total


def _load_probes(
    dataset_dir: Path,
    probes: Sequence[LfwSplitProbe],
    cache: RawEmbeddingCache,
    on_progress: Callable[[str, int, int], None] | None,
) -> tuple[list[_ValidProbe], list[tuple[LfwSplitProbe, str]]]:
    """从原始缓存恢复探针向量，并把模型 FTE 单独返回。"""

    valid: list[_ValidProbe] = []
    rejected: list[tuple[LfwSplitProbe, str]] = []
    for index, probe in enumerate(probes, start=1):
        entry = _required_cache_entry(dataset_dir, probe.relative_path, cache)
        if entry.status == "failed":
            rejected.append((probe, entry.reason or "cached_failure"))
        else:
            if entry.embedding is None or entry.metrics is None:
                raise RuntimeError(f"cached probe data is incomplete: {probe.relative_path}")
            valid.append(_ValidProbe(probe, _normalize(entry.embedding), entry.metrics))
        if on_progress is not None:
            on_progress("probe-cache", index, len(probes))
    return valid, rejected


def _required_cache_entry(
    dataset_dir: Path,
    relative_path: str,
    cache: RawEmbeddingCache,
):
    """校验图片内容后读取缓存；缺少记录时立即失败而不运行模型。"""

    image_path = dataset_dir / relative_path
    if not image_path.is_file():
        raise FileNotFoundError(f"protocol image does not exist: {image_path}")
    entry = cache.get(relative_path, file_sha256(image_path))
    if entry is None:
        raise RuntimeError(f"embedding cache is incomplete or stale: {relative_path}")
    return entry


def _build_gallery_matrices(
    gallery_embeddings: Mapping[str, Sequence[np.ndarray]],
) -> _GalleryMatrices:
    """按人员稳定排序并构建批量相似度计算所需矩阵。"""

    person_ids = tuple(sorted(gallery_embeddings))
    samples: list[np.ndarray] = []
    starts: list[int] = []
    counts: list[int] = []
    prototypes: list[np.ndarray] = []
    dimension: int | None = None
    for person_id in person_ids:
        embeddings = [_normalize(item) for item in gallery_embeddings[person_id]]
        if not embeddings:
            continue
        if dimension is None:
            dimension = int(embeddings[0].size)
        if any(embedding.size != dimension for embedding in embeddings):
            raise ValueError("Gallery embeddings must have the same dimension")
        starts.append(len(samples))
        counts.append(len(embeddings))
        samples.extend(embeddings)
        prototypes.append(_normalize(np.mean(embeddings, axis=0)))
    if not samples:
        raise ValueError("Gallery contains no embeddings")
    return _GalleryMatrices(
        person_ids=person_ids,
        sample_matrix=np.stack(samples).astype(np.float32),
        starts=np.asarray(starts, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        prototypes=np.stack(prototypes).astype(np.float32),
    )


def _write_score_batches(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    probes: Sequence[_ValidProbe],
    gallery: _GalleryMatrices,
    batch_size: int,
    on_progress: Callable[[str, int, int], None] | None,
) -> int:
    """批量计算六种人员得分并写入第一、第二候选及候选分差。"""

    total_written = 0
    for start in range(0, len(probes), batch_size):
        batch_started_at = perf_counter()
        batch = probes[start : start + batch_size]
        query_matrix = np.stack([item.embedding for item in batch]).astype(np.float32)
        if query_matrix.shape[1] != gallery.sample_matrix.shape[1]:
            raise ValueError("Probe and Gallery embedding dimensions do not match")
        sample_scores = query_matrix @ gallery.sample_matrix.T
        method_scores = _aggregate_batch_scores(query_matrix, sample_scores, gallery)
        per_probe_scoring_ms = (
            (perf_counter() - batch_started_at) * 1000 / len(batch) if batch else 0.0
        )
        rows: list[tuple[object, ...]] = []
        for method, top_k, scores in method_scores:
            for probe_index, probe in enumerate(batch):
                top_index, second_index = _top_two(scores[probe_index])
                top_score = float(scores[probe_index, top_index])
                second_score = float(scores[probe_index, second_index])
                top_person_id = gallery.person_ids[top_index]
                second_person_id = gallery.person_ids[second_index]
                rows.append(
                    (
                        run_id,
                        probe.protocol.split,
                        probe.protocol.relative_path,
                        probe.protocol.source_identity,
                        probe.protocol.expected_person_id,
                        method,
                        top_k,
                        top_person_id,
                        top_score,
                        second_person_id,
                        second_score,
                        max(0.0, top_score - second_score),
                        int(
                            probe.protocol.expected_person_id is not None
                            and top_person_id == probe.protocol.expected_person_id
                        ),
                        json.dumps(probe.quality_metrics, ensure_ascii=False, sort_keys=True),
                        per_probe_scoring_ms,
                    )
                )
        connection.executemany(
            """
            INSERT INTO decision_scores (
                run_id, split, relative_path, source_identity, expected_person_id,
                method, top_k, top_person_id, top_score, second_person_id,
                second_score, score_gap, top_is_correct, probe_quality_json,
                scoring_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        total_written += len(rows)
        connection.commit()
        if on_progress is not None:
            on_progress("scoring", min(start + len(batch), len(probes)), len(probes))
    return total_written


def _aggregate_batch_scores(
    query_matrix: np.ndarray,
    sample_scores: np.ndarray,
    gallery: _GalleryMatrices,
) -> list[tuple[str, int, np.ndarray]]:
    """一次矩阵计算生成六种无阈值人员级分数。"""

    single_scores = sample_scores[:, gallery.starts]
    max_scores = np.maximum.reduceat(sample_scores, gallery.starts, axis=1)
    prototype_scores = query_matrix @ gallery.prototypes.T
    results: list[tuple[str, int, np.ndarray]] = [
        ("single", 0, single_scores),
        ("max", 0, max_scores),
        ("mean_prototype", 0, prototype_scores),
    ]
    for top_k in (2, 3, 5):
        aggregated = np.empty((query_matrix.shape[0], len(gallery.person_ids)), dtype=np.float32)
        for person_index, (person_start, count) in enumerate(
            zip(gallery.starts, gallery.counts, strict=True)
        ):
            person_values = sample_scores[
                :, int(person_start) : int(person_start + count)
            ]
            effective_k = min(top_k, int(count))
            if effective_k == int(count):
                aggregated[:, person_index] = np.mean(person_values, axis=1)
            else:
                strongest = np.partition(person_values, -effective_k, axis=1)[:, -effective_k:]
                aggregated[:, person_index] = np.mean(strongest, axis=1)
        results.append(("top_k_mean", top_k, aggregated))
    return results


def _top_two(scores: np.ndarray) -> tuple[int, int]:
    """返回一行人员分数中的第一、第二候选下标，同分时按身份顺序决定。"""

    if scores.ndim != 1 or scores.size < 2:
        raise ValueError("at least two valid Gallery identities are required")
    first = int(np.argmax(scores))
    without_first = scores.copy()
    without_first[first] = -np.inf
    second = int(np.argmax(without_first))
    return first, second


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """把缓存向量恢复为单位向量，并拒绝空向量和零向量。"""

    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm == 0.0:
        raise ValueError("embedding must be a non-zero one-dimensional vector")
    return vector / norm


def _create_schema(connection: sqlite3.Connection) -> None:
    """创建只保存实验元数据、无阈值候选分数和拒绝项的表。"""

    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            protocol_sha256 TEXT NOT NULL,
            embedding_extraction_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            gallery_image_total INTEGER NOT NULL,
            gallery_valid_image_total INTEGER NOT NULL,
            gallery_rejection_total INTEGER NOT NULL,
            probe_total INTEGER NOT NULL,
            valid_probe_total INTEGER NOT NULL,
            probe_rejection_total INTEGER NOT NULL,
            score_record_total INTEGER NOT NULL
        );

        CREATE TABLE decision_scores (
            run_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('calibration', 'evaluation')),
            relative_path TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            expected_person_id TEXT,
            method TEXT NOT NULL CHECK(method IN ('single', 'max', 'mean_prototype', 'top_k_mean')),
            top_k INTEGER NOT NULL,
            top_person_id TEXT NOT NULL,
            top_score REAL NOT NULL,
            second_person_id TEXT NOT NULL,
            second_score REAL NOT NULL,
            score_gap REAL NOT NULL,
            top_is_correct INTEGER NOT NULL CHECK(top_is_correct IN (0, 1)),
            probe_quality_json TEXT NOT NULL,
            scoring_latency_ms REAL NOT NULL,
            PRIMARY KEY (run_id, relative_path, method, top_k)
        );

        CREATE INDEX decision_scores_split_method
        ON decision_scores(run_id, split, method, top_k);

        CREATE TABLE rejections (
            run_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('gallery', 'probe')),
            split TEXT,
            relative_path TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            expected_person_id TEXT,
            reason TEXT NOT NULL,
            PRIMARY KEY (run_id, role, relative_path)
        );
        """
    )


def _write_rejections(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    gallery_rejections: Sequence[tuple[str, str, str]],
    probe_rejections: Sequence[tuple[LfwSplitProbe, str]],
) -> None:
    """保存模型无法提取主体脸的 Gallery 与 Probe 路径和原因。"""

    rows = [
        (run_id, "gallery", None, relative_path, person_id, person_id, reason)
        for person_id, relative_path, reason in gallery_rejections
    ]
    rows.extend(
        (
            run_id,
            "probe",
            probe.split,
            probe.relative_path,
            probe.source_identity,
            probe.expected_person_id,
            reason,
        )
        for probe, reason in probe_rejections
    )
    connection.executemany(
        """
        INSERT INTO rejections (
            run_id, role, split, relative_path, source_identity, expected_person_id, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
