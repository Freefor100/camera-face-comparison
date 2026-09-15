from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from .experiment_artifacts import file_sha256
from .joint_calibration import JointScoreRow
from .lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from .raw_embedding_cache import RawEmbeddingCache, RawEmbeddingEntry

ImageDomain = Literal["natural", "xqlfw"]
GalleryDomain = Literal["natural", "mixed", "xqlfw"]


@dataclass(frozen=True)
class CrossQualityScenario:
    """一个固定 Gallery 域和 Probe 域组成的开放集实验场景。"""

    name: str
    gallery_domain: GalleryDomain
    probe_domain: ImageDomain
    participates_in_selection: bool


CROSS_QUALITY_SCENARIOS = (
    CrossQualityScenario("natural_gallery__natural_probe", "natural", "natural", True),
    CrossQualityScenario("natural_gallery__xqlfw_probe", "natural", "xqlfw", True),
    CrossQualityScenario("mixed_gallery__natural_probe", "mixed", "natural", True),
    CrossQualityScenario("mixed_gallery__xqlfw_probe", "mixed", "xqlfw", True),
    CrossQualityScenario("xqlfw_gallery__natural_probe", "xqlfw", "natural", False),
    CrossQualityScenario("xqlfw_gallery__xqlfw_probe", "xqlfw", "xqlfw", False),
)


@dataclass(frozen=True)
class CrossQualityExportSummary:
    """跨质量无阈值分数库的覆盖统计。"""

    scenario_total: int
    score_record_total: int
    probe_rejection_total: int
    gallery_rejection_total: int


@dataclass(frozen=True)
class _GalleryMatrices:
    """一个场景中连续排列的 Gallery 样本与身份原型。"""

    person_ids: tuple[str, ...]
    sample_matrix: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    prototypes: np.ndarray


@dataclass(frozen=True)
class _CachedDomain:
    """同一批协议路径在一个图像域中的缓存结果。"""

    entries: Mapping[str, RawEmbeddingEntry]


def mixed_gallery_domains(
    enrollment: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> dict[str, ImageDomain]:
    """为混合质量 Gallery 生成与输入字典顺序无关的固定域分配。

    多样本身份按路径哈希排序后交替使用自然与 XQLFW 图片；单样本身份
    使用身份哈希固定选择域。这样既保持可复现，也能在可能时真实形成同人质量不齐。
    """

    assignments: dict[str, ImageDomain] = {}
    for person_id in sorted(enrollment):
        paths = tuple(sorted(set(enrollment[person_id])))
        if len(paths) == 1:
            digest = hashlib.sha256(f"{seed}:{person_id}".encode()).digest()
            assignments[paths[0]] = "xqlfw" if digest[0] % 2 else "natural"
            continue
        ranked = sorted(
            paths,
            key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).digest(),
        )
        for index, path in enumerate(ranked):
            assignments[path] = "natural" if index % 2 == 0 else "xqlfw"
    return assignments


def load_joint_score_rows(
    path: Path,
    *,
    run_id: str,
    split: Literal["calibration", "evaluation"],
    scenario_names: Sequence[str],
    method: str,
    top_k: int,
) -> tuple[tuple[JointScoreRow, ...], dict[str, int]]:
    """从跨质量分数库恢复一个聚合方法的候选排序和端到端 Known 分母。

    模型 FTE 不会出现在有效分数行中，但会计入 `protocol_known_totals`，供
    后续同时报告有效样本 TPIR 和端到端 TPIR。
    """

    if split not in {"calibration", "evaluation"}:
        raise ValueError(f"unsupported score split: {split}")
    names = tuple(dict.fromkeys(scenario_names))
    if not names:
        raise ValueError("at least one cross-quality scenario is required")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows: list[JointScoreRow] = []
        protocol_known_totals: dict[str, int] = {}
        for scenario in names:
            raw_rows = connection.execute(
                """
                SELECT source_identity, expected_person_id, top_is_correct,
                       candidate_count, candidate_scores
                FROM rankings
                WHERE run_id = ? AND split = ? AND scenario = ? AND method = ? AND top_k = ?
                ORDER BY relative_path
                """,
                (run_id, split, scenario, method, top_k),
            ).fetchall()
            for source_identity, expected_person_id, top_is_correct, count, blob in raw_rows:
                scores = np.frombuffer(blob, dtype=np.float32)
                if scores.size != int(count):
                    raise RuntimeError("stored candidate score count does not match its BLOB")
                rows.append(
                    JointScoreRow(
                        scenario=scenario,
                        source_identity=str(source_identity),
                        expected_person_id=(
                            None if expected_person_id is None else str(expected_person_id)
                        ),
                        top_is_correct=bool(top_is_correct),
                        candidate_scores=tuple(float(value) for value in scores),
                    )
                )
            valid_known = sum(row[1] is not None for row in raw_rows)
            rejected_known = connection.execute(
                """
                SELECT COUNT(*)
                FROM rejections
                WHERE run_id = ? AND scenario = ? AND role = 'probe'
                      AND split = ? AND expected_person_id IS NOT NULL
                """,
                (run_id, scenario, split),
            ).fetchone()[0]
            protocol_known_totals[scenario] = int(valid_known + rejected_known)
    finally:
        connection.close()
    return tuple(rows), protocol_known_totals


def export_cross_quality_scores(
    *,
    natural_dataset_dir: Path,
    xqlfw_dataset_dir: Path,
    protocol: LfwSplitProtocol,
    natural_cache: RawEmbeddingCache,
    xqlfw_cache: RawEmbeddingCache,
    output_path: Path,
    run_id: str,
    protocol_sha256: str,
    embedding_extraction_id: str,
    seed: int = 2026,
    batch_size: int = 32,
    max_candidates: int = 32,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> CrossQualityExportSummary:
    """从两个原始缓存导出六个场景的无阈值人员候选排序。

    参数：
        natural_dataset_dir：自然 LFW 图片根目录。
        xqlfw_dataset_dir：XQLFW 同路径变体根目录。
        protocol：Phase 4 已固定的身份互斥协议。
        natural_cache、xqlfw_cache：两个图像域的完整原始 embedding 缓存。
        output_path：原子生成的 SQLite 分数库。
        run_id、protocol_sha256、embedding_extraction_id：绑定本次实验的标识。
        seed：混合 Gallery 的固定随机来源。
        batch_size：一次矩阵打分的 Probe 数量。
        max_candidates：每条记录保留的最高身份候选数。
    返回：
        场景、候选分数和模型失败的实际数量。
    前置条件：
        两个缓存必须覆盖协议全部图片；本函数不初始化模型，也不接受判定阈值。
    """

    if not run_id or not protocol_sha256 or not embedding_extraction_id:
        raise ValueError("cross-quality run identifiers must not be empty")
    if batch_size < 1 or max_candidates < 2:
        raise ValueError("batch_size must be positive and max_candidates at least two")

    all_paths = tuple(
        sorted(
            {
                *(path for paths in protocol.enrollment.values() for path in paths),
                *(probe.relative_path for probe in protocol.probes),
            }
        )
    )
    domains = {
        "natural": _CachedDomain(
            _load_domain_entries(natural_dataset_dir, all_paths, natural_cache)
        ),
        "xqlfw": _CachedDomain(
            _load_domain_entries(xqlfw_dataset_dir, all_paths, xqlfw_cache)
        ),
    }
    mixed = mixed_gallery_domains(protocol.enrollment, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_path)
    total_scores = total_probe_rejections = total_gallery_rejections = 0
    try:
        _create_schema(connection)
        for scenario_index, scenario in enumerate(CROSS_QUALITY_SCENARIOS, start=1):
            gallery_embeddings, gallery_rejections = _scenario_gallery(
                protocol.enrollment,
                scenario.gallery_domain,
                domains,
                mixed,
            )
            gallery = _build_gallery_matrices(gallery_embeddings)
            _write_scenario(connection, run_id, scenario, gallery, protocol, gallery_rejections)
            valid_probes, probe_rejections = _scenario_probes(
                protocol.probes,
                domains[scenario.probe_domain],
            )
            total_probe_rejections += len(probe_rejections)
            total_gallery_rejections += len(gallery_rejections)
            _write_probe_rejections(connection, run_id, scenario.name, probe_rejections)
            total_scores += _write_rankings(
                connection,
                run_id=run_id,
                scenario=scenario,
                probes=valid_probes,
                gallery=gallery,
                batch_size=batch_size,
                max_candidates=max_candidates,
                on_progress=on_progress,
            )
            connection.commit()
            if on_progress is not None:
                on_progress("scenario", scenario_index, len(CROSS_QUALITY_SCENARIOS))
        summary = CrossQualityExportSummary(
            scenario_total=len(CROSS_QUALITY_SCENARIOS),
            score_record_total=total_scores,
            probe_rejection_total=total_probe_rejections,
            gallery_rejection_total=total_gallery_rejections,
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                protocol_sha256,
                embedding_extraction_id,
                seed,
                datetime.now(UTC).isoformat(),
                summary.scenario_total,
                summary.score_record_total,
                summary.probe_rejection_total,
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


def _load_domain_entries(
    dataset_dir: Path,
    relative_paths: Sequence[str],
    cache: RawEmbeddingCache,
) -> dict[str, RawEmbeddingEntry]:
    """校验图片哈希并一次恢复一个域的全部协议缓存。"""

    entries: dict[str, RawEmbeddingEntry] = {}
    for relative_path in relative_paths:
        image_path = dataset_dir / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"cross-quality image is missing: {image_path}")
        entry = cache.get(relative_path, file_sha256(image_path))
        if entry is None:
            raise RuntimeError(f"cross-quality cache is incomplete or stale: {relative_path}")
        entries[relative_path] = entry
    return entries


def _scenario_gallery(
    enrollment: Mapping[str, Sequence[str]],
    gallery_domain: GalleryDomain,
    domains: Mapping[ImageDomain, _CachedDomain],
    mixed: Mapping[str, ImageDomain],
) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, str, str, str]]]:
    """恢复一个场景的有效 Gallery，并保留每张失败图片的来源域。"""

    gallery: dict[str, list[np.ndarray]] = {}
    rejections: list[tuple[str, str, str, str]] = []
    for person_id, paths in enrollment.items():
        for relative_path in paths:
            domain: ImageDomain = (
                mixed[relative_path] if gallery_domain == "mixed" else gallery_domain
            )
            entry = domains[domain].entries[relative_path]
            if entry.status == "failed":
                rejections.append(
                    (person_id, relative_path, domain, entry.reason or "cached_failure")
                )
            else:
                if entry.embedding is None:
                    raise RuntimeError(f"cached Gallery embedding is missing: {relative_path}")
                gallery.setdefault(person_id, []).append(_normalize(entry.embedding))
    return gallery, rejections


def _scenario_probes(
    probes: Sequence[LfwSplitProbe],
    domain: _CachedDomain,
) -> tuple[list[tuple[LfwSplitProbe, RawEmbeddingEntry]], list[tuple[LfwSplitProbe, str]]]:
    """把一个 Probe 域拆成有效记录和模型 FTE。"""

    valid: list[tuple[LfwSplitProbe, RawEmbeddingEntry]] = []
    rejected: list[tuple[LfwSplitProbe, str]] = []
    for probe in probes:
        entry = domain.entries[probe.relative_path]
        if entry.status == "failed":
            rejected.append((probe, entry.reason or "cached_failure"))
        else:
            if entry.embedding is None or entry.metrics is None:
                raise RuntimeError(f"cached Probe data is incomplete: {probe.relative_path}")
            valid.append((probe, entry))
    return valid, rejected


def _build_gallery_matrices(
    gallery_embeddings: Mapping[str, Sequence[np.ndarray]],
) -> _GalleryMatrices:
    """按身份稳定排序构建全部人员聚合共享的矩阵。"""

    person_ids = tuple(sorted(person_id for person_id, rows in gallery_embeddings.items() if rows))
    if len(person_ids) < 2:
        raise ValueError("cross-quality experiment requires at least two Gallery identities")
    samples: list[np.ndarray] = []
    starts: list[int] = []
    counts: list[int] = []
    prototypes: list[np.ndarray] = []
    dimension: int | None = None
    for person_id in person_ids:
        embeddings = [_normalize(item) for item in gallery_embeddings[person_id]]
        if dimension is None:
            dimension = int(embeddings[0].size)
        if any(item.size != dimension for item in embeddings):
            raise ValueError("cross-quality Gallery embeddings must share a dimension")
        starts.append(len(samples))
        counts.append(len(embeddings))
        samples.extend(embeddings)
        prototypes.append(_normalize(np.mean(embeddings, axis=0)))
    return _GalleryMatrices(
        person_ids=person_ids,
        sample_matrix=np.stack(samples).astype(np.float32),
        starts=np.asarray(starts, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        prototypes=np.stack(prototypes).astype(np.float32),
    )


def _aggregate_batch_scores(
    query_matrix: np.ndarray,
    sample_scores: np.ndarray,
    gallery: _GalleryMatrices,
) -> list[tuple[str, int, np.ndarray]]:
    """由同一次样本矩阵乘法生成六种人员聚合分数。"""

    results: list[tuple[str, int, np.ndarray]] = [
        ("single", 0, sample_scores[:, gallery.starts]),
        ("max", 0, np.maximum.reduceat(sample_scores, gallery.starts, axis=1)),
        ("mean_prototype", 0, query_matrix @ gallery.prototypes.T),
    ]
    for top_k in (2, 3, 5):
        aggregated = np.empty((query_matrix.shape[0], len(gallery.person_ids)), dtype=np.float32)
        for person_index, (start, count) in enumerate(
            zip(gallery.starts, gallery.counts, strict=True)
        ):
            values = sample_scores[:, int(start) : int(start + count)]
            effective_k = min(top_k, int(count))
            strongest = (
                values
                if effective_k == int(count)
                else np.partition(values, -effective_k, axis=1)[:, -effective_k:]
            )
            aggregated[:, person_index] = np.mean(strongest, axis=1)
        results.append(("top_k_mean", top_k, aggregated))
    return results


def _write_rankings(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    scenario: CrossQualityScenario,
    probes: Sequence[tuple[LfwSplitProbe, RawEmbeddingEntry]],
    gallery: _GalleryMatrices,
    batch_size: int,
    max_candidates: int,
    on_progress: Callable[[str, int, int], None] | None,
) -> int:
    """批量打分并以紧凑 BLOB 保存前若干身份的索引和分数。"""

    total_written = 0
    candidate_count = min(max_candidates, len(gallery.person_ids))
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        query_matrix = np.stack([_normalize(entry.embedding) for _, entry in batch]).astype(
            np.float32
        )
        started_at = perf_counter()
        sample_scores = query_matrix @ gallery.sample_matrix.T
        method_scores = _aggregate_batch_scores(query_matrix, sample_scores, gallery)
        per_probe_ms = (perf_counter() - started_at) * 1000 / len(batch)
        rows: list[tuple[object, ...]] = []
        for method, top_k, scores in method_scores:
            for probe_index, (probe, entry) in enumerate(batch):
                ranking = np.argsort(-scores[probe_index], kind="stable")[:candidate_count]
                ranked_scores = scores[probe_index, ranking].astype(np.float32)
                top_person = gallery.person_ids[int(ranking[0])]
                rows.append(
                    (
                        run_id,
                        scenario.name,
                        probe.split,
                        probe.relative_path,
                        probe.source_identity,
                        probe.expected_person_id,
                        method,
                        top_k,
                        candidate_count,
                        ranking.astype(np.int32).tobytes(),
                        ranked_scores.tobytes(),
                        int(
                            probe.expected_person_id is not None
                            and top_person == probe.expected_person_id
                        ),
                        json.dumps(entry.metrics, ensure_ascii=False, sort_keys=True),
                        float(entry.latency_ms),
                        per_probe_ms,
                    )
                )
        connection.executemany(
            """
            INSERT INTO rankings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        total_written += len(rows)
        connection.commit()
        if on_progress is not None:
            on_progress(f"scoring:{scenario.name}", min(start + len(batch), len(probes)), len(probes))
    return total_written


def _write_scenario(
    connection: sqlite3.Connection,
    run_id: str,
    scenario: CrossQualityScenario,
    gallery: _GalleryMatrices,
    protocol: LfwSplitProtocol,
    gallery_rejections: Sequence[tuple[str, str, str, str]],
) -> None:
    """保存场景定义、身份索引和 Gallery 失败。"""

    gallery_total = sum(len(paths) for paths in protocol.enrollment.values())
    connection.execute(
        "INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            scenario.name,
            scenario.gallery_domain,
            scenario.probe_domain,
            int(scenario.participates_in_selection),
            gallery_total,
            gallery_total - len(gallery_rejections),
            len(gallery.person_ids),
        ),
    )
    connection.executemany(
        "INSERT INTO identities VALUES (?, ?, ?, ?)",
        (
            (run_id, scenario.name, index, person_id)
            for index, person_id in enumerate(gallery.person_ids)
        ),
    )
    connection.executemany(
        "INSERT INTO rejections VALUES (?, ?, 'gallery', NULL, ?, ?, ?, ?, ?)",
        (
            (run_id, scenario.name, relative_path, person_id, person_id, domain, reason)
            for person_id, relative_path, domain, reason in gallery_rejections
        ),
    )


def _write_probe_rejections(
    connection: sqlite3.Connection,
    run_id: str,
    scenario_name: str,
    rejections: Sequence[tuple[LfwSplitProbe, str]],
) -> None:
    """保存一个场景中没有 embedding 的全部 Probe。"""

    connection.executemany(
        "INSERT INTO rejections VALUES (?, ?, 'probe', ?, ?, ?, ?, NULL, ?)",
        (
            (
                run_id,
                scenario_name,
                probe.split,
                probe.relative_path,
                probe.source_identity,
                probe.expected_person_id,
                reason,
            )
            for probe, reason in rejections
        ),
    )


def _normalize(embedding: np.ndarray | None) -> np.ndarray:
    """把缓存 embedding 恢复为有限单位向量。"""

    if embedding is None:
        raise ValueError("embedding is missing")
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must have a finite positive norm")
    return vector / norm


def _create_schema(connection: sqlite3.Connection) -> None:
    """创建不含任何判定参数的跨质量实验表。"""

    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            protocol_sha256 TEXT NOT NULL,
            embedding_extraction_id TEXT NOT NULL,
            seed INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
            scenario_total INTEGER NOT NULL,
            score_record_total INTEGER NOT NULL,
            probe_rejection_total INTEGER NOT NULL
        );
        CREATE TABLE scenarios (
            run_id TEXT NOT NULL,
            name TEXT NOT NULL,
            gallery_domain TEXT NOT NULL,
            probe_domain TEXT NOT NULL,
            participates_in_selection INTEGER NOT NULL,
            gallery_image_total INTEGER NOT NULL,
            gallery_valid_image_total INTEGER NOT NULL,
            gallery_identity_total INTEGER NOT NULL,
            PRIMARY KEY (run_id, name)
        );
        CREATE TABLE identities (
            run_id TEXT NOT NULL,
            scenario TEXT NOT NULL,
            identity_index INTEGER NOT NULL,
            person_id TEXT NOT NULL,
            PRIMARY KEY (run_id, scenario, identity_index)
        );
        CREATE TABLE rankings (
            run_id TEXT NOT NULL,
            scenario TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('calibration', 'evaluation')),
            relative_path TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            expected_person_id TEXT,
            method TEXT NOT NULL CHECK(method IN ('single', 'max', 'mean_prototype', 'top_k_mean')),
            top_k INTEGER NOT NULL,
            candidate_count INTEGER NOT NULL,
            candidate_indices BLOB NOT NULL,
            candidate_scores BLOB NOT NULL,
            top_is_correct INTEGER NOT NULL CHECK(top_is_correct IN (0, 1)),
            probe_metrics_json TEXT NOT NULL,
            extraction_latency_ms REAL NOT NULL,
            scoring_latency_ms REAL NOT NULL,
            PRIMARY KEY (run_id, scenario, relative_path, method, top_k)
        );
        CREATE INDEX rankings_lookup
        ON rankings(run_id, split, scenario, method, top_k);
        CREATE TABLE rejections (
            run_id TEXT NOT NULL,
            scenario TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('gallery', 'probe')),
            split TEXT,
            relative_path TEXT NOT NULL,
            source_identity TEXT NOT NULL,
            expected_person_id TEXT,
            image_domain TEXT,
            reason TEXT NOT NULL,
            PRIMARY KEY (run_id, scenario, role, relative_path)
        );
        """
    )
