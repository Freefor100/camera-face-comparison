from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cross_quality_experiment import CROSS_QUALITY_SCENARIOS, mixed_gallery_domains
from .experiment_artifacts import file_sha256
from .joint_calibration import (
    JointOperatingPoint,
    JointScoreRow,
    evaluate_joint_operating_point,
)
from .lfw_dataset import LfwSplitProtocol
from .raw_embedding_cache import RawEmbeddingCache, RawEmbeddingEntry


@dataclass(frozen=True)
class GalleryScaleInputs:
    """只含 Calibration 身份和向量的小 Gallery 稳定性实验输入。"""

    gallery_samples: dict[str, dict[str, tuple[np.ndarray, ...]]]
    known_probes: dict[str, dict[str, tuple[np.ndarray, ...]]]
    unknown_probes: dict[str, tuple[np.ndarray, ...]]
    known_protocol_counts: dict[str, int]
    unknown_protocol_total: int
    coverage: dict[str, object]


def load_gallery_scale_inputs(
    *,
    natural_dataset_dir: Path,
    xqlfw_dataset_dir: Path,
    protocol: LfwSplitProtocol,
    natural_cache: RawEmbeddingCache,
    xqlfw_cache: RawEmbeddingCache,
    seed: int,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> GalleryScaleInputs:
    """从两域原始缓存恢复 Calibration 小库实验所需向量。

    参数：
        natural_dataset_dir、xqlfw_dataset_dir：同路径的两个图片域。
        protocol：固定身份互斥协议，仅读取其中 Calibration Probe。
        natural_cache、xqlfw_cache：已完成的无阈值 embedding 缓存。
        seed：混合 Gallery 域分配使用的固定种子。
    返回：
        自然/混合 Gallery、自然/XQLFW Probe 和包含 FTE 的协议分母。
    前置条件：
        两个缓存必须覆盖协议使用的全部路径；本函数不会初始化人脸模型。
    """

    mixed_assignments = mixed_gallery_domains(protocol.enrollment, seed=seed)
    gallery: dict[str, dict[str, list[np.ndarray]]] = {"natural": {}, "mixed": {}}
    gallery_fte = {"natural": 0, "mixed": 0}
    gallery_total = sum(len(paths) for paths in protocol.enrollment.values())
    done = 0
    for person_id in sorted(protocol.enrollment):
        for relative_path in protocol.enrollment[person_id]:
            natural = _required_cache_entry(
                natural_dataset_dir, relative_path, natural_cache
            )
            xqlfw = _required_cache_entry(xqlfw_dataset_dir, relative_path, xqlfw_cache)
            _append_entry(gallery["natural"], person_id, natural, gallery_fte, "natural")
            mixed_domain = mixed_assignments[relative_path]
            mixed_entry = natural if mixed_domain == "natural" else xqlfw
            _append_entry(gallery["mixed"], person_id, mixed_entry, gallery_fte, "mixed")
            done += 1
            if on_progress is not None:
                on_progress("gallery", done, gallery_total)

    known: dict[str, dict[str, list[np.ndarray]]] = {"natural": {}, "xqlfw": {}}
    unknown: dict[str, list[np.ndarray]] = {"natural": [], "xqlfw": []}
    known_protocol_counts: dict[str, int] = {}
    unknown_protocol_total = 0
    probe_fte = {"natural": 0, "xqlfw": 0}
    calibration_probes = [probe for probe in protocol.probes if probe.split == "calibration"]
    for index, probe in enumerate(calibration_probes, start=1):
        if probe.expected_person_id is None:
            unknown_protocol_total += 1
        else:
            known_protocol_counts[probe.expected_person_id] = (
                known_protocol_counts.get(probe.expected_person_id, 0) + 1
            )
        for domain, dataset_dir, cache in (
            ("natural", natural_dataset_dir, natural_cache),
            ("xqlfw", xqlfw_dataset_dir, xqlfw_cache),
        ):
            entry = _required_cache_entry(dataset_dir, probe.relative_path, cache)
            if entry.status == "failed":
                probe_fte[domain] += 1
                continue
            vector = _entry_vector(entry, probe.relative_path)
            if probe.expected_person_id is None:
                unknown[domain].append(vector)
            else:
                known[domain].setdefault(probe.expected_person_id, []).append(vector)
        if on_progress is not None:
            on_progress("probe", index, len(calibration_probes))

    return GalleryScaleInputs(
        gallery_samples={
            domain: {
                person_id: tuple(vectors)
                for person_id, vectors in sorted(people.items())
                if vectors
            }
            for domain, people in gallery.items()
        },
        known_probes={
            domain: {
                person_id: tuple(vectors)
                for person_id, vectors in sorted(people.items())
                if vectors
            }
            for domain, people in known.items()
        },
        unknown_probes={domain: tuple(vectors) for domain, vectors in unknown.items()},
        known_protocol_counts=known_protocol_counts,
        unknown_protocol_total=unknown_protocol_total,
        coverage={
            "source_split": "calibration",
            "gallery_image_total": gallery_total,
            "gallery_fte_by_domain": gallery_fte,
            "probe_protocol_total": len(calibration_probes),
            "probe_fte_by_domain": probe_fte,
        },
    )


def run_gallery_scale_experiment(
    inputs: GalleryScaleInputs,
    *,
    selected: JointOperatingPoint,
    gallery_sizes: Sequence[int],
    repeats: int,
    seed: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """在 Calibration 身份中重放冻结策略的多种 Gallery 规模。

    该实验不重新选阈值，也没有 Evaluation 输入。每个规模在自然/混合 Gallery
    与自然/XQLFW Probe 的四个主要场景使用完全相同的身份子集和部署参数。
    """

    sizes = tuple(int(value) for value in gallery_sizes)
    if not sizes or any(value < 2 for value in sizes):
        raise ValueError("gallery sizes must all be at least two")
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    identity_pool = tuple(
        sorted(
            set(inputs.known_protocol_counts)
            & set(inputs.gallery_samples["natural"])
            & set(inputs.gallery_samples["mixed"])
        )
    )
    if max(sizes) > len(identity_pool):
        raise ValueError(
            f"gallery size exceeds Calibration identity pool: {len(identity_pool)}"
        )

    scenarios = tuple(item for item in CROSS_QUALITY_SCENARIOS if item.participates_in_selection)
    total = len(sizes) * repeats * len(scenarios)
    done = 0
    runs: list[dict[str, object]] = []
    for repeat in range(repeats):
        order = _shuffled_identities(identity_pool, seed=seed, repeat=repeat)
        for gallery_size in sizes:
            selected_ids = order[:gallery_size]
            for scenario in scenarios:
                rows = _score_scenario(
                    inputs,
                    gallery_domain=scenario.gallery_domain,
                    probe_domain=scenario.probe_domain,
                    scenario_name=scenario.name,
                    selected_ids=selected_ids,
                    aggregation_method=selected.aggregation_method,
                    top_k=selected.top_k,
                )
                protocol_known_total = sum(
                    inputs.known_protocol_counts[person_id] for person_id in selected_ids
                )
                result = evaluate_joint_operating_point(
                    rows,
                    selected=selected,
                    protocol_known_totals={scenario.name: protocol_known_total},
                )
                runs.append(
                    {
                        "gallery_size": gallery_size,
                        "repeat": repeat,
                        "scenario": scenario.name,
                        "gallery_ids": list(selected_ids),
                        "metrics": asdict(result.scenarios[0]),
                    }
                )
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
    return {
        "source_split": "calibration",
        "gallery_sizes": list(sizes),
        "repeats": repeats,
        "seed": seed,
        "identity_pool_total": len(identity_pool),
        "unknown_protocol_total": inputs.unknown_protocol_total,
        "selected_policy": {
            "aggregation_method": selected.aggregation_method,
            "top_k": selected.top_k,
            "rule": selected.rule,
            "minimum_score": selected.minimum_score,
            "minimum_gap": selected.minimum_gap,
            "minimum_probability": selected.minimum_probability,
            "nac_neighbors": selected.nac_neighbors,
        },
        "runs": runs,
        "summary": _summarize_runs(runs, sizes),
    }


def _score_scenario(
    inputs: GalleryScaleInputs,
    *,
    gallery_domain: str,
    probe_domain: str,
    scenario_name: str,
    selected_ids: Sequence[str],
    aggregation_method: str,
    top_k: int,
) -> tuple[JointScoreRow, ...]:
    """为一个小库场景生成当前聚合方法的全部有效 Calibration 排序。"""

    person_ids = tuple(selected_ids)
    samples = inputs.gallery_samples[gallery_domain]
    sample_vectors: list[np.ndarray] = []
    starts: list[int] = []
    counts: list[int] = []
    prototypes: list[np.ndarray] = []
    for person_id in person_ids:
        vectors = samples[person_id]
        starts.append(len(sample_vectors))
        counts.append(len(vectors))
        sample_vectors.extend(vectors)
        prototypes.append(_normalize(np.mean(vectors, axis=0)))

    queries: list[np.ndarray] = []
    expected: list[str | None] = []
    sources: list[str] = []
    for person_id in person_ids:
        for index, vector in enumerate(inputs.known_probes[probe_domain].get(person_id, ())):
            queries.append(vector)
            expected.append(person_id)
            sources.append(f"{person_id}:{index}")
    for index, vector in enumerate(inputs.unknown_probes[probe_domain]):
        queries.append(vector)
        expected.append(None)
        sources.append(f"unknown:{index}")
    if not queries:
        raise ValueError(f"small Gallery scenario has no valid Probe: {scenario_name}")

    query_matrix = np.stack(queries).astype(np.float32)
    sample_matrix = np.stack(sample_vectors).astype(np.float32)
    sample_scores = query_matrix @ sample_matrix.T
    person_scores = _aggregate_scores(
        query_matrix,
        sample_scores,
        starts=np.asarray(starts, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        prototypes=np.stack(prototypes).astype(np.float32),
        method=aggregation_method,
        top_k=top_k,
    )
    rankings = np.argsort(-person_scores, axis=1, kind="stable")
    rows = []
    for index, ranking in enumerate(rankings):
        ranked_scores = tuple(float(value) for value in person_scores[index, ranking])
        top_person = person_ids[int(ranking[0])]
        rows.append(
            JointScoreRow(
                scenario=scenario_name,
                source_identity=sources[index],
                expected_person_id=expected[index],
                top_is_correct=expected[index] is not None and top_person == expected[index],
                candidate_scores=ranked_scores,
            )
        )
    return tuple(rows)


def _aggregate_scores(
    queries: np.ndarray,
    sample_scores: np.ndarray,
    *,
    starts: np.ndarray,
    counts: np.ndarray,
    prototypes: np.ndarray,
    method: str,
    top_k: int,
) -> np.ndarray:
    """按冻结方法把样本相似度转换为人员分数。"""

    if method == "single":
        return sample_scores[:, starts]
    if method == "max":
        return np.maximum.reduceat(sample_scores, starts, axis=1)
    if method == "mean_prototype":
        return queries @ prototypes.T
    if method != "top_k_mean" or top_k not in {2, 3, 5}:
        raise ValueError(f"unsupported aggregation method: {method}:{top_k}")
    result = np.empty((queries.shape[0], starts.size), dtype=np.float32)
    for person_index, (start, count) in enumerate(zip(starts, counts, strict=True)):
        values = sample_scores[:, int(start) : int(start + count)]
        effective_k = min(top_k, int(count))
        strongest = (
            values
            if effective_k == int(count)
            else np.partition(values, -effective_k, axis=1)[:, -effective_k:]
        )
        result[:, person_index] = np.mean(strongest, axis=1)
    return result


def _append_entry(
    target: dict[str, list[np.ndarray]],
    person_id: str,
    entry: RawEmbeddingEntry,
    fte: dict[str, int],
    domain: str,
) -> None:
    """把有效 Gallery 向量加入身份，或累计当前域的 FTE。"""

    if entry.status == "failed":
        fte[domain] += 1
        return
    target.setdefault(person_id, []).append(_entry_vector(entry, person_id))


def _entry_vector(entry: RawEmbeddingEntry, source: str) -> np.ndarray:
    """从成功缓存项读取并归一化 embedding。"""

    if entry.embedding is None:
        raise RuntimeError(f"cached embedding is missing: {source}")
    return _normalize(entry.embedding)


def _required_cache_entry(
    dataset_dir: Path,
    relative_path: str,
    cache: RawEmbeddingCache,
) -> RawEmbeddingEntry:
    """读取文件哈希一致的缓存项，拒绝不完整或已过期缓存。"""

    image_path = dataset_dir / relative_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Gallery scale image is missing: {image_path}")
    entry = cache.get(relative_path, file_sha256(image_path))
    if entry is None:
        raise RuntimeError(f"embedding cache is incomplete or stale: {relative_path}")
    return entry


def _normalize(embedding: np.ndarray) -> np.ndarray:
    """把 embedding 转为有限 float32 单位向量。"""

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding must have a finite positive norm")
    return vector / norm


def _shuffled_identities(
    identities: Sequence[str],
    *,
    seed: int,
    repeat: int,
) -> tuple[str, ...]:
    """使用固定摘要种子生成与 Python 哈希随机化无关的身份顺序。"""

    digest = hashlib.sha256(f"{seed}:{repeat}:calibration".encode()).digest()
    values = list(identities)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    return tuple(values)


def _summarize_runs(
    runs: Sequence[Mapping[str, object]],
    sizes: Sequence[int],
) -> list[dict[str, object]]:
    """按 Gallery 规模和场景汇总重复实验的范围与均值。"""

    summary: list[dict[str, object]] = []
    scenarios = sorted({str(run["scenario"]) for run in runs})
    for size in sizes:
        for scenario in scenarios:
            selected = [
                run
                for run in runs
                if int(run["gallery_size"]) == size and run["scenario"] == scenario
            ]
            metrics = [run["metrics"] for run in selected]
            summary.append(
                {
                    "gallery_size": size,
                    "scenario": scenario,
                    "run_total": len(selected),
                    "fpir_valid": _numeric_summary(
                        [float(item["fpir_valid"]) for item in metrics]
                    ),
                    "tpir_e2e": _numeric_summary(
                        [float(item["tpir_e2e"]) for item in metrics]
                    ),
                    "rank1_rate_valid": _numeric_summary(
                        [
                            float(item["rank1_correct"]) / int(item["valid_known_total"])
                            for item in metrics
                        ]
                    ),
                }
            )
    return summary


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    """返回重复值的均值、最小值和最大值。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
