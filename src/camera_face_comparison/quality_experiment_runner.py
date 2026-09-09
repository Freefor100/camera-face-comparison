from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings
from .evaluation_cache import file_sha256
from .face_engine import FaceInputError, FaceObservation
from .image_input import ImageInput, apply_quality_policy, measure_quality
from .quality_degradation import (
    DegradationSpec,
    apply_degradation,
    degradation_specs,
    select_primary_face,
)
from .quality_experiment import QualityExperimentProtocol
from .quality_experiment_store import QualityExperimentEntry, QualityExperimentStore


class QualityFaceEngine(Protocol):
    """Phase 3 执行器需要的人脸模型最小接口。"""

    def extract_faces(self, frame: np.ndarray) -> list[FaceObservation]:
        """返回当前图片中的全部模型检测结果。"""


@dataclass(frozen=True)
class ExperimentSource:
    """一张图片在质量实验中的角色和身份标签。"""

    role: str
    identity: str
    relative_path: str


@dataclass(frozen=True)
class QualityExperimentRunSummary:
    """一次可恢复质量实验的覆盖和缓存使用摘要。"""

    source_total: int
    condition_total: int
    request_total: int
    cache_hit_total: int
    inference_total: int
    failure_total: int
    gallery_references: dict[str, str]


def measure_degraded_frame(
    *,
    relative_path: str,
    file_sha256: str,
    frame: np.ndarray,
    spec: DegradationSpec,
    baseline_bbox: tuple[float, float, float, float] | None,
    face_engine: QualityFaceEngine,
    store: QualityExperimentStore,
) -> tuple[QualityExperimentEntry, bool]:
    """读取缓存或运行一次退化图片的主脸提取和原始质量测量。

    返回值中的布尔量表示是否直接命中缓存，供实验报告核对恢复行为。
    """

    cached = store.get(relative_path, file_sha256, spec)
    if cached is not None:
        return cached, True

    degraded = apply_degradation(frame, spec, baseline_bbox)
    started_at = perf_counter()
    faces = face_engine.extract_faces(degraded)
    latency_ms = (perf_counter() - started_at) * 1000
    try:
        primary = select_primary_face(faces)
    except FaceInputError as error:
        entry = QualityExperimentEntry(
            status="failed",
            embedding=None,
            metrics=None,
            bbox=None,
            face_count=len(faces),
            latency_ms=latency_ms,
            reason=str(error),
        )
        store.put_failed(
            relative_path=relative_path,
            file_sha256=file_sha256,
            spec=spec,
            face_count=len(faces),
            latency_ms=latency_ms,
            reason=str(error),
        )
        return entry, False

    metrics = measure_quality(degraded, primary)
    entry = QualityExperimentEntry(
        status="observed",
        embedding=primary.embedding,
        metrics=metrics,
        bbox=primary.bbox,
        face_count=len(faces),
        latency_ms=latency_ms,
        reason=None,
    )
    store.put_observed(
        relative_path=relative_path,
        file_sha256=file_sha256,
        spec=spec,
        embedding=primary.embedding,
        metrics=metrics,
        bbox=primary.bbox,
        face_count=len(faces),
        latency_ms=latency_ms,
    )
    return entry, False


def run_quality_experiment(
    *,
    protocol: QualityExperimentProtocol,
    dataset_dir: Path,
    settings: Settings,
    face_engine: QualityFaceEngine,
    store: QualityExperimentStore,
    specs: Sequence[DegradationSpec] | None = None,
    commit_every: int = 20,
    on_progress: Callable[[int, int, ExperimentSource, DegradationSpec, bool], None]
    | None = None,
) -> QualityExperimentRunSummary:
    """对固定 Known、Unknown 和参考图执行全部单因素退化条件。

    Gallery 参考图从每个身份的候选中选择第一张通过当前质量策略的基线图片。
    选择只决定干净参考起点，不会阻止后续低质量参考图实验。
    """

    if commit_every < 1:
        raise ValueError("commit_every must be at least one")
    selected_specs = tuple(specs or degradation_specs())
    if not selected_specs or selected_specs[0].kind != "baseline":
        raise ValueError("the first degradation condition must be baseline")

    counters = _RunCounters()
    gallery_references = _select_gallery_references(
        protocol=protocol,
        dataset_dir=dataset_dir,
        settings=settings,
        face_engine=face_engine,
        store=store,
        counters=counters,
    )
    sources = [
        ExperimentSource("gallery_reference", person_id, relative_path)
        for person_id, relative_path in sorted(gallery_references.items())
    ]
    sources.extend(
        ExperimentSource("known_probe", case.person_id, case.probe_path)
        for case in protocol.known
    )
    sources.extend(
        ExperimentSource("unknown_probe", case.source_identity, case.probe_path)
        for case in protocol.unknown
    )
    total = len(sources) * len(selected_specs)
    completed = 0
    pending_writes = 0
    for source in sources:
        image_path = dataset_dir / source.relative_path
        image_input = ImageInput.from_file(image_path, source_type="dataset")
        digest = file_sha256(image_path)
        baseline_entry, baseline_hit = measure_degraded_frame(
            relative_path=source.relative_path,
            file_sha256=digest,
            frame=image_input.frame,
            spec=selected_specs[0],
            baseline_bbox=None,
            face_engine=face_engine,
            store=store,
        )
        counters.record(baseline_entry, baseline_hit)
        completed += 1
        pending_writes += int(not baseline_hit)
        if on_progress is not None:
            on_progress(completed, total, source, selected_specs[0], baseline_hit)

        for spec in selected_specs[1:]:
            if spec.kind == "face_size" and baseline_entry.bbox is None:
                entry, cache_hit = _put_baseline_failure(
                    source=source,
                    digest=digest,
                    spec=spec,
                    store=store,
                )
            else:
                entry, cache_hit = measure_degraded_frame(
                    relative_path=source.relative_path,
                    file_sha256=digest,
                    frame=image_input.frame,
                    spec=spec,
                    baseline_bbox=baseline_entry.bbox,
                    face_engine=face_engine,
                    store=store,
                )
            counters.record(entry, cache_hit)
            completed += 1
            pending_writes += int(not cache_hit)
            if pending_writes >= commit_every:
                store.commit()
                pending_writes = 0
            if on_progress is not None:
                on_progress(completed, total, source, spec, cache_hit)
    store.commit()
    return QualityExperimentRunSummary(
        source_total=len(sources),
        condition_total=len(selected_specs),
        request_total=counters.requests,
        cache_hit_total=counters.cache_hits,
        inference_total=counters.inferences,
        failure_total=counters.failures,
        gallery_references=gallery_references,
    )


@dataclass
class _RunCounters:
    """在参考图选择和正式条件循环之间共享运行计数。"""

    requests: int = 0
    cache_hits: int = 0
    inferences: int = 0
    failures: int = 0

    def record(self, entry: QualityExperimentEntry, cache_hit: bool) -> None:
        """累计一次缓存请求的命中、推理和失败状态。"""

        self.requests += 1
        self.cache_hits += int(cache_hit)
        self.inferences += int(not cache_hit)
        self.failures += int(entry.status == "failed")


def _select_gallery_references(
    *,
    protocol: QualityExperimentProtocol,
    dataset_dir: Path,
    settings: Settings,
    face_engine: QualityFaceEngine,
    store: QualityExperimentStore,
    counters: _RunCounters,
) -> dict[str, str]:
    """为每个 Known 身份选择第一张通过当前策略的干净参考图。"""

    baseline = DegradationSpec("baseline", 1.0)
    selected: dict[str, str] = {}
    for case in protocol.known:
        for relative_path in case.gallery_candidates:
            image_path = dataset_dir / relative_path
            image_input = ImageInput.from_file(image_path, source_type="dataset")
            digest = file_sha256(image_path)
            entry, cache_hit = measure_degraded_frame(
                relative_path=relative_path,
                file_sha256=digest,
                frame=image_input.frame,
                spec=baseline,
                baseline_bbox=None,
                face_engine=face_engine,
                store=store,
            )
            counters.record(entry, cache_hit)
            if (
                entry.status == "observed"
                and entry.metrics is not None
                and apply_quality_policy(entry.metrics, settings).tier != "reject"
            ):
                selected[case.person_id] = relative_path
                break
        if case.person_id not in selected:
            raise RuntimeError(f"no accepted baseline Gallery image for {case.person_id}")
    store.commit()
    return selected


def _put_baseline_failure(
    *,
    source: ExperimentSource,
    digest: str,
    spec: DegradationSpec,
    store: QualityExperimentStore,
) -> tuple[QualityExperimentEntry, bool]:
    """在基线无人脸时记录无法构造目标人脸尺寸条件。"""

    cached = store.get(source.relative_path, digest, spec)
    if cached is not None:
        return cached, True
    reason = "baseline_face_unavailable"
    store.put_failed(
        relative_path=source.relative_path,
        file_sha256=digest,
        spec=spec,
        face_count=0,
        latency_ms=0.0,
        reason=reason,
    )
    return QualityExperimentEntry("failed", None, None, None, 0, 0.0, reason), False
