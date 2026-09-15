from __future__ import annotations

import sqlite3

import numpy as np

from experiments.face_evaluation.cross_quality_experiment import (
    CROSS_QUALITY_SCENARIOS,
    export_cross_quality_scores,
    load_joint_score_rows,
    mixed_gallery_domains,
)
from experiments.face_evaluation.experiment_artifacts import file_sha256
from experiments.face_evaluation.lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from experiments.face_evaluation.raw_embedding_cache import RawEmbeddingCache


def test_mixed_gallery_is_reproducible_and_mixes_multi_sample_identities() -> None:
    """路径顺序变化不能改变混合结果，多样本身份不能全部落入同一域。"""

    enrollment = {
        "Alice": ("Alice/2.jpg", "Alice/1.jpg", "Alice/3.jpg"),
        "Bob": ("Bob/1.jpg",),
    }

    first = mixed_gallery_domains(enrollment, seed=2026)
    second = mixed_gallery_domains(
        {"Bob": enrollment["Bob"], "Alice": tuple(reversed(enrollment["Alice"]))},
        seed=2026,
    )

    assert first == second
    assert {first[path] for path in enrollment["Alice"]} == {"natural", "xqlfw"}


def test_cross_quality_export_saves_rankings_without_decision_parameters(tmp_path) -> None:
    """六场景分数库必须保存候选排序和 FTE，但不能保存任何接收阈值。"""

    natural_dir = tmp_path / "natural"
    xqlfw_dir = tmp_path / "xqlfw"
    enrollment = {
        "Alice": ("Alice/Alice_0001.jpg", "Alice/Alice_0002.jpg"),
        "Bob": ("Bob/Bob_0001.jpg", "Bob/Bob_0002.jpg"),
    }
    probes = (
        LfwSplitProbe("Alice/Alice_0003.jpg", "Alice", "Alice", "calibration"),
        LfwSplitProbe("Unknown/Unknown_0001.jpg", None, "Unknown", "evaluation"),
    )
    protocol = LfwSplitProtocol(
        enrollment=enrollment,
        probes=probes,
        seed=2026,
        calibration_fraction=0.5,
        source_protocol_sha256="source",
    )
    vectors = {
        "Alice/Alice_0001.jpg": np.array([1.0, 0.0], dtype=np.float32),
        "Alice/Alice_0002.jpg": np.array([0.8, 0.2], dtype=np.float32),
        "Bob/Bob_0001.jpg": np.array([0.0, 1.0], dtype=np.float32),
        "Bob/Bob_0002.jpg": np.array([0.2, 0.8], dtype=np.float32),
        "Alice/Alice_0003.jpg": np.array([0.9, 0.1], dtype=np.float32),
        "Unknown/Unknown_0001.jpg": np.array([0.6, 0.8], dtype=np.float32),
    }
    metrics = {"face_size_px": 120.0, "blur_variance": 100.0}
    for directory in (natural_dir, xqlfw_dir):
        for relative_path in vectors:
            path = directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{directory.name}:{relative_path}".encode())

    natural_cache_path = tmp_path / "natural.sqlite"
    xqlfw_cache_path = tmp_path / "xqlfw.sqlite"
    with RawEmbeddingCache(natural_cache_path, "natural", "model") as cache:
        for relative_path, vector in vectors.items():
            cache.put_observed(
                relative_path=relative_path,
                file_sha256=file_sha256(natural_dir / relative_path),
                embedding=vector,
                metrics=metrics,
                face_count=1,
                latency_ms=1.0,
            )
    with RawEmbeddingCache(xqlfw_cache_path, "xqlfw", "model") as cache:
        for relative_path, vector in vectors.items():
            if relative_path == "Unknown/Unknown_0001.jpg":
                cache.put_failed(
                    relative_path=relative_path,
                    file_sha256=file_sha256(xqlfw_dir / relative_path),
                    face_count=0,
                    latency_ms=2.0,
                    reason="no_face_detected",
                )
            else:
                cache.put_observed(
                    relative_path=relative_path,
                    file_sha256=file_sha256(xqlfw_dir / relative_path),
                    embedding=vector[::-1].copy(),
                    metrics=metrics,
                    face_count=1,
                    latency_ms=2.0,
                )

    output = tmp_path / "scores.sqlite"
    with (
        RawEmbeddingCache(natural_cache_path, "natural", "model") as natural_cache,
        RawEmbeddingCache(xqlfw_cache_path, "xqlfw", "model") as xqlfw_cache,
    ):
        summary = export_cross_quality_scores(
            natural_dataset_dir=natural_dir,
            xqlfw_dataset_dir=xqlfw_dir,
            protocol=protocol,
            natural_cache=natural_cache,
            xqlfw_cache=xqlfw_cache,
            output_path=output,
            run_id="run-a",
            protocol_sha256="protocol",
            embedding_extraction_id="model",
            seed=2026,
        )

    assert summary.scenario_total == len(CROSS_QUALITY_SCENARIOS) == 6
    assert summary.score_record_total == 54
    assert summary.probe_rejection_total == 3

    connection = sqlite3.connect(output)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(rankings)")}
    score_rows = connection.execute(
        "SELECT candidate_count, candidate_indices, candidate_scores "
        "FROM rankings ORDER BY scenario, relative_path, method, top_k"
    ).fetchall()
    rejections = connection.execute(
        "SELECT scenario, relative_path, reason FROM rejections ORDER BY scenario"
    ).fetchall()
    scenario_rows = connection.execute(
        "SELECT name, participates_in_selection FROM scenarios ORDER BY name"
    ).fetchall()
    connection.close()

    assert "minimum_score" not in columns
    assert "minimum_gap" not in columns
    assert all(row[0] == 2 for row in score_rows)
    assert all(len(np.frombuffer(row[1], dtype=np.int32)) == 2 for row in score_rows)
    assert all(len(np.frombuffer(row[2], dtype=np.float32)) == 2 for row in score_rows)
    assert len(rejections) == 3
    assert all(row[1:] == ("Unknown/Unknown_0001.jpg", "no_face_detected") for row in rejections)
    assert sum(row[1] for row in scenario_rows) == 4

    primary_scenarios = tuple(
        scenario.name for scenario in CROSS_QUALITY_SCENARIOS if scenario.participates_in_selection
    )
    calibration_rows, protocol_known_totals = load_joint_score_rows(
        output,
        run_id="run-a",
        split="calibration",
        scenario_names=primary_scenarios,
        method="mean_prototype",
        top_k=0,
    )
    assert len(calibration_rows) == 4
    assert {row.scenario for row in calibration_rows} == set(primary_scenarios)
    assert protocol_known_totals == {scenario: 1 for scenario in primary_scenarios}
