from __future__ import annotations

import json
import sqlite3

import numpy as np

from experiments.face_evaluation.decision_scores import (
    export_lfw_decision_scores,
    summarize_decision_scores,
)
from experiments.face_evaluation.experiment_artifacts import (
    embedding_extraction_id,
    file_sha256,
)
from experiments.face_evaluation.lfw_dataset import LfwSplitProbe, LfwSplitProtocol
from experiments.face_evaluation.raw_embedding_cache import RawEmbeddingCache


def test_embedding_id_only_depends_on_model_identity() -> None:
    """特征提取标识只能随模型变化，不能包含后续判定参数。"""

    assert embedding_extraction_id() == embedding_extraction_id("buffalo_l")
    assert embedding_extraction_id("buffalo_l") != embedding_extraction_id("antelopev2")


def test_cache_only_export_saves_six_threshold_free_scores_and_rejections(tmp_path) -> None:
    """缓存导出必须保存六种聚合结果，且不需要人脸引擎或判定阈值。"""

    dataset_dir = tmp_path / "lfw"
    paths = {
        "a1": "Alice/Alice_0001.jpg",
        "a2": "Alice/Alice_0002.jpg",
        "b1": "Bob/Bob_0001.jpg",
        "b2": "Bob/Bob_0002.jpg",
        "known": "Alice/Alice_0003.jpg",
        "unknown": "Unknown/Unknown_0001.jpg",
        "rejected": "Rejected/Rejected_0001.jpg",
    }
    for relative_path in paths.values():
        path = dataset_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode("utf-8"))

    protocol = LfwSplitProtocol(
        enrollment={
            "Alice": (paths["a1"], paths["a2"]),
            "Bob": (paths["b1"], paths["b2"]),
        },
        probes=(
            LfwSplitProbe(paths["known"], "Alice", "Alice", "calibration"),
            LfwSplitProbe(paths["unknown"], None, "Unknown", "evaluation"),
            LfwSplitProbe(paths["rejected"], None, "Rejected", "evaluation"),
        ),
        seed=2026,
        calibration_fraction=0.5,
        source_protocol_sha256="source-hash",
    )
    metrics = {
        "detection_score": 0.95,
        "face_size_px": 160.0,
        "blur_variance": 140.0,
        "brightness": 120.0,
        "contrast": 30.0,
    }
    embeddings = {
        paths["a1"]: np.array([1.0, 0.0], dtype=np.float32),
        paths["a2"]: np.array([0.8, 0.6], dtype=np.float32),
        paths["b1"]: np.array([0.0, 1.0], dtype=np.float32),
        paths["b2"]: np.array([0.6, 0.8], dtype=np.float32),
        paths["known"]: np.array([1.0, 0.0], dtype=np.float32),
        paths["unknown"]: np.array([0.70710677, 0.70710677], dtype=np.float32),
    }
    cache_path = tmp_path / "embeddings.sqlite"
    with RawEmbeddingCache(cache_path, "lfw", "model-a") as cache:
        for relative_path, embedding in embeddings.items():
            cache.put_observed(
                relative_path=relative_path,
                file_sha256=file_sha256(dataset_dir / relative_path),
                embedding=embedding,
                metrics=metrics,
                face_count=1,
                latency_ms=10.0,
            )
        cache.put_failed(
            relative_path=paths["rejected"],
            file_sha256=file_sha256(dataset_dir / paths["rejected"]),
            face_count=0,
            latency_ms=10.0,
            reason="no_face_detected",
        )
        cache.commit()
        summary = export_lfw_decision_scores(
            dataset_dir=dataset_dir,
            protocol=protocol,
            cache=cache,
            output_path=tmp_path / "decision_scores.sqlite",
            run_id="run-a",
            protocol_sha256="split-hash",
            embedding_extraction_id="model-a",
        )

    assert summary.gallery_valid_image_total == 4
    assert summary.valid_probe_total == 2
    assert summary.probe_rejection_total == 1
    assert summary.score_record_total == 12

    connection = sqlite3.connect(tmp_path / "decision_scores.sqlite")
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(decision_scores)").fetchall()
    }
    run_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
    }
    rows = connection.execute(
        "SELECT method, top_k, top_person_id, top_score, second_score, score_gap "
        "FROM decision_scores WHERE relative_path = ? ORDER BY method, top_k",
        (paths["known"],),
    ).fetchall()
    methods = {(row[0], row[1]) for row in rows}
    connection.close()

    assert "minimum_score" not in columns
    assert "minimum_gap" not in columns
    assert "probe_quality_tier" not in columns
    assert "probe_quality_summary" not in columns
    assert "embedding_extraction_id" in run_columns
    assert methods == {
        ("single", 0),
        ("max", 0),
        ("mean_prototype", 0),
        ("top_k_mean", 2),
        ("top_k_mean", 3),
        ("top_k_mean", 5),
    }
    assert all(row[2] == "Alice" for row in rows)
    assert all(row[3] >= row[4] for row in rows)
    assert all(row[5] >= 0.0 for row in rows)

    calibration = summarize_decision_scores(
        tmp_path / "decision_scores.sqlite", run_id="run-a", split="calibration"
    )
    evaluation = summarize_decision_scores(
        tmp_path / "decision_scores.sqlite", run_id="run-a", split="evaluation"
    )
    assert calibration["valid_probe_total"] == 1
    assert calibration["known_probe_total"] == 1
    assert evaluation["valid_probe_total"] == 1
    assert evaluation["unknown_probe_total"] == 1
    assert len(calibration["aggregation_variants"]) == 6
    assert "minimum_score" not in json.dumps(calibration)
    assert "minimum_gap" not in json.dumps(calibration)
