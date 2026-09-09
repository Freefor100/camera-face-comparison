from __future__ import annotations

import numpy as np

from camera_face_comparison.evaluation_cache import file_sha256
from camera_face_comparison.lfw_dataset import LfwProbe, LfwProtocol
from camera_face_comparison.qmul_evaluation import (
    export_qmul_pressure_scores,
    summarize_qmul_pressure_scores,
)
from camera_face_comparison.raw_embedding_cache import RawEmbeddingCache


def test_qmul_pressure_export_separates_rank1_fte_and_transferred_policy(tmp_path) -> None:
    """QMUL 压力结果必须区分排序错误、模型 FTE 和外部工作点拒绝。"""

    dataset_dir = tmp_path / "Face_Identification_Test_Set"
    paths = {
        "gallery/a.jpg": np.array([1.0, 0.0], dtype=np.float32),
        "gallery/b.jpg": np.array([0.0, 1.0], dtype=np.float32),
        "mated_probe/a.jpg": np.array([0.9, 0.1], dtype=np.float32),
        "mated_probe/b_wrong.jpg": np.array([1.0, 0.0], dtype=np.float32),
        "unmated_probe/u.jpg": np.array([-1.0, 0.0], dtype=np.float32),
    }
    for relative_path in (*paths, "unmated_probe/failed.jpg"):
        path = dataset_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode())

    cache_path = tmp_path / "raw.sqlite"
    with RawEmbeddingCache(cache_path, "qmul-test", "model-test") as cache:
        for relative_path, embedding in paths.items():
            cache.put_observed(
                relative_path=relative_path,
                file_sha256=file_sha256(dataset_dir / relative_path),
                embedding=embedding,
                metrics={"quality_score": 0.5},
                face_count=1,
                latency_ms=2.0,
            )
        failed_path = "unmated_probe/failed.jpg"
        cache.put_failed(
            relative_path=failed_path,
            file_sha256=file_sha256(dataset_dir / failed_path),
            face_count=0,
            latency_ms=1.0,
            reason="no_face",
        )
        cache.commit()
        protocol = LfwProtocol(
            enrollment={"a": ("gallery/a.jpg",), "b": ("gallery/b.jpg",)},
            probes=(
                LfwProbe("mated_probe/a.jpg", "a"),
                LfwProbe("mated_probe/b_wrong.jpg", "b"),
                LfwProbe("unmated_probe/u.jpg", None),
                LfwProbe(failed_path, None),
            ),
        )
        scores_path = tmp_path / "scores.sqlite"
        summary = export_qmul_pressure_scores(
            dataset_dir=dataset_dir,
            protocol=protocol,
            cache=cache,
            output_path=scores_path,
            run_id="run-test",
            protocol_sha256="protocol-test",
            embedding_extraction_id="model-test",
            batch_size=2,
        )

    assert summary.gallery_valid_identity_total == 2
    assert summary.mated_valid_probe_total == 2
    assert summary.unmated_valid_probe_total == 1
    assert summary.probe_fte_total == 1
    report = summarize_qmul_pressure_scores(
        scores_path,
        run_id="run-test",
        transferred_policy={
            "source": "lfw-test",
            "method": "mean_prototype",
            "match_threshold": -1.0,
            "use_score_gap": True,
            "min_score_gap": 0.5,
        },
    )
    assert report["mated"]["rank1_correct"] == 1
    assert report["mated"]["valid_rank1_rate"] == 0.5
    assert report["unmated"]["model_fte"] == 1
    assert report["transferred_policy"]["known_true_accepts"] == 1
    assert report["transferred_policy"]["known_wrong_accepts"] == 1
    assert report["transferred_policy"]["unknown_false_accepts"] == 1
    assert report["failure_reasons"]["no_face"] == 1
