from __future__ import annotations

import numpy as np

from camera_face_comparison.gallery_scale_experiment import (
    GalleryScaleInputs,
    run_gallery_scale_experiment,
)
from camera_face_comparison.joint_calibration import JointOperatingPoint


def test_gallery_scale_replay_is_deterministic_and_uses_calibration_only() -> None:
    """小库门槛必须固定复现，且输出不得读取或产生 Evaluation 分区结果。"""

    vectors = {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "carol": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    inputs = GalleryScaleInputs(
        gallery_samples={
            domain: {person_id: (vector,) for person_id, vector in vectors.items()}
            for domain in ("natural", "mixed")
        },
        known_probes={
            domain: {person_id: (vector,) for person_id, vector in vectors.items()}
            for domain in ("natural", "xqlfw")
        },
        unknown_probes={
            "natural": (np.array([-1.0, 0.0, 0.0], dtype=np.float32),),
            "xqlfw": (np.array([0.0, -1.0, 0.0], dtype=np.float32),),
        },
        known_protocol_counts={person_id: 1 for person_id in vectors},
        unknown_protocol_total=1,
        coverage={"source": "synthetic"},
    )
    selected = JointOperatingPoint(
        aggregation_method="mean_prototype",
        top_k=0,
        rule="score_threshold",
        target_fpir=0.01,
        minimum_score=0.5,
        minimum_gap=None,
        minimum_probability=None,
        nac_neighbors=None,
        meets_all_scenarios=True,
        worst_tpir_e2e=1.0,
        average_tpir_e2e=1.0,
        worst_fpir_valid=0.0,
        scenarios=(),
    )

    first = run_gallery_scale_experiment(
        inputs,
        selected=selected,
        gallery_sizes=(2,),
        repeats=2,
        seed=2026,
    )
    second = run_gallery_scale_experiment(
        inputs,
        selected=selected,
        gallery_sizes=(2,),
        repeats=2,
        seed=2026,
    )

    assert first == second
    assert len(first["runs"]) == 8
    assert {run["scenario"] for run in first["runs"]} == {
        "natural_gallery__natural_probe",
        "natural_gallery__xqlfw_probe",
        "mixed_gallery__natural_probe",
        "mixed_gallery__xqlfw_probe",
    }
    assert "evaluation" not in repr(first).lower()
