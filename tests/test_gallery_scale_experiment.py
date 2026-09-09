from __future__ import annotations

import numpy as np

from camera_face_comparison.calibration import OperatingPoint
from camera_face_comparison.gallery_scale_experiment import (
    GalleryScaleInputs,
    run_gallery_scale_experiment,
)


def test_gallery_scale_repeats_are_deterministic_and_identity_disjoint() -> None:
    """固定种子必须复现规模实验，且 Calibration/Evaluation 身份不能重叠。"""

    vectors = {
        "c1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "c3": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "e1": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "e2": np.array([0.0, 0.9, 0.1], dtype=np.float32),
        "e3": np.array([0.1, 0.0, 0.9], dtype=np.float32),
    }
    inputs = GalleryScaleInputs(
        prototypes=vectors,
        known_probes={
            "calibration": {name: (vector,) for name, vector in vectors.items() if name[0] == "c"},
            "evaluation": {name: (vector,) for name, vector in vectors.items() if name[0] == "e"},
        },
        unknown_probes={
            "calibration": (np.array([-1.0, 0.0, 0.0], dtype=np.float32),),
            "evaluation": (np.array([0.0, -1.0, 0.0], dtype=np.float32),),
        },
        gallery_image_total=6,
        gallery_valid_image_total=6,
        gallery_fte_total=0,
        probe_total=8,
        probe_valid_total=8,
        probe_fte_total=0,
    )
    transferred = OperatingPoint(
        target_fpir=0.003,
        use_score_gap=True,
        match_threshold=-1.0,
        min_score_gap=0.5,
        meets_target=True,
        known_total=1,
        rank1_correct=1,
        known_true_accepts=1,
        known_wrong_accepts=0,
        unknown_total=1,
        unknown_false_accepts=0,
        fpir=0.0,
        tpir=1.0,
        fnir=0.0,
    )

    first = run_gallery_scale_experiment(
        inputs,
        gallery_sizes=(2,),
        repeats=2,
        seed=2026,
        target_fpir=0.003,
        transferred_policy=transferred,
    )
    second = run_gallery_scale_experiment(
        inputs,
        gallery_sizes=(2,),
        repeats=2,
        seed=2026,
        target_fpir=0.003,
        transferred_policy=transferred,
    )

    assert first == second
    assert len(first["runs"]) == 2
    for run in first["runs"]:
        assert set(run["calibration_gallery_ids"]).isdisjoint(run["evaluation_gallery_ids"])
