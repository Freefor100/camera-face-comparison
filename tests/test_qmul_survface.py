from __future__ import annotations

import numpy as np

from camera_face_comparison.qmul_survface import build_qmul_protocol


def test_qmul_protocol_uses_mat_labels_and_unmated_directory(monkeypatch, tmp_path) -> None:
    """QMUL 协议必须使用官方 MAT 身份标签，并把 unmated probe 设为 Unknown。"""

    root = tmp_path / "QMUL-SurvFace"
    test_set = root / "Face_Identification_Test_Set"
    for directory in (test_set / "gallery", test_set / "mated_probe", test_set / "unmated_probe"):
        directory.mkdir(parents=True)
    (test_set / "gallery" / "100_cam1_1.jpg").write_bytes(b"gallery")
    (test_set / "mated_probe" / "100_cam3_1.jpg").write_bytes(b"mated")
    (test_set / "unmated_probe" / "999_cam2_1.jpg").write_bytes(b"unmated")

    from camera_face_comparison import qmul_survface

    def fake_loadmat(path):
        """按标签文件名返回最小的官方字段结构。"""

        if path.name == "gallery_img_ID_pairs.mat":
            return {
                "gallery_ids": np.array([[7]], dtype=np.uint16),
                "gallery_set": np.array([["100_cam1_1.jpg"]]),
            }
        return {
            "mated_probe_ids": np.array([[7]], dtype=np.uint16),
            "mated_probe_set": np.array([["100_cam3_1.jpg"]]),
        }

    monkeypatch.setattr(qmul_survface, "_load_mat", fake_loadmat)
    protocol = build_qmul_protocol(root)

    assert protocol.enrollment == {"7": ("gallery/100_cam1_1.jpg",)}
    assert protocol.probes[0].relative_path == "mated_probe/100_cam3_1.jpg"
    assert protocol.probes[0].expected_person_id == "7"
    assert protocol.probes[1].relative_path == "unmated_probe/999_cam2_1.jpg"
    assert protocol.probes[1].expected_person_id is None
