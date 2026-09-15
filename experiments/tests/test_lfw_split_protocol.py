from __future__ import annotations

import json

from experiments.face_evaluation.lfw_dataset import (
    LfwProbe,
    LfwProtocol,
    read_lfw_split_protocol,
    split_lfw_protocol,
    write_lfw_split_protocol,
)


def test_lfw_split_protocol_is_reproducible_and_keeps_probe_identities_disjoint(
    tmp_path,
) -> None:
    """固定种子必须复现分区，且同一探针身份不能跨越两个分区。"""

    protocol = LfwProtocol(
        enrollment={name: (f"{name}/{name}_0001.jpg",) for name in ("A", "B", "C", "D")},
        probes=tuple(
            [LfwProbe(f"{name}/{name}_0002.jpg", name) for name in ("A", "B", "C", "D")]
            + [LfwProbe(f"U{index}/U{index}_0001.jpg", None) for index in range(1, 5)]
        ),
    )

    first = split_lfw_protocol(protocol, seed=2026, calibration_fraction=0.5)
    second = split_lfw_protocol(protocol, seed=2026, calibration_fraction=0.5)

    assert first == second
    assert first.enrollment == protocol.enrollment
    assignments: dict[str, set[str]] = {}
    for probe in first.probes:
        assignments.setdefault(probe.source_identity, set()).add(probe.split)
    assert all(len(splits) == 1 for splits in assignments.values())
    assert {probe.split for probe in first.probes} == {"calibration", "evaluation"}

    output = tmp_path / "protocol.json"
    write_lfw_split_protocol(first, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["protocol"] == "lfw-decision-split-v1"
    assert payload["seed"] == 2026
    assert payload["calibration_fraction"] == 0.5
    assert payload["source_protocol_sha256"]
    assert read_lfw_split_protocol(output) == first
