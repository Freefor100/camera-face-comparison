from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.lfw_dataset import read_lfw_split_protocol
from camera_face_comparison.quality_experiment import (
    build_quality_experiment_protocol,
    write_quality_experiment_protocol,
)


def main() -> int:
    """从 Phase 2 分区和无阈值分数生成 Phase 3 固定实验协议。"""

    parser = argparse.ArgumentParser(
        description="Prepare a Calibration-only LFW face-quality experiment protocol."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--split-protocol", type=Path)
    parser.add_argument("--decision-scores", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unknown-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    phase2_dir = args.data_dir / "experiments" / "phase2"
    protocol_path = args.split_protocol or phase2_dir / "protocol.json"
    scores_path = args.decision_scores or phase2_dir / "decision_scores.sqlite"
    output_path = args.output or args.data_dir / "experiments" / "phase3" / "protocol.json"
    try:
        split_protocol = read_lfw_split_protocol(protocol_path)
        protocol = build_quality_experiment_protocol(
            split_protocol,
            scores_path,
            unknown_count=args.unknown_count,
            seed=args.seed,
        )
        write_quality_experiment_protocol(protocol, output_path)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(output_path),
                "known_identity_total": len(protocol.known),
                "unknown_identity_total": len(protocol.unknown),
                "seed": protocol.seed,
                "source_split_protocol_sha256": protocol.source_split_protocol_sha256,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
