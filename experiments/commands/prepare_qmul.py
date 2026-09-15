from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.face_evaluation.qmul_survface import build_qmul_protocol, write_qmul_protocol


def main() -> int:
    """读取 QMUL 官方 MAT 协议，检查图片并保存可回放协议摘要。"""

    parser = argparse.ArgumentParser(description="Prepare the QMUL-SurvFace open-set protocol.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "qmul-survface" / "QMUL-SurvFace",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "logs" / "qmul_survface_protocol.json",
    )
    args = parser.parse_args()
    try:
        protocol = build_qmul_protocol(args.dataset_root)
        write_qmul_protocol(protocol, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    known_probe_count = sum(probe.expected_person_id is not None for probe in protocol.probes)
    unknown_probe_count = len(protocol.probes) - known_probe_count
    report = {
        "protocol": "qmul-survface-open-set-v1",
        "gallery_person_count": len(protocol.enrollment),
        "gallery_image_total": sum(len(paths) for paths in protocol.enrollment.values()),
        "known_probe_total": known_probe_count,
        "unknown_probe_total": unknown_probe_count,
        "probe_total": len(protocol.probes),
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
