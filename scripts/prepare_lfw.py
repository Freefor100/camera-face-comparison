from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings
from camera_face_comparison.lfw_dataset import (
    build_lfw_protocol,
    ensure_lfw_dataset,
    write_lfw_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible LFW open-set face-recognition evaluation split."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--download", action="store_true", help="download LFW only when it is absent")
    parser.add_argument("--known-identities", type=int, default=3)
    parser.add_argument("--unknown-identities", type=int, default=3)
    parser.add_argument("--enrollment-per-identity", type=int, default=5)
    parser.add_argument("--probes-per-identity", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.data_dir)
    try:
        dataset_dir = ensure_lfw_dataset(settings.data_dir, download=args.download)
        protocol = build_lfw_protocol(
            dataset_dir,
            known_identity_count=args.known_identities,
            unknown_identity_count=args.unknown_identities,
            enrollment_per_identity=args.enrollment_per_identity,
            probes_per_identity=args.probes_per_identity,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    output = args.output or settings.data_dir / "datasets" / "lfw_open_set_protocol.json"
    write_lfw_protocol(protocol, output)
    print(
        json.dumps(
            {
                "protocol": "lfw-open-set-v1",
                "output": str(output),
                "enrolled_identities": len(protocol.enrollment),
                "probes": len(protocol.probes),
                "unknown_probes": sum(probe.expected_person_id is None for probe in protocol.probes),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
