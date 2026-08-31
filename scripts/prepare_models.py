from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera_face_comparison.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the InsightFace buffalo_l model for later offline use."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    settings = load_settings(args.data_dir)
    target = settings.models_dir / "buffalo_l"
    if target.is_dir():
        print(f"Offline model is already available at {target}")
        return 0
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("InsightFace is not installed. Install project dependencies first.", file=sys.stderr)
        return 1

    print("Downloading buffalo_l into the local data directory. This step needs network access.")
    FaceAnalysis(
        name="buffalo_l",
        root=str(settings.data_dir),
        providers=["CPUExecutionProvider"],
    )
    if not target.is_dir():
        print(f"Model download did not create {target}", file=sys.stderr)
        return 1
    print(f"Offline model prepared at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
