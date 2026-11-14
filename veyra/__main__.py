"""Launch with python -m veyra; startup never downloads a model."""

import argparse
from pathlib import Path
import sys

from .core import Config, DEFAULT_MODEL, validate_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Veyra: local estimated relative 3D face mesh (CPU)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="local official baseline model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="requested capture width (advisory)")
    parser.add_argument("--height", type=int, default=480, help="requested capture height (advisory)")
    parser.add_argument("--fps", type=int, default=30, help="maximum processing rate, 1–60 (default: 30)")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera preview")
    parser.add_argument("--duration", type=float, default=0, help="auto-close after N seconds; 0 means until quit")
    parser.add_argument("--self-test", action="store_true", help="test CPU model with blank frames; no camera or GUI")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if not (160 <= args.width <= 1920 and 120 <= args.height <= 1080):
        parser.error("capture size must be 160–1920 wide and 120–1080 high")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if not 0 <= args.duration <= 86400:
        parser.error("--duration must be 0–86400 seconds")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_model(args.model)
        if args.self_test:
            from .tracking import self_test
            self_test(args.model)
            return 0
        from .app import run
        return run(Config(args.model, args.camera, args.width, args.height, args.fps, not args.no_mirror),
                   args.duration)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Veyra: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
