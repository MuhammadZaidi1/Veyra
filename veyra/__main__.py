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
    parser.add_argument("--renderer", choices=("auto", "opengl", "software"), default="auto", help="surface OpenGL or legacy software wireframe")
    parser.add_argument("--mode", choices=("wireframe", "solid", "combined"), default="solid")
    parser.add_argument("--no-smoothing", action="store_true", help="disable additional viewer smoothing")
    parser.add_argument("--smoothing-cutoff", type=float, default=4.0, help="minimum response frequency, 1–10 Hz; higher reduces lag")
    parser.add_argument("--smoothing-beta", type=float, default=20.0, help="motion adaptation, 0–100 (default: 20)")
    parser.add_argument("--benchmark", action="store_true", help="synthetic smoothing and renderer benchmark; no camera")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if not (160 <= args.width <= 1920 and 120 <= args.height <= 1080):
        parser.error("capture size must be 160–1920 wide and 120–1080 high")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if not 0 <= args.duration <= 86400:
        parser.error("--duration must be 0–86400 seconds")
    if not 1 <= args.smoothing_cutoff <= 10 or not 0 <= args.smoothing_beta <= 100:
        parser.error("smoothing cutoff must be 1–10 Hz and beta 0–100")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_model(args.model)
        if args.benchmark:
            from .diagnostics import benchmark
            return benchmark(args.model)
        if args.self_test:
            from .tracking import self_test
            self_test(args.model)
            return 0
        from .app import run
        return run(Config(model=args.model, camera=args.camera, width=args.width, height=args.height,
                          fps=args.fps, mirror=not args.no_mirror, renderer=args.renderer,
                          display_mode={"solid": "Solid", "wireframe": "Wireframe", "combined": "Solid + Wireframe"}[args.mode],
                          smoothing=not args.no_smoothing, smoothing_cutoff=args.smoothing_cutoff,
                          smoothing_beta=args.smoothing_beta),
                   args.duration)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Veyra: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
