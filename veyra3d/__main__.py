"""Explicit local research commands; no camera, download, or baseline switch."""

import argparse
import json
from pathlib import Path
import sys

from .config import ExperimentConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Veyra3D research foundation; no validated facial model yet")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("doctor", "probe forward/backward on synthetic data"),
                        ("smoke", "four optimizer steps + temporary checkpoint/resume; non-face synthetic data only")):
        command = commands.add_parser(name, help=help_)
        command.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    training = commands.add_parser("train", help="explicit configured training; do not run full experiments before data approval")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    evaluation = commands.add_parser("evaluate", help="evaluate a local checkpoint on a configured held-out split")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    for command in (training, evaluation):
        command.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    args = parser.parse_args(argv)
    try:
        if args.command in ("doctor", "smoke"):
            from .diagnostics import doctor, smoke
            result = (doctor if args.command == "doctor" else smoke)(args.device)
        else:
            config = ExperimentConfig.load(args.config)
            if args.device:
                config.device = args.device
            from .experiment import evaluate, train
            result = train(config, args.resume) if args.command == "train" else evaluate(config, args.checkpoint, args.split)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"Veyra3D: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
