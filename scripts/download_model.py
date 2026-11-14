#!/usr/bin/env python3
"""Fetch the pinned, public pretrained baseline; never downloads at app launch."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import ssl
import sys
import tempfile
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_SIZE = 3_758_596
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Model is missing: {path}\nRun ./setup.sh to download it.")
    if path.stat().st_size != MODEL_SIZE or checksum(path) != MODEL_SHA256:
        raise ValueError(
            f"Model checksum does not match the pinned official baseline: {path}\n"
            "Move or remove that file, then run ./setup.sh again."
        )


def download(path: Path) -> None:
    if path.exists():
        verify(path)
        print(f"Model already verified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".face_landmarker.", suffix=".part", dir=path.parent, delete=False
        ) as output:
            partial = Path(output.name)
            request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Veyra-setup/0.1"})
            with urllib.request.urlopen(
                request, timeout=60, context=ssl.create_default_context()
            ) as response:
                total = 0
                while chunk := response.read(256 * 1024):
                    total += len(chunk)
                    if total > MODEL_SIZE:
                        raise ValueError("Downloaded model exceeds the pinned size; refusing it.")
                    output.write(chunk)
        verify(partial)
        partial.replace(path)
        print(f"Downloaded and verified official Face Landmarker v1: {path}")
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Model destination")
    parser.add_argument("--check", action="store_true", help="Verify the local model without network")
    args = parser.parse_args()
    try:
        if args.check:
            verify(args.output)
            print(f"Model verified: {args.output}")
        else:
            download(args.output)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        print(
            "Setup needs HTTPS access to storage.googleapis.com. Check your connection and "
            "trusted TLS certificates, then retry; certificate checks are never disabled.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
