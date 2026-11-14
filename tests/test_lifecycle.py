"""Opt-in native Cocoa lifecycle regressions; all displayed data is synthetic.

Run on macOS with VEYRA_NATIVE_UI_TEST=1. Each test opens brief native windows
and uses the actual spawned-process/shared-memory transport, without a camera.
Offscreen Qt does not reproduce Cocoa's recursive quit behavior.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_ENABLED = sys.platform == "darwin" and os.environ.get("VEYRA_NATIVE_UI_TEST") == "1"


@unittest.skipUnless(NATIVE_ENABLED, "set VEYRA_NATIVE_UI_TEST=1 on macOS for native lifecycle checks")
class NativeLifecycleTests(unittest.TestCase):
    def run_native(self, body):
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "cocoa"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
        environment["XDG_CACHE_HOME"] = str(ROOT / ".cache")
        prefix = """
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path.cwd() / "tests"))
            from synthetic_worker import run_synthetic_worker
            from veyra import app
            from veyra.core import Config
            app.run_worker = run_synthetic_worker
        """
        command = [
            "/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)",
            sys.executable, "-u", "-c", textwrap.dedent(prefix) + textwrap.dedent(body),
        ]
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=15)
        diagnostic = result.stdout + "\n" + result.stderr
        self.assertEqual(result.returncode, 0, diagnostic)
        self.assertIn("Camera active: 640 × 480", result.stdout, diagnostic)
        self.assertRegex(result.stdout, r"Displayed [1-9][0-9]* previews\.", diagnostic)
        self.assertEqual(result.stdout.count("Veyra stopped; camera worker released."), 1,
                         diagnostic)
        self.assertIn("SYNTHETIC_RESULT=0", result.stdout, diagnostic)
        self.assertNotIn("leaked semaphore", result.stderr, diagnostic)
        self.assertNotIn("leaked shared_memory", result.stderr, diagnostic)
        self.assertNotIn("Traceback", result.stderr, diagnostic)

    def test_duration_quit_releases_spawned_worker_and_shared_memory(self):
        self.run_native("""
            result = app.run(Config(), 0.8)
            print(f"SYNTHETIC_RESULT={result}", flush=True)
            raise SystemExit(result)
        """)

    def test_preview_close_releases_worker_and_closes_both_windows(self):
        self.run_native("""
            from PySide6 import QtCore, QtWidgets
            qt_app = QtWidgets.QApplication([])
            controller = app.Controller(qt_app, Config())
            QtCore.QTimer.singleShot(800, controller.preview.close)
            qt_app.exec()
            controller.shutdown()
            result = int(controller.error or not controller.stopped
                         or controller.preview.isVisible() or controller.geometry.isVisible())
            print(f"SYNTHETIC_RESULT={result}", flush=True)
            raise SystemExit(result)
        """)


if __name__ == "__main__":
    unittest.main()
