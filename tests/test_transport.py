"""Synthetic volatile transport checks; no camera or real face data."""

import multiprocessing
import time
import unittest

import numpy as np

from veyra.transport import FrameMailbox


def packet(value=1, face=True):
    return {
        "rgb": np.full((12, 16, 3), value, dtype=np.uint8),
        "points": np.array([[-0.1, 0.1, 0.0], [0.1, -0.1, 0.05]], dtype=np.float32) if face else None,
        "edges": np.array([[0, 1]], dtype=np.int32),
        "fps": 25.0, "inference_ms": 8.0, "captured_at": 123.0,
    }


def child_publish(mailbox):
    try:
        mailbox.publish(packet(37))
    finally:
        mailbox.close()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("spawn")
        self.mailbox = FrameMailbox(self.context)
        self.addCleanup(self.mailbox.unlink)
        self.addCleanup(self.mailbox.close)

    def test_latest_frame_replaces_older_and_read_once(self):
        self.assertIsNone(self.mailbox.read_latest())
        self.assertTrue(self.mailbox.publish(packet(1)))
        self.assertTrue(self.mailbox.publish(packet(2)))
        received = self.mailbox.read_latest()
        np.testing.assert_array_equal(received["rgb"], packet(2)["rgb"])
        self.assertEqual(received["resolution"], (16, 12))
        self.assertEqual(received["fps"], 25)
        self.assertIsNone(self.mailbox.read_latest())

    def test_no_face_clears_previous_geometry(self):
        self.mailbox.publish(packet())
        self.mailbox.read_latest()
        self.mailbox.publish(packet(face=False))
        self.assertIsNone(self.mailbox.read_latest()["points"])

    def test_updates_and_reads_own_independent_copies(self):
        source = packet()
        self.mailbox.publish(source)
        source["rgb"][:] = 99
        source["points"][:] = 99
        received = self.mailbox.read_latest()
        self.assertEqual(int(received["rgb"][0, 0, 0]), 1)
        self.assertLess(float(received["points"].max()), 1)
        self.mailbox.publish(packet(2))
        self.assertEqual(int(received["rgb"][0, 0, 0]), 1)

    def test_busy_writer_cannot_block_gui_reader(self):
        self.mailbox._lock.acquire()
        try:
            start = time.perf_counter()
            self.assertIsNone(self.mailbox.read_latest())
            self.assertLess(time.perf_counter() - start, 0.05)
            self.assertFalse(self.mailbox.publish(packet()))
        finally:
            self.mailbox._lock.release()

    def test_invalid_payload_never_changes_last_complete_frame(self):
        self.mailbox.publish(packet(7))
        invalid = packet()
        invalid["rgb"] = np.zeros((1081, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.mailbox.publish(invalid)
        self.assertEqual(int(self.mailbox.read_latest()["rgb"][0, 0, 0]), 7)
        for change in (
            {"rgb": np.zeros((4, 4), dtype=np.uint8)},
            {"rgb": np.zeros((4, 4, 3), dtype=np.float32)},
            {"points": np.zeros((479, 3), dtype=np.float32)},
            {"points": np.full((2, 3), np.nan)},
            {"edges": np.array([[0, 2]])},
            {"fps": float("nan")},
        ):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    self.mailbox.publish(packet() | change)

    def test_spawned_process_shares_payload_without_queue_feeder(self):
        worker = self.context.Process(target=child_publish, args=(self.mailbox,))
        worker.start()
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
            self.fail("Synthetic transport worker failed to exit")
        self.assertEqual(worker.exitcode, 0)
        received = self.mailbox.read_latest()
        self.assertIsNotNone(received)
        self.assertEqual(int(received["rgb"][0, 0, 0]), 37)
        worker.close()


if __name__ == "__main__":
    unittest.main()
