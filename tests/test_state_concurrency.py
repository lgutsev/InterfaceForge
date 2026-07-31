from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from interfaceforge.state import StateStore


class StateConcurrencyTests(unittest.TestCase):
    def test_concurrent_events_are_not_lost(self) -> None:
        # Simulates parallel Slurm array tasks (or overlapping `iface`
        # invocations) each recording one provenance event at roughly the
        # same time. Without locking around the read-modify-write in
        # StateStore.event(), the last writer to save() wins and earlier
        # threads' events are silently dropped.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_count = 24

            def record(index: int) -> None:
                StateStore(root).event("write", worker=index)

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(record, range(worker_count)))

            state = StateStore(root).load()
            recorded_workers = {event["details"]["worker"] for event in state["events"]}
            self.assertEqual(len(state["events"]), worker_count)
            self.assertEqual(recorded_workers, set(range(worker_count)))

    def test_concurrent_artifacts_are_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_count = 16

            def record(index: int) -> None:
                source = root / f"artifact_{index}.txt"
                source.write_text(str(index), encoding="utf-8")
                StateStore(root).artifact(f"artifact_{index}", source)

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(record, range(worker_count)))

            state = StateStore(root).load()
            self.assertEqual(len(state["artifacts"]), worker_count)


if __name__ == "__main__":
    unittest.main()
