"""Small file-backed provenance store for restartable campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class StateLockTimeout(RuntimeError):
    """Raised when the campaign state lock cannot be acquired in time."""


class _FileLock:
    """Minimal cross-platform advisory lock via O_CREAT|O_EXCL.

    StateStore.event()/artifact() are otherwise an unprotected
    read-modify-write of state.json: concurrent callers (parallel Slurm
    array tasks, or overlapping `iface` invocations) can each load the
    same old state and the last writer silently discards the others'
    events/artifacts. A stale lock (left behind by a killed process) is
    broken after `stale_after` seconds so one crashed job cannot wedge
    provenance writes for the rest of the campaign.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 30.0,
        poll: float = 0.05,
        stale_after: float = 300.0,
    ) -> None:
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after
        self._fd: int | None = None

    def __enter__(self) -> _FileLock:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except (FileExistsError, PermissionError):
                # On Windows, a contested O_CREAT|O_EXCL under heavy
                # contention can surface as PermissionError rather than
                # FileExistsError; treat both as "lock currently held".
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_after:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise StateLockTimeout(f"Timed out waiting for state lock: {self.path}") from None
                time.sleep(self.poll)

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


class StateStore:
    """Atomic JSON state with intentionally minimal workflow semantics."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.root = Path(campaign_root).resolve()
        self.directory = self.root / ".interfaceforge"
        self.path = self.directory / "state.json"
        self._lock_path = self.directory / "state.lock"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": 1,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "host": platform.node(),
                "python": sys.version.split()[0],
                "events": [],
                "artifacts": {},
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        temporary = self.path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def event(self, action: str, **details: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with _FileLock(self._lock_path):
            state = self.load()
            state.setdefault("events", []).append(
                {"time": utc_now(), "action": action, "details": details}
            )
            self.save(state)

    def artifact(self, name: str, path: str | Path, **metadata: Any) -> None:
        artifact_path = Path(path).resolve()
        record: dict[str, Any] = {
            "path": str(artifact_path),
            "updated_at": utc_now(),
            **metadata,
        }
        if artifact_path.is_file():
            record["sha256"] = sha256_file(artifact_path)
            record["size_bytes"] = artifact_path.stat().st_size
        self.directory.mkdir(parents=True, exist_ok=True)
        with _FileLock(self._lock_path):
            state = self.load()
            state.setdefault("artifacts", {})[name] = record
            self.save(state)
