"""Small file-backed provenance store for restartable campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
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


class StateStore:
    """Atomic JSON state with intentionally minimal workflow semantics."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.root = Path(campaign_root).resolve()
        self.directory = self.root / ".interfaceforge"
        self.path = self.directory / "state.json"

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
        state = self.load()
        state.setdefault("events", []).append(
            {"time": utc_now(), "action": action, "details": details}
        )
        self.save(state)

    def artifact(self, name: str, path: str | Path, **metadata: Any) -> None:
        artifact_path = Path(path).resolve()
        state = self.load()
        record: dict[str, Any] = {
            "path": str(artifact_path),
            "updated_at": utc_now(),
            **metadata,
        }
        if artifact_path.is_file():
            record["sha256"] = sha256_file(artifact_path)
            record["size_bytes"] = artifact_path.stat().st_size
        state.setdefault("artifacts", {})[name] = record
        self.save(state)
