"""Shared provenance helpers: source commit and order-independent hashing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def interfaceforge_commit() -> dict[str, Any]:
    """Return the git commit of the InterfaceForge source tree, if it is a checkout.

    An installed wheel has no git metadata; that is reported explicitly rather
    than guessed so a manifest never claims a commit it cannot prove.
    """

    source = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "not a git checkout"}
    return {"commit": commit or None, "dirty": bool(status), "source": "git rev-parse HEAD"}


def stable_json_hash(payload: Any) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace variance)."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
