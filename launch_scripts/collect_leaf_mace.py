#!/usr/bin/env python3
"""Collect terminal VASP calculations into heritage-safe MACE extxyz splits."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from interfaceforge.leaf_collect import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(default_engine="mace"))
