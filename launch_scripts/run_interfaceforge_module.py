"""Run an InterfaceForge module from a source checkout.

This bootstrap is intentionally dependency-free.  It is invoked by containerised
site Python wrappers that may discard PYTHONPATH, but can still read the bound
InterfaceForge checkout.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_interfaceforge_module.py MODULE [ARG ...]")
    module = sys.argv[1]
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
