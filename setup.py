"""Build hook: ship the canonical ``schemas/*.json`` as ``interfaceforge/schemas`` package data.

All project metadata lives in ``pyproject.toml``. ``schemas/`` at the repository root is the only
copy of the JSON Schemas in the source tree; this hook copies it into the wheel at build time.
Editable/source-tree runs read ``schemas/`` directly (see ``interfaceforge.schema_check``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SCHEMA_SOURCE = Path(__file__).resolve().parent / "schemas"


class build_py_with_schemas(build_py):  # noqa: N801 - setuptools command naming
    def run(self) -> None:
        super().run()
        if self.editable_mode:
            return
        target = Path(self.build_lib) / "interfaceforge" / "schemas"
        target.mkdir(parents=True, exist_ok=True)
        for schema in sorted(SCHEMA_SOURCE.glob("*.json")):
            shutil.copyfile(schema, target / schema.name)


setup(cmdclass={"build_py": build_py_with_schemas})
