"""Requested vs executed density initialization for one run (read-only).

Campaign audits should not depend on reading job logs, so every run gets the
same four fields plus the fallback record::

    density_init_requested   "neural-paw" | "standard"
    density_init_compatible  True | False | None (not known until the job ran)
    density_init_executed    True only when a generated density is what VASP reads
    density_init_status      see STATUSES

Sources: the Step1 manifest row (what was requested / skipped at preparation),
``density_init.json`` (what the initializer did) and ``density_init_fallback.json``
(written by the launcher when the hook failed).  For ``--precondition`` runs
the initializer seeds ``precondition/``, so both files are looked for there too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .launch import FALLBACK_NAME
from .workflow import REPORT_FORMAT, REPORT_NAME

#: ``density_init_status`` values beyond the initializer's own report statuses
#: (``PROMOTED``, ``STAGED``, ``STANDARD_START``).
STATUSES = (
    "NOT_REQUESTED",
    "SKIPPED",
    "PENDING",
    "PROMOTED",
    "STAGED",
    "STANDARD_START",
    "UNSUPPORTED_POTCAR_SCHEMA",
    "POTCAR_VARIANT_NOT_ALLOWED",
    "BACKEND_UNAVAILABLE",
    "FAILED",
)
_INCOMPATIBLE = {"UNSUPPORTED_POTCAR_SCHEMA", "POTCAR_VARIANT_NOT_ALLOWED"}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _locate(run: Path, name: str) -> Path | None:
    for directory in (run, run / "precondition"):
        if (directory / name).is_file():
            return directory / name
    return None


def density_init_run_status(run: str | Path, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Status fields for ``run``; ``plan`` is its ``step1_manifest.json`` ``density_init`` entry."""

    run = Path(run)
    report_path = _locate(run, REPORT_NAME)
    report = _json(report_path) if report_path else {}
    if report.get("format") != REPORT_FORMAT:
        report, report_path = {}, None
    fallback_path = _locate(run, FALLBACK_NAME)
    fallback = _json(fallback_path) if fallback_path else {}

    requested = (plan or {}).get("requested") or (
        report.get("backend") if report.get("backend") not in (None, "standard") else None
    )
    fields: dict[str, Any] = {
        "density_init_requested": requested or "standard",
        "density_init_compatible": None,
        "density_init_executed": False,
        "density_init_status": "NOT_REQUESTED",
        "density_init_fallback_occurred": False,
        "density_init_fallback_action": None,
        "density_init_failure": None,
        "density_init_report": str(report_path) if report_path else None,
    }
    if not requested:
        return fields
    if plan is not None and plan.get("applied") is False:
        fields["density_init_status"] = "SKIPPED"
        fields["density_init_failure"] = plan.get("reason")
        return fields

    if fallback:
        fields["density_init_fallback_occurred"] = True
        fields["density_init_fallback_action"] = fallback.get("action")
    status = report.get("status")
    if status == "FAILED":
        code = report.get("failure_code")
        fields["density_init_status"] = code if code in STATUSES else "FAILED"
        fields["density_init_failure"] = report.get("error")
        if code in _INCOMPATIBLE:
            fields["density_init_compatible"] = False
    elif status in {"PROMOTED", "STAGED"}:
        fields["density_init_status"] = status
        fields["density_init_compatible"] = True
        fields["density_init_executed"] = status == "PROMOTED" and bool(report.get("active"))
    elif status:
        fields["density_init_status"] = status
    elif fallback:
        # The hook failed before the initializer could write a report (e.g. no Python env).
        fields["density_init_status"] = "FAILED"
        fields["density_init_failure"] = f"hook exited {fallback.get('exit_code')} before writing {REPORT_NAME}"
    else:
        fields["density_init_status"] = "PENDING"
    if fallback and status in {"PROMOTED", "STAGED"}:
        # A later successful hook removes the record; both present means it is stale or tampered.
        fields["density_init_executed"] = False
        fields["density_init_failure"] = f"{FALLBACK_NAME} present alongside a {status} report"
    return fields
