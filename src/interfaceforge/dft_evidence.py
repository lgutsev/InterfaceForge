"""DFT input identity checks for heterogeneous interface/reference cells."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import SafetyError
from .vasp_provenance import _equivalent_value, build_vasp_reference_record

# POTIM, spin moment, atom count and k meshes need not match different phases.
TAGS = ("ENCUT", "GGA", "METAGGA", "LHFCALC", "AEXX", "HFSCREEN", "LDAU", "IVDW")


def _same_setting(left: str, right: str) -> bool:
    aliases = {".TRUE.": "T", "TRUE": "T", ".FALSE.": "F", "FALSE": "F"}
    left = aliases.get(left.strip().upper(), left)
    right = aliases.get(right.strip().upper(), right)
    return _equivalent_value(left, right)


def structure_evidence(run: Path, composition: Mapping[str, int]) -> dict[str, Any]:
    path = run / "OUTCAR"
    if not path.is_file():
        return {"composition": {"status": "NOT_CHECKED", "note": "OUTCAR missing"},
                "provenance": None}
    record = build_vasp_reference_record(
        source_leaf=run, source_outcar=path, staged_leaf=str(run),
        included_files=("INCAR", "OUTCAR", "POTCAR", "KPOINTS"), required_incar_tags=TAGS,
    )
    counts = None
    nions = None
    with path.open(errors="replace") as handle:
        for line in handle:
            if "ions per type" in line and "=" in line:
                try:
                    counts = [int(v) for v in line.split("=", 1)[1].split()]
                    if not counts or any(n < 0 for n in counts):
                        raise ValueError("invalid counts")
                except ValueError as exc:
                    raise SafetyError(f"{run}: malformed OUTCAR ions per type") from exc
            match = re.search(r"\bNIONS\s*=\s*(\d+)", line)
            if match:
                nions = int(match.group(1))
    total = sum(composition.values())
    if nions is not None and nions != total:
        raise SafetyError(f"{run}: OUTCAR NIONS={nions} differs from structure atom count {total}")
    if counts is not None and sum(counts) != total:
        raise SafetyError(f"{run}: OUTCAR ions per type {counts} differs from structure {dict(composition)}")
    titles = record["potcar_titles"]
    symbols = []
    for title in titles:
        fields = title.split()
        symbols.append(fields[1].split("_", 1)[0] if len(fields) > 1 else "")
    check: dict[str, Any] = {"status": "NOT_CHECKED", "ion_counts": counts, "nions": nions,
                             "note": "Need OUTCAR TITEL species and ions per type"}
    if counts is not None and len(symbols) == len(counts) and all(symbols):
        actual: Counter[str] = Counter()
        for symbol, count in zip(symbols, counts, strict=True):
            actual[symbol] += count
        if dict(actual) != dict(composition):
            raise SafetyError(f"{run}: OUTCAR composition {dict(actual)} differs from structure {dict(composition)}")
        check.update(status="PASS", outcar_composition=dict(actual), note="Species-resolved counts agree")
    return {"composition": check, "provenance": record}


def audit_provenance(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare executed settings and shared-element titles, keeping unknowns explicit.

    Titles establish potential labels, not byte-identical per-element datasets.
    Whole POTCAR hashes are retained as evidence, never compared across chemistry.
    """
    issues: list[str] = []
    missing: list[str] = []
    settings: dict[str, tuple[str, str]] = {}
    potentials: dict[str, tuple[str, str]] = {}
    for name, entry in evidence.items():
        record = entry.get("provenance")
        if not record:
            missing.append(f"{name}: OUTCAR provenance missing")
            continue
        executed = record["outcar_executed_tags"]
        inputs = record["incar_tags"]
        for tag in TAGS:
            value = executed.get(tag)
            if not value:
                missing.append(f"{name}: executed {tag} unknown")
                continue
            if tag in inputs and not _same_setting(inputs[tag], value):
                issues.append(f"{name}: INCAR/OUTCAR {tag} differs ({inputs[tag]} vs {value})")
            if tag in settings and not _same_setting(settings[tag][1], value):
                issues.append(f"{name}: {tag}={value} differs from {settings[tag][0]} ({settings[tag][1]})")
            else:
                settings[tag] = (name, value)
        if str(executed.get("LDAU", "")).upper() in {"T", ".TRUE.", "TRUE"}:
            missing.append(f"{name}: species-resolved Hubbard parameters require review")
        if not record["potcar_titles"]:
            missing.append(f"{name}: OUTCAR POTCAR TITEL missing")
        for title in record["potcar_titles"]:
            fields = title.split()
            if len(fields) < 2:
                missing.append(f"{name}: unreadable POTCAR title {title}")
                continue
            symbol = fields[1].split("_", 1)[0]
            if symbol in potentials and potentials[symbol][1] != title:
                issues.append(f"{name}: {symbol} POTCAR {title} differs from {potentials[symbol]}")
            else:
                potentials[symbol] = (name, title)
    return {"status": "CHECK" if issues else "NOT_CHECKED" if missing or not evidence else "PASS",
            "issues": issues, "missing_evidence": missing,
            "note": "POTCAR compatibility is title-level; hashes retained for provenance. "
                    "Different k meshes and molecular spin states are allowed; convergence is not established here."}
