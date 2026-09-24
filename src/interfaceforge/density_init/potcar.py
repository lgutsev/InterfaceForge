"""POTCAR provenance for density initialization and its benchmark.

POTCAR *selection* belongs to the VASP workflow (e.g. ``POTCAR_gen`` with the
production ``POTCAR_DEFS.txt`` or, for controlled tests only, the Materials
Project-compatible ``POTCAR_DEFS_MP.txt``), never to the initializer.  This
module only *records* where a run's POTCAR came from and checks the record
against the POTCAR itself:

* the actual ``POTCAR`` is authoritative: its SHA-256 and the dataset symbol
  per element (from ``TITEL``) are always recorded;
* a declared definitions file (``--potcar-definitions``) is parsed the way
  ``POTCAR_gen`` parses it, and every element's declared variant must match
  the dataset actually present -- a declaration that disagrees with the
  POTCAR is refused rather than recorded.  ``$POTCAR_DEFS`` is deliberately
  *not* read: a job environment may export it for an unrelated POTCAR.

Nothing here regenerates, rewrites or substitutes a POTCAR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import SafetyError
from .inputs import nonempty, potcar_entries, sha256_file


def read_potcar_definitions(path: str | Path) -> dict[str, str]:
    """``{element: variant}`` from a ``POTCAR_gen`` definitions file (``El|variant`` lines).

    Blank lines and ``#`` comments are skipped and carriage returns removed,
    as ``POTCAR_gen`` does; a later line for the same element wins.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    mapping: dict[str, str] = {}
    for number, raw in enumerate(source.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw.replace("\r", "")
        element, _, variant = line.partition("|")
        element, variant = element.strip(), variant.strip()
        if not element or element.startswith("#"):
            continue
        if not variant:
            raise SafetyError(f"{source}:{number}: invalid POTCAR mapping for element {element!r}")
        mapping[element] = variant
    return mapping


def potcar_provenance(
    potcar: str | Path,
    elements: list[str],
    *,
    definitions: str | Path | None = None,
    generator: str | None = None,
) -> dict[str, Any]:
    """Provenance of ``potcar`` for the POSCAR species ``elements`` (in POSCAR order).

    Raises :class:`SafetyError` when a declared definitions file disagrees
    with the POTCAR actually present.
    """

    path = Path(potcar)
    record: dict[str, Any] = {
        "potcar_generator": generator,
        "potcar_definitions": None,
        "potcar_definitions_sha256": None,
        "potcar_sha256": None,
        "potcar_variants": None,
        "potcar_titels": None,
        "declared_variants": None,
        "consistent_with_definitions": None,
    }
    defs_path = Path(definitions).expanduser().resolve() if definitions else None
    if defs_path is not None:
        mapping = read_potcar_definitions(defs_path)
        record["potcar_definitions"] = str(defs_path)
        record["potcar_definitions_sha256"] = sha256_file(defs_path)
    if not nonempty(path):
        return record  # POTCAR generated later (e.g. at launch); checked where it exists
    entries = potcar_entries(path)
    record["potcar_sha256"] = sha256_file(path)
    record["potcar_titels"] = [entry["titel"] for entry in entries]
    unique = list(dict.fromkeys(elements))
    if len(entries) == len(unique):
        record["potcar_variants"] = {
            element: entry["symbol"] for element, entry in zip(unique, entries, strict=True)
        }
    else:
        record["potcar_variants"] = {entry["element"]: entry["symbol"] for entry in entries}

    if defs_path is None:
        return record
    missing = [element for element in unique if element not in mapping]
    if missing:
        raise SafetyError(f"{defs_path} has no POTCAR definition for {', '.join(missing)}")
    declared = {element: mapping[element] for element in unique}
    record["declared_variants"] = declared
    actual = record["potcar_variants"] or {}
    mismatches = [
        f"{element}: declared {declared[element]!r}, POTCAR has {actual.get(element)!r}"
        for element in unique
        if actual.get(element) != declared[element]
    ]
    if mismatches:
        raise SafetyError(
            f"{path} does not match the declared POTCAR definitions {defs_path}: "
            + "; ".join(mismatches)
            + ". The POTCAR is authoritative: regenerate it with the intended mapping or fix the declaration"
        )
    record["consistent_with_definitions"] = True
    return record
