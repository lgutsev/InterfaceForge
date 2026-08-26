"""Reproducibility records for VASP reference trajectories.

The record deliberately retains the complete parsed INCAR rather than a small
hand-picked subset.  Required/consistent tags are policy; the full input and
file digests are provenance.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from .vasp import parse_incar

DEFAULT_REQUIRED_INCAR_TAGS = ("ENCUT", "IVDW", "POTIM")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _outcar_fingerprint(
    path: Path,
    *,
    tracked_tags: Sequence[str],
    header_lines: int = 20000,
) -> dict[str, Any]:
    """Hash an OUTCAR while collecting identity and ionic-frame metadata."""
    digest = sha256()
    version = ""
    potcar_titles: list[str] = []
    nkpts = ""
    ionic_frames = 0
    executed_tags: dict[str, str] = {}
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            if b"POSITION" in raw and b"TOTAL-FORCE" in raw:
                ionic_frames += 1
            if index >= header_lines:
                continue
            line = raw.decode("utf-8", errors="ignore")
            stripped = line.strip()
            if not version:
                match = re.search(r"\bvasp\.\S+", stripped, flags=re.IGNORECASE)
                if match:
                    version = match.group(0)
            if "TITEL" in line and "=" in line:
                title = line.split("=", 1)[1].strip()
                if title and title not in potcar_titles:
                    potcar_titles.append(title)
            if not nkpts:
                match = re.search(r"\bNKPTS\s*=\s*(\d+)", line)
                if match:
                    nkpts = match.group(1)
            for tag in tracked_tags:
                match = re.search(rf"\b{re.escape(tag)}\s*=\s*([^\s;]+)", line)
                if match:
                    executed_tags[tag] = match.group(1).strip()
    return {
        "outcar_sha256": digest.hexdigest(),
        "ionic_frames_detected": ionic_frames,
        "vasp_version": version,
        "potcar_titles": potcar_titles,
        "nkpts": nkpts,
        "outcar_executed_tags": executed_tags,
    }


def _equivalent_value(left: str, right: str) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    except ValueError:
        return left.strip().upper() == right.strip().upper()


def build_vasp_reference_record(
    *,
    source_leaf: Path,
    source_outcar: Path,
    staged_leaf: str,
    included_files: Iterable[str],
    file_paths: Mapping[str, Path] | None = None,
    required_incar_tags: Sequence[str] = DEFAULT_REQUIRED_INCAR_TAGS,
) -> dict[str, Any]:
    resolved_paths = dict(file_paths or {})
    incar_path = resolved_paths.get("INCAR", source_leaf / "INCAR")
    incar_tags = parse_incar(incar_path)
    required = [str(tag).upper() for tag in required_incar_tags]
    missing = [tag for tag in required if not incar_tags.get(tag)]
    identity = _outcar_fingerprint(source_outcar, tracked_tags=required)
    hashes = {}
    for name in included_files:
        path = resolved_paths.get(name, source_leaf / name)
        if not path.is_file():
            continue
        hashes[name] = (
            identity["outcar_sha256"] if name == "OUTCAR" else sha256_file(path)
        )
    return {
        "schema_version": 1,
        "source_leaf": str(source_leaf),
        "source_outcar": str(source_outcar),
        "staged_leaf": staged_leaf,
        "file_sha256": hashes,
        "resolved_input_paths": {
            name: str(path) for name, path in sorted(resolved_paths.items())
        },
        "incar_tags": dict(sorted(incar_tags.items())),
        "required_incar_tags": required,
        "missing_required_incar_tags": missing,
        **identity,
    }


def audit_vasp_reference_records(
    records: Sequence[dict[str, Any]],
    *,
    consistent_incar_tags: Sequence[str] = DEFAULT_REQUIRED_INCAR_TAGS,
) -> dict[str, Any]:
    consistent = [str(tag).upper() for tag in consistent_incar_tags]
    problems: list[dict[str, Any]] = []
    for record in records:
        missing = list(record.get("missing_required_incar_tags", []))
        if missing:
            problems.append(
                {
                    "staged_leaf": record["staged_leaf"],
                    "issues": [f"missing explicit INCAR tag: {tag}" for tag in missing],
                }
            )
        incar_tags = record.get("incar_tags", {})
        executed_tags = record.get("outcar_executed_tags", {})
        mismatches = [
            tag
            for tag in record.get("required_incar_tags", [])
            if tag in executed_tags
            and tag in incar_tags
            and not _equivalent_value(str(incar_tags[tag]), str(executed_tags[tag]))
        ]
        if mismatches:
            problems.append(
                {
                    "staged_leaf": record["staged_leaf"],
                    "issues": [
                        f"INCAR/OUTCAR mismatch for {tag}: input={incar_tags[tag]}, "
                        f"executed={executed_tags[tag]}"
                        for tag in mismatches
                    ],
                }
            )
        hashes = record.get("file_sha256", {})
        if not hashes.get("KPOINTS") and not record.get("nkpts"):
            problems.append(
                {
                    "staged_leaf": record["staged_leaf"],
                    "issues": ["missing both KPOINTS hash and OUTCAR NKPTS provenance"],
                }
            )
        if not hashes.get("POTCAR") and not record.get("potcar_titles"):
            problems.append(
                {
                    "staged_leaf": record["staged_leaf"],
                    "issues": ["missing both POTCAR hash and OUTCAR TITEL provenance"],
                }
            )

    unique_values: dict[str, list[str]] = {}
    for tag in consistent:
        values = sorted(
            {str(record.get("incar_tags", {}).get(tag, "<MISSING>")) for record in records}
        )
        unique_values[tag] = values
        if len(values) > 1:
            problems.append(
                {
                    "staged_leaf": "__dataset__",
                    "issues": [f"inconsistent {tag}: {', '.join(values)}"],
                }
            )

    versions = sorted({str(record.get("vasp_version") or "<UNKNOWN>") for record in records})
    if len(versions) != 1 or versions == ["<UNKNOWN>"]:
        problems.append(
            {
                "staged_leaf": "__dataset__",
                "issues": [f"mixed or unknown VASP versions: {', '.join(versions)}"],
            }
        )
    all_incar_tags = sorted(
        {tag for record in records for tag in record.get("incar_tags", {})}
    )
    incar_tag_values = {
        tag: sorted(
            {str(record.get("incar_tags", {}).get(tag, "<MISSING>")) for record in records}
        )
        for tag in all_incar_tags
    }
    differing_incar_tags = {
        tag: values for tag, values in incar_tag_values.items() if len(values) > 1
    }
    incar_hashes = sorted(
        {
            str(record.get("file_sha256", {}).get("INCAR", "<MISSING>"))
            for record in records
        }
    )
    frame_counts = [int(record.get("ionic_frames_detected", 0)) for record in records]
    for record, count in zip(records, frame_counts, strict=True):
        if count == 0:
            problems.append(
                {
                    "staged_leaf": record["staged_leaf"],
                    "issues": ["no ionic POSITION/TOTAL-FORCE frames detected in OUTCAR"],
                }
            )
    potcar_sets = sorted(
        {
            " | ".join(record.get("potcar_titles", [])) or "<UNKNOWN>"
            for record in records
        }
    )
    return {
        "schema_version": 1,
        "status": "FAILED" if problems else "OK",
        "records": len(records),
        "consistent_incar_tags": consistent,
        "incar_tag_values": unique_values,
        "exact_incar_files_identical": len(incar_hashes) == 1,
        "unique_incar_sha256": incar_hashes,
        "differing_incar_tags": differing_incar_tags,
        "ionic_frame_counts": {
            "minimum": min(frame_counts, default=0),
            "maximum": max(frame_counts, default=0),
            "unique": sorted(set(frame_counts)),
        },
        "vasp_versions": versions,
        "potcar_title_sets": potcar_sets,
        "file_hash_coverage": {
            name: sum(name in record.get("file_sha256", {}) for record in records)
            for name in sorted(
                {name for record in records for name in record.get("file_sha256", {})}
            )
        },
        "unique_file_sha256": {
            name: sorted(
                {
                    record.get("file_sha256", {}).get(name, "<MISSING>")
                    for record in records
                }
            )
            for name in sorted(
                {name for record in records for name in record.get("file_sha256", {})}
            )
        },
        "outcar_echo_coverage": {
            tag: sum(tag in record.get("outcar_executed_tags", {}) for record in records)
            for tag in consistent
        },
        "problems": problems,
    }


def write_vasp_reference_provenance(
    records: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    output: Path,
) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "reference_provenance.json"
    csv_path = output / "reference_provenance.csv"
    audit_path = output / "reference_provenance_audit.json"
    records_path.write_text(json.dumps({"schema_version": 1, "records": records}, indent=2) + "\n")
    csv_rows = []
    for record in records:
        tags = record.get("incar_tags", {})
        hashes = record.get("file_sha256", {})
        csv_rows.append(
            {
                "staged_leaf": record["staged_leaf"],
                "source_outcar": record["source_outcar"],
                "ENCUT": tags.get("ENCUT", ""),
                "IVDW": tags.get("IVDW", ""),
                "POTIM": tags.get("POTIM", ""),
                "vasp_version": record.get("vasp_version", ""),
                "nkpts": record.get("nkpts", ""),
                "ionic_frames_detected": record.get("ionic_frames_detected", 0),
                "incar_sha256": hashes.get("INCAR", ""),
                "outcar_sha256": hashes.get("OUTCAR", ""),
                "kpoints_sha256": hashes.get("KPOINTS", ""),
                "potcar_sha256": hashes.get("POTCAR", ""),
                "potcar_titles": " | ".join(record.get("potcar_titles", [])),
                "missing_required_tags": ",".join(
                    record.get("missing_required_incar_tags", [])
                ),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return {
        "records": str(records_path),
        "csv": str(csv_path),
        "audit": str(audit_path),
    }
