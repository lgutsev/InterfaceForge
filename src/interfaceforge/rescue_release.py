"""Frozen split ledger and same-geometry partial replacement for the SiN/TiN rescue.

``freeze``   read an existing exported deck ({train,valid,test}.extxyz plus its
             DeePMD tree), record every (source_run, source_frame) with its
             historical split, geometry and label hashes, and verify that the
             DeePMD export holds exactly the same frames with the same numbers.
``plan``     the same decisions without writing a release: every check of
             ``assemble`` that needs no deck arrays (rules, replacements, added
             frames, holdouts, digest-level leakage), plus tolerance-level
             leakage against the deck when ``--deck`` is given.
``assemble`` build a release from the frozen ledger and a release policy:
             per-source keep/exclude/replace, where replacements are reviewed
             relabels of the *same* geometry. Every replaced frame keeps its
             historical split; energy, forces and virial are replaced
             together. Missing, duplicate, unknown or wrong-source replacements
             are errors. Nothing is re-split and no old export is modified.

Label masks: a source rule or an added-frame entry may set ``label_use``
(``all``, ``energy_forces`` = virial not trained, ``forces_virial`` = energy
not trained). A masked term is left out of the release's training files: the
extxyz frame has no REF_virial / REF_energy and carries ``label_use``, and the
frame lands in its own DeePMD system under ``deepmd/<split>/<label_use>/``
without ``virial.npy`` / ``energy.npy``. DeePMD-kit 3.x (checked against
v3.2.0b0, PyTorch backend) loads a missing optional label as zeros with
``find_<key> = 0`` and multiplies that loss prefactor by it, per batch (a batch
is drawn from one system), and skips such systems in the energy-bias
statistics; lcurve prints NaN for that term on such batches. The original
labels are never edited: they stay in the deck and the relabel files.

Added frames are new labelled frames not in the historical deck (strain cells),
each entry with its split (train/valid/test/holdout). Holdouts are written to
``holdout.extxyz`` only, never to a DeePMD system, and are checked against every
release split like the policy's leakage references.

Labels follow the deck: REF_energy = E_sigma->0 (eV), REF_forces (eV/A),
REF_virial = -V * stress (eV, nine components).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import SafetyError
from .vasp_provenance import sha256_file

SPLITS = ("train", "valid", "test")
ADDED_SPLITS = (*SPLITS, "holdout")
GEOMETRY_TOLERANCE_A = 1e-6
# Leakage is looser than same-geometry: a rounded or rewrapped copy of a holdout is still the holdout.
LEAKAGE_TOLERANCE_A = 1e-3
ACTIONS = {"keep", "exclude", "replace_all", "replace_subset"}
UNREPLACED = {"keep", "hold"}
# Label terms each label_use trains; a term not listed is absent from the release's training files.
LABEL_USES = {
    "all": ("energy", "forces", "virial"),
    "energy_forces": ("energy", "forces"),
    "forces_virial": ("forces", "virial"),
}
# Collector manifests: "bulk_X source_frame 5 (split train)" or "<deck>, source_run X, source_frame 5 (split test)".
SOURCE_STRUCTURE = re.compile(r"(?:source_run\s+)?([A-Za-z0-9_.\-]+),?\s+source_frame\s+(\d+)\s+\(split\s+(\w+)\)\s*$")


# --------------------------------------------------------------------------- reading


def iter_extxyz(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (file index, raw frame text) without parsing every frame."""
    with path.open(encoding="utf-8") as handle:
        index = 0
        while True:
            head = handle.readline()
            if not head.strip():
                return
            natoms = int(head)
            text = head + handle.readline() + "".join(handle.readline() for _ in range(natoms))
            yield index, text
            index += 1


def parse_frame(text: str):
    from ase.io import read

    return read(io.StringIO(text), format="extxyz", index=0)


def frame_arrays(atoms) -> dict[str, Any]:
    info = atoms.info
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell, dtype=float),
        "positions": np.asarray(atoms.positions, dtype=float),
        "energy": float(info["REF_energy"]),
        "forces": np.asarray(atoms.arrays["REF_forces"], dtype=float),
        "virial": np.asarray(info["REF_virial"], dtype=float).reshape(3, 3) if "REF_virial" in info else None,
        "move_mask": np.asarray(atoms.arrays.get("move_mask", np.ones(len(atoms), dtype=int))),
    }


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for item in arrays:
        if isinstance(item, np.ndarray):
            h.update(np.ascontiguousarray(item, dtype=float).tobytes())
        else:
            h.update(json.dumps(item).encode())
    return h.hexdigest()[:20]


def geometry_digest(arrays: dict[str, Any]) -> str:
    """Identity of species order, cell and positions at 1e-6 A (for duplicate/overlap checks)."""
    return _digest(arrays["symbols"], np.round(arrays["cell"], 6), np.round(arrays["positions"], 6))


def label_digest(arrays: dict[str, Any]) -> str:
    virial = arrays["virial"] if arrays["virial"] is not None else np.zeros((3, 3))
    return _digest(np.asarray([arrays["energy"]]), arrays["forces"], virial)


def _info_value(text: str, key: str) -> str | None:
    match = re.search(rf'\b{key}=("[^"]*"|\S+)', text.split("\n", 2)[1])
    return match.group(1).strip('"') if match else None


# --------------------------------------------------------------------------- freeze


LEDGER_COLUMNS = (
    "source_run",
    "source_frame",
    "split",
    "leaf",
    "natoms",
    "deck_file",
    "deck_index",
    "geometry_digest",
    "label_digest",
    "ref_energy_ev",
    "source_path",
)


def freeze(deck: Path, output: Path, *, deepmd: Path | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: dict[tuple[str, int], str] = {}
    for split in SPLITS:
        path = deck / f"{split}.extxyz"
        for index, text in iter_extxyz(path):
            atoms = parse_frame(text)
            arrays = frame_arrays(atoms)
            run = str(atoms.info.get("source_run"))
            frame = int(atoms.info.get("source_frame"))
            recorded = str(atoms.info.get("split"))
            if recorded != split:
                errors.append(f"{path.name}[{index}] records split {recorded}")
            key = (run, frame)
            if key in seen:
                errors.append(f"{run} frame {frame} in both {seen[key]} and {split}")
            seen[key] = split
            finite = [arrays["positions"], arrays["cell"], arrays["forces"], np.asarray([arrays["energy"]])]
            if arrays["virial"] is not None:
                finite.append(arrays["virial"])
            if not all(np.all(np.isfinite(a)) for a in finite):
                errors.append(f"{path.name}[{index}] nonfinite label")
            rows.append(
                {
                    "source_run": run,
                    "source_frame": frame,
                    "split": split,
                    "leaf": atoms.info.get("IF_leaf"),
                    "natoms": len(atoms),
                    "deck_file": path.name,
                    "deck_index": index,
                    "geometry_digest": geometry_digest(arrays),
                    "label_digest": label_digest(arrays),
                    "ref_energy_ev": repr(arrays["energy"]),
                    "source_path": atoms.info.get("source_path"),
                    "_arrays": arrays if deepmd is not None else None,
                }
            )
    geometry_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        geometry_splits[row["geometry_digest"]].add(row["split"])
    overlapping = sum(len(s) > 1 for s in geometry_splits.values())
    duplicates = len(rows) - len(geometry_splits)
    deepmd_report = verify_deepmd(deepmd, rows) if deepmd is not None else None
    if deepmd_report:
        errors.extend(deepmd_report["errors"])
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["source_run"]][row["split"]] += 1
    with (output / "split_ledger.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, LEDGER_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["source_run"], r["source_frame"])))
    report = {
        "schema_version": 1,
        "deck": str(deck),
        "deck_sha256": {f"{s}.extxyz": sha256_file(deck / f"{s}.extxyz") for s in SPLITS},
        "frames": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "runs": len(counts),
        "per_run": {run: dict(c) for run, c in sorted(counts.items())},
        "duplicate_geometries": duplicates,
        "geometries_in_more_than_one_split": overlapping,
        "deepmd": {k: v for k, v in (deepmd_report or {}).items() if k != "errors"} or None,
        "errors": errors,
        "ledger_sha256": sha256_file(output / "split_ledger.tsv"),
    }
    (output / "freeze_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def verify_deepmd(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every DeePMD system's frame_map and arrays against the extxyz rows of the same leaf/split."""
    by_leaf: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_leaf[(row["split"], row["leaf"])][row["source_frame"]] = row
    errors, systems, frames, worst = [], 0, 0, Counter()
    for (split, leaf), expected in sorted(by_leaf.items()):
        system = root / split / leaf
        if not (system / "frame_map.csv").is_file():
            errors.append(f"deepmd {split}/{leaf}: missing")
            continue
        systems += 1
        with (system / "frame_map.csv").open(encoding="utf-8") as handle:
            mapped = [int(r["source_frame"]) for r in csv.DictReader(handle)]
        if sorted(mapped) != sorted(expected) or len(set(mapped)) != len(mapped):
            errors.append(f"deepmd {split}/{leaf}: membership differs from extxyz")
            continue
        data = system / "set.000"
        coord = np.load(data / "coord.npy")
        box = np.load(data / "box.npy")
        energy = np.load(data / "energy.npy").reshape(-1)
        force = np.load(data / "force.npy")
        virial = np.load(data / "virial.npy") if (data / "virial.npy").is_file() else None
        type_map = (system / "type_map.raw").read_text().split()
        types = [type_map[int(t)] for t in (system / "type.raw").read_text().split()]
        if virial is None:
            errors.append(f"deepmd {split}/{leaf}: no virial.npy")
        for local, source_frame in enumerate(mapped):
            arrays = expected[source_frame]["_arrays"]
            frames += 1
            if types != arrays["symbols"]:
                errors.append(f"deepmd {split}/{leaf} frame {source_frame}: species order")
                continue
            diffs = {
                "coord_a": np.max(np.abs(coord[local] - arrays["positions"].reshape(-1))),
                "box_a": np.max(np.abs(box[local] - arrays["cell"].reshape(-1))),
                "energy_ev": abs(energy[local] - arrays["energy"]),
                "force_ev_a": np.max(np.abs(force[local] - arrays["forces"].reshape(-1))),
                "virial_ev": (
                    np.max(np.abs(virial[local] - arrays["virial"].reshape(-1)))
                    if virial is not None and arrays["virial"] is not None
                    else 0.0
                ),
            }
            for key, value in diffs.items():
                worst[key] = max(worst[key], float(value))
            if any(v > 1e-6 for v in diffs.values()):
                errors.append(f"deepmd {split}/{leaf} frame {source_frame}: numbers differ {diffs}")
    return {"systems": systems, "frames": frames, "max_abs_difference": dict(worst), "errors": errors}


def read_ledger(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["source_frame"] = int(row["source_frame"])
        row["deck_index"] = int(row["deck_index"])
        row["natoms"] = int(row["natoms"])
    return rows


# --------------------------------------------------------------------------- policy


def load_policy(path: Path) -> dict[str, Any]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise SafetyError(f"{path}: schema_version must be 1")
    for name, rule in policy.get("sources", {}).items():
        if rule.get("action") not in ACTIONS:
            raise SafetyError(f"{path}: {name}: action must be one of {sorted(ACTIONS)}")
        if rule["action"] == "replace_subset" and rule.get("unreplaced") not in UNREPLACED:
            raise SafetyError(f"{path}: {name}: replace_subset needs unreplaced: keep|hold")
        if rule["action"] in {"exclude", "replace_subset"} and not rule.get("reason"):
            raise SafetyError(f"{path}: {name}: {rule['action']} needs a reason")
        if rule.get("label_use", "all") not in LABEL_USES:
            raise SafetyError(f"{path}: {name}: label_use must be one of {sorted(LABEL_USES)}")
    ids: Counter = Counter()
    for item in policy.get("replacements", []):
        if not {"id", "path", "sha256"} <= set(item):
            raise SafetyError(f"{path}: every replacement needs id, path and sha256")
        if "manifest" in item and "manifest_sha256" not in item:
            raise SafetyError(f"{path}: replacement {item['id']}: a manifest needs its manifest_sha256")
        if "label_use" in item:
            raise SafetyError(f"{path}: replacement {item['id']}: label_use belongs to the source rule")
        ids[item["id"]] += 1
    for item in policy.get("added_frames", []):
        if not {"id", "path", "sha256", "split", "label_use", "reason"} <= set(item):
            raise SafetyError(f"{path}: every added_frames entry needs id, path, sha256, split, label_use and reason")
        if item["split"] not in ADDED_SPLITS:
            raise SafetyError(f"{path}: added {item['id']}: split must be one of {list(ADDED_SPLITS)}")
        if item["label_use"] not in LABEL_USES:
            raise SafetyError(f"{path}: added {item['id']}: label_use must be one of {sorted(LABEL_USES)}")
        ids[item["id"]] += 1
    for item in policy.get("leakage_references", []):
        if not {"id", "path", "sha256", "forbid"} <= set(item):
            raise SafetyError(f"{path}: every leakage reference needs id, path, sha256 and forbid")
        if not item["forbid"] or not set(item["forbid"]) <= set(SPLITS):
            raise SafetyError(f"{path}: leakage reference {item['id']}: forbid must list splits from {list(SPLITS)}")
        ids[item["id"]] += 1
    repeated = sorted(k for k, n in ids.items() if n > 1)
    if repeated:
        raise SafetyError(f"{path}: ids must be unique across replacements, added_frames and references: {repeated}")
    return policy


def source_rule(policy: dict[str, Any], run: str) -> tuple[str, dict[str, Any]]:
    """Exact name first, then the first matching glob; unmatched sources are an error."""
    sources = policy.get("sources", {})
    if run in sources:
        return run, sources[run]
    for pattern, rule in sources.items():
        if fnmatch.fnmatchcase(run, pattern):
            return pattern, rule
    raise SafetyError(f"No release-policy rule for source {run}; every source needs an explicit decision")


# --------------------------------------------------------------------------- policy inputs


def _resolve(base: Path, value: str) -> Path:
    return Path(value) if Path(value).is_absolute() else (base / value).resolve()


def _listed_frames(item: dict[str, Any], base: Path, kind: str) -> list[tuple[int, Any]]:
    """(file index, atoms) of a hash-pinned extxyz, filtered by ``select`` and counted against ``expected_frames``."""
    path = _resolve(base, item["path"])
    if sha256_file(path) != item["sha256"]:
        raise SafetyError(f"{kind} {item['id']} changed: {path}")
    select = item.get("select") or {}
    frames = [
        (index, atoms)
        for index, text in iter_extxyz(path)
        for atoms in [parse_frame(text)]
        if all(str(atoms.info.get(k)) == str(v) for k, v in select.items())
    ]
    expected = item.get("expected_frames")
    if expected is not None and len(frames) != int(expected):
        raise SafetyError(f"{kind} {item['id']}: {len(frames)} frames selected, the policy expects {expected}")
    return frames


def _manifest_sources(item: dict[str, Any], base: Path) -> dict[str, tuple[str, int, str]]:
    """case_id -> (source_run, source_frame, split) from a collector package manifest."""
    path = _resolve(base, item["manifest"])
    if sha256_file(path) != item["manifest_sha256"]:
        raise SafetyError(f"Replacement {item['id']}: manifest changed: {path}")
    found = {}
    for frame in json.loads(path.read_text(encoding="utf-8"))["frames"]:
        match = SOURCE_STRUCTURE.search(str(frame.get("source_structure", "")))
        if not match:
            raise SafetyError(f"Replacement {item['id']}: {frame['case_id']} names no deck frame")
        found[frame["case_id"]] = (match.group(1), int(match.group(2)), match.group(3))
    return found


def _finite_labels(arrays: dict[str, Any]) -> bool:
    return bool(
        math.isfinite(arrays["energy"])
        and np.all(np.isfinite(arrays["forces"]))
        and arrays["virial"] is not None
        and np.all(np.isfinite(arrays["virial"]))
    )


def load_replacements(policy: dict[str, Any], base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Reviewed relabels by deck key. A frame names its deck frame itself (source_run/source_frame) or,
    for collector output, through its package manifest (case_id -> source_structure)."""
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for item in policy.get("replacements", []):
        manifest = _manifest_sources(item, base) if "manifest" in item else None
        for index, atoms in _listed_frames(item, base, "Replacement"):
            manifest_split = None
            if manifest is not None:
                case = atoms.info.get("case_id")
                if case not in manifest:
                    raise SafetyError(f"{item['id']}[{index}]: case_id {case} is not in the manifest")
                stamped = atoms.info.get("manifest_sha256")
                if stamped is not None and stamped != item["manifest_sha256"]:
                    raise SafetyError(f"{item['id']}[{index}]: collected from a different manifest")
                run, frame, manifest_split = manifest[case]
            else:
                run, frame = atoms.info.get("source_run"), atoms.info.get("source_frame")
                if run is None or frame is None:
                    raise SafetyError(f"{item['id']}[{index}]: no source_run/source_frame")
            key = (str(run), int(frame))
            if key in found:
                raise SafetyError(
                    f"Duplicate replacement for {key[0]} frame {key[1]} ({found[key]['replacement']} and {item['id']})"
                )
            arrays = frame_arrays(atoms)
            if arrays["virial"] is None:
                raise SafetyError(f"{item['id']}[{index}]: replacement without REF_virial")
            if not _finite_labels(arrays):
                raise SafetyError(f"{item['id']}[{index}]: nonfinite replacement label")
            found[key] = {
                "replacement": item["id"],
                "index": index,
                "arrays": arrays,
                "label_source": atoms.info.get("label_source", item["id"]),
                "manifest_split": manifest_split,
            }
    return found


def load_added(policy: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    """New labelled frames (not in the deck), one decision row each, keyed (added:<id>, file index)."""
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, int], str] = {}
    for item in policy.get("added_frames", []):
        for index, atoms in _listed_frames(item, base, "Added frames"):
            claim = (item["sha256"], index)
            if claim in seen:
                raise SafetyError(f"Added frame {item['path']}[{index}] is selected by {seen[claim]} and {item['id']}")
            seen[claim] = item["id"]
            arrays = frame_arrays(atoms)
            if not _finite_labels(arrays):
                raise SafetyError(f"Added {item['id']}[{index}]: missing or nonfinite label")
            rows.append(
                {
                    "source_run": f"added:{item['id']}",
                    "source_frame": index,
                    "split": item["split"],
                    "leaf": item.get("leaf", f"added/{item['id']}"),
                    "natoms": len(atoms),
                    "status": "holdout" if item["split"] == "holdout" else "added",
                    "label_use": item["label_use"],
                    "label_source": atoms.info.get("label_source", item["id"]),
                    "rule": f"added_frames:{item['id']}",
                    "reason": item["reason"],
                    "case_id": atoms.info.get("case_id", ""),
                    "source_path": atoms.info.get("source_path", item["path"]),
                    "geometry_digest": geometry_digest(arrays),
                    "label_digest": label_digest(arrays),
                    "arrays": arrays,
                    "info": dict(atoms.info),
                }
            )
    return rows


def _geometry(atoms) -> dict[str, Any]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell, dtype=float),
        "positions": np.asarray(atoms.positions, dtype=float),
    }


def load_references(policy: dict[str, Any], base: Path, added: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Geometries that must not appear in the listed release splits: the policy's leakage references
    (labels not needed) and every added holdout, which is forbidden in every split."""
    refs = []
    for item in policy.get("leakage_references", []):
        for index, atoms in _listed_frames(item, base, "Leakage reference"):
            geometry = _geometry(atoms)
            refs.append(
                {
                    "id": item["id"],
                    "index": index,
                    "case_id": atoms.info.get("case_id", ""),
                    "forbid": tuple(item["forbid"]),
                    "geometry": geometry,
                    "geometry_digest": geometry_digest(geometry),
                }
            )
    for row in added:
        if row["split"] == "holdout":
            refs.append(
                {
                    "id": row["source_run"],
                    "index": row["source_frame"],
                    "case_id": row["case_id"],
                    "forbid": SPLITS,
                    "geometry": row["arrays"],
                    "geometry_digest": row["geometry_digest"],
                }
            )
    return refs


# --------------------------------------------------------------------------- plan


def _same_geometry(
    original: dict[str, Any], new: dict[str, Any], tolerance: float = GEOMETRY_TOLERANCE_A
) -> str | None:
    if original["symbols"] != new["symbols"]:
        return "species order differs"
    if np.max(np.abs(original["cell"] - new["cell"])) > tolerance:
        return "cell differs"
    delta = new["positions"] - original["positions"]
    frac = np.linalg.solve(original["cell"].T, delta.T).T
    frac -= np.round(frac)
    worst = float(np.max(np.linalg.norm(frac @ original["cell"], axis=1)))
    return None if worst <= tolerance else f"positions differ by up to {worst:.2e} A"


def plan_release(
    ledger: list[dict[str, Any]], policy: dict[str, Any], replacements: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Decide every ledger frame; raise on any replacement that does not belong."""
    keys = {(r["source_run"], r["source_frame"]) for r in ledger}
    unknown = sorted(set(replacements) - keys)
    if unknown:
        raise SafetyError(f"Replacements for frames absent from the frozen ledger: {unknown[:5]}")
    decisions = []
    missing: dict[str, int] = Counter()
    for row in ledger:
        key = (row["source_run"], row["source_frame"])
        pattern, rule = source_rule(policy, row["source_run"])
        action = rule["action"]
        replacement = replacements.get(key)
        if replacement and action not in {"replace_all", "replace_subset"}:
            raise SafetyError(f"Replacement for {key[0]} frame {key[1]}, whose rule '{pattern}' is {action}")
        if replacement and replacement["manifest_split"] not in {None, row["split"]}:
            raise SafetyError(
                f"{replacement['replacement']}: {key[0]} frame {key[1]} selected as {replacement['manifest_split']},"
                f" the frozen ledger says {row['split']}"
            )
        if action == "exclude":
            status, label = "excluded", None
        elif action == "keep":
            status, label = "kept", "original"
        elif replacement:
            status, label = "replaced", replacement["label_source"]
        elif action == "replace_all":
            missing[row["source_run"]] += 1
            continue
        else:
            status, label = ("kept", "original") if rule["unreplaced"] == "keep" else ("held", None)
        decisions.append(
            {
                **row,
                "status": status,
                "label_source": label,
                "label_use": rule.get("label_use", "all") if status in {"kept", "replaced"} else "",
                "rule": pattern,
                "reason": rule.get("reason", ""),
            }
        )
    if missing:
        raise SafetyError(f"replace_all sources lack replacements: {dict(missing)}")
    return decisions


def find_leaks(frames: list[dict[str, Any]], refs: list[dict[str, Any]]) -> list[str]:
    """Reference geometries found in a forbidden split: by digest, and within LEAKAGE_TOLERANCE_A
    for every frame that brings its geometry (``geometry`` or ``arrays``)."""
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_symbols: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_digest[frame["geometry_digest"]].append(frame)
        geometry = frame.get("geometry") or frame.get("arrays")
        if geometry is not None:
            by_symbols[tuple(geometry["symbols"])].append(frame)
    leaks = []
    for ref in refs:
        hits = {id(f): f for f in by_digest.get(ref["geometry_digest"], [])}
        for frame in by_symbols.get(tuple(ref["geometry"]["symbols"]), []):
            geometry = frame.get("geometry") or frame["arrays"]
            if _same_geometry(ref["geometry"], geometry, LEAKAGE_TOLERANCE_A) is None:
                hits[id(frame)] = frame
        for frame in hits.values():
            if frame["split"] in ref["forbid"] and frame["source_run"] != ref["id"]:
                leaks.append(
                    f"{ref['id']}[{ref['index']}] {ref['case_id']} is {frame['source_run']} frame"
                    f" {frame['source_frame']} in {frame['split']}"
                )
    return leaks


def check_added(decisions: list[dict[str, Any]], added: list[dict[str, Any]]) -> None:
    """An added training frame must not repeat a deck frame or another added frame in any split."""
    splits: dict[str, set[str]] = defaultdict(set)
    for d in decisions:
        if d["status"] in {"kept", "replaced"}:
            splits[d["geometry_digest"]].add(d["split"])
    problems = []
    for row in added:
        if row["split"] == "holdout":
            continue
        if splits[row["geometry_digest"]]:
            seen = sorted(splits[row["geometry_digest"]])
            problems.append(f"{row['source_run']}[{row['source_frame']}] repeats a {seen} frame")
        splits[row["geometry_digest"]].add(row["split"])
    if problems:
        raise SafetyError("Added frames duplicate release geometries: " + "; ".join(problems[:5]))


def _count(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row[key]][row[value]] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


def scan_deck_geometries(deck: Path, symbols: set[tuple[str, ...]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Geometry of every deck frame whose species order is one of ``symbols`` (no other frame can match)."""
    found = {}
    natoms = {len(s) for s in symbols}
    for split in SPLITS:
        for _, text in iter_extxyz(deck / f"{split}.extxyz"):
            if int(text.split("\n", 1)[0]) not in natoms:
                continue
            atoms = parse_frame(text)
            if tuple(atoms.get_chemical_symbols()) in symbols:
                found[(str(atoms.info["source_run"]), int(atoms.info["source_frame"]))] = _geometry(atoms)
    return found


def plan(ledger_path: Path, policy_path: Path, deck: Path | None = None) -> dict[str, Any]:
    """Every release decision and check that does not write a release."""
    ledger = read_ledger(ledger_path)
    policy = load_policy(policy_path)
    replacements = load_replacements(policy, policy_path.parent)
    decisions = plan_release(ledger, policy, replacements)
    added = load_added(policy, policy_path.parent)
    check_added(decisions, added)
    refs = load_references(policy, policy_path.parent, added)
    active = [d for d in decisions if d["status"] in {"kept", "replaced"}]
    training = [r for r in added if r["split"] != "holdout"]
    if deck is not None:
        geometries = scan_deck_geometries(deck, {tuple(r["geometry"]["symbols"]) for r in refs})
        active = [{**d, "geometry": geometries.get((d["source_run"], d["source_frame"]))} for d in active]
    frames = active + training
    leaks = find_leaks(frames, refs)
    if leaks:
        raise SafetyError(f"{len(leaks)} holdout/reference geometries in forbidden splits: " + "; ".join(leaks[:5]))
    return {
        "schema_version": 1,
        "ledger_sha256": sha256_file(ledger_path),
        "policy_sha256": sha256_file(policy_path),
        "status_counts": dict(Counter(d["status"] for d in decisions + added)),
        "split_counts": dict(Counter(d["split"] for d in frames)),
        "label_use_by_split": {
            split: dict(Counter(d["label_use"] for d in frames if d["split"] == split)) for split in SPLITS
        },
        "replaced_by_source": dict(Counter(d["source_run"] for d in decisions if d["status"] == "replaced")),
        "added_by_entry": _count(added, "source_run", "split"),
        "leakage_references": dict(Counter(r["id"] for r in refs)),
        "leakage_check": "digest and tolerance against the deck" if deck is not None else "digest against the ledger",
        "label_use_by_source": {run: use for run, use in sorted({(d["source_run"], d["label_use"]) for d in frames})},
        "per_source": _count(decisions, "source_run", "status"),
    }


# --------------------------------------------------------------------------- assemble


def _release_atoms(arrays: dict[str, Any], info: dict[str, Any], label_use: str, label_source: str, *, full: bool):
    """One output frame; masked terms are left out unless ``full`` (holdouts keep every label)."""
    from ase import Atoms

    terms = LABEL_USES["all"] if full else LABEL_USES[label_use]
    out = Atoms(symbols=arrays["symbols"], positions=arrays["positions"], cell=arrays["cell"], pbc=True)
    out.info.update({k: v for k, v in info.items() if k not in {"REF_energy", "REF_virial", "REF_stress_GPa"}})
    if "energy" in terms:
        out.info["REF_energy"] = arrays["energy"]
    if "virial" in terms:
        out.info["REF_virial"] = arrays["virial"].reshape(-1)
    out.info["label_use"] = label_use
    out.info["label_source"] = label_source
    out.arrays["REF_forces"] = arrays["forces"]
    out.arrays["move_mask"] = arrays["move_mask"]
    return out


def assemble(ledger_path: Path, policy_path: Path, deck: Path, output: Path) -> dict[str, Any]:
    from ase import Atoms
    from ase.io import write

    from .data import Frame, SourceTrajectory, _write_deepmd_system

    if output.exists() and any(output.iterdir()):
        raise SafetyError(f"Release directory is not empty: {output}; releases are never overwritten")
    ledger = read_ledger(ledger_path)
    policy = load_policy(policy_path)
    replacements = load_replacements(policy, policy_path.parent)
    decisions = plan_release(ledger, policy, replacements)
    added = load_added(policy, policy_path.parent)
    check_added(decisions, added)
    refs = load_references(policy, policy_path.parent, added)
    ref_symbols = {tuple(r["geometry"]["symbols"]) for r in refs}
    # The input may be the old deck or a fresh admission-gated recollection of the same
    # sources: frames are matched by (source_run, source_frame), never by file position.
    by_key = {(d["source_run"], d["source_frame"]): d for d in decisions}
    active = {k for k, d in by_key.items() if d["status"] in {"kept", "replaced"}}
    emitted: set[tuple[str, int]] = set()
    output.mkdir(parents=True, exist_ok=True)
    systems: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    compared: list[dict[str, Any]] = []
    geometry_mismatch = []
    for split in SPLITS:
        target = output / f"{split}.extxyz"
        for index, text in iter_extxyz(deck / f"{split}.extxyz"):
            run, frame = _info_value(text, "source_run"), _info_value(text, "source_frame")
            key = (str(run), int(frame) if frame is not None else -1)
            decision = by_key.get(key)
            if decision is None:
                raise SafetyError(f"{split}.extxyz[{index}] ({key[0]} frame {key[1]}) is not in the frozen ledger")
            if decision["split"] != split:
                raise SafetyError(f"{key[0]} frame {key[1]} is in {split}; the frozen ledger says {decision['split']}")
            if key not in active:
                continue
            if key in emitted:
                raise SafetyError(f"{key[0]} frame {key[1]} appears twice in the input")
            emitted.add(key)
            atoms = parse_frame(text)
            original = frame_arrays(atoms)
            if (
                geometry_digest(original) != decision["geometry_digest"]
                or label_digest(original) != decision["label_digest"]
            ):
                raise SafetyError(f"{split}.extxyz[{index}] changed since the ledger was frozen")
            arrays = original
            if decision["status"] == "replaced":
                new = replacements[(decision["source_run"], decision["source_frame"])]["arrays"]
                problem = _same_geometry(original, new)
                if problem:
                    geometry_mismatch.append(f"{decision['source_run']} frame {decision['source_frame']}: {problem}")
                    continue
                # Same geometry: every label channel comes from the one replacement calculation.
                arrays = {**original, "energy": new["energy"], "forces": new["forces"], "virial": new["virial"]}
            if "virial" in LABEL_USES[decision["label_use"]] and arrays["virial"] is None:
                raise SafetyError(f"{key[0]} frame {key[1]}: label_use {decision['label_use']} but no virial")
            if tuple(arrays["symbols"]) in ref_symbols:
                compared.append({**decision, "geometry": original})
            out = _release_atoms(arrays, atoms.info, decision["label_use"], decision["label_source"], full=False)
            write(str(target), out, format="extxyz", append=target.exists())
            systems[(split, decision["label_use"], decision["leaf"])].append((decision, arrays))
    absent = sorted(active - emitted)
    if absent:
        raise SafetyError(f"{len(absent)} active ledger frames are missing from the input, e.g. {absent[:3]}")
    if geometry_mismatch:
        raise SafetyError("Replacement geometry differs from the frozen frame: " + "; ".join(geometry_mismatch[:5]))
    leaks = find_leaks(compared + [r for r in added if r["split"] != "holdout"], refs)
    if leaks:
        raise SafetyError(f"{len(leaks)} holdout/reference geometries in forbidden splits: " + "; ".join(leaks[:5]))
    for row in added:
        holdout = row["split"] == "holdout"
        out = _release_atoms(row["arrays"], row["info"], row["label_use"], row["label_source"], full=holdout)
        out.info.update(source_run=row["source_run"], source_frame=row["source_frame"], split=row["split"])
        target = output / f"{row['split']}.extxyz"
        write(str(target), out, format="extxyz", append=target.exists())
        if not holdout:
            # One DeePMD system per species order: an added file may hold several materials.
            symbols = row["arrays"]["symbols"]
            leaf = f"{row['leaf']}/{Atoms(symbols).get_chemical_formula()}_{len(symbols)}"
            systems[(row["split"], row["label_use"], leaf)].append((row, row["arrays"]))
    type_map = policy.get("type_map") or sorted(
        {s for items in systems.values() for _, a in items for s in a["symbols"]}
    )
    index = []
    for (split, label_use, leaf), items in sorted(systems.items()):
        if len({tuple(a["symbols"]) for _, a in items}) > 1:
            raise SafetyError(f"deepmd {split}/{label_use}/{leaf}: frames differ in species order")
        terms = LABEL_USES[label_use]
        frames = [
            Frame(
                source_index=d["source_frame"],
                atoms=Atoms(symbols=a["symbols"], positions=a["positions"], cell=a["cell"], pbc=True),
                energy=a["energy"],
                forces=a["forces"],
                move_mask=a["move_mask"],
                virial=a["virial"] if "virial" in terms else None,
            )
            for d, a in items
        ]
        source = SourceTrajectory(path=Path(items[0][0]["source_path"]), run_id=leaf, category="", group="")
        system = _write_deepmd_system(
            output / "deepmd" / split / label_use, source, frames, type_map, include_energy="energy" in terms
        )
        data = system / "set.000"
        present = {"energy": (data / "energy.npy").is_file(), "virial": (data / "virial.npy").is_file()}
        if present != {"energy": "energy" in terms, "virial": "virial" in terms}:
            raise SafetyError(f"deepmd {split}/{label_use}/{leaf}: label files {present} do not match {label_use}")
        record = {
            "system": str(system.relative_to(output)).replace("\\", "/"),
            "split": split,
            "label_use": label_use,
            "frames": len(frames),
            "trained_terms": list(terms),
            "absent_files": [f"{t}.npy" for t in ("energy", "virial") if t not in terms],
        }
        (system / "release_labels.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        index.append(record)
    (output / "deepmd").mkdir(exist_ok=True)
    (output / "deepmd" / "systems.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    rows = decisions + added
    with (output / "release_ledger.tsv").open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "source_run",
            "source_frame",
            "split",
            "status",
            "label_use",
            "label_source",
            "rule",
            "reason",
            "case_id",
            "geometry_digest",
            "label_digest",
        )
        writer = csv.DictWriter(handle, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = release_report(output, rows, ledger_path, policy_path)
    report["leakage_references"] = dict(Counter(r["id"] for r in refs))
    (output / "release_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def release_report(output: Path, decisions: list[dict[str, Any]], ledger: Path, policy: Path) -> dict[str, Any]:
    status = Counter(d["status"] for d in decisions)
    active = [d for d in decisions if d["status"] in {"kept", "replaced", "added"}]
    geometry_splits: dict[str, set[str]] = defaultdict(set)
    for d in active:
        geometry_splits[d["geometry_digest"]].add(d["split"])
    files = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "release_report.json")
    return {
        "schema_version": 1,
        "ledger_sha256": sha256_file(ledger),
        "policy_sha256": sha256_file(policy),
        "status_counts": dict(status),
        "split_counts": dict(Counter(d["split"] for d in active)),
        "replaced_by_split": dict(Counter(d["split"] for d in active if d["status"] == "replaced")),
        "added_by_split": dict(Counter(d["split"] for d in decisions if d["status"] in {"added", "holdout"})),
        "label_use_by_split": {
            split: dict(Counter(d["label_use"] for d in active if d["split"] == split)) for split in SPLITS
        },
        "per_source": {
            run: dict(Counter(d["status"] for d in decisions if d["source_run"] == run))
            for run in sorted({d["source_run"] for d in decisions})
        },
        "geometries_in_more_than_one_split": sum(len(s) > 1 for s in geometry_splits.values()),
        "files_sha256": {str(p.relative_to(output)).replace("\\", "/"): sha256_file(p) for p in files},
    }


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("freeze", help="Record the historical split of every deck frame")
    f.add_argument("deck", type=Path, help="Folder with {train,valid,test}.extxyz")
    f.add_argument("--deepmd", type=Path, help="The deck's DeePMD tree, checked frame by frame")
    f.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("plan", help="Check a policy against the ledger without writing a release")
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--deck", type=Path, help="Also compare holdout/reference geometries with the deck's frames")
    p.add_argument("--report", type=Path, help="Write the plan summary here as JSON")
    a = commands.add_parser("assemble", help="Build a release from the ledger, a policy and replacements")
    a.add_argument("--ledger", type=Path, required=True)
    a.add_argument("--policy", type=Path, required=True)
    a.add_argument("--deck", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            report = freeze(args.deck, args.output, deepmd=args.deepmd)
            print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, indent=1))
            return 0 if not report["errors"] else 2
        if args.command == "plan":
            report = plan(args.ledger, args.policy, args.deck)
            if args.report:
                args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "per_source"}, indent=1))
            return 0
        report = assemble(args.ledger, args.policy, args.deck, args.output)
    except SafetyError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
