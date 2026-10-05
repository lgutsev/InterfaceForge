"""Frozen split ledger and same-geometry partial replacement for the SiN/TiN rescue.

``freeze``   read an existing exported deck ({train,valid,test}.extxyz plus its
             DeePMD tree), record every (source_run, source_frame) with its
             historical split, geometry and label hashes, and verify that the
             DeePMD export holds exactly the same frames with the same numbers.
``assemble`` build a release from the frozen ledger and a release policy:
             per-source keep/exclude/replace, where replacements are reviewed
             relabels of the *same* geometry. Every replaced frame keeps its
             historical split; energy, forces and virial are replaced
             together. Missing, duplicate, unknown or wrong-source replacements
             are errors. Nothing is re-split and no old export is modified.

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
GEOMETRY_TOLERANCE_A = 1e-6
ACTIONS = {"keep", "exclude", "replace_all", "replace_subset"}
UNREPLACED = {"keep", "hold"}


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
    for item in policy.get("replacements", []):
        if not {"id", "path", "sha256"} <= set(item):
            raise SafetyError(f"{path}: every replacement needs id, path and sha256")
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


# --------------------------------------------------------------------------- assemble


def load_replacements(policy: dict[str, Any], base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for item in policy.get("replacements", []):
        path = (base / item["path"]).resolve() if not Path(item["path"]).is_absolute() else Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise SafetyError(f"Replacement {item['id']} changed: {path}")
        for index, text in iter_extxyz(path):
            atoms = parse_frame(text)
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
            if not (
                math.isfinite(arrays["energy"])
                and np.all(np.isfinite(arrays["forces"]))
                and np.all(np.isfinite(arrays["virial"]))
            ):
                raise SafetyError(f"{item['id']}[{index}]: nonfinite replacement label")
            found[key] = {
                "replacement": item["id"],
                "index": index,
                "arrays": arrays,
                "label_source": atoms.info.get("label_source", item["id"]),
            }
    return found


def _same_geometry(original: dict[str, Any], new: dict[str, Any]) -> str | None:
    if original["symbols"] != new["symbols"]:
        return "species order differs"
    if np.max(np.abs(original["cell"] - new["cell"])) > GEOMETRY_TOLERANCE_A:
        return "cell differs"
    delta = new["positions"] - original["positions"]
    frac = np.linalg.solve(original["cell"].T, delta.T).T
    frac -= np.round(frac)
    worst = float(np.max(np.linalg.norm(frac @ original["cell"], axis=1)))
    return None if worst <= GEOMETRY_TOLERANCE_A else f"positions differ by up to {worst:.2e} A"


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
            {**row, "status": status, "label_source": label, "rule": pattern, "reason": rule.get("reason", "")}
        )
    if missing:
        raise SafetyError(f"replace_all sources lack replacements: {dict(missing)}")
    return decisions


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
    # The input may be the old deck or a fresh admission-gated recollection of the same
    # sources: frames are matched by (source_run, source_frame), never by file position.
    by_key = {(d["source_run"], d["source_frame"]): d for d in decisions}
    active = {k for k, d in by_key.items() if d["status"] in {"kept", "replaced"}}
    emitted: set[tuple[str, int]] = set()
    output.mkdir(parents=True, exist_ok=True)
    systems: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
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
            out = Atoms(symbols=arrays["symbols"], positions=arrays["positions"], cell=arrays["cell"], pbc=True)
            out.info.update({k: v for k, v in atoms.info.items() if k not in {"REF_energy", "REF_virial"}})
            out.info["REF_energy"] = arrays["energy"]
            out.info["REF_virial"] = arrays["virial"].reshape(-1)
            out.info["label_source"] = decision["label_source"]
            out.arrays["REF_forces"] = arrays["forces"]
            out.arrays["move_mask"] = arrays["move_mask"]
            write(str(target), out, format="extxyz", append=target.exists())
            systems[(split, decision["leaf"])].append((decision, arrays))
    absent = sorted(active - emitted)
    if absent:
        raise SafetyError(f"{len(absent)} active ledger frames are missing from the input, e.g. {absent[:3]}")
    if geometry_mismatch:
        raise SafetyError("Replacement geometry differs from the frozen frame: " + "; ".join(geometry_mismatch[:5]))
    type_map = policy.get("type_map") or sorted(
        {s for items in systems.values() for _, a in items for s in a["symbols"]}
    )
    for (split, leaf), items in sorted(systems.items()):
        frames = [
            Frame(
                source_index=d["source_frame"],
                atoms=Atoms(symbols=a["symbols"], positions=a["positions"], cell=a["cell"], pbc=True),
                energy=a["energy"],
                forces=a["forces"],
                move_mask=a["move_mask"],
                virial=a["virial"],
            )
            for d, a in items
        ]
        source = SourceTrajectory(path=Path(items[0][0]["source_path"]), run_id=leaf, category="", group="")
        _write_deepmd_system(output / "deepmd" / split, source, frames, type_map)
    with (output / "release_ledger.tsv").open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "source_run",
            "source_frame",
            "split",
            "status",
            "label_source",
            "rule",
            "reason",
            "geometry_digest",
            "label_digest",
        )
        writer = csv.DictWriter(handle, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(decisions)
    report = release_report(output, decisions, ledger_path, policy_path)
    (output / "release_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def release_report(output: Path, decisions: list[dict[str, Any]], ledger: Path, policy: Path) -> dict[str, Any]:
    status = Counter(d["status"] for d in decisions)
    active = [d for d in decisions if d["status"] in {"kept", "replaced"}]
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
        report = assemble(args.ledger, args.policy, args.deck, args.output)
    except SafetyError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
