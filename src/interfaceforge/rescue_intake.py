"""Laptop-side intake of resettled A2 DFT packages for the SiN/TiN rescue.

Reads verified local copies only; never resolves LONI paths as local files and
never edits a package. Three read-only operations:

``intake``   verify each package against its RESETTLE_AUDIT.tsv, bind every
             manifest calculation to its ``runs/`` (inputs + outputs) and
             ``outputs/`` (export) copies, and choose one canonical record.
``pairs``    join each package-21 orig/kconv pair to its stored deck frame and
             recompute orig-vs-stored and kconv-vs-orig E/F/full stress.
``compare``  side-by-side numerics and complete SCF traces of two statics
             (frame 0008 versus 0009).

Units: energies eV (E_sigma->0 and TOTEN kept separate), forces eV/A, stress
GPa with tensile positive (ASE sign), virial eV with virial = -V * stress.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .source_audit import parse_oszicar, scan_outcar
from .vasp import parse_incar
from .vasp_provenance import sha256_file

EV_A3_TO_GPA = 160.21766208
OUTPUT_FILES = ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR")
INPUT_FILES = ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
# VASP defaults when a tag is absent from the original MD INCAR.
VASP_DEFAULTS = {"EDIFF": "1E-4", "NELM": "60", "ALGO": "Normal"}


# --------------------------------------------------------------------------- intake


def read_resettle_audit(package: Path) -> list[dict[str, str]]:
    rows = []
    lines = (package / "RESETTLE_AUDIT.tsv").read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if line and not line.startswith("#")]
    for row in csv.DictReader(body, delimiter="\t"):
        rows.append(row)
    return rows


def verify_resettle(package: Path) -> dict[str, Any]:
    """Hash every audited member; list missing/mismatched/unlisted relative paths."""
    rows = read_resettle_audit(package)
    listed, verified, mismatched, missing, links = set(), 0, [], [], []
    for row in rows:
        relative = row["path"]
        listed.add(relative)
        local = package / relative
        if row.get("target"):
            links.append({"path": relative, "target": row["target"], "present": local.exists()})
            continue
        if not local.is_file():
            missing.append(relative)
            continue
        if int(row["bytes"]) != local.stat().st_size or sha256_file(local) != row["sha256"]:
            mismatched.append(relative)
            continue
        verified += 1
    unlisted = sorted(
        str(path.relative_to(package)).replace("\\", "/")
        for path in package.rglob("*")
        if path.is_file()
        and str(path.relative_to(package)).replace("\\", "/") not in listed
        and path.name != "RESETTLE_AUDIT.tsv"
    )
    header = [line for line in (package / "RESETTLE_AUDIT.tsv").read_text(encoding="utf-8").splitlines()[:3]]
    return {
        "header": header,
        "rows": len(rows),
        "verified": verified,
        "mismatched": mismatched,
        "missing": missing,
        "links": links,
        "unlisted_local": unlisted,
        "complete": not mismatched and not missing,
    }


def _hashes(directory: Path, names: Iterable[str]) -> dict[str, str | None]:
    return {name: (sha256_file(directory / name) if (directory / name).is_file() else None) for name in names}


def _incar(path: Path) -> dict[str, str]:
    return {k.upper(): str(v) for k, v in parse_incar(path).items()} if path.is_file() else {}


def _same_tags(a: dict[str, str], b: dict[str, str]) -> bool:
    from .source_audit import equivalent

    keys = set(a) | set(b)
    return all(k in a and k in b and equivalent(a[k], b[k], k) for k in keys)


def scf_summary(run: Path, incar: dict[str, str]) -> dict[str, Any]:
    ediff = float(incar.get("EDIFF", VASP_DEFAULTS["EDIFF"]).replace("D", "E"))
    nelm = int(incar.get("NELM", VASP_DEFAULTS["NELM"]))
    frames, issues = parse_oszicar(run / "OSZICAR", nelm=nelm, ediff=ediff)
    final = frames[-1] if frames else {}
    trace = electronic_trace(run / "OSZICAR")
    # Energy rise after the SCF first settled (|dE| < 1 meV); startup steps are excluded.
    settled = next((i for i, step in enumerate(trace) if i > 0 and abs(step["de"]) < 1e-3), len(trace))
    rises = [b["energy"] - a["energy"] for a, b in zip(trace[settled:], trace[settled + 1 :], strict=False)]
    return {
        "ionic_frames": len(frames),
        "ediff_ev": ediff,
        "nelm": nelm,
        "algo": incar.get("ALGO", VASP_DEFAULTS["ALGO"]),
        "scf_steps": final.get("last_iteration"),
        "nelm_reached": bool(final) and final["last_iteration"] >= nelm,
        "final_de_ev": final.get("last_de_ev"),
        "final_deps_ev": final.get("last_deps_ev"),
        "energy_criterion_met": final.get("energy_criterion_met"),
        "last_rms_c": final.get("last_rms_c"),
        "last_rms_c_iteration": final.get("last_rms_c_iteration"),
        "final_step_rms_c": final.get("final_rms_c"),
        "residual_jumps": final.get("residual_jumps"),
        "settled_iteration": trace[settled]["iteration"] if settled < len(trace) else None,
        "max_rise_after_settled_ev": max(rises, default=0.0),
        "issue_codes": sorted({item["code"] for item in issues}),
    }


def calculation_record(package: Path, frame: dict[str, Any], level: str) -> dict[str, Any]:
    name = frame["frame"]
    run_dir = package / "runs" / name / level
    out_dir = package / "outputs" / name / level
    run_out = _hashes(run_dir, OUTPUT_FILES)
    export_out = _hashes(out_dir, OUTPUT_FILES)
    run_in = _hashes(run_dir, INPUT_FILES)
    staged = package / "inputs" / name
    staged_hashes = {
        "INCAR": sha256_file(staged / f"INCAR.{level}") if (staged / f"INCAR.{level}").is_file() else None,
        "KPOINTS": sha256_file(staged / "KPOINTS") if (staged / "KPOINTS").is_file() else None,
        "POSCAR": sha256_file(staged / "POSCAR") if (staged / "POSCAR").is_file() else None,
    }
    incar = _incar(run_dir / "INCAR")
    staged_incar = _incar(staged / f"INCAR.{level}")
    # Relaxations start from the staged POSCAR; later levels may chain CONTCARs.
    duplicate_outputs = all(run_out[k] and run_out[k] == export_out[k] for k in ("OUTCAR", "OSZICAR"))
    problems = []
    if not run_out["OUTCAR"]:
        problems.append("runs/ OUTCAR missing")
    if not all(run_in.values()):
        problems.append("runs/ inputs missing: " + ",".join(k for k, v in run_in.items() if not v))
    if export_out["OUTCAR"] and not duplicate_outputs:
        problems.append("runs/ and outputs/ OUTCAR or OSZICAR differ")
    if staged_incar and incar and not _same_tags(staged_incar, incar):
        problems.append("executed INCAR differs from staged INCAR")
    if staged_hashes["KPOINTS"] and run_in["KPOINTS"] != staged_hashes["KPOINTS"]:
        problems.append("KPOINTS differs from staged")
    outcar = scan_outcar(run_dir / "OUTCAR") if run_out["OUTCAR"] else {}
    return {
        "package": package.name,
        "frame": name,
        "level": level,
        "case_id": frame.get("case_id"),
        "role": frame.get("role"),
        "material": frame.get("material"),
        "natoms": frame.get("natoms"),
        "geometry_key": frame.get("geometry_key"),
        "canonical": f"runs/{name}/{level}",
        "export_copy": f"outputs/{name}/{level}" if export_out["OUTCAR"] else None,
        "export_copy_identical": duplicate_outputs if export_out["OUTCAR"] else None,
        "inputs_sha256": run_in,
        "staged_inputs_sha256": staged_hashes,
        "outputs_sha256": run_out,
        "vasp_version": outcar.get("vasp_version"),
        "nkpts": outcar.get("nkpts"),
        "completed": outcar.get("completed"),
        "ionic_frames_outcar": outcar.get("frames"),
        "potcar_titles": outcar.get("potcar_titles"),
        "scf": scf_summary(run_dir, incar) if run_out["OSZICAR"] else None,
        "problems": problems,
    }


def intake_package(package: Path) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    calcs = [calculation_record(package, frame, level) for frame in manifest["frames"] for level in frame["levels"]]
    return {
        "package": package.name,
        "manifest": {
            "package": manifest.get("package"),
            "version": manifest.get("version"),
            "generator_commit": manifest.get("generator_commit"),
            "config_sha256": manifest.get("config_sha256"),
            "sha256": sha256_file(package / "manifest.json"),
        },
        "roles": sorted({frame.get("role") for frame in manifest["frames"]}),
        "resettle": verify_resettle(package),
        "calculations": calcs,
    }


INTAKE_COLUMNS = (
    "package",
    "frame",
    "level",
    "case_id",
    "role",
    "material",
    "natoms",
    "canonical",
    "export_copy_identical",
    "vasp_version",
    "nkpts",
    "completed",
    "outcar_sha256",
    "oszicar_sha256",
    "incar_sha256",
    "kpoints_sha256",
    "poscar_sha256",
    "potcar_sha256",
    "ediff_ev",
    "nelm",
    "algo",
    "scf_steps",
    "nelm_reached",
    "final_de_ev",
    "final_deps_ev",
    "last_rms_c",
    "residual_jumps",
    "max_rise_after_settled_ev",
    "problems",
)


def _intake_row(calc: dict[str, Any]) -> dict[str, Any]:
    scf = calc["scf"] or {}
    return {
        **{k: calc.get(k) for k in INTAKE_COLUMNS if k in calc},
        "outcar_sha256": calc["outputs_sha256"]["OUTCAR"],
        "oszicar_sha256": calc["outputs_sha256"]["OSZICAR"],
        **{f"{k.lower()}_sha256": calc["inputs_sha256"][k] for k in INPUT_FILES},
        **{
            k: scf.get(k)
            for k in (
                "ediff_ev",
                "nelm",
                "algo",
                "scf_steps",
                "nelm_reached",
                "final_de_ev",
                "final_deps_ev",
                "last_rms_c",
                "residual_jumps",
                "max_rise_after_settled_ev",
            )
        },
        "problems": "; ".join(calc["problems"]),
    }


def run_intake(packages: Sequence[Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reports = [intake_package(package) for package in packages]
    rows = [_intake_row(calc) for report in reports for calc in report["calculations"]]
    with (output / "intake.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, INTAKE_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    missing = [f"{r['package']}/{path}" for r in reports for path in r["resettle"]["missing"]]
    (output / "missing_for_dispatch.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    summary = {
        "packages": {
            r["package"]: {
                "resettle_rows": r["resettle"]["rows"],
                "verified": r["resettle"]["verified"],
                "mismatched": len(r["resettle"]["mismatched"]),
                "missing": len(r["resettle"]["missing"]),
                "unlisted_local": len(r["resettle"]["unlisted_local"]),
                "calculations": len(r["calculations"]),
                "duplicate_pairs_identical": sum(bool(c["export_copy_identical"]) for c in r["calculations"]),
                "calculations_with_problems": sum(bool(c["problems"]) for c in r["calculations"]),
                "roles": r["roles"],
            }
            for r in reports
        },
    }
    summary["duplicate_pairs_identical"] = sum(p["duplicate_pairs_identical"] for p in summary["packages"].values())
    summary["calculations"] = sum(p["calculations"] for p in summary["packages"].values())
    (output / "intake.json").write_text(
        json.dumps({"summary": summary, "packages": reports}, indent=1), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- labels


def electronic_trace(oszicar: Path) -> list[dict[str, Any]]:
    """All electronic steps of a single-point OSZICAR (first ionic step only)."""
    from .source_audit import _number, _scf_columns

    trace = []
    for line in oszicar.read_text(encoding="utf-8", errors="replace").splitlines():
        scf = re.match(r"\s*(DAV|RMM|CG|DMP)\s*:\s*(\d+)\s+(.*)", line)
        if scf:
            fields = _scf_columns(scf.group(3))
            trace.append(
                {
                    "algorithm": scf.group(1),
                    "iteration": int(scf.group(2)),
                    "energy": _number(fields["energy"]),
                    "de": _number(fields["de"]),
                    "deps": _number(fields["deps"]),
                    "ncg": int(fields["ncg"]),
                    "rms": _number(fields["rms"]),
                    "rms_c": _number(fields["rmsc"]) if fields["rmsc"] else None,
                }
            )
        elif re.match(r"\s*\d+\s+(?:T=|F=)", line):
            break
    return trace


def outcar_label(outcar: Path) -> dict[str, Any]:
    """Final-image label from an OUTCAR, with both energy conventions."""
    from ase.io import read

    atoms = read(outcar, index=-1, format="vasp-out")
    energy_zero = atoms.get_potential_energy()
    free = atoms.calc.results.get("free_energy")
    stress = atoms.get_stress(voigt=False) * EV_A3_TO_GPA
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions(),
        "cell": np.asarray(atoms.cell),
        "energy_sigma0": float(energy_zero),
        "free_energy": None if free is None else float(free),
        "forces": atoms.get_forces(),
        "stress_gpa": stress,
        "volume": float(atoms.get_volume()),
    }


def _extxyz_atoms(text: str):
    from ase.io import read

    return read(io.StringIO(text), format="extxyz", index=0)


def extxyz_label(atoms) -> dict[str, Any]:
    info = atoms.info
    virial = np.asarray(info["REF_virial"], dtype=float).reshape(3, 3)
    volume = atoms.get_volume()
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions(),
        "cell": np.asarray(atoms.cell),
        "energy_sigma0": float(info["REF_energy"]),
        "free_energy": None,
        "forces": np.asarray(atoms.arrays["REF_forces"], dtype=float),
        "stress_gpa": -virial / volume * EV_A3_TO_GPA,
        "virial": virial,
        "volume": float(volume),
        "info": {k: v for k, v in info.items() if isinstance(v, (str, int, float))},
    }


def deck_frames(deck: Path, wanted: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Stream ``{split}.extxyz`` and return the wanted (split, source_run, source_frame) frames."""
    found: dict[tuple[str, str, int], dict[str, Any]] = {}
    for split in sorted({key[0] for key in wanted}):
        path = deck / f"{split}.extxyz"
        with path.open(encoding="utf-8") as handle:
            index = 0
            while True:
                count_line = handle.readline()
                if not count_line:
                    break
                natoms = int(count_line)
                comment = handle.readline()
                run = re.search(r"\bsource_run=(\S+)", comment)
                frame = re.search(r"\bsource_frame=(\d+)", comment)
                key = (split, run.group(1) if run else "", int(frame.group(1)) if frame else -1)
                if key in wanted:
                    body = "".join(handle.readline() for _ in range(natoms))
                    label = extxyz_label(_extxyz_atoms(count_line + comment + body))
                    label["deck_file"] = path.name
                    label["deck_index"] = index
                    found[key] = label
                    if len(found) == len(wanted):
                        return found
                else:
                    for _ in range(natoms):
                        handle.readline()
                index += 1
    return found


def geometry_difference(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    same_species = a["symbols"] == b["symbols"]
    cell_diff = float(np.max(np.abs(a["cell"] - b["cell"])))
    if not same_species or len(a["positions"]) != len(b["positions"]):
        return {"same_species_order": False, "max_cell_diff_a": cell_diff, "max_position_diff_a": None}
    # Minimum-image position difference through fractional coordinates of a.
    delta = b["positions"] - a["positions"]
    frac = np.linalg.solve(a["cell"].T, delta.T).T
    frac -= np.round(frac)
    delta = frac @ a["cell"]
    return {
        "same_species_order": True,
        "max_cell_diff_a": cell_diff,
        "max_position_diff_a": float(np.max(np.linalg.norm(delta, axis=1))),
    }


def label_difference(test: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """``test - ref``: energies, full force RMSE/max and all nine stress components."""
    natoms = len(ref["symbols"])
    df = test["forces"] - ref["forces"]
    ds = test["stress_gpa"] - ref["stress_gpa"]
    out = {
        **geometry_difference(ref, test),
        "dE_sigma0_ev": test["energy_sigma0"] - ref["energy_sigma0"],
        "dE_sigma0_mev_atom": 1000 * (test["energy_sigma0"] - ref["energy_sigma0"]) / natoms,
        "dF_free_ev": (
            test["free_energy"] - ref["free_energy"]
            if test["free_energy"] is not None and ref["free_energy"] is not None
            else None
        ),
        "force_rmse_ev_a": float(np.sqrt(np.mean(df**2))),
        "force_max_component_ev_a": float(np.max(np.abs(df))),
        "force_max_atom_norm_ev_a": float(np.max(np.linalg.norm(df, axis=1))),
        "stress_diff_gpa": ds.round(6).tolist(),
        "stress_max_abs_diff_gpa": float(np.max(np.abs(ds))),
    }
    if out["dF_free_ev"] is not None:
        out["dF_free_mev_atom"] = 1000 * out["dF_free_ev"] / natoms
    return out


def _training_settings(package: Path, frame: dict[str, Any]) -> dict[str, str]:
    """The original MD INCAR tags for this frame's training run, with VASP defaults."""
    # TRAINING_RUN: the run path on its first line, then key=value notes.
    run = (package / "inputs" / frame["frame"] / "TRAINING_RUN").read_text(encoding="utf-8").splitlines()[0].strip()
    leaf = run.rstrip("/").split("/")[-1]
    path = package / "outputs" / frame["frame"] / f"training_INCAR.{leaf}"
    tags = _incar(path)
    return {
        "training_run": run,
        "training_incar": f"outputs/{frame['frame']}/training_INCAR.{leaf}" if path.is_file() else None,
        **{k: tags.get(k, f"{v} (VASP default)") for k, v in VASP_DEFAULTS.items()},
        **{k: tags.get(k) for k in ("PREC", "LREAL", "ENCUT", "ISMEAR", "SIGMA", "IVDW") if k in tags},
    }


def _source_key(frame: dict[str, Any]) -> tuple[str, str, int]:
    ref = frame["dataset_reference"]
    match = re.search(r"source_run (\S+), source_frame (\d+) \(split (\w+)\)", frame["source_structure"])
    if not match or int(match.group(2)) != int(ref["source_frame"]) or match.group(3) != ref["split"]:
        raise ValueError(f"{frame['frame']}: source_structure and dataset_reference disagree")
    return ref["split"], match.group(1), int(ref["source_frame"])


def run_pairs(package: Path, deck: Path, output: Path) -> list[dict[str, Any]]:
    """Paired package-21 comparison bound to the stored deck through the manifest."""
    from ase.io import read

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    frames = {f["case_id"]: f for f in manifest["frames"]}
    stems = sorted({cid.rsplit("_", 1)[0] for cid in frames}, key=lambda s: frames[s + "_orig"]["index"])
    keys = {stem: _source_key(frames[stem + "_orig"]) for stem in stems}
    deck_labels = deck_frames(deck, set(keys.values()))
    stored = read(package / "dataset_frames.extxyz", index=":", format="extxyz")
    rows = []
    for stem in stems:
        orig, kconv = frames[stem + "_orig"], frames[stem + "_kconv"]
        if _source_key(kconv) != keys[stem]:
            raise ValueError(f"{stem}: orig and kconv point at different source frames")
        ref = orig["dataset_reference"]
        stored_label = extxyz_label(stored[ref["dataset_index"]])
        deck_label = deck_labels.get(keys[stem])
        labels = {}
        for case, frame in (("orig", orig), ("kconv", kconv)):
            run = package / "runs" / frame["frame"] / "static"
            labels[case] = outcar_label(run / "OUTCAR")
            labels[case]["scf"] = scf_summary(run, _incar(run / "INCAR"))
            labels[case]["run_sha256"] = _hashes(run, ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "OUTCAR", "OSZICAR"))
            labels[case]["poscar"] = outcar_label_from_poscar(run / "POSCAR")
            labels[case]["kpoints"] = (
                " ".join(frame["kpoints"])
                if isinstance(frame["kpoints"][0], str)
                else "x".join(str(k) for k in frame["kpoints"])
            )
            labels[case]["vasp"] = scan_outcar(run / "OUTCAR")
        row = {
            "pair": stem,
            "orig_frame": orig["frame"],
            "kconv_frame": kconv["frame"],
            "material": orig["material"],
            "natoms": orig["natoms"],
            "source_split": keys[stem][0],
            "source_run": keys[stem][1],
            "source_frame_zero_based": keys[stem][2],
            "source_outcar_remote": ref["source_path"],
            "deck_file": deck_label["deck_file"] if deck_label else None,
            "deck_index": deck_label["deck_index"] if deck_label else None,
            "stored_equals_deck": _stored_equals_deck(stored_label, deck_label),
            "poscar_vs_deck": geometry_difference(deck_label, labels["orig"]["poscar"]) if deck_label else None,
            "kconv_poscar_vs_orig_poscar": geometry_difference(labels["orig"]["poscar"], labels["kconv"]["poscar"]),
            "orig_minus_stored": label_difference(labels["orig"], deck_label or stored_label),
            "kconv_minus_orig": label_difference(labels["kconv"], labels["orig"]),
            "training_settings": _training_settings(package, orig),
        }
        for case in ("orig", "kconv"):
            label = labels[case]
            row[case] = {
                "kpoints": label["kpoints"],
                "nkpts": label["vasp"].get("nkpts"),
                "vasp_version": label["vasp"].get("vasp_version"),
                "energy_sigma0_ev": label["energy_sigma0"],
                "free_energy_ev": label["free_energy"],
                "scf": label["scf"],
                "sha256": label["run_sha256"],
            }
        rows.append(row)
    (output / "pairs.json").write_text(json.dumps(rows, indent=1, default=_json_default), encoding="utf-8")
    _write_pairs_tsv(rows, output / "pairs.tsv")
    return rows


def outcar_label_from_poscar(poscar: Path) -> dict[str, Any]:
    from ase.io import read

    atoms = read(poscar, format="vasp")
    return {"symbols": atoms.get_chemical_symbols(), "positions": atoms.get_positions(), "cell": np.asarray(atoms.cell)}


def _stored_equals_deck(stored: dict[str, Any], deck: dict[str, Any] | None) -> bool | None:
    if deck is None:
        return None
    return bool(
        stored["symbols"] == deck["symbols"]
        and np.allclose(stored["positions"], deck["positions"], atol=1e-8)
        and np.allclose(stored["cell"], deck["cell"], atol=1e-8)
        and math.isclose(stored["energy_sigma0"], deck["energy_sigma0"], abs_tol=1e-8)
        and np.allclose(stored["forces"], deck["forces"], atol=1e-8)
        and np.allclose(stored["virial"], deck["virial"], atol=1e-6)
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value))


PAIR_COLUMNS = (
    "pair",
    "orig_frame",
    "kconv_frame",
    "material",
    "natoms",
    "source_split",
    "source_run",
    "source_frame_zero_based",
    "deck_index",
    "stored_equals_deck",
    "poscar_max_pos_diff_a",
    "orig_kpoints",
    "kconv_kpoints",
    "orig_nkpts",
    "kconv_nkpts",
    "md_ediff",
    "md_nelm",
    "md_algo",
    "orig_ediff",
    "orig_nelm",
    "orig_algo",
    "orig_scf_steps",
    "orig_last_rms_c",
    "orig_residual_jumps",
    "kconv_scf_steps",
    "kconv_last_rms_c",
    "kconv_residual_jumps",
    "kconv_max_rise_after_settled_ev",
    "orig_vs_stored_dE_mev_atom",
    "orig_vs_stored_force_rmse",
    "orig_vs_stored_force_max",
    "orig_vs_stored_stress_max_gpa",
    "kconv_vs_orig_dE_sigma0_mev_atom",
    "kconv_vs_orig_dF_free_mev_atom",
    "kconv_vs_orig_force_rmse",
    "kconv_vs_orig_force_max",
    "kconv_vs_orig_stress_max_gpa",
    "orig_outcar_sha256",
    "kconv_outcar_sha256",
)


def _write_pairs_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    def fmt(value: Any) -> Any:
        return f"{value:.6g}" if isinstance(value, float) else value

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, PAIR_COLUMNS, delimiter="\t")
        writer.writeheader()
        for r in rows:
            os_, ko = r["orig_minus_stored"], r["kconv_minus_orig"]
            md = r["training_settings"]
            writer.writerow(
                {
                    k: fmt(v)
                    for k, v in {
                        **{k: r[k] for k in PAIR_COLUMNS if k in r},
                        "poscar_max_pos_diff_a": (r["poscar_vs_deck"] or {}).get("max_position_diff_a"),
                        "orig_kpoints": r["orig"]["kpoints"],
                        "kconv_kpoints": r["kconv"]["kpoints"],
                        "orig_nkpts": r["orig"]["nkpts"],
                        "kconv_nkpts": r["kconv"]["nkpts"],
                        "md_ediff": md["EDIFF"],
                        "md_nelm": md["NELM"],
                        "md_algo": md["ALGO"],
                        "orig_ediff": r["orig"]["scf"]["ediff_ev"],
                        "orig_nelm": r["orig"]["scf"]["nelm"],
                        "orig_algo": r["orig"]["scf"]["algo"],
                        "orig_scf_steps": r["orig"]["scf"]["scf_steps"],
                        "orig_last_rms_c": r["orig"]["scf"]["last_rms_c"],
                        "orig_residual_jumps": r["orig"]["scf"]["residual_jumps"],
                        "kconv_scf_steps": r["kconv"]["scf"]["scf_steps"],
                        "kconv_last_rms_c": r["kconv"]["scf"]["last_rms_c"],
                        "kconv_residual_jumps": r["kconv"]["scf"]["residual_jumps"],
                        "kconv_max_rise_after_settled_ev": r["kconv"]["scf"]["max_rise_after_settled_ev"],
                        "orig_vs_stored_dE_mev_atom": os_["dE_sigma0_mev_atom"],
                        "orig_vs_stored_force_rmse": os_["force_rmse_ev_a"],
                        "orig_vs_stored_force_max": os_["force_max_component_ev_a"],
                        "orig_vs_stored_stress_max_gpa": os_["stress_max_abs_diff_gpa"],
                        "kconv_vs_orig_dE_sigma0_mev_atom": ko["dE_sigma0_mev_atom"],
                        "kconv_vs_orig_dF_free_mev_atom": ko.get("dF_free_mev_atom"),
                        "kconv_vs_orig_force_rmse": ko["force_rmse_ev_a"],
                        "kconv_vs_orig_force_max": ko["force_max_component_ev_a"],
                        "kconv_vs_orig_stress_max_gpa": ko["stress_max_abs_diff_gpa"],
                        "orig_outcar_sha256": r["orig"]["sha256"]["OUTCAR"],
                        "kconv_outcar_sha256": r["kconv"]["sha256"]["OUTCAR"],
                    }.items()
                }
            )


# --------------------------------------------------------------------------- compare


OUTCAR_SCALARS = {
    "NELECT": r"\bNELECT\s*=\s*([-\d.Ee+]+)",
    "NBANDS": r"\bNBANDS\s*=\s*(\d+)",
    "NKPTS": r"\bNKPTS\s*=\s*(\d+)",
    "ISTART": r"\bISTART\s*=\s*(\d+)",
    "ICHARG": r"\bICHARG\s*=\s*(\d+)",
    "ALGO_IALGO": r"\bIALGO\s*=\s*(\d+)",
    "ISMEAR": r"\bISMEAR\s*=\s*(-?\d+)",
    "SIGMA": r"\bSIGMA\s*=\s*([-\d.Ee+]+)",
    "EDIFF": r"\bEDIFF\s*=\s*([-\d.Ee+]+)",
    "NELM": r"\bNELM\s*=\s*(\d+)",
    "NELMDL": r"\bNELMDL\s*=\s*(-?\d+)",
    "AMIX": r"\bAMIX\s*=\s*([-\d.Ee+]+)",
    "BMIX": r"\bBMIX\s*=\s*([-\d.Ee+]+)",
    "LREAL": r"\bLREAL\s*=\s*(\S+)",
    "PREC": r"\bPREC\s*=\s*(\S+)",
    "ENCUT": r"\bENCUT\s*=\s*([-\d.Ee+]+)",
}


def outcar_numerics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {"sha256": sha256_file(path)}
    for key, pattern in OUTCAR_SCALARS.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else None
    out["potcar_titles"] = list(dict.fromkeys(re.findall(r"TITEL\s*=\s*(.+)", text)))
    out["vrhfin"] = list(dict.fromkeys(m.strip() for m in re.findall(r"VRHFIN\s*=\s*(.+)", text)))
    out["eddrmm_zhegv_warnings"] = len(re.findall(r"WARNING in EDDRMM: call to ZHEGV failed", text))
    out["warnings"] = len(re.findall(r"\bWARNING\b", text))
    fermi = re.findall(r"E-fermi\s*:\s*([-\d.]+)", text)
    out["efermi_final"] = float(fermi[-1]) if fermi else None
    for key, pattern in (
        ("PSCENC", r"PSCENC\s*=\s*([-\d.]+)"),
        ("TEWEN", r"TEWEN\s*=\s*([-\d.]+)"),
        ("EENTRO", r"EENTRO\s*=\s*([-\d.]+)"),
        ("EATOM", r"EATOM\s*=\s*([-\d.]+)"),
        ("Edisp", r"Edisp \(eV\):\s*([-\d.]+)"),
    ):
        values = re.findall(pattern, text)
        out[key + "_final"] = float(values[-1]) if values else None
    kpoints_header = re.search(r"Dimension of arrays:.*?k-points\s+NKPTS\s*=\s*(\d+)", text, re.S)
    out["nkpts_irreducible"] = int(kpoints_header.group(1)) if kpoints_header else None
    mesh = re.search(r"generate k-points for:\s*(\d+)\s+(\d+)\s+(\d+)", text)
    out["generated_mesh"] = "x".join(mesh.groups()) if mesh else None
    occupations = _final_occupations(text)
    out.update(occupations)
    out["toten_per_iteration"] = [float(v) for v in re.findall(r"free energy\s+TOTEN\s*=\s*([-\d.]+)", text)]
    out["rms_total_per_iteration"] = [float(v) for v in re.findall(r"rms\(total\)\s*=\s*([\d.Ee+-]+)", text)]
    return out


def _final_occupations(text: str) -> dict[str, Any]:
    """Partial-occupation summary of the last eigenvalue block (band, energy, occupation lines)."""
    start = text.rfind("E-fermi")
    if start < 0:
        return {}
    block = text[start:]
    end = block.find("--------------------------------------------------------------------------------")
    block = block[: end if end > 0 else None]
    occ = [float(m.group(2)) for m in re.finditer(r"^\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", block, re.M)]
    if not occ:
        return {}
    arr = np.asarray(occ)
    return {
        "occupation_values": len(arr),
        "occupation_sum_per_kpoint_mean": None,
        "fractional_occupations": int(np.sum((arr > 0.01) & (arr < 0.99 * arr.max()))),
        "max_occupation": float(arr.max()),
    }


def evaluate_tolerances(
    difference: dict[str, Any], scf: dict[str, Any], numerics: dict[str, Any], tolerances: dict[str, Any], kind: str
) -> dict[str, Any]:
    """Judge ``candidate - reference`` and the candidate's SCF against a declared tolerance class."""
    limits = tolerances["classes"][kind]
    checks = {}
    if "energy_mev_atom" in limits:
        checks["energy_sigma0"] = (abs(difference["dE_sigma0_mev_atom"]), limits["energy_mev_atom"])
        if difference.get("dF_free_mev_atom") is not None:
            checks["free_energy"] = (abs(difference["dF_free_mev_atom"]), limits["energy_mev_atom"])
    for key, limit in (
        ("force_rmse_ev_a", "force_rmse_ev_a"),
        ("force_max_component_ev_a", "force_max_component_ev_a"),
        ("stress_max_abs_diff_gpa", "stress_component_gpa"),
    ):
        if limit in limits:
            checks[key] = (difference[key], limits[limit])
    gate = tolerances.get("scf_qualification", {})
    scf_checks = {
        "nelm_reached": scf.get("nelm_reached") is gate.get("nelm_reached", False),
        "residual_jumps": (scf.get("residual_jumps") or 0) <= gate.get("residual_jumps", 0),
        "energy_rise_after_settled": (scf.get("max_rise_after_settled_ev") or 0.0)
        <= gate.get("max_energy_rise_after_settled_ev", 1e-3),
        "energy_criterion_met": scf.get("energy_criterion_met") is gate.get("energy_criterion_met", True),
        "eddrmm_zhegv_warnings": numerics.get("eddrmm_zhegv_warnings", 0) <= gate.get("eddrmm_zhegv_warnings", 0),
    }
    passed = all(value <= limit for value, limit in checks.values()) and all(scf_checks.values())
    return {
        "class": kind,
        "tolerance_status": tolerances.get("status"),
        "labels": {k: {"value": v, "limit": lim, "ok": v <= lim} for k, (v, lim) in checks.items()},
        "scf": scf_checks,
        "verdict": "PASS" if passed else "HOLD",
    }


def run_compare(
    a: Path,
    b: Path,
    output: Path,
    *,
    labels: tuple[str, str] = ("a", "b"),
    tolerances: Path | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Second run ``b`` is the candidate, first run ``a`` the reference."""
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for label, run in zip(labels, (a, b), strict=True):
        result[label] = {
            "run": str(run),
            "inputs_sha256": _hashes(run, INPUT_FILES),
            "restart_files_present": {
                n: (run / n).is_file() and (run / n).stat().st_size > 0 for n in ("WAVECAR", "CHGCAR", "CHG")
            },
            "kpoints_text": (run / "KPOINTS").read_text(encoding="utf-8") if (run / "KPOINTS").is_file() else None,
            "ibzkpt_count": _ibzkpt_count(run / "IBZKPT"),
            "numerics": outcar_numerics(run / "OUTCAR"),
            "scf": scf_summary(run, _incar(run / "INCAR")),
            "trace": electronic_trace(run / "OSZICAR"),
            "incar": _incar(run / "INCAR"),
        }
    la, lb = (outcar_label(a / "OUTCAR"), outcar_label(b / "OUTCAR"))
    pa, pb = outcar_label_from_poscar(a / "POSCAR"), outcar_label_from_poscar(b / "POSCAR")
    result["poscar_geometry"] = geometry_difference(pa, pb)
    result["poscar_identical_bytes"] = (
        result[labels[0]]["inputs_sha256"]["POSCAR"] == result[labels[1]]["inputs_sha256"]["POSCAR"]
    )
    result["potcar_identical_bytes"] = (
        result[labels[0]]["inputs_sha256"]["POTCAR"] == result[labels[1]]["inputs_sha256"]["POTCAR"]
    )
    result[f"{labels[1]}_minus_{labels[0]}"] = label_difference(lb, la)
    result["incar_differences"] = {
        k: (result[labels[0]]["incar"].get(k), result[labels[1]]["incar"].get(k))
        for k in sorted(set(result[labels[0]]["incar"]) | set(result[labels[1]]["incar"]))
        if result[labels[0]]["incar"].get(k) != result[labels[1]]["incar"].get(k)
    }
    if tolerances is not None:
        import yaml

        result["evaluation"] = evaluate_tolerances(
            result[f"{labels[1]}_minus_{labels[0]}"],
            result[labels[1]]["scf"],
            result[labels[1]]["numerics"],
            yaml.safe_load(tolerances.read_text(encoding="utf-8")),
            kind or "convergence",
        )
    (output / "compare.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    with (output / "scf_traces.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["run", "algorithm", "iteration", "energy_ev", "de_ev", "deps_ev", "ncg", "rms", "rms_c"])
        for label in labels:
            for step in result[label]["trace"]:
                writer.writerow([label, *step.values()])
    return result


def _ibzkpt_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return int(lines[1].split()[0]) if len(lines) > 1 else None


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    intake = commands.add_parser("intake", help="Verify packages and bind canonical calculations")
    intake.add_argument("packages", nargs="+", type=Path)
    intake.add_argument("--output", type=Path, required=True)
    pairs = commands.add_parser("pairs", help="Package-21 orig/kconv/stored comparison")
    pairs.add_argument("package", type=Path)
    pairs.add_argument("--deck", type=Path, required=True, help="Local deck folder with {split}.extxyz")
    pairs.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Numerics and SCF traces of two static runs")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--labels", nargs=2, default=("a", "b"))
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--tolerances", type=Path, help="numerical_tolerances.yaml; judges the second run")
    compare.add_argument("--class", dest="kind", default="convergence", help="tolerance class (default: convergence)")
    args = parser.parse_args(argv)
    if args.command == "intake":
        summary = run_intake(args.packages, args.output)
        print(json.dumps(summary, indent=1))
        return 0 if all(not p["mismatched"] and not p["missing"] for p in summary["packages"].values()) else 2
    if args.command == "pairs":
        rows = run_pairs(args.package, args.deck, args.output)
        print(f"{len(rows)} pairs -> {args.output / 'pairs.tsv'}")
        return 0
    result = run_compare(
        args.a, args.b, args.output, labels=tuple(args.labels), tolerances=args.tolerances, kind=args.kind
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if not isinstance(v, dict) or "_minus_" in k or k == "evaluation"},
            indent=1,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
