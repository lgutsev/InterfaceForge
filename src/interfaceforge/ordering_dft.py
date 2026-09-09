# ruff: noqa: E501
"""DFT verification of MLIP-predicted chemical ordering at fixed composition.

``iface swap-mc export`` writes one POSCAR per candidate N/O arrangement. This
module turns that export into a VASP benchmark and closes the loop:

``dft-prepare``
    Build a run tree in which *every candidate differs only in its POSCAR*.
    INCAR (apart from ``SYSTEM``), KPOINTS and POTCAR are byte-identical across
    candidates, the cell and composition are verified identical, and the species
    blocks are canonicalized so one POTCAR is valid everywhere. Relative energies
    at fixed composition are only meaningful under exactly these conditions, so
    they are checked rather than assumed.
``dft-launch``
    Preflight (hashes unchanged, no pre-existing outputs) then submit.
``dft-collect``
    Read ``energy(sigma->0)`` from the last completed ionic step of each run with
    the same parsing ``iface audit`` uses.
``dft-compare``
    Join DFT against the MLIP energies at a common reference configuration and
    report whether the search's favourable arrangements stay favourable under
    DFT -- correlation, ranking agreement, and the searched-vs-random contrast
    that is the actual claim of the ordering study.

Two stages are supported. ``static`` (``IBRION=-1``, ``NSW=0``) scores the MLIP
geometries as they are: it isolates the energy ranking from any geometry
difference and is the cheap first test. ``relax`` (``IBRION=2``, ``ISIF=2``)
re-relaxes at fixed cell with the same frozen layers, which is the quantity the
MLIP search itself approximated. Run ``static`` first on the whole shortlist,
``relax`` on the subset that matters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adhesion import _collapse_blank_runs, _drop_orphaned_section_banners, _incar_tag
from .audit import audit_run
from .errors import DependencyError, SafetyError
from .state import sha256_file, utc_now
from .vasp import (
    _potcar_elements,
    assemble_potcar,
    resolve_launcher,
    resolve_potcar_root,
    submit_run,
)

SCHEMA_VERSION = 1
QUANTITY = "chemical_ordering_dft_benchmark"
STAGES = ("static", "relax")
MLIP_ENERGY_CHOICES = ("best", "refine", "screen")

#: Outputs whose presence means a run directory has already been used.
RUNTIME_MARKERS = ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR", "WAVECAR")

#: Tags that only mean something under ``IBRION=0``; neither stage runs MD.
_MD_ONLY_TAGS = {
    "TEBEG", "TEEND", "SMASS", "MDALGO",
    "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L", "PMASS", "ANDERSEN_PROB",
}

CAVEATS: tuple[str, ...] = (
    "Relative energies are only comparable when the cell, composition, k-mesh, POTCAR, "
    "and every electronic INCAR tag are identical across candidates. dft-prepare enforces "
    "this and records the hashes; do not edit a single run's INCAR afterwards.",
    "ISYM=0 is set for every run. Different N/O arrangements have different symmetry, and "
    "symmetry-reduced k-point sets would otherwise make their energies inconsistent at the "
    "meV level that ordering differences live at.",
    "The static stage scores MLIP geometries with DFT: it tests the energy ranking, not the "
    "geometries. A good static correlation with poor relaxed agreement means the MLIP is "
    "ranking arrangements correctly but relaxing them wrongly.",
    "Agreement on one composition does not transfer to another. Repeat a small DFT check at "
    "each oxygen content, especially after a search deliberately leaves the random-arrangement "
    "distribution the model was trained near.",
)

CITATION: dict[str, Any] = {
    "method": "DFT verification of MLIP chemical-ordering energies at fixed composition",
    "convention": "energy(sigma->0) of the last completed ionic step, total energy in eV",
}


# --------------------------------------------------------------------------- POSCAR


@dataclass(frozen=True)
class Poscar:
    """Enough of a POSCAR to compare, canonicalize, and rewrite one."""

    comment: str
    scale: str
    lattice: tuple[tuple[float, float, float], ...]
    species: tuple[str, ...]
    counts: tuple[int, ...]
    selective: bool
    coordinate_mode: str
    positions: tuple[tuple[str, str, str], ...]
    flags: tuple[tuple[str, str, str] | None, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        out: list[str] = []
        for symbol, count in zip(self.species, self.counts, strict=True):
            out.extend([symbol] * count)
        return tuple(out)

    @property
    def natoms(self) -> int:
        return sum(self.counts)

    def composition(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for symbol, count in zip(self.species, self.counts, strict=True):
            totals[symbol] = totals.get(symbol, 0) + count
        return totals


_ELEMENT = re.compile(r"[A-Z][a-z]?")


def _read_poscar(path: Path) -> Poscar:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 8:
        raise SafetyError(f"POSCAR is too short: {path}")
    comment = lines[0]
    scale = lines[1].strip()
    try:
        lattice = tuple(
            tuple(float(value) for value in lines[index].split()[:3]) for index in (2, 3, 4)
        )
    except (ValueError, IndexError) as exc:
        raise SafetyError(f"Could not read the lattice from {path}: {exc}") from exc
    if any(len(row) != 3 for row in lattice):
        raise SafetyError(f"Lattice rows are not three numbers each: {path}")
    tokens = lines[5].split()
    if tokens and all(re.fullmatch(r"\d+", token) for token in tokens):
        species = tuple(comment.split())  # legacy: species on the comment line
        counts_index = 5
    else:
        species = tuple(tokens)
        counts_index = 6
    if not species or any(not _ELEMENT.fullmatch(token) for token in species):
        raise SafetyError(f"{path} has no usable species line (VASP 5+ or legacy)")
    counts_tokens = lines[counts_index].split()
    if not counts_tokens or not all(re.fullmatch(r"\d+", token) for token in counts_tokens):
        raise SafetyError(f"{path} has no valid ion-count line")
    counts = tuple(int(token) for token in counts_tokens)
    if len(counts) != len(species):
        raise SafetyError(f"{path} has {len(species)} species but {len(counts)} counts")
    cursor = counts_index + 1
    selective = cursor < len(lines) and lines[cursor].strip().casefold().startswith("s")
    if selective:
        cursor += 1
    if cursor >= len(lines):
        raise SafetyError(f"{path} ends before its coordinate mode line")
    coordinate_mode = lines[cursor].strip()
    cursor += 1
    natoms = sum(counts)
    rows = lines[cursor : cursor + natoms]
    if len(rows) != natoms:
        raise SafetyError(f"{path} has {len(rows)} coordinate rows for {natoms} ions")
    positions: list[tuple[str, str, str]] = []
    flags: list[tuple[str, str, str] | None] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 3:
            raise SafetyError(f"Short coordinate row in {path}: {row!r}")
        positions.append((fields[0], fields[1], fields[2]))
        if selective:
            if len(fields) < 6:
                raise SafetyError(f"Selective-dynamics row without F/T flags in {path}: {row!r}")
            flags.append((fields[3], fields[4], fields[5]))
        else:
            flags.append(None)
    return Poscar(
        comment=comment,
        scale=scale,
        lattice=lattice,
        species=species,
        counts=counts,
        selective=selective,
        coordinate_mode=coordinate_mode,
        positions=tuple(positions),
        flags=tuple(flags),
    )


def _canonical_species_order(poscar: Poscar) -> tuple[str, ...]:
    """One deterministic species order shared by every candidate.

    Alphabetical, so the order does not depend on which arrangement happened to
    be written first, and so a POTCAR built for one candidate is valid for all.
    """
    return tuple(sorted(set(poscar.species)))


def _canonical_text(poscar: Poscar, species_order: Sequence[str], *, comment: str) -> str:
    """Rewrite a POSCAR with atoms grouped in ``species_order``.

    Reordering is stable within a species, and selective-dynamics flags travel
    with their atom, so the frozen-layer definition is preserved exactly.
    """

    symbols = poscar.symbols
    order = [index for symbol in species_order for index, value in enumerate(symbols) if value == symbol]
    if len(order) != poscar.natoms:
        missing = sorted(set(symbols) - set(species_order))
        raise SafetyError(f"species order {list(species_order)} does not cover {missing}")
    counts = [sum(1 for value in symbols if value == symbol) for symbol in species_order]
    out = [comment, poscar.scale]
    out.extend("  ".join(f"{value: .16f}" for value in row) for row in poscar.lattice)
    out.append("  ".join(f"{symbol:>4}" for symbol in species_order))
    out.append("  ".join(f"{value:>4d}" for value in counts))
    if poscar.selective:
        out.append("Selective dynamics")
    out.append(poscar.coordinate_mode)
    for index in order:
        x, y, z = poscar.positions[index]
        row = f"  {x:>20} {y:>20} {z:>20}"
        flag = poscar.flags[index]
        if flag is not None:
            row += "   " + " ".join(f"{value:>1}" for value in flag)
        out.append(row)
    return "\n".join(out) + "\n"


def _lattice_matches(a: Poscar, b: Poscar, tolerance: float) -> bool:
    if a.scale.strip() != b.scale.strip():
        return False
    return all(
        abs(x - y) <= tolerance
        for row_a, row_b in zip(a.lattice, b.lattice, strict=True)
        for x, y in zip(row_a, row_b, strict=True)
    )


def _frozen_signature(poscar: Poscar) -> tuple[tuple[str, int], ...]:
    """Frozen-atom count per species: the constraint definition, order-independent."""

    frozen: dict[str, int] = {}
    for symbol, flag in zip(poscar.symbols, poscar.flags, strict=True):
        if flag is not None and all(value.upper().startswith("F") for value in flag):
            frozen[symbol] = frozen.get(symbol, 0) + 1
    return tuple(sorted(frozen.items()))


# ----------------------------------------------------------------------------- INCAR


def _render_incar(base: str, overrides: Sequence[tuple[str, str]], *, remove: Iterable[str] = ()) -> str:
    """Keep the reference INCAR verbatim except for the tags this workflow owns."""

    changed = {tag.upper() for tag, _ in overrides} | {tag.upper() for tag in remove}
    kept = []
    for line in base.splitlines():
        tag = _incar_tag(line)
        if tag in changed or (tag is not None and tag.startswith("ML_")):
            continue
        kept.append(line)
    kept = _collapse_blank_runs(_drop_orphaned_section_banners(kept))
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend(["", "# Set by InterfaceForge for the fixed-composition ordering benchmark"])
    kept.extend(f"{tag:<15} = {value}" for tag, value in overrides)
    return "\n".join(kept) + "\n"


def _stage_overrides(stage: str, *, label: str, ediff: str, ediffg: str, nsw: int) -> list[tuple[str, str]]:
    common = [("ISIF", "2"), ("ISYM", "0"), ("EDIFF", ediff),
              ("LWAVE", ".FALSE."), ("LCHARG", ".FALSE."), ("LVHAR", ".FALSE.")]
    if stage == "static":
        # POTIM/EDIFFG are ionic-step settings; a single point has no ionic step,
        # and an inherited AIMD POTIM left in place only misleads the next reader.
        return [("SYSTEM", f"ordering static {label}"), ("IBRION", "-1"), ("NSW", "0"), *common]
    return [
        ("SYSTEM", f"ordering relax {label}"),
        ("IBRION", "2"), ("NSW", str(nsw)), ("POTIM", "0.20"), ("EDIFFG", ediffg),
        *common,
    ]


def _incar_body_hash(text: str) -> str:
    """Hash an INCAR ignoring SYSTEM, the one tag allowed to differ per candidate."""

    body = [line for line in text.splitlines() if _incar_tag(line) != "SYSTEM"]
    return hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------- discovery


def _load_export(export_dir: Path) -> list[dict[str, Any]]:
    """Candidate rows from an ``iface swap-mc export`` tree (or a plain POSCAR tree)."""

    manifest_path = export_dir / "manifest.json"
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("candidates", []):
            directory = Path(entry["directory"])
            if not directory.is_absolute():
                directory = export_dir / directory
            if not directory.is_dir():  # tree moved since export; fall back to the id
                directory = export_dir / entry["cand_id"]
            rows.append({"cand_id": entry["cand_id"], "directory": directory})
    else:
        for directory in sorted(path for path in export_dir.iterdir() if path.is_dir()):
            if (directory / "POSCAR").is_file():
                rows.append({"cand_id": directory.name, "directory": directory})
    if not rows:
        raise SafetyError(f"No exported candidates found in {export_dir}")

    for row in rows:
        poscar = row["directory"] / "POSCAR"
        if not poscar.is_file():
            raise SafetyError(f"{row['cand_id']} has no POSCAR at {poscar}")
        row["poscar"] = poscar
        ordering = row["directory"] / "ordering.json"
        row["ordering"] = json.loads(ordering.read_text(encoding="utf-8")) if ordering.is_file() else {}
    return rows


def _group_of(ordering: Mapping[str, Any]) -> str:
    """Classify a candidate for the searched-vs-random contrast."""

    roles = {str(value) for value in (ordering.get("roles") or [])}
    if "random-baseline" in roles:
        return "random"
    if "initial" in roles:
        return "initial"
    role = str(ordering.get("role") or "")
    if role == "baseline":
        return "baseline"
    return "searched"


def _mlip_energy(ordering: Mapping[str, Any], preference: str) -> float | None:
    refine = ordering.get("energy_refine_ev")
    screen = ordering.get("energy_screen_ev")
    best = ordering.get("energy_ev")
    if preference == "refine":
        return refine
    if preference == "screen":
        return screen if screen is not None else best
    return best if best is not None else refine


# --------------------------------------------------------------------------- prepare


def prepare_ordering_dft(
    export_dir: str | Path,
    output_dir: str | Path,
    *,
    reference: str | Path,
    stages: Sequence[str] = ("static",),
    incar: str | None = None,
    kpoints: str | None = None,
    potcar: str | None = None,
    launcher: str | None = None,
    propagate_launcher: bool = True,
    potcar_root: str | Path | None = None,
    potcar_mapping: str | Path | None = None,
    ediff: str = "1E-6",
    ediffg: str = "-0.02",
    nsw: int = 99,
    lattice_tolerance: float = 1e-4,
    mlip_energy: str = "best",
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build a VASP tree in which candidates differ only in their POSCAR.

    ``reference`` is an existing converged run for the same interface: its
    INCAR supplies every electronic setting (ENCUT, PREC, ISPIN, LDAU, smearing,
    parallelization), its KPOINTS is copied verbatim, and its POTCAR is reused
    when the species order matches. Nothing in the reference directory is
    modified. This never launches VASP.
    """

    emit = progress or (lambda _message: None)
    bad_stages = [stage for stage in stages if stage not in STAGES]
    if bad_stages:
        raise SafetyError(f"Unknown stage(s) {bad_stages}; choose from {list(STAGES)}")
    if not stages:
        raise SafetyError("At least one stage is required")
    if mlip_energy not in MLIP_ENERGY_CHOICES:
        raise SafetyError(f"--mlip-energy must be one of {list(MLIP_ENERGY_CHOICES)}")

    export = Path(export_dir).expanduser().resolve()
    if not export.is_dir():
        raise NotADirectoryError(export)
    source = Path(reference).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise SafetyError(f"{output} already exists and is not empty; refusing to overwrite prepared runs")

    incar_path = (source / (incar or "INCAR")).resolve()
    kpoints_path = (source / (kpoints or "KPOINTS")).resolve()
    for path, label in ((incar_path, "INCAR"), (kpoints_path, "KPOINTS")):
        if not path.is_file():
            raise FileNotFoundError(f"reference {label} not found: {path}")
    base_incar = incar_path.read_text(encoding="utf-8")

    launcher_path: Path | None = None
    if propagate_launcher:
        if launcher:
            launcher_path = (source / launcher).resolve()
            if not launcher_path.is_file():
                raise FileNotFoundError(f"--launcher not found: {launcher_path}")
        else:
            for name in ("runvasp.sh", "run.slurm"):
                if (source / name).is_file():
                    launcher_path = (source / name).resolve()
                    break

    rows = _load_export(export)
    emit(f"read {len(rows)} exported candidate(s) from {export}")
    for row in rows:
        row["poscar_parsed"] = _read_poscar(row["poscar"])

    reference_row = rows[0]
    reference_poscar: Poscar = reference_row["poscar_parsed"]
    species_order = _canonical_species_order(reference_poscar)
    problems: list[str] = []
    for row in rows[1:]:
        other: Poscar = row["poscar_parsed"]
        if other.composition() != reference_poscar.composition():
            problems.append(
                f"{row['cand_id']}: composition {other.composition()} differs from "
                f"{reference_row['cand_id']} {reference_poscar.composition()}"
            )
        if not _lattice_matches(reference_poscar, other, lattice_tolerance):
            problems.append(f"{row['cand_id']}: cell differs from {reference_row['cand_id']} beyond {lattice_tolerance} A")
        if other.selective != reference_poscar.selective:
            problems.append(f"{row['cand_id']}: selective dynamics {other.selective}, expected {reference_poscar.selective}")
        elif _frozen_signature(other) != _frozen_signature(reference_poscar):
            problems.append(
                f"{row['cand_id']}: frozen atoms per species {_frozen_signature(other)} differ from "
                f"{_frozen_signature(reference_poscar)}"
            )
    if problems:
        raise SafetyError(
            "Exported candidates are not comparable at fixed composition; relative energies "
            "would be meaningless:\n  " + "\n  ".join(problems)
        )

    warnings: list[str] = []
    upper_incar = {
        (_incar_tag(line) or ""): line.split("=", 1)[1].strip()
        for line in base_incar.splitlines()
        if _incar_tag(line)
    }
    for tag, why in (
        ("ENCUT", "the plane-wave cutoff would then be taken from the POTCAR default"),
        ("PREC", "the precision setting would fall back to VASP's default"),
    ):
        if tag not in upper_incar:
            warnings.append(f"reference INCAR sets no {tag}; {why}. Set it explicitly before comparing energies.")
    if not reference_poscar.selective:
        warnings.append(
            "Exported POSCARs carry no selective dynamics: every atom will relax in the relax stage, "
            "which does not match a frozen-substrate MLIP search."
        )

    output.mkdir(parents=True, exist_ok=True)
    shared = output / "shared"
    shared.mkdir(exist_ok=True)
    shutil.copy2(kpoints_path, shared / "KPOINTS")

    potcar_target = shared / "POTCAR"
    potcar_source = (source / (potcar or "POTCAR")).resolve()
    potcar_provenance: dict[str, Any]
    if potcar_source.is_file() and potcar_source.stat().st_size:
        elements = _potcar_elements(potcar_source)
        if tuple(elements) == species_order:
            shutil.copy2(potcar_source, potcar_target)
            potcar_provenance = {"origin": "reference", "path": str(potcar_source), "elements": elements}
        else:
            root = resolve_potcar_root(potcar_root)
            built = assemble_potcar(
                rows[0]["directory"] / "POSCAR", potcar_target,
                pseudopotential_root=root, mapping_file=potcar_mapping, force=True,
            )
            warnings.append(
                f"reference POTCAR order {elements} does not match the canonical species order "
                f"{list(species_order)}; a matching POTCAR was assembled from {root}."
            )
            potcar_provenance = {"origin": "assembled", **built}
    else:
        root = resolve_potcar_root(potcar_root)
        built = assemble_potcar(
            rows[0]["directory"] / "POSCAR", potcar_target,
            pseudopotential_root=root, mapping_file=potcar_mapping, force=True,
        )
        potcar_provenance = {"origin": "assembled", **built}

    # The POTCAR was assembled for the *canonical* order; rewriting each POSCAR
    # into that order is what makes one POTCAR valid for every candidate.
    potcar_elements = _potcar_elements(potcar_target)
    if tuple(potcar_elements) != species_order:
        raise SafetyError(
            f"POTCAR species order {potcar_elements} does not match the canonical POSCAR order "
            f"{list(species_order)}; every run would use the wrong pseudopotentials"
        )

    candidates: list[dict[str, Any]] = []
    incar_hashes: dict[str, set[str]] = {stage: set() for stage in stages}
    for row in rows:
        ordering = row["ordering"]
        entry: dict[str, Any] = {
            "cand_id": row["cand_id"],
            "role": ordering.get("role"),
            "roles": ordering.get("roles"),
            "group": _group_of(ordering),
            "n_substituent": ordering.get("n_substituent"),
            "substituent_sites": ordering.get("substituent_sites"),
            "mlip_energy_ev": _mlip_energy(ordering, mlip_energy),
            "mlip_energy_screen_ev": ordering.get("energy_screen_ev"),
            "mlip_energy_refine_ev": ordering.get("energy_refine_ev"),
            "force_std_ev_ang": ordering.get("force_std_ev_ang"),
            "source_poscar": str(row["poscar"]),
            "runs": {},
        }
        for stage in stages:
            run = output / stage / row["cand_id"]
            run.mkdir(parents=True, exist_ok=True)
            text = _canonical_text(
                row["poscar_parsed"], species_order,
                comment=f"{row['cand_id']} {stage} ordering benchmark",
            )
            (run / "POSCAR").write_text(text, encoding="utf-8")
            drop = set(_MD_ONLY_TAGS)
            if stage == "static":
                drop |= {"POTIM", "EDIFFG"}
            incar_text = _render_incar(
                base_incar,
                _stage_overrides(stage, label=row["cand_id"], ediff=ediff, ediffg=ediffg, nsw=nsw),
                remove=drop,
            )
            (run / "INCAR").write_text(incar_text, encoding="utf-8")
            incar_hashes[stage].add(_incar_body_hash(incar_text))
            _place(shared / "KPOINTS", run / "KPOINTS")
            _place(potcar_target, run / "POTCAR")
            if launcher_path is not None:
                shutil.copy2(launcher_path, run / launcher_path.name)
            names = ["INCAR", "POSCAR", "KPOINTS", "POTCAR"]
            if launcher_path is not None:
                names.append(launcher_path.name)
            entry["runs"][stage] = {
                "relative_path": f"{stage}/{row['cand_id']}",
                "directory": str(run),
                "launcher": launcher_path.name if launcher_path else None,
                "sha256": {name: sha256_file(run / name) for name in names},
            }
            emit(f"prepared {stage}/{row['cand_id']}")
        candidates.append(entry)

    for stage, hashes in incar_hashes.items():
        if len(hashes) != 1:
            raise SafetyError(f"stage {stage} produced non-identical INCARs; energies would not be comparable")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quantity": QUANTITY,
        "created_at": utc_now(),
        "export_dir": str(export),
        "reference_dir": str(source),
        "stages": list(stages),
        "n_candidates": len(candidates),
        "mlip_energy_preference": mlip_energy,
        "settings": {"ediff": ediff, "ediffg": ediffg, "nsw": nsw, "isym": 0, "isif": 2},
        "shared_inputs": {
            "species_order": list(species_order),
            "n_ions": reference_poscar.natoms,
            "composition": reference_poscar.composition(),
            "selective_dynamics": reference_poscar.selective,
            "frozen_per_species": dict(_frozen_signature(reference_poscar)),
            "lattice": [list(row) for row in reference_poscar.lattice],
            "kpoints_sha256": sha256_file(shared / "KPOINTS"),
            "potcar_sha256": sha256_file(potcar_target),
            "incar_body_sha256": {stage: sorted(hashes)[0] for stage, hashes in incar_hashes.items()},
            "source_incar": str(incar_path),
            "potcar": potcar_provenance,
        },
        "launcher": launcher_path.name if launcher_path else None,
        "candidates": candidates,
        "warnings": warnings,
        "caveats": list(CAVEATS),
        "citation": CITATION,
        "next_steps": [
            "iface swap-mc dft-launch OUTPUT --stage static   (dry run; add --execute to submit)",
            "iface swap-mc dft-collect OUTPUT --stage static",
            "iface swap-mc dft-compare OUTPUT --stage static",
        ],
    }
    (output / "ordering_dft_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "manifest": str(output / "ordering_dft_manifest.json"),
        "stages": list(stages),
        "n_candidates": len(candidates),
        "runs": len(candidates) * len(stages),
        "warnings": warnings,
        "submission": "not performed; run iface swap-mc dft-launch after review",
    }


def _place(source: Path, target: Path) -> None:
    """Hard-link a shared input, falling back to a copy across filesystems."""

    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


# ---------------------------------------------------------------------------- launch


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "ordering_dft_manifest.json"
    if not path.is_file():
        raise SafetyError(f"No ordering_dft_manifest.json in {root}; run 'iface swap-mc dft-prepare' first")
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_runs(root: Path, manifest: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in manifest.get("stages", []):
        raise SafetyError(f"{root} has no prepared '{stage}' stage (prepared: {manifest.get('stages')})")
    runs = []
    for entry in manifest["candidates"]:
        run = entry["runs"].get(stage)
        if run is None:
            continue
        directory = (root / run["relative_path"]).resolve()
        if not directory.is_relative_to(root):
            raise SafetyError(f"Unsafe run path outside the tree: {directory}")
        runs.append({**run, "directory": directory, "cand_id": entry["cand_id"], "group": entry.get("group"),
                     "role": entry.get("role"), "mlip_energy_ev": entry.get("mlip_energy_ev")})
    if not runs:
        raise SafetyError(f"No {stage} runs recorded in {root}")
    return runs


def launch_ordering_dft(
    root: str | Path,
    *,
    stage: str = "static",
    execute: bool = False,
    launcher: str | None = None,
    limit: int | None = None,
    potcar_root: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Preflight every prepared run, then optionally submit them.

    A run whose inputs changed since preparation, or that already holds VASP
    output, is never submitted: the first would break the identical-inputs
    guarantee the comparison rests on, the second would be a duplicate.
    """

    emit = progress or (lambda _message: None)
    root_path = Path(root).expanduser().resolve()
    manifest = _load_manifest(root_path)
    runs = _stage_runs(root_path, manifest, stage)

    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in runs:
        directory: Path = item["directory"]
        if not directory.is_dir():
            raise SafetyError(f"prepared run directory is missing: {directory}")
        for name, expected in item["sha256"].items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                raise SafetyError(
                    f"{path} changed since dft-prepare; every candidate must share identical "
                    "inputs. Re-run dft-prepare instead of editing a single run."
                )
        existing = [name for name in RUNTIME_MARKERS if (directory / name).exists()]
        if existing:
            skipped.append({**_row(item), "status": "SKIPPED_EXISTING_OUTPUT", "detail": ", ".join(existing)})
            emit(f"skip {item['relative_path']}: already holds {', '.join(existing)}")
            continue
        script = resolve_launcher(directory, launcher or item.get("launcher"))
        if script.name not in item["sha256"]:
            raise SafetyError(f"launcher {script.name} was not part of the prepared inputs for {item['relative_path']}")
        emit(f"preflight OK: {item['relative_path']} (launcher={script.name})")
        planned.append({**item, "launcher_name": script.name})

    if limit is not None:
        if limit < 1:
            raise SafetyError("--limit must be positive")
        planned = planned[:limit]
    if not planned:
        return {
            "mode": "nothing-to-do",
            "root": str(root_path), "stage": stage,
            "runs": 0, "skipped": skipped,
            "submission": "not performed; every run already holds output",
        }
    if not execute:
        return {
            "mode": "dry-run",
            "root": str(root_path), "stage": stage,
            "runs": len(planned), "preflight": "PASS",
            "planned": [_row(item) for item in planned],
            "skipped": skipped,
            "submission": "not performed; pass --execute after review",
        }

    rows: list[dict[str, Any]] = list(skipped)
    failure: str | None = None
    for index, item in enumerate(planned, start=1):
        emit(f"[{index}/{len(planned)}] sbatch {item['relative_path']}")
        try:
            job_id = submit_run(item["directory"], item["launcher_name"], potcar_root=potcar_root)
            rows.append({**_row(item), "status": "SUBMITTED", "job_id": job_id, "detail": ""})
        except Exception as exc:  # a scheduler failure must stop the batch, not continue blindly
            failure = f"{item['relative_path']}: {exc}"
            rows.append({**_row(item), "status": "FAILED", "job_id": "", "detail": str(exc)})
            break

    payload = {
        "format": "interfaceforge-ordering-dft-launch",
        "schema_version": SCHEMA_VERSION,
        "status": "FAILED" if failure else "SUBMITTED",
        "root": str(root_path),
        "stage": stage,
        "runs": rows,
    }
    json_path = root_path / f"ordering_dft_launch_{stage}.json"
    tsv_path = root_path / f"ordering_dft_launch_{stage}.tsv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_delimited(tsv_path, rows, ("status", "job_id", "cand_id", "relative_path", "detail"), delimiter="\t")
    if failure is not None:
        raise SafetyError(f"Ordering DFT launch stopped after a submission failure ({failure}); review {json_path}")
    return {
        "mode": "submitted",
        "root": str(root_path), "stage": stage,
        "submitted": sum(1 for row in rows if row["status"] == "SUBMITTED"),
        "skipped": len(skipped),
        "reports": [str(json_path), str(tsv_path)],
        "jobs": rows,
    }


def _row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cand_id": item["cand_id"],
        "relative_path": item["relative_path"],
        "directory": str(item["directory"]),
        "launcher": item.get("launcher_name") or item.get("launcher"),
    }


def _write_delimited(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, delimiter: str = ",") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


# --------------------------------------------------------------------------- collect


def collect_ordering_dft(
    root: str | Path,
    *,
    stage: str = "static",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Read each run's converged energy with the same parsing ``iface audit`` uses.

    The energy is ``energy(sigma->0)`` of the last completed ionic step, the
    convention used throughout this project. Unfinished runs are reported, not
    silently dropped.
    """

    emit = progress or (lambda _message: None)
    root_path = Path(root).expanduser().resolve()
    manifest = _load_manifest(root_path)
    runs = _stage_runs(root_path, manifest, stage)

    rows: list[dict[str, Any]] = []
    for item in runs:
        directory: Path = item["directory"]
        audited = audit_run(directory, directory) if (directory / "OUTCAR").is_file() else {}
        energy = audited.get("sigma0_energy_ev_last")
        row = {
            "cand_id": item["cand_id"],
            "relative_path": item["relative_path"],
            "group": item.get("group"),
            "role": item.get("role"),
            "mlip_energy_ev": item.get("mlip_energy_ev"),
            "dft_energy_ev": energy,
            "finished_normally": bool(audited.get("finished_normally")),
            "opt_converged": audited.get("opt_converged"),
            "ionic_steps": audited.get("ionic_steps"),
            "max_force_ev_a_last": audited.get("max_force_ev_a_last"),
            "health": audited.get("health"),
            "warnings": audited.get("warnings"),
            "usable": bool(audited.get("finished_normally")) and energy is not None,
        }
        if stage == "relax" and row["usable"] and audited.get("opt_converged") is False:
            row["usable"] = False
            row["warnings"] = (row["warnings"] or "") + "; ionic relaxation did not reach EDIFFG"
        rows.append(row)
        emit(f"{item['relative_path']}: {'usable' if row['usable'] else 'not usable'}")

    rows.sort(key=lambda row: (row["dft_energy_ev"] is None, row["dft_energy_ev"] or 0.0))
    usable = [row for row in rows if row["usable"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "quantity": f"{QUANTITY}_energies",
        "root": str(root_path),
        "stage": stage,
        "collected_at": utc_now(),
        "n_runs": len(rows),
        "n_usable": len(usable),
        "energy_convention": "energy(sigma->0), last completed ionic step, eV",
        "runs": rows,
        "caveats": list(CAVEATS),
    }
    stage_dir = root_path / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "ordering_dft_energies.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_delimited(
        stage_dir / "ordering_dft_energies.csv", rows,
        ("cand_id", "group", "role", "dft_energy_ev", "mlip_energy_ev", "finished_normally",
         "opt_converged", "ionic_steps", "max_force_ev_a_last", "usable"),
    )
    payload["outputs"] = {
        "json": str(stage_dir / "ordering_dft_energies.json"),
        "csv": str(stage_dir / "ordering_dft_energies.csv"),
    }
    return payload


# --------------------------------------------------------------------------- compare


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((value - mx) ** 2 for value in x)
    syy = sum((value - my) ** 2 for value in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    return _pearson(_ranks(x), _ranks(y))


def _group_stats(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def compare_ordering(
    root: str | Path,
    *,
    stage: str = "static",
    reference_cand: str | None = None,
    per_atom: bool = False,
    figure: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compare DFT and MLIP ordering energies against a common reference.

    At fixed composition both methods are compared through
    ``dE_i = E_i - E_reference`` evaluated on the *same* configuration, so the
    arbitrary total-energy offset between DFT and an MLIP cancels exactly and no
    fitted shift is applied. The headline number is whether DFT reproduces the
    energy gain the search claims over random arrangements.
    """

    emit = progress or (lambda _message: None)
    root_path = Path(root).expanduser().resolve()
    manifest = _load_manifest(root_path)
    energies = collect_ordering_dft(root_path, stage=stage, progress=progress)
    rows = [row for row in energies["runs"] if row["usable"] and row["mlip_energy_ev"] is not None]
    unusable = [row["cand_id"] for row in energies["runs"] if not row["usable"]]
    missing_mlip = [row["cand_id"] for row in energies["runs"] if row["usable"] and row["mlip_energy_ev"] is None]
    if len(rows) < 2:
        raise SafetyError(
            f"only {len(rows)} candidate(s) in {root_path}/{stage} have both a finished DFT energy "
            "and an MLIP energy; nothing to compare yet"
        )

    natoms = int(manifest["shared_inputs"]["n_ions"])
    if reference_cand is not None:
        chosen = next((row for row in rows if row["cand_id"] == reference_cand), None)
        if chosen is None:
            raise SafetyError(f"reference candidate {reference_cand!r} is not among the usable rows")
        policy = "explicit"
    else:
        chosen = next((row for row in rows if row["group"] == "initial"), None)
        policy = "initial-arrangement"
        if chosen is None:
            randoms = [row for row in rows if row["group"] == "random"]
            if randoms:
                chosen = min(randoms, key=lambda row: row["mlip_energy_ev"])
                policy = "lowest-MLIP random baseline"
            else:
                chosen = max(rows, key=lambda row: row["dft_energy_ev"])
                policy = "highest-DFT candidate (no baseline in the export)"
    emit(f"reference configuration: {chosen['cand_id']} ({policy})")

    scale = 1000.0 / natoms if per_atom else 1.0
    unit = "meV/atom" if per_atom else "eV"
    compared: list[dict[str, Any]] = []
    for row in rows:
        d_dft = (row["dft_energy_ev"] - chosen["dft_energy_ev"]) * scale
        d_mlip = (row["mlip_energy_ev"] - chosen["mlip_energy_ev"]) * scale
        compared.append({
            "cand_id": row["cand_id"],
            "group": row["group"],
            "role": row["role"],
            "dft_energy_ev": row["dft_energy_ev"],
            "mlip_energy_ev": row["mlip_energy_ev"],
            "delta_dft": d_dft,
            "delta_mlip": d_mlip,
            "residual": d_mlip - d_dft,
        })
    compared.sort(key=lambda row: row["delta_dft"])

    dft_values = [row["delta_dft"] for row in compared]
    mlip_values = [row["delta_mlip"] for row in compared]
    residuals = [row["residual"] for row in compared]
    dft_ranks = _ranks(dft_values)
    n = len(compared)
    concordant = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (dft_values[i] - dft_values[j]) * (mlip_values[i] - mlip_values[j]) > 0
    )
    pairs = n * (n - 1) // 2

    mlip_best_index = min(range(n), key=lambda index: compared[index]["delta_mlip"])
    dft_best_index = min(range(n), key=lambda index: compared[index]["delta_dft"])
    mlip_best, dft_best = compared[mlip_best_index], compared[dft_best_index]
    mlip_best_dft_rank = int(dft_ranks[mlip_best_index])

    group_table = {
        name: {
            "dft": _group_stats([row for row in compared if row["group"] == name], "delta_dft"),
            "mlip": _group_stats([row for row in compared if row["group"] == name], "delta_mlip"),
        }
        for name in sorted({row["group"] for row in compared})
    }
    searched = [row for row in compared if row["group"] == "searched"]
    random_rows = [row for row in compared if row["group"] == "random"]
    verdict: dict[str, Any] = {"testable": bool(searched and random_rows)}
    if searched and random_rows:
        dft_gain = min(row["delta_dft"] for row in searched) - (
            sum(row["delta_dft"] for row in random_rows) / len(random_rows)
        )
        mlip_gain = min(row["delta_mlip"] for row in searched) - (
            sum(row["delta_mlip"] for row in random_rows) / len(random_rows)
        )
        verdict.update({
            "definition": "best searched arrangement minus the mean random arrangement, same reference",
            "dft_gain": dft_gain,
            "mlip_gain": mlip_gain,
            "unit": unit,
            "sign_agrees": (dft_gain < 0) == (mlip_gain < 0),
            "dft_confirms_stabilization": dft_gain < 0,
            "overestimate": mlip_gain - dft_gain,
            "n_searched": len(searched),
            "n_random": len(random_rows),
        })

    statistics = {
        "n_compared": n,
        "unit": unit,
        "reference_cand_id": chosen["cand_id"],
        "reference_policy": policy,
        "pearson_r": _pearson(dft_values, mlip_values),
        "spearman_rho": _spearman(dft_values, mlip_values),
        "pairwise_order_agreement": concordant / pairs if pairs else None,
        "mae": sum(abs(value) for value in residuals) / n,
        "rmse": math.sqrt(sum(value * value for value in residuals) / n),
        "max_abs_residual": max(abs(value) for value in residuals),
        "mean_signed_residual": sum(residuals) / n,
        "dft_spread": max(dft_values) - min(dft_values),
        "mlip_spread": max(mlip_values) - min(mlip_values),
        "mlip_best_cand_id": mlip_best["cand_id"],
        "dft_best_cand_id": dft_best["cand_id"],
        "mlip_best_is_dft_best": mlip_best["cand_id"] == dft_best["cand_id"],
        "mlip_best_dft_rank": mlip_best_dft_rank,
        "mlip_best_dft_penalty": mlip_best["delta_dft"] - dft_best["delta_dft"],
    }

    interpretation: list[str] = []
    if statistics["max_abs_residual"] > 0.5 * statistics["dft_spread"] and statistics["dft_spread"] > 0:
        interpretation.append(
            "The largest MLIP error is comparable to the whole DFT spread: the ranking is not "
            "resolved by this model. Add these configurations to training before drawing conclusions."
        )
    if statistics["spearman_rho"] is not None and statistics["spearman_rho"] < 0.5:
        interpretation.append(
            "Rank correlation below 0.5: the MLIP does not order these arrangements as DFT does."
        )
    if not statistics["mlip_best_is_dft_best"]:
        interpretation.append(
            f"The MLIP's best arrangement is rank {mlip_best_dft_rank} under DFT, "
            f"{statistics['mlip_best_dft_penalty']:+.3f} {unit} above the DFT best."
        )
    if verdict.get("testable") and not verdict.get("sign_agrees", True):
        interpretation.append(
            "DFT does not reproduce the sign of the searched-vs-random energy gain: the search is "
            "most likely exploiting model error. Retrain on these configurations and repeat."
        )
    if not interpretation:
        interpretation.append(
            "DFT reproduces the MLIP ordering on this shortlist within the reported statistics."
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "quantity": f"{QUANTITY}_comparison",
        "root": str(root_path),
        "stage": stage,
        "compared_at": utc_now(),
        "per_atom": per_atom,
        "n_ions": natoms,
        "mlip_energy_preference": manifest.get("mlip_energy_preference"),
        "statistics": statistics,
        "searched_vs_random": verdict,
        "groups": group_table,
        "candidates": compared,
        "excluded_unusable_dft": unusable,
        "excluded_missing_mlip_energy": missing_mlip,
        "interpretation": interpretation,
        "caveats": list(CAVEATS),
        "citation": CITATION,
    }

    stage_dir = root_path / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    json_path = stage_dir / "ordering_comparison.json"
    csv_path = stage_dir / "ordering_comparison.csv"
    md_path = stage_dir / "ordering_comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _write_delimited(
        csv_path, compared,
        ("cand_id", "group", "role", "dft_energy_ev", "mlip_energy_ev", "delta_dft", "delta_mlip", "residual"),
    )
    md_path.write_text(_comparison_markdown(payload), encoding="utf-8")
    outputs = {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
    if figure:
        try:
            outputs["figure"] = _write_comparison_figure(payload, stage_dir / "ordering_comparison.png")
        except DependencyError as exc:
            payload.setdefault("warnings", []).append(str(exc))
    payload["outputs"] = outputs
    return payload


def _comparison_markdown(payload: Mapping[str, Any]) -> str:
    stats = payload["statistics"]
    unit = stats["unit"]
    lines = [
        f"# DFT vs MLIP chemical ordering ({payload['stage']} stage)",
        "",
        f"Reference configuration `{stats['reference_cand_id']}` ({stats['reference_policy']}); "
        f"every energy below is relative to it, in {unit}.",
        "",
        "| Statistic | Value |",
        "|---|---|",
    ]

    def fmt(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    for label, key in (
        ("Candidates compared", "n_compared"),
        ("Pearson r", "pearson_r"),
        ("Spearman rho", "spearman_rho"),
        ("Pairwise order agreement", "pairwise_order_agreement"),
        (f"MAE ({unit})", "mae"),
        (f"RMSE ({unit})", "rmse"),
        (f"Max |residual| ({unit})", "max_abs_residual"),
        (f"DFT spread ({unit})", "dft_spread"),
        (f"MLIP spread ({unit})", "mlip_spread"),
        ("MLIP best is DFT best", "mlip_best_is_dft_best"),
        ("DFT rank of the MLIP best", "mlip_best_dft_rank"),
    ):
        lines.append(f"| {label} | {fmt(stats.get(key))} |")

    verdict = payload["searched_vs_random"]
    lines.extend(["", "## Searched vs random", ""])
    if verdict.get("testable"):
        lines.extend([
            f"- DFT gain: **{verdict['dft_gain']:.4f} {unit}** "
            f"({verdict['n_searched']} searched, {verdict['n_random']} random)",
            f"- MLIP gain: {verdict['mlip_gain']:.4f} {unit}",
            f"- MLIP overestimates the gain by {verdict['overestimate']:.4f} {unit}",
            f"- Sign agrees: {verdict['sign_agrees']}",
        ])
    else:
        lines.append("- Not testable: the export has no searched and random arrangements to contrast.")

    lines.extend(["", "## Candidates", "",
                  f"| Candidate | Group | dE DFT ({unit}) | dE MLIP ({unit}) | Residual ({unit}) |",
                  "|---|---|---|---|---|"])
    for row in payload["candidates"]:
        lines.append(
            f"| {row['cand_id']} | {row['group']} | {row['delta_dft']:.4f} | "
            f"{row['delta_mlip']:.4f} | {row['residual']:+.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in payload["interpretation"])
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in payload["caveats"])
    return "\n".join(lines) + "\n"


_GROUP_STYLE = {
    "searched": ("#0072B2", "o", "searched"),
    "random": ("#D55E00", "s", "random baseline"),
    "initial": ("#009E73", "D", "initial"),
    "baseline": ("#CC79A7", "^", "baseline"),
}


def _write_comparison_figure(payload: Mapping[str, Any], path: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "The ordering comparison figure requires matplotlib; install InterfaceForge with interfaceforge[report]"
        ) from exc

    rows = payload["candidates"]
    unit = payload["statistics"]["unit"]
    with plt.rc_context({"font.size": 8.0, "axes.linewidth": 0.7, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(figsize=(3.4, 3.2), dpi=300)
        limits = [
            min(min(row["delta_dft"] for row in rows), min(row["delta_mlip"] for row in rows)),
            max(max(row["delta_dft"] for row in rows), max(row["delta_mlip"] for row in rows)),
        ]
        pad = 0.05 * (limits[1] - limits[0] or 1.0)
        axes.plot([limits[0] - pad, limits[1] + pad], [limits[0] - pad, limits[1] + pad],
                  color="#666666", linewidth=0.7, zorder=1)
        for group, (color, marker, label) in _GROUP_STYLE.items():
            subset = [row for row in rows if row["group"] == group]
            if not subset:
                continue
            axes.scatter([row["delta_dft"] for row in subset], [row["delta_mlip"] for row in subset],
                         s=22, c=color, marker=marker, label=label, zorder=2, edgecolors="none")
        axes.set_xlabel(f"DFT dE relative to reference ({unit})")
        axes.set_ylabel(f"MLIP dE relative to reference ({unit})")
        axes.set_title(f"Chemical ordering, {payload['stage']} stage", fontsize=9.0)
        axes.legend(frameon=False, fontsize=7.0, loc="best")
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
    return str(path)


# ------------------------------------------------------------------------- CLI glue


def cmd_dft_prepare(args: argparse.Namespace) -> int:
    print(json.dumps(
        prepare_ordering_dft(
            args.export_dir, args.output_dir,
            reference=args.reference,
            stages=tuple(dict.fromkeys(args.stage or ["static"])),
            incar=args.incar, kpoints=args.kpoints, potcar=args.potcar,
            launcher=args.launcher, propagate_launcher=not args.no_launcher,
            potcar_root=args.potcar_root, potcar_mapping=args.potcar_mapping,
            ediff=args.ediff, ediffg=args.ediffg, nsw=args.nsw,
            mlip_energy=args.mlip_energy, force=args.force,
            progress=lambda message: print(message, flush=True),
        ),
        indent=2, default=str,
    ))
    return 0


def cmd_dft_launch(args: argparse.Namespace) -> int:
    print(json.dumps(
        launch_ordering_dft(
            args.root, stage=args.stage, execute=args.execute, launcher=args.launcher,
            limit=args.limit, potcar_root=args.potcar_root,
            progress=lambda message: print(message, flush=True),
        ),
        indent=2, default=str,
    ))
    return 0


def cmd_dft_collect(args: argparse.Namespace) -> int:
    payload = collect_ordering_dft(args.root, stage=args.stage)
    payload.pop("caveats", None)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_dft_compare(args: argparse.Namespace) -> int:
    payload = compare_ordering(
        args.root, stage=args.stage, reference_cand=args.reference_cand,
        per_atom=args.per_atom, figure=not args.no_figure,
    )
    print(json.dumps(
        {key: payload[key] for key in ("statistics", "searched_vs_random", "groups", "interpretation", "outputs")},
        indent=2, default=str,
    ))
    return 0


def register_dft_commands(commands: Any) -> None:
    """Attach dft-prepare/dft-launch/dft-collect/dft-compare to the swap-mc subparsers."""

    prepare = commands.add_parser(
        "dft-prepare",
        help="Build a VASP tree from exported candidates in which only POSCAR differs",
    )
    prepare.add_argument("export_dir", help="Directory written by 'iface swap-mc export'")
    prepare.add_argument("output_dir")
    prepare.add_argument("--reference", required=True, help="Converged run supplying INCAR/KPOINTS/POTCAR/launcher")
    prepare.add_argument("--stage", action="append", choices=STAGES, help="Repeat for both (default: static)")
    prepare.add_argument("--incar", help="INCAR name inside the reference directory (default INCAR)")
    prepare.add_argument("--kpoints", help="KPOINTS name inside the reference directory")
    prepare.add_argument("--potcar", help="POTCAR name inside the reference directory")
    prepare.add_argument("--launcher", help="Launcher name to propagate (default runvasp.sh then run.slurm)")
    prepare.add_argument("--no-launcher", action="store_true", help="Do not copy any launcher")
    prepare.add_argument("--potcar-root", help="Licensed PBE PAW tree, if a POTCAR must be assembled")
    prepare.add_argument("--potcar-mapping", help="Element to POTCAR variant YAML")
    prepare.add_argument("--ediff", default="1E-6", help="Electronic convergence for every run (default 1E-6)")
    prepare.add_argument("--ediffg", default="-0.02", help="Force criterion for the relax stage (default -0.02)")
    prepare.add_argument("--nsw", type=int, default=99, help="Ionic step cap for the relax stage (default 99)")
    prepare.add_argument("--mlip-energy", default="best", choices=MLIP_ENERGY_CHOICES,
                         help="Which archived MLIP energy to carry into the comparison (default best)")
    prepare.add_argument("--force", action="store_true", help="Write into a non-empty output directory")
    prepare.set_defaults(func=cmd_dft_prepare)

    launch = commands.add_parser("dft-launch", help="Preflight and (with --execute) submit the prepared ordering runs")
    launch.add_argument("root", help="Directory written by dft-prepare")
    launch.add_argument("--stage", default="static", choices=STAGES)
    launch.add_argument("--execute", action="store_true", help="Actually submit; without it this is a dry run")
    launch.add_argument("--launcher", help="Override the launcher name")
    launch.add_argument("--limit", type=int, help="Submit at most this many runs")
    launch.add_argument("--potcar-root")
    launch.set_defaults(func=cmd_dft_launch)

    collect = commands.add_parser("dft-collect", help="Read finished DFT energies from a prepared ordering tree")
    collect.add_argument("root")
    collect.add_argument("--stage", default="static", choices=STAGES)
    collect.set_defaults(func=cmd_dft_collect)

    compare = commands.add_parser("dft-compare", help="Compare DFT and MLIP ordering energies at a common reference")
    compare.add_argument("root")
    compare.add_argument("--stage", default="static", choices=STAGES)
    compare.add_argument("--reference-cand", help="Candidate id to use as the zero of both energy scales")
    compare.add_argument("--per-atom", action="store_true", help="Report meV/atom instead of eV")
    compare.add_argument("--no-figure", action="store_true", help="Skip the scatter figure")
    compare.set_defaults(func=cmd_dft_compare)
