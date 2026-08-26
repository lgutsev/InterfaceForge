"""Bulk VASP-MLFF training-campaign generation for a family x termination x
composition grid of interface structures (e.g. Real/Ideal x N_Term/Ti_Term x
x=0,0.25,0.5,0.75,1.0), plus a throttled mass-launcher and a grid-aware audit
rollup.

This module deliberately does not duplicate the existing campaign machinery
(``prepare_campaign``/``submit_campaign``/``stage_tags``/``run_audit``): its
job is (1) turning a reviewed source manifest into a correct ``campaign.yaml``
that machinery already knows how to prepare, submit, and audit, and (2) the
two genuinely new pieces that machinery does not provide -- Slurm-array
throttled mass submission, and a rollup of ``iface audit``'s flat run table
grouped by the family/termination/x grid instead of by individual run.

Source discovery is intentionally two-step (discover -> human review of the
written CSV -> build) rather than one-shot: this repo has no visibility into
how any particular Step2-style source tree is actually laid out, so
auto-discovery is best-effort and reported, never silently trusted.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .audit import find_runs, run_audit
from .campaign import submission_candidates
from .config import Campaign
from .errors import SafetyError
from .scheduler import render_job, write_job

DEFAULT_FAMILIES = ("Real", "Ideal")
DEFAULT_TERMS = ("N_Term", "Ti_Term")
DEFAULT_X_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _slug(value: Any) -> str:
    # Underscores are preserved (not folded into the "-" separator) so a
    # generated system id like "real-n_term-x0" stays visually split into
    # family/term/x for a human browsing runs/vasp/, even though nothing in
    # this module parses that id back apart -- mass_audit_mlff_interfaces
    # reads family/term/x from campaign.systems[i].tags instead.
    return re.sub(r"[^A-Za-z0-9_]+", "-", str(value).strip()).strip("-").lower()


def _x_label(value: float) -> str:
    return f"{value:g}"


def discover_mlff_interface_sources(
    source_root: str | Path,
    output_manifest: str | Path,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    x_values: tuple[float, ...] = DEFAULT_X_VALUES,
    structure_name: str = "CONTCAR",
) -> dict[str, Any]:
    """Best-effort discovery of one equilibrated structure per grid cell.

    Matches each ``structure_name`` file found under ``source_root`` against
    every (family, term, x) combination by simple case-insensitive token
    matching against its path components -- "real"/"ideal",
    "n_term"/"nterm"/"n-term" (and the ti_term equivalents), and an
    ``x``-prefixed or bare numeric directory matching that x value. A cell
    with zero or more than one match is reported, never guessed: the written
    CSV's ``match_status`` is ``matched``, ``missing``, or ``ambiguous``
    (with candidates listed), and this function does not raise on an
    incomplete grid -- review and hand-fix the CSV, then pass it to
    ``generate_mlff_interfaces_campaign``.
    """

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise SafetyError(f"Source root is not a directory: {root}")
    candidates = sorted(root.rglob(structure_name))

    def _term_tokens(term: str) -> list[str]:
        base = term.lower()
        return [base, base.replace("_", "-"), base.replace("_", "")]

    # Match x by numeric value rather than a single formatted string: a
    # directory can spell x=1.0 as "x1", "x1.0", "x=1.00", etc.
    number_token = re.compile(r"(?<![a-z0-9.])x?[-_=]?(\d+(?:\.\d+)?)(?![a-z0-9])", re.I)

    def _path_x_values(text: str) -> list[float]:
        values = []
        for match in number_token.finditer(text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
        return values

    rows: list[dict[str, Any]] = []
    for family in families:
        for term in terms:
            for x in x_values:
                matches: list[Path] = []
                for path in candidates:
                    text = str(path.relative_to(root)).lower()
                    if not re.search(rf"(?<![a-z]){re.escape(family.lower())}(?![a-z])", text):
                        continue
                    term_hit = any(
                        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text)
                        for token in _term_tokens(term)
                    )
                    if not term_hit:
                        continue
                    path_x_values = _path_x_values(text)
                    is_x_match = any(math.isclose(candidate, x, abs_tol=1e-6) for candidate in path_x_values)
                    # A real observed convention: the x=0 (oxygen-free) baseline
                    # carries no numeric suffix at all (e.g. "SiN_TiN_N-term"),
                    # unlike the oxygen-substituted cells ("..._O_x0.25"). When
                    # x==0 and nothing else under this family/term has *any*
                    # x-like numeric token, treat the bare leaf as the match.
                    is_bare_zero = x == 0.0 and not path_x_values
                    if not (is_x_match or is_bare_zero):
                        continue
                    matches.append(path)
                if len(matches) == 1:
                    status = "matched"
                elif not matches:
                    status = "missing"
                else:
                    status = "ambiguous"
                rows.append(
                    {
                        "system_id": f"{_slug(family)}-{_slug(term)}-x{_x_label(x)}",
                        "family": family,
                        "term": term,
                        "x": _x_label(x),
                        "structure_path": str(matches[0]) if len(matches) == 1 else "",
                        "match_status": status,
                        "candidates": "; ".join(str(m) for m in matches),
                    }
                )

    manifest_path = Path(output_manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["system_id", "family", "term", "x", "structure_path", "match_status", "candidates"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["match_status"] for row in rows)
    return {
        "source_root": str(root),
        "manifest": str(manifest_path),
        "grid_size": len(rows),
        "structure_files_seen": len(candidates),
        "status_counts": dict(counts),
        "rows": rows,
    }


def generate_mlff_interfaces_campaign(
    manifest_source: str | Path,
    campaign_root: str | Path,
    *,
    profile_path: str | Path,
    profile_name: str = "vasp_train",
    encut: float = 520.0,
    ivdw: int = 11,
    tebeg: float = 300.0,
    teend: float = 600.0,
    train_nsw: int = 100000,
    refit_nsw: int = 0,
    stability_nsw: int = 20000,
    potim: float = 1.0,
    kpoints_source: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a ready-to-``iface prepare`` campaign.yaml from a reviewed manifest.

    Every row must have ``match_status == "matched"`` (see
    ``discover_mlff_interface_sources``); this refuses to generate a
    campaign from an incomplete or ambiguous grid rather than silently
    dropping cells. Deliberately does not set a shared reference POTCAR:
    each system's own POSCAR (from its own structure) carries its own
    species, so ``iface vasp submit`` generates the correct POTCAR per
    leaf automatically -- required here since the oxygen-free and
    oxygen-containing interfaces in this grid cannot share one POTCAR.
    """

    manifest_path = Path(manifest_source).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SafetyError(f"No rows in manifest: {manifest_path}")
    bad = [row["system_id"] for row in rows if row.get("match_status") != "matched"]
    if bad:
        raise SafetyError(
            f"Manifest has {len(bad)} unresolved grid cell(s), refusing to generate a "
            f"campaign: {', '.join(bad)}. Fix structure_path/match_status in {manifest_path} first."
        )

    root = Path(campaign_root).resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise SafetyError(f"Refusing to write into a nonempty campaign root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)

    incar_path = root / "inputs" / "INCAR"
    if force or not incar_path.is_file():
        incar_path.write_text(
            "# InterfaceForge generated shell for the MLFF thermal-conductivity "
            "training grid.\n"
            "# Add converged electronic-structure settings (EDIFF, ISMEAR, "
            "sigma, ...); ENCUT/IVDW below are the ones explicitly requested.\n"
            f"ENCUT = {encut:g}\n"
            f"IVDW = {int(ivdw)}\n",
            encoding="utf-8",
        )
    kpoints_path = root / "inputs" / "KPOINTS"
    if kpoints_source is not None:
        kpoints_path.write_text(Path(kpoints_source).read_text(encoding="utf-8"), encoding="utf-8")
    elif force or not kpoints_path.is_file():
        kpoints_path.write_text("Automatic\n0\nGamma\n1 1 1\n", encoding="utf-8")

    systems = [
        {
            "id": row["system_id"],
            "kind": "interface",
            "structure": row["structure_path"],
            "tags": {"family": row["family"], "term": row["term"], "x": row["x"]},
        }
        for row in rows
    ]

    campaign_data = {
        "schema_version": 1,
        "project": {
            "name": "mlff-thermal-conductivity-grid",
            "description": (
                "Real/Ideal x N_Term/Ti_Term x composition MLFF training grid "
                "for VASP ML_LHEAT Green-Kubo thermal conductivity."
            ),
        },
        "profile": str(Path(profile_path).resolve()),
        "systems": systems,
        "reference": {"engine": "vasp", "inputs": {"INCAR": "inputs/INCAR", "KPOINTS": "inputs/KPOINTS"}},
        "stages": {
            "vasp_mlff": {
                "enabled": True,
                "train": {
                    "temperature": tebeg,
                    "teend": teend,
                    "nsw": train_nsw,
                    "potim": potim,
                    "profile": profile_name,
                },
                "refit": {"nsw": refit_nsw, "potim": potim, "profile": profile_name},
                "stability": {
                    "temperature": tebeg,
                    "nsw": stability_nsw,
                    "potim": potim,
                    "profile": profile_name,
                },
            }
        },
        "dataset": {"strategy": "grouped", "ratios": [0.8, 0.1, 0.1]},
        "models": {},
        "active_learning": {"enabled": False},
        "exploration": {},
        "validation": {},
    }
    campaign_path = root / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign_data, sort_keys=False), encoding="utf-8")

    return {
        "campaign_root": str(root),
        "campaign": str(campaign_path),
        "systems": len(systems),
        "grid": sorted({(row["family"], row["term"]) for row in rows}),
        "x_values": sorted({row["x"] for row in rows}),
        "note": (
            "No reference POTCAR was set; iface vasp submit generates the correct "
            "POTCAR per system from its own POSCAR species."
        ),
    }


def write_throttled_array_launcher(
    campaign: Campaign,
    *,
    stage: str = "train",
    concurrency: int = 4,
    array_profile_name: str = "vasp_train_array",
    output: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one Slurm array job that runs every prepared leaf's run.slurm,
    throttled to ``concurrency`` concurrent tasks via Slurm's native
    ``--array=0-N%K`` syntax.

    Requires ``iface prepare`` to have already run (each leaf's run.slurm
    must exist). Each array task runs exactly one leaf's already-generated
    ``run.slurm`` as a plain shell script (its ``#SBATCH`` lines are inert
    when executed this way) rather than re-``sbatch``-ing it, so this
    remains one job for Slurm to schedule/throttle rather than N independent
    submissions racing the fairshare queue.
    """

    if concurrency < 1:
        raise SafetyError("concurrency must be at least 1")
    launchers = submission_candidates(campaign, stage=stage)
    if not launchers:
        raise SafetyError(
            f"No prepared run.slurm files found for stage {stage!r}; run iface prepare first"
        )
    array_dir = Path(output).resolve() if output else campaign.root / "runs" / "vasp" / "_arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = array_dir / f"{stage}_array_manifest.txt"
    manifest_path.write_text("\n".join(str(path) for path in launchers) + "\n", encoding="utf-8")

    profile = yaml.safe_load(campaign.profile_path.read_text(encoding="utf-8"))
    if array_profile_name not in profile.get("jobs", {}):
        raise SafetyError(
            f"Profile has no job named {array_profile_name!r}; add one sized for a single "
            f"{stage} leaf (the array multiplies it by concurrency, not by task count)"
        )
    command = (
        f'MANIFEST="{manifest_path}"\n'
        'LEAF_SCRIPT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")\n'
        'if [[ -z "$LEAF_SCRIPT" ]]; then echo "No manifest line for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; fi\n'
        'cd "$(dirname "$LEAF_SCRIPT")"\n'
        'bash "$(basename "$LEAF_SCRIPT")"\n'
    )
    script = render_job(
        profile,
        array_profile_name,
        command=command,
        job_name=f"{campaign.name}_{stage}_array",
        array=f"0-{len(launchers) - 1}%{concurrency}",
    )
    script_path = array_dir / f"{stage}_array.slurm"
    write_job(script_path, script, force=force)
    return {
        "stage": stage,
        "leaves": len(launchers),
        "concurrency": concurrency,
        "manifest": str(manifest_path),
        "launcher": str(script_path),
        "submitted": False,
    }


def mass_audit_mlff_interfaces(
    campaign: Campaign,
    *,
    readiness_profile: str = "general",
) -> dict[str, Any]:
    """Audit the whole grid via the existing ``iface audit`` engine, then
    roll the flat per-run table up by (family, term, x) -- one row per grid
    cell aggregating its train/refit/stability runs, plus totals per family
    and per term.

    Family/term/x for each run come from ``campaign.systems[i].tags`` (set
    by ``generate_mlff_interfaces_campaign``) looked up by the first path
    component of each run's ``relative_path`` (its system id), rather than
    re-parsing that id textually -- slugified family/term names can contain
    the same separator characters used between id components, which makes
    reconstructing them from the string alone ambiguous.
    """

    runs_root = campaign.root / "runs" / "vasp"
    if not find_runs(runs_root):
        raise SafetyError(f"No run directories found below {runs_root}; run iface prepare first")
    payload = run_audit(runs_root, readiness_profile=readiness_profile)

    tags_by_system = {system.id: system.tags for system in campaign.systems}
    cells: dict[str, dict[str, Any]] = {}
    unparsed: list[str] = []
    for row in payload["runs"]:
        relative = row["relative_path"].replace("\\", "/")
        if "/" not in relative:
            unparsed.append(row["relative_path"])
            continue
        system_id, stage = relative.split("/", 1)
        tags = tags_by_system.get(system_id)
        if tags is None:
            unparsed.append(row["relative_path"])
            continue
        cell = cells.setdefault(
            system_id,
            {"family": tags.get("family"), "term": tags.get("term"), "x": tags.get("x"), "stages": {}},
        )
        cell["stages"][stage] = {"health": row["health"], "next_action": row["next_action"]}

    by_family: dict[str, Counter] = defaultdict(Counter)
    by_term: dict[str, Counter] = defaultdict(Counter)
    for cell in cells.values():
        train_health = cell["stages"].get("train", {}).get("health", "not started")
        by_family[cell["family"]][train_health] += 1
        by_term[cell["term"]][train_health] += 1

    return {
        "schema_version": 1,
        "campaign_root": str(campaign.root),
        "readiness_profile": readiness_profile,
        "grid_cells": len(cells),
        "cells": dict(sorted(cells.items())),
        "train_health_by_family": {family: dict(counts) for family, counts in by_family.items()},
        "train_health_by_term": {term: dict(counts) for term, counts in by_term.items()},
        "unparsed_runs": unparsed,
        "flat_audit_outputs": payload["outputs"],
    }
