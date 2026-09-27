"""File-safe orchestration of one density initialization (``iface vasp initialize-density``).

Guarantees, in order of importance:

1. The run's ``INCAR``/``POSCAR``/``POTCAR``/``KPOINTS`` are never handed to
   the initializer; it works on copies in a private scratch directory, and the
   originals are re-hashed afterwards (and restored from a pristine copy if
   anything changed).
2. An existing ``CHGCAR`` that InterfaceForge did not itself promote (a
   converged or user-supplied density) is never replaced unless ``overwrite``
   is given, and even then it is first renamed to a timestamped backup.
3. The generated density is always written to ``CHGCAR.neural_init`` first
   and only then promoted to ``CHGCAR``.
4. The only INCAR change is ``ICHARG = 1`` (so VASP actually reads the
   promoted density).  The original is kept as ``INCAR.pre_density_init`` and
   the change is verified to be the *only* difference in active tags.
5. Every failure after validation rolls back promotion and INCAR edits and
   leaves a machine-readable ``density_init.json`` with ``status = FAILED``.
6. Re-running with unchanged inputs is a no-op (``ALREADY_INITIALIZED``); a
   previously promoted density whose inputs have since changed is demoted
   (renamed ``CHGCAR.neural_init.stale-<time>``) before regeneration.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..errors import DependencyError, InterfaceForgeError, SafetyError
from ..vasp import parse_incar, update_incar
from .base import (
    MAGMOM_SOURCES,
    SPIN_CHANNEL_MODES,
    BackendResult,
    DensityInitializer,
    InitializationPlan,
    StandardInitializer,
    resolve_magnetism,
)
from .inputs import (
    REQUIRED_INPUTS,
    grid_from_chgcar,
    grid_from_file,
    grid_from_incar,
    input_hashes,
    lmaxmix,
    magnetic_settings,
    missing_inputs,
    nelect,
    nonempty,
    poscar_species,
    sha256_file,
    structure_identity,
)
from .neural_paw import InferenceError, NeuralPawInitializer
from .potcar import potcar_provenance

REPORT_NAME = "density_init.json"
LOG_NAME = "density_init.log"
STAGED_NAME = "CHGCAR.neural_init"
INCAR_BACKUP_NAME = "INCAR.pre_density_init"
LOCK_NAME = ".density_init.lock"
REPORT_FORMAT = "interfaceforge-density-init"
STARTED_MARKERS = ("OUTCAR", "OSZICAR", "vasprun.xml")

BACKENDS: dict[str, type[DensityInitializer]] = {}


def register_backend(cls: type[DensityInitializer]) -> type[DensityInitializer]:
    BACKENDS[cls.name] = cls
    return cls


register_backend(StandardInitializer)
register_backend(NeuralPawInitializer)


def make_backend(name: str, **options: Any) -> DensityInitializer:
    if name not in BACKENDS:
        raise SafetyError(f"Unknown density-init backend {name!r}; choose from {', '.join(sorted(BACKENDS))}")
    cls = BACKENDS[name]
    if cls is StandardInitializer:
        return cls()
    return cls(**{key: value for key, value in options.items() if value is not None})


# ----------------------------------------------------------------------------
# provenance helpers
# ----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def interfaceforge_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {"version": __version__, "commit": None, "dirty": None}
    source = Path(__file__).resolve().parents[3]
    if (source / ".git").exists():
        try:
            info["commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
            info["dirty"] = bool(status.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _read_report(run: Path) -> dict[str, Any]:
    path = run / REPORT_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) and payload.get("format") == REPORT_FORMAT else {}


def _write_report(run: Path, payload: dict[str, Any]) -> Path:
    path = run / REPORT_NAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_install(source: Path, target: Path, *, link: bool) -> None:
    """Place ``source`` at ``target`` atomically (hard link when allowed, else copy)."""

    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    if link:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


class _Lock:
    def __init__(self, run: Path) -> None:
        self.path = run / LOCK_NAME
        self.fd: int | None = None

    def __enter__(self) -> _Lock:
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise SafetyError(
                f"{self.path} exists: another density initialization is running (or crashed); "
                "remove the lock file only after confirming no job is using this directory"
            ) from exc
        os.write(self.fd, f"{os.getpid()} {_now()}\n".encode())
        return self

    def __exit__(self, *exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ----------------------------------------------------------------------------
# grid resolution
# ----------------------------------------------------------------------------


def discover_grid_by_dry_run(
    run: Path, command: str, scratch: Path, *, timeout: float | None = 3600
) -> tuple[tuple[int, int, int], float]:
    """Run a throw-away ``NELM=1`` VASP step on copies to read VASP's own FFT grid.

    Only non-grid tags are changed (``NELM``, ``NSW``, ``IBRION``, output
    switches, fresh start), so ``NGXF/NGYF/NGZF`` are exactly the target
    run's.  Its cost is counted as initialization overhead.
    """

    probe = scratch / "grid_probe"
    probe.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_INPUTS:
        if (run / name).is_file():
            shutil.copy2(run / name, probe / name)
    update_incar(
        probe / "INCAR",
        {
            "NELM": 1,
            "NELMDL": 0,
            "NSW": 0,
            "IBRION": -1,
            "ISTART": 0,
            "ICHARG": 2,
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "LORBIT": 0,
        },
    )
    started = time.perf_counter()
    with (probe / "vasp.out").open("wb") as log:
        try:
            subprocess.run(
                command, shell=True, cwd=probe, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            pass
    elapsed = time.perf_counter() - started
    outcar = probe / "OUTCAR"
    grid = grid_from_file(outcar) if outcar.is_file() else None
    if grid is None:
        raise SafetyError(f"grid dry run ({command!r}) produced no NGXF/NGYF/NGZF line; see {probe / 'vasp.out'}")
    return grid, elapsed


def resolve_grid(
    run: Path,
    incar: dict[str, str],
    *,
    grid: tuple[int, int, int] | list[int] | None,
    grid_from: str | Path | None,
) -> tuple[tuple[int, int, int] | None, str]:
    if grid is not None:
        values = tuple(int(value) for value in grid)
        if len(values) != 3 or any(value <= 0 for value in values):
            raise SafetyError(f"--grid needs three positive integers, got {grid}")
        return values, "explicit --grid"  # type: ignore[return-value]
    from_incar = grid_from_incar(incar)
    if from_incar is not None:
        return from_incar, "INCAR NGXF/NGYF/NGZF"
    if grid_from is not None:
        path = Path(grid_from).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        found = grid_from_file(path)
        if found is None:
            raise SafetyError(f"No FFT grid found in {path}")
        return found, f"--grid-from {path}"
    return None, "unresolved"


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------


def _options_fingerprint(
    backend: DensityInitializer, magmom_source: str, spin_channel: str, grid: Any
) -> dict[str, Any]:
    settings = backend.settings() if hasattr(backend, "settings") else {}
    settings = {key: value for key, value in settings.items() if key not in {"python", "timeout_s"}}
    return {
        "backend": backend.name,
        "magmom_source": magmom_source,
        "spin_channel": spin_channel,
        "grid": list(grid) if grid else None,
        "backend_settings": settings,
    }


def _already_initialized(
    run: Path, previous: dict[str, Any], fingerprint: dict[str, Any], hashes: dict[str, str | None]
) -> tuple[bool, bool]:
    """``(up_to_date, stale_promotion)`` for a previous successful report."""

    if previous.get("status") not in {"PROMOTED", "STAGED"}:
        return False, False
    files = previous.get("files") or {}
    promoted_hash = files.get("promoted_sha256")
    chgcar = run / "CHGCAR"
    ours_active = bool(promoted_hash) and nonempty(chgcar) and sha256_file(chgcar) == promoted_hash
    before = (previous.get("inputs") or {}).get("sha256_after") or {}
    same_inputs = all(hashes.get(name) == before.get(name) for name in REQUIRED_INPUTS)
    same_options = previous.get("request") == fingerprint
    staged_hash = files.get("staged_sha256")
    staged_ok = bool(staged_hash) and nonempty(run / STAGED_NAME) and sha256_file(run / STAGED_NAME) == staged_hash
    if same_inputs and same_options and staged_ok and (ours_active or previous.get("status") == "STAGED"):
        return True, False
    return False, ours_active and not (same_inputs and same_options)


def initialize_density(
    run_dir: str | Path,
    *,
    backend: str | DensityInitializer = "neural-paw",
    magmom_source: str = "incar",
    spin_channel: str = "auto",
    grid: tuple[int, int, int] | list[int] | None = None,
    grid_from: str | Path | None = None,
    grid_dry_run_command: str | None = None,
    dry_run: bool = False,
    stage_only: bool = False,
    overwrite: bool = False,
    force: bool = False,
    set_icharg: bool = True,
    backend_options: dict[str, Any] | None = None,
    potcar_definitions: str | Path | None = None,
    potcar_generator: str | None = None,
) -> dict[str, Any]:
    """Generate (and by default promote) an initial density for one VASP run.

    ``potcar_definitions`` / ``potcar_generator`` only *declare* how the POTCAR
    was made; the declaration is recorded and must match the POTCAR actually present.
    """

    if magmom_source not in MAGMOM_SOURCES:
        raise SafetyError(f"--magmom-source must be one of {', '.join(MAGMOM_SOURCES)}")
    if spin_channel not in SPIN_CHANNEL_MODES:
        raise SafetyError(f"--spin-channel must be one of {', '.join(SPIN_CHANNEL_MODES)}")
    if stage_only and overwrite:
        raise SafetyError("--stage-only and --overwrite cannot be combined")
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    if isinstance(backend, DensityInitializer):
        initializer = backend
    else:
        initializer = make_backend(backend, **(backend_options or {}))

    # ---------------- validation (no writes) ----------------
    missing = missing_inputs(run)
    if missing:
        raise SafetyError(f"{run} is missing required VASP input(s): {', '.join(missing)}")
    started = [name for name in STARTED_MARKERS if nonempty(run / name)]
    if started:
        raise SafetyError(
            f"{run} already has VASP output ({', '.join(started)}); density initialization only "
            "seeds a fresh calculation. Prepare a clean directory (e.g. 'iface vasp density-init-bench')"
        )
    incar = parse_incar(run / "INCAR")
    species = poscar_species(run / "POSCAR")
    icharg = incar.get("ICHARG")
    if icharg is not None and int(float(icharg)) >= 10:
        raise SafetyError(f"ICHARG = {icharg} is a non-self-consistent run; density initialization refused")
    if icharg is not None and int(float(icharg)) not in {0, 1, 2}:
        raise SafetyError(f"ICHARG = {icharg} is not a standard start; density initialization refused")
    istart = incar.get("ISTART")
    if nonempty(run / "WAVECAR") and (istart is None or int(float(istart)) != 0):
        raise SafetyError(
            f"{run} has a nonempty WAVECAR and ISTART != 0: VASP would restart from those orbitals. "
            "A neural density is only meaningful for a fresh electronic start; set ISTART = 0 or "
            "remove the WAVECAR"
        )
    potcar_record = potcar_provenance(
        run / "POTCAR", species, definitions=potcar_definitions, generator=potcar_generator
    )
    settings = magnetic_settings(incar, len(species))
    magnetism = resolve_magnetism(
        settings, magmom_source=magmom_source, spin_channel=spin_channel, backend=initializer
    )
    warnings: list[str] = list(magnetism.get("warnings", []))
    if istart is not None and int(float(istart)) != 0 and not nonempty(run / "WAVECAR"):
        warnings.append(f"ISTART = {istart} but no WAVECAR: VASP starts with random orbitals")

    resolved_grid, grid_source = (None, "not required")
    if initializer.needs_grid:
        resolved_grid, grid_source = resolve_grid(run, incar, grid=grid, grid_from=grid_from)
        if resolved_grid is None and grid_dry_run_command is None:
            raise SafetyError(
                "Cannot determine the target FFT grid. Pass --grid NGXF NGYF NGZF, --grid-from an "
                "OUTCAR/CHGCAR of a run with identical ENCUT/PREC/ENAUG and lattice, set NGXF/NGYF/NGZF "
                "in the INCAR, or --grid-dry-run-command 'srun vasp_std' (one NELM=1 step on copies)"
            )
    nelect_value, nelect_source = nelect(run, incar, species)
    plan = InitializationPlan(
        run_dir=run,
        species=species,
        grid=resolved_grid,
        grid_source=grid_source if resolved_grid else ("dry run" if initializer.needs_grid else grid_source),
        nelect=nelect_value,
        nelect_source=nelect_source,
        lmaxmix=lmaxmix(incar),
        magnetism=magnetism,
    )
    hashes = input_hashes(run)
    fingerprint = _options_fingerprint(initializer, magmom_source, spin_channel, resolved_grid)
    previous = _read_report(run)
    up_to_date, stale = _already_initialized(run, previous, fingerprint, hashes)

    existing_chgcar = nonempty(run / "CHGCAR")
    previous_promoted = (previous.get("files") or {}).get("promoted_sha256")
    chgcar_is_ours = existing_chgcar and bool(previous_promoted) and sha256_file(run / "CHGCAR") == previous_promoted
    will_promote = initializer.writes_chgcar and not stage_only and (
        not existing_chgcar or chgcar_is_ours or overwrite
    )
    if up_to_date and previous.get("status") == "STAGED" and will_promote:
        up_to_date = False  # the blocker (a foreign CHGCAR / --stage-only) is gone: promote now
    base: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "schema_version": 1,
        "run_dir": str(run),
        "backend": initializer.name,
        "request": fingerprint,
        "interfaceforge": interfaceforge_provenance(),
        "structure": structure_identity(run / "POSCAR"),
        "inputs": {"required": list(REQUIRED_INPUTS), "sha256_before": hashes},
        "potcar": potcar_record,
        "grid": {"dims": list(resolved_grid) if resolved_grid else None, "source": plan.grid_source},
        "nelect": {"value": nelect_value, "source": nelect_source},
        "lmaxmix": plan.lmaxmix,
        "magnetism": {key: value for key, value in magnetism.items() if key not in {"warnings"}},
        "incar": {"ICHARG_before": icharg, "ISTART": istart},
        "warnings": warnings,
    }

    if up_to_date and not force:
        payload = dict(previous)
        payload["mode"] = "already-initialized"
        payload["checked_at"] = _now()
        payload["status_note"] = "inputs, options and generated files unchanged; nothing regenerated"
        return payload

    if dry_run:
        probe = initializer.probe() if initializer.writes_chgcar else {"available": True}
        would = []
        if initializer.writes_chgcar:
            would.append(STAGED_NAME)
            if will_promote:
                would.append("CHGCAR")
                if set_icharg and icharg != "1":
                    would.extend([f"INCAR (ICHARG {icharg or 'default'} -> 1)", INCAR_BACKUP_NAME])
        if stale:
            would.insert(0, "demote stale promoted CHGCAR")
        if existing_chgcar and not chgcar_is_ours and not overwrite and initializer.writes_chgcar:
            warnings.append(
                "existing CHGCAR was not written by density initialization; it will be kept and the new "
                f"density only staged as {STAGED_NAME} (pass --overwrite to promote with a backup)"
            )
        return {
            **base,
            "mode": "dry-run",
            "status": "PLANNED",
            "backend_probe": probe,
            "would_write": would,
            "grid_dry_run_command": grid_dry_run_command,
            "up_to_date": up_to_date,
        }

    # ---------------- execution ----------------
    record: dict[str, Any] = {**base, "mode": "executed", "started_at": _now(), "files": {}, "incar_changes": []}
    files = record["files"]
    timing: dict[str, float] = {}
    wall_start = time.perf_counter()
    rollback: list[tuple[str, Path, Path | None]] = []  # (kind, target, backup)
    scratch: Path | None = None
    with _Lock(run):
        try:
            scratch = Path(tempfile.mkdtemp(prefix=".density_init_", dir=run))
            pristine = scratch / "pristine"
            work = scratch / "work"
            pristine.mkdir()
            work.mkdir()
            for name in REQUIRED_INPUTS:
                if (run / name).is_file():
                    shutil.copy2(run / name, pristine / name)
                    shutil.copy2(run / name, work / name)

            if stale:
                demoted = run / f"{STAGED_NAME}.stale-{_stamp()}"
                os.replace(run / "CHGCAR", demoted)
                files["demoted_stale"] = demoted.name
                warnings.append(f"previously promoted CHGCAR no longer matches the inputs; moved to {demoted.name}")
                previous_before = (previous.get("incar") or {}).get("ICHARG_before")
                if icharg == "1" and (previous.get("incar_changes") or []):
                    if previous_before is None:
                        update_incar(run / "INCAR", {}, delete=("ICHARG",))
                    else:
                        update_incar(run / "INCAR", {"ICHARG": previous_before})
                    record["incar_changes"].append(
                        {"tag": "ICHARG", "before": "1", "after": previous_before, "reason": "stale demotion"}
                    )
                    icharg = previous_before
                    shutil.copy2(run / "INCAR", pristine / "INCAR")
                    shutil.copy2(run / "INCAR", work / "INCAR")
                    hashes = input_hashes(run)
                    record["inputs"]["sha256_before"] = hashes
                existing_chgcar = False
                chgcar_is_ours = False

            if initializer.needs_grid and plan.grid is None:
                assert grid_dry_run_command is not None
                plan.grid, timing["grid_discovery_s"] = discover_grid_by_dry_run(run, grid_dry_run_command, scratch)
                plan.grid_source = f"VASP NELM=1 dry run: {grid_dry_run_command}"
                record["grid"] = {"dims": list(plan.grid), "source": plan.grid_source}

            if initializer.writes_chgcar and hasattr(initializer, "require_available"):
                record["backend_probe"] = initializer.require_available()

            result: BackendResult = initializer.generate(plan, work)
            timing.update(result.timing)
            record["backend_info"] = result.backend_info
            record["initializer_output"] = result.extra
            warnings.extend(result.warnings)
            magnetism_record = record["magnetism"]
            magnetism_record["initializer_moments"] = result.initializer_moments
            magnetism_record["spin_channel_written"] = result.spin_channel_written
            if magmom_source == "incar" and result.initializer_moments is not None:
                raise InferenceError("backend returned initializer moments although INCAR MAGMOM is authoritative")

            after = input_hashes(run)
            if after != hashes:
                for name in REQUIRED_INPUTS:
                    if after.get(name) != hashes.get(name) and (pristine / name).is_file():
                        shutil.copy2(pristine / name, run / name)
                raise SafetyError("run inputs changed during initialization; restored the pristine copies")

            if not initializer.writes_chgcar:
                record["status"] = "STANDARD_START"
                record["active"] = False
                record["status_note"] = "VASP default start; nothing generated"
            else:
                generated = result.chgcar
                if generated is None or not nonempty(generated):
                    raise InferenceError("backend produced no CHGCAR")
                produced_grid = grid_from_chgcar(generated)
                if produced_grid is not None and tuple(produced_grid) != tuple(plan.grid or ()):
                    raise InferenceError(f"generated CHGCAR grid {produced_grid} != target grid {plan.grid}")
                if poscar_species(generated) != species:
                    raise InferenceError("generated CHGCAR ion order/species differ from POSCAR")

                staged = run / STAGED_NAME
                if staged.exists():
                    files["replaced_staged_sha256"] = sha256_file(staged)
                _atomic_install(generated, staged, link=False)
                files["staged"] = STAGED_NAME
                files["staged_sha256"] = sha256_file(staged)
                files["generated"] = [STAGED_NAME]

                promote = not stage_only and (not existing_chgcar or chgcar_is_ours or overwrite)
                if promote:
                    target = run / "CHGCAR"
                    if existing_chgcar and not chgcar_is_ours:
                        backup = run / f"CHGCAR.pre_density_init.{_stamp()}"
                        os.replace(target, backup)
                        rollback.append(("restore", target, backup))
                        files["backed_up_chgcar"] = backup.name
                        warnings.append(f"--overwrite: existing CHGCAR preserved as {backup.name}")
                    else:
                        rollback.append(("remove", target, None))
                    _atomic_install(staged, target, link=True)
                    files["promoted"] = "CHGCAR"
                    files["promoted_sha256"] = sha256_file(target)
                    files["generated"].append("CHGCAR")

                    if set_icharg and icharg != "1":
                        before_tags = parse_incar(run / "INCAR")
                        backup_incar = run / INCAR_BACKUP_NAME
                        if backup_incar.exists() and sha256_file(backup_incar) != sha256_file(run / "INCAR"):
                            backup_incar = run / f"{INCAR_BACKUP_NAME}.{_stamp()}"
                        shutil.copy2(pristine / "INCAR", backup_incar)
                        rollback.append(("incar", run / "INCAR", pristine / "INCAR"))
                        update_incar(run / "INCAR", {"ICHARG": 1})
                        after_tags = parse_incar(run / "INCAR")
                        changed = {
                            tag
                            for tag in set(before_tags) | set(after_tags)
                            if before_tags.get(tag) != after_tags.get(tag)
                        }
                        if changed != {"ICHARG"}:
                            raise SafetyError(f"INCAR edit changed more than ICHARG: {sorted(changed)}")
                        record["incar_changes"].append(
                            {"tag": "ICHARG", "before": icharg, "after": "1", "reason": "read promoted CHGCAR"}
                        )
                        files["incar_backup"] = backup_incar.name
                    elif not set_icharg and icharg != "1":
                        warnings.append(
                            "--no-set-icharg: INCAR ICHARG is not 1, so VASP will ignore the promoted CHGCAR"
                        )
                    record["status"] = "PROMOTED"
                    record["active"] = icharg == "1" or (set_icharg and bool(record["incar_changes"]))
                else:
                    record["status"] = "STAGED"
                    record["active"] = False
                    if existing_chgcar and not chgcar_is_ours:
                        warnings.append(
                            f"existing CHGCAR kept untouched; new density staged as {STAGED_NAME} only "
                            "(pass --overwrite to promote it with a backup)"
                        )
            record["inputs"]["sha256_after"] = input_hashes(run)
            record["inputs"]["unchanged_except_incar_icharg"] = all(
                record["inputs"]["sha256_after"][name] == hashes[name] for name in REQUIRED_INPUTS if name != "INCAR"
            )
        except BaseException as exc:
            for kind, target, backup in reversed(rollback):
                try:
                    if kind == "remove" and target.exists():
                        target.unlink()
                    elif kind == "restore" and backup is not None and backup.exists():
                        os.replace(backup, target)
                    elif kind == "incar" and backup is not None:
                        shutil.copy2(backup, target)
                except OSError:
                    pass
            record["status"] = "FAILED"
            record["active"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["failure_code"] = getattr(exc, "code", None) or (
                "BACKEND_UNAVAILABLE" if isinstance(exc, DependencyError) else None
            )
            record["rolled_back"] = [kind for kind, _, _ in rollback]
            timing["total_s"] = time.perf_counter() - wall_start
            record["timing"] = timing
            record["finished_at"] = _now()
            _finish_logs(run, scratch, record)
            _write_report(run, record)
            if not isinstance(exc, Exception) or isinstance(
                exc, InterfaceForgeError | FileNotFoundError | ValueError
            ):
                raise
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

    timing["total_s"] = time.perf_counter() - wall_start
    timing["overhead_s"] = timing["total_s"]
    record["timing"] = timing
    record["finished_at"] = _now()
    record["grid_dry_run_command"] = grid_dry_run_command
    _finish_logs(run, scratch, record)
    _write_report(run, record)
    record["report"] = str(run / REPORT_NAME)
    return record


def _finish_logs(run: Path, scratch: Path | None, record: dict[str, Any]) -> None:
    if scratch is None:
        return
    logs = [scratch / "work" / "worker.log", scratch / "grid_probe" / "vasp.out"]
    parts = []
    for log in logs:
        if log.is_file():
            parts.append(f"===== {log.relative_to(scratch)} =====\n" + log.read_text(encoding="utf-8", errors="ignore"))
    if parts:
        (run / LOG_NAME).write_text("\n".join(parts), encoding="utf-8")
        record.setdefault("files", {})["log"] = LOG_NAME
    shutil.rmtree(scratch, ignore_errors=True)


#: backend option name -> ``initialize-density`` flag (value flags; booleans are switches)
BACKEND_OPTION_FLAGS = {
    "python": "--ndi-python",
    "device": "--device",
    "weights_dir": "--weights-dir",
    "config": "--ndi-config",
    "electrafi_checkpoint": "--electrafi-checkpoint",
    "augnet_total_checkpoint": "--augnet-total-checkpoint",
    "augnet_mag_checkpoint": "--augnet-mag-checkpoint",
    "timeout": "--inference-timeout",
    "allow_potcar_variant": "--allow-potcar-variant",
}


def backend_option_args(options: dict[str, Any]) -> list[str]:
    """Translate backend options into ``initialize-density`` command-line flags."""

    args: list[str] = []
    for key, value in sorted(options.items()):
        if value is None or value is False:
            continue
        if key not in BACKEND_OPTION_FLAGS:
            raise SafetyError(f"Unknown density-init backend option {key!r}")
        flag = BACKEND_OPTION_FLAGS[key]
        args.extend([flag] if value is True else [flag, str(value)])
    return args


def potcar_declaration_args(definitions: str | Path | None, generator: str | None) -> list[str]:
    """``initialize-density`` flags declaring how the POTCAR was generated."""

    args: list[str] = []
    if definitions:
        args += ["--potcar-definitions", str(definitions)]
    if generator:
        args += ["--potcar-generator", str(generator)]
    return args


def hook_command(
    *,
    interfaceforge: str,
    backend: str = "neural-paw",
    magmom_source: str = "incar",
    spin_channel: str = "auto",
    extra: list[str] | None = None,
) -> str:
    """Shell command a launcher runs immediately before VASP."""

    parts = [
        *shlex.split(interfaceforge),
        "vasp",
        "initialize-density",
        ".",
        "--backend",
        backend,
        "--magmom-source",
        magmom_source,
        "--spin-channel",
        spin_channel,
        *(extra or []),
    ]
    return " ".join(shlex.quote(part) for part in parts)
