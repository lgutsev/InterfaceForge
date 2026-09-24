"""Synthetic Step1 AIMD fixtures shared by the Step1 recovery tests.

Not a test module (the name does not match ``test*.py``).  Every trajectory
here is fabricated: a two-ion H/O cell whose H ion drifts along x by a fixed
fractional amount per ionic step, with an OSZICAR written in VASP's MD-line
layout.  The builders only aim to reproduce the *file-level* signatures of
real NiO Step1 runs (interrupted, runaway, startup transient, ramp, repaired)
so the recovery logic can be regression tested without VASP.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

LATTICE = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 20.0))
H_START = 0.10
H_STEP = 0.0005  # fractional x per ionic step -> 0.005 A per step on a 10 A axis
O_POSITION = (0.20, 0.20, 0.50)

Series = float | Sequence[float | None] | Callable[[int], float | None] | None


def h_position(step: int) -> float:
    """Fractional x of the H ion after ``step`` ionic steps (step 0 = POSCAR)."""

    return H_START + H_STEP * step


def _value(series: Series, step: int, default: float | None) -> float | None:
    if series is None:
        return default
    if callable(series):
        return series(step)
    if isinstance(series, (int, float)):
        return float(series)
    return series[step - 1]


def _header(comment: str) -> str:
    rows = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in LATTICE)
    return f"{comment}\n1.0\n{rows}\nH O\n1 1\n"


def poscar_text(step: int = 0, *, comment: str = "step1 fixture", velocities: bool = False) -> str:
    text = (
        _header(comment)
        + "Selective dynamics\nDirect\n"
        + f"{h_position(step):.8f} 0.10000000 0.50000000 T T T\n"
        + f"{O_POSITION[0]:.8f} {O_POSITION[1]:.8f} {O_POSITION[2]:.8f} F F F\n"
    )
    if velocities:
        text += "\n  0.01000000  0.00000000  0.00000000\n  0.00000000  0.00000000  0.00000000\n"
    return text


def contcar_text(step: int, *, velocities: bool = True, h_override: float | None = None) -> str:
    """A VASP-like MD CONTCAR after ``step`` completed ionic steps.

    VASP writes the positions for the *next* step, so the H ion sits at
    ``h_position(step + 1)``; ``h_override`` fabricates a CONTCAR that does
    not match the trajectory.
    """

    x = h_position(step + 1) if h_override is None else h_override
    text = (
        _header("step1 fixture CONTCAR")
        + "Selective dynamics\nDirect\n"
        + f"{x:.8f} 0.10000000 0.50000000 T T T\n"
        + f"{O_POSITION[0]:.8f} {O_POSITION[1]:.8f} {O_POSITION[2]:.8f} F F F\n"
    )
    if velocities:
        text += "\n  0.01234567  0.00000000  0.00000000\n  0.00000000  0.00000000  0.00000000\n"
    return text


def xdatcar_text(n_steps: int, *, nblock: int = 4) -> str:
    """XDATCAR with one Direct frame every ``nblock`` ionic steps up to ``n_steps``."""

    frames = []
    for index, step in enumerate(range(nblock, n_steps + 1, nblock), start=1):
        frames.append(
            f"Direct configuration= {index:6d}\n"
            f"  {h_position(step):.8f}  0.10000000  0.50000000\n"
            f"  {O_POSITION[0]:.8f}  {O_POSITION[1]:.8f}  {O_POSITION[2]:.8f}\n"
        )
    return _header("step1 fixture") + "".join(frames)


def oszicar_text(
    n_steps: int,
    *,
    temperatures: Series = None,
    energies: Series = None,
    scf_iterations: int | Sequence[int] | Callable[[int], int] = 3,
    partial_scf_tail: int = 0,
) -> str:
    """VASP MD OSZICAR: electronic lines then one ``N T= ... F= ...`` line per step.

    ``None`` in ``temperatures``/``energies`` renders as VASP's ``******``
    overflow marker.  ``partial_scf_tail`` appends electronic lines with no MD
    line, as when a job is killed in the middle of an SCF cycle.
    """

    lines: list[str] = []
    for step in range(1, n_steps + 1):
        if callable(scf_iterations):
            iterations = scf_iterations(step)
        elif isinstance(scf_iterations, int):
            iterations = scf_iterations
        else:
            iterations = scf_iterations[step - 1]
        for electronic in range(1, iterations + 1):
            lines.append(f"RMM: {electronic:3d}   -0.100000E+02   -0.1E-04   -0.1E-04    10   0.1E-03\n")
        temperature = _value(temperatures, step, 300.0)
        energy = _value(energies, step, -10.0)
        t_text = "******" if temperature is None else f"{temperature:7.1f}"
        f_text = "****************" if energy is None else f"{energy:.8E}"
        e_text = "****************" if energy is None else f"{energy + 1.0:.8E}"
        lines.append(
            f"{step:5d} T= {t_text} E= {e_text} F= {f_text} E0= {f_text} EK= 0.10000E+01 SP= 0.00E+00 SK= 0.00E+00\n"
        )
    for electronic in range(1, partial_scf_tail + 1):
        lines.append(f"RMM: {electronic:3d}   -0.100000E+02   -0.1E-04   -0.1E-04    10   0.1E-03\n")
    return "".join(lines)


def incar_text(
    *,
    nsw: int = 400,
    potim: float = 1.0,
    tebeg: float = 300.0,
    teend: float | None = 300.0,
    nblock: int = 4,
    nelm: int = 60,
    algo: str = "Fast",
    istart: int = 1,
    smass: int | None = -1,
    extra: Mapping[str, Any] | None = None,
) -> str:
    tags: dict[str, Any] = {
        "SYSTEM": "Step1_preheat_fixture",
        "ISTART": istart,
        "ENCUT": 400,
        "PREC": "Normal",
        "EDIFF": "1E-4",
        "NELM": nelm,
        "ALGO": algo,
        "ISPIN": 2,
        "LDAU": ".TRUE.",
        "LDAUU": "4.6 0.0",
        "MAGMOM": "1*0.0 1*0.0",
        "IBRION": 0,
        "NSW": nsw,
        "POTIM": f"{potim:g}",
        "NBLOCK": nblock,
        "TEBEG": f"{tebeg:g}",
    }
    if teend is not None:
        tags["TEEND"] = f"{teend:g}"
    if smass is not None:
        tags["SMASS"] = smass
    tags.update(extra or {})
    return "".join(f"{key} = {value}\n" for key, value in tags.items())


PLAIN_LAUNCHER = "#!/bin/bash\n#SBATCH -N 1\nmodule load vasp\nsrun -n4 vasp_std\n"
FINISHED_OUTCAR = " General timing and accounting informations for this job:\n"
RUNNING_OUTCAR = " running output, no timing block yet\n"
ERROR_OUTCAR = " ZBRENT: fatal error in bracketing\n"


def set_age(path: Path, hours: float) -> None:
    """Backdate every regular file directly inside ``path`` (non-recursive)."""

    moment = time.time() - hours * 3600.0
    for item in path.iterdir():
        if item.is_file():
            os.utime(item, (moment, moment))


def write_step1_run(
    root: Path,
    name: str = "run",
    *,
    nsw: int = 400,
    potim: float = 1.0,
    tebeg: float = 300.0,
    teend: float | None = 300.0,
    nblock: int = 4,
    nelm: int = 60,
    algo: str = "Fast",
    istart: int = 1,
    smass: int | None = -1,
    incar_extra: Mapping[str, Any] | None = None,
    steps: int = 33,
    temperatures: Series = None,
    energies: Series = None,
    scf_iterations: int | Sequence[int] | Callable[[int], int] = 3,
    partial_scf_tail: int = 0,
    contcar: str | None = "good",
    xdatcar: bool = True,
    outcar: str | None = "running",
    launcher: str | None = "plain",
    wavecar: bool = False,
    age_hours: float = 10.0,
    poscar_velocities: bool = False,
) -> Path:
    """Create one synthetic Step1 run directory and return it.

    ``contcar``: ``"good"`` (matches the trajectory), ``"missing"``/``None``,
    ``"truncated"`` (header only), ``"far"`` (H displaced by 4 A), or
    ``"nan"`` (non-finite velocity block).  ``outcar``: ``"running"``,
    ``"finished"`` (VASP timing footer), ``"error"`` or ``None``.
    ``launcher``: ``"plain"``, ``"precondition"`` (INCAR.precondition +
    wrapped launcher, as step1-prepare --precondition writes), or ``None``.
    """

    from interfaceforge.vasp import build_precondition_incar, wrap_launcher_with_precondition

    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    incar = incar_text(
        nsw=nsw,
        potim=potim,
        tebeg=tebeg,
        teend=teend,
        nblock=nblock,
        nelm=nelm,
        algo=algo,
        istart=istart,
        smass=smass,
        extra=incar_extra,
    )
    (run / "INCAR").write_text(incar, encoding="utf-8")
    (run / "POSCAR").write_text(poscar_text(0, velocities=poscar_velocities), encoding="utf-8")
    (run / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (run / "POTCAR").write_text("fixture POTCAR H O\n", encoding="utf-8")
    if steps or partial_scf_tail:
        (run / "OSZICAR").write_text(
            oszicar_text(
                steps,
                temperatures=temperatures,
                energies=energies,
                scf_iterations=scf_iterations,
                partial_scf_tail=partial_scf_tail,
            ),
            encoding="utf-8",
        )
    if xdatcar and steps:
        (run / "XDATCAR").write_text(xdatcar_text(steps, nblock=nblock), encoding="utf-8")
    if contcar and contcar != "missing" and steps:
        if contcar == "good":
            text = contcar_text(steps)
        elif contcar == "truncated":
            text = _header("truncated CONTCAR")
        elif contcar == "far":
            text = contcar_text(steps, h_override=h_position(steps) + 0.4)
        elif contcar == "nan":
            text = contcar_text(steps, velocities=False) + "\n  nan  0.0  0.0\n  0.0  0.0  0.0\n"
        else:
            raise ValueError(contcar)
        (run / "CONTCAR").write_text(text, encoding="utf-8")
    if outcar == "running":
        (run / "OUTCAR").write_text(RUNNING_OUTCAR, encoding="utf-8")
    elif outcar == "finished":
        (run / "OUTCAR").write_text(FINISHED_OUTCAR, encoding="utf-8")
    elif outcar == "error":
        (run / "OUTCAR").write_text(ERROR_OUTCAR, encoding="utf-8")
    elif outcar is not None:
        raise ValueError(outcar)
    if launcher == "plain":
        (run / "runvasp.sh").write_text(PLAIN_LAUNCHER, encoding="utf-8")
    elif launcher == "precondition":
        (run / "runvasp.sh").write_text(
            wrap_launcher_with_precondition(PLAIN_LAUNCHER, launcher_name="runvasp.sh"), encoding="utf-8"
        )
        (run / "INCAR.precondition").write_text(
            build_precondition_incar(incar, system="Step1_preheat_fixture_precondition"), encoding="utf-8"
        )
    elif launcher is not None:
        raise ValueError(launcher)
    if wavecar:
        (run / "WAVECAR").write_bytes(b"fixture WAVECAR" * 8)
    set_age(run, age_hours)
    return run


def linear_ramp(t0: float, t1: float, n: int) -> Callable[[int], float]:
    """Temperature that follows a VASP TEBEG->TEEND ramp exactly, step 1..n."""

    return lambda step: t0 + (t1 - t0) * step / n


def write_manifest(step1_root: Path, runs: Iterable[Path]) -> Path:
    """A step1_manifest.json listing ``runs`` as step1-prepare would (hashes of INCAR/POSCAR)."""

    rows = []
    for run in runs:
        rows.append(
            {
                "relative_path": run.relative_to(step1_root).as_posix(),
                "step1_incar_sha256": _sha256(run / "INCAR"),
                "step1_poscar_sha256": _sha256(run / "POSCAR"),
            }
        )
    path = step1_root / "step1_manifest.json"
    path.write_text(json.dumps({"format": "interfaceforge-step1-series", "runs": rows}, indent=2), encoding="utf-8")
    return path


def write_legacy_repair_record(run: Path, **fields: Any) -> Path:
    """A schema-1 step1_repair.json exactly as InterfaceForge wrote it before generations existed."""

    record: dict[str, Any] = {
        "format": "interfaceforge-step1-repair",
        "schema_version": 1,
        "run": str(run),
        "status": "PREPARED",
        "safe_prefix_steps": 12,
        "safe_segment_steps": 12,
        "previous_safe_prefix_steps": 0,
        "rewind_frame": 3,
        "original_nsw": 400,
        "repair_nsw": 388,
        "original_potim_fs": 1.0,
        "repair_potim_fs": 0.5,
        "repair_algo": "Normal",
        "repair_electronic": {"EDIFF": "1E-5", "NELM": "120", "NELMIN": "6"},
        "repair_langevin_gamma": None,
        "repair_ramp_from_k": 100.0,
        "repair_precondition": True,
        "source": "XDATCAR",
        "archive": str(run / ".interfaceforge" / "archive" / "step1_repair_20260901T000000Z"),
    }
    record.update(fields)
    path = run / "step1_repair.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_legacy_launch_ledger(directory: Path, rows: Sequence[Mapping[str, Any]], *, age_hours: float = 0.0) -> Path:
    """A schema-1 step1_launch.json (the file whose historical SUBMITTED row blocked relaunch)."""

    payload = {
        "format": "interfaceforge-step1-launch",
        "schema_version": 1,
        "status": "SUBMITTED",
        "root": str(directory),
        "preflight": "PASS",
        "runs": [dict(row) for row in rows],
    }
    path = directory / "step1_launch.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if age_hours:
        moment = time.time() - age_hours * 3600.0
        os.utime(path, (moment, moment))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    """Every file and directory below ``root`` with size, mtime_ns and content hash.

    Two equal snapshots prove a command performed zero mutation.
    """

    snapshot: dict[str, tuple[Any, ...]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            snapshot[relative + "/"] = ("dir",)
        else:
            stat = path.stat()
            snapshot[relative] = (stat.st_size, stat.st_mtime_ns, _sha256(path))
    return snapshot


def fake_guard(active: Mapping[str | Path, str] | None = None, *, verified: bool = True) -> Any:
    """A SchedulerGuard whose snapshots always list ``active`` ({run_dir: STATE})."""

    from interfaceforge.step1_scheduler import SchedulerGuard

    jobs = [
        {"job_id": str(9000 + index), "state": state, "workdir": str(path)}
        for index, (path, state) in enumerate((active or {}).items())
    ]
    return SchedulerGuard.fixed(jobs, verified=verified)


def sequence_guard(snapshots: Sequence[Mapping[str | Path, str]], *, verified: bool = True) -> Any:
    """A SchedulerGuard returning a new snapshot per refresh (last one repeats).

    Used to simulate a job that becomes active between planning and mutation.
    """

    from interfaceforge.step1_scheduler import SchedulerGuard, SchedulerSnapshot

    state = {"index": 0}

    def factory() -> SchedulerSnapshot:
        index = min(state["index"], len(snapshots) - 1)
        state["index"] += 1
        jobs = [
            {
                "job_id": str(9100 + number),
                "state": job_state,
                "workdir": str(path),
                "workdir_real": os.path.realpath(str(path)),
            }
            for number, (path, job_state) in enumerate(snapshots[index].items())
        ]
        return SchedulerSnapshot(
            requested="slurm",
            mode="slurm",
            verified=verified,
            reason=f"fixture snapshot {index}",
            taken_at="2026-09-23T00:00:00+00:00",
            jobs=jobs,
            monotonic=time.monotonic(),
        )

    return SchedulerGuard("slurm", recheck_seconds=0.0, snapshot_factory=factory)


@contextmanager
def mock_sbatch(first_job_id: int = 5001) -> Iterator[list[tuple[list[str], str]]]:
    """Patch sbatch (``interfaceforge.vasp.subprocess.run``); yields the recorded calls."""

    calls: list[tuple[list[str], str]] = []

    def fake(command: Sequence[str], cwd: Any = None, **_kwargs: Any) -> Mock:
        calls.append((list(command), str(cwd)))
        result = Mock()
        result.stdout = f"Submitted batch job {first_job_id + len(calls) - 1}\n"
        return result

    with patch("interfaceforge.vasp.subprocess.run", side_effect=fake):
        yield calls
