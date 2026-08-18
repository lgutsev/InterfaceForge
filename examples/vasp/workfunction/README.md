# Work-function plotting for the LONI VASP workflow

This directory holds a standalone, unattended work-function plotter for
surface-based VASP optimizations, intended to live outside InterfaceForge on
the compute side (e.g. `/home/$USER/bin` on LONI) and run automatically at
the end of `runvasp.sh`.

There are two ways to get a work-function plot from a finished run. Use
whichever matches where you are running from:

| Situation | Use |
|---|---|
| Unattended, at the end of a LONI Slurm job, no InterfaceForge install on the compute node | `plot_workfunc.py` in this directory |
| Interactive, from a controller/analysis environment with InterfaceForge (and ASE + matplotlib) installed | `iface vasp workfunction LOCPOT OUTCAR --plot-output workfunction.png` ([docs/vasp.md](../../../docs/vasp.md)) |

Both require the same INCAR tag and read the same LOCPOT/OUTCAR files; they
differ only in deployment, not in what they measure. The `iface` path is
additionally unit tested (`tests/test_workfunction.py`) and writes a JSON
summary alongside the plot; prefer it when it is available.

## Requirement: `LVHAR = .TRUE.`

VASP only writes the local **electrostatic potential** into `LOCPOT` when
`LVHAR = .TRUE.` is set. Without it, `LOCPOT` holds the charge density
instead, and a work-function plot computed from it looks plausible but is
wrong. `plot_workfunc.py` checks the run's `INCAR` for this tag before
plotting and refuses to run if it cannot confirm it (override with
`--allow-missing-lvhar` if you are certain the LOCPOT is already the
potential, e.g. it came from elsewhere).

To have InterfaceForge set this automatically when preparing a surface
static or relaxation INCAR:

```bash
iface vasp incar static INCAR --workfunction
iface vasp incar relax INCAR --workfunction
```

This adds `LVHAR = .TRUE.` without touching any other tag already present.

## Deploying `plot_workfunc.py` to LONI

```bash
mkdir -p ~/bin
cp examples/vasp/workfunction/plot_workfunc.py ~/bin/
chmod +x ~/bin/plot_workfunc.py
```

The script needs `ase`, `numpy`, and `matplotlib` importable by whichever
`python` runs it. On LONI that is typically the same environment already
used to run/post-process VASP jobs (e.g. the module-provided Python, or a
small conda environment), not the isolated AI2-Kit controller environment
described in [docs/ai2kit.md](../../../docs/ai2kit.md) — that environment is
deliberately minimal and unrelated to this script.

## Wiring into `runvasp.sh`

[launch_scripts/runvasp.sh](../../../launch_scripts/runvasp.sh) and
[launch_scripts/runvasp_bigmem.sh](../../../launch_scripts/runvasp_bigmem.sh)
call the plotter after VASP exits, guarded so it only fires for jobs that
actually requested a work function:

```bash
if grep -Eqi '^\s*LVHAR\s*=\s*\.?(TRUE|T)\.?' INCAR && [ -s LOCPOT ]; then
    python "$HOME/bin/plot_workfunc.py" --title "$SLURM_JOB_NAME" || true
fi
```

The `|| true` means a plotting failure (missing matplotlib, a malformed
LOCPOT, etc.) cannot flip the Slurm job's exit status or be mistaken for a
failed VASP run — check `Workfunction.png`/`locpot.dat` for that job
separately. For any other job (no `LVHAR = .TRUE.`, e.g. a bulk relaxation
or a training run), the guard is false and this block is a no-op.

If you maintain other launcher variants of your own, copy the same guarded
block into them; InterfaceForge does not generate `runvasp.sh` for you (see
[docs/vasp.md](../../../docs/vasp.md#one-command-mlff-continuation) — it is
only discovered and preferred as a launcher at submission time).

## Output

- `locpot.dat`: the raw planar-averaged potential, Fermi-shifted when OUTCAR
  is present.
- `Workfunction.png`: the plotted curve. Read the work function off the
  **flat vacuum plateau**, not necessarily the single maximum grid point —
  inspect the curve, especially for a slab with a nonzero dipole where a
  correction may be needed.
