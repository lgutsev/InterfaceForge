# LONI AI2-Kit controller environment setup

The batch job in this directory creates or validates the small, process-isolated
AI2-Kit controller environment used by the InterfaceForge adapter:

```text
/project/lgutsev/env/iface_ai2kit_controller
```

## Status

On 2026-08-17, job 968863 created the Python 3.11 controller environment on
LONI and a follow-up isolated validation confirmed `pip check`,
`ai2-kit==1.0.9`, `oh-my-batch==0.7.5`, and both command-line interfaces.
Environment creation is therefore human-tested on LONI. No end-to-end
active-learning campaign has yet completed through exploration, VASP labeling,
dataset import, and retraining.

## Submit

From LONI:

```bash
sbatch examples/ai2kit/setup_iface_ai2kit_controller.sh
```

The checked-in header uses account `loni_perovsk27` and partition `single`.
Change only the `#SBATCH` account or partition fields if the allocation naming
changes.

Inspect the result with:

```bash
squeue --me --name=setup_ai2kit
tail -n 100 setup_ai2kit_*.out
tail -n 100 setup_ai2kit_*.err
```

## Behavior

The job is unattended:

1. It validates an existing target environment by checking Python, exact
   `ai2-kit==1.0.9` and `oh-my-batch==0.7.5` versions, both command-line
   interfaces, and `pip check`.
2. If validation succeeds, it exits without changing the environment.
3. If validation fails, it renames the existing directory to a timestamped
   `.incomplete-YYYYmmdd-HHMMSS` backup instead of deleting it.
4. It creates a Python 3.11 environment from conda-forge, disables Conda's
   offline flag for the job, suppresses interactive prompts, installs the pinned
   controller packages, and runs the same validation again.

A timestamped incomplete backup is intentionally retained until the new
environment has been inspected. It may be deleted manually afterward if it is
no longer useful.

`CONDA_OFFLINE=false` corrects an accidentally enabled Conda offline setting;
it cannot provide network access when a node is genuinely disconnected.

The script also clears inherited `PYTHONPATH`/`PYTHONHOME` and sets
`PYTHONNOUSERSITE=1`. This is required on the tested LONI account because an
otherwise clean controller prefix could see unrelated user-site packages such
as phonopy, sumo, and symfc, causing a false failure from `pip check`.

## Execution boundary

This Slurm job only builds the environment. It does **not** run the active-learning
controller, submit nested jobs, modify the MACE runtime, or label/retrain data.

Run `iface active-learning ai2kit ... --execute` later from a LONI login or
other permitted service host, never from inside a Slurm allocation. InterfaceForge
intentionally refuses nested controller execution when `SLURM_JOB_ID` is set.

The controller prefix supplies the explicit `controller_python`, `ai2-kit`,
and `omb` executables. Continue to invoke the `iface` command from the
InterfaceForge installation or editable checkout described in
[`docs/ai2kit.md`](../../docs/ai2kit.md).
