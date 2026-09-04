# Getting started

This guide covers installation and helps route a new calculation to the right
InterfaceForge workflow. Detailed options and recovery procedures live in the
topic guides linked below.

## Installation

For a broad local installation:

```bash
git clone https://github.com/lgutsev/InterfaceForge.git
cd InterfaceForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

Use a narrower extra when the environment has one job:

| Installation | Intended use |
|---|---|
| `pip install -e .` | Configuration, lightweight audits, and validation helpers |
| `pip install -e ".[vasp]"` | ASE-backed geometry and OUTCAR collection |
| `pip install -e ".[slab-align]"` | Slab-alignment plotting and SUMO-oriented analysis |
| `pip install -e ".[report]"` | Matplotlib/openpyxl reports |
| `pip install -e ".[mace-roi]"` | Experimental MACE-ROI training |
| `pip install -e ".[ai2kit]"` | AI2-Kit/oh-my-batch controller support |
| `pip install -e ".[intermat]"` | InterMat geometry generation |
| `pip install -e ".[dev]"` | Complete tests and development tools |

The exploration GPU environment for AI2-Kit must additionally provide MACE,
OpenMM, and OpenMM-ML. Licensed VASP executables and pseudopotentials are never
distributed with InterfaceForge.

## Create a campaign

```bash
iface init my-interface
cd my-interface
```

Edit `campaign.yaml`, choose or customize a scheduler profile, and add the
structures and explicit VASP settings required by the campaign. Then validate
the graph before generating anything:

```bash
iface plan
iface prepare
iface submit                                      # dry-run
iface submit --system interface_ab_300k --stage train --execute
```

The campaign schema and directory/state model are described in
[Campaign format](campaign.md).

## Choose the next workflow

### Existing VASP trajectories → MLIP datasets

```bash
iface audit runs/vasp
iface collect
iface train mace
iface train deepmd
```

When reference trajectories are scattered across unrelated roots, use the
[mapped leaf-campaign workflow](../examples/mapped-leaf-campaign/README.md) to
create synchronized extxyz and DeePMD datasets with a visual cross-audit.

### Generated slabs → optimized structures → AIMD

Use the guarded VASP sequence:

```bash
iface vasp opt-prepare OPT --manifest OPT/manifest_batch.csv --dry-run
iface vasp opt-prepare OPT --manifest OPT/manifest_batch.csv
iface vasp opt-launch OPT                         # dry-run
iface vasp opt-launch OPT --execute

iface vasp step1-prepare OPT --protocol training --dry-run
iface vasp step1-prepare OPT --protocol training
iface vasp step1-launch Step1                     # dry-run
iface vasp step1-launch Step1 --execute

iface vasp step2-prepare Step1 --temperatures 300 450 600 --protocol training --inherit-wavecar
iface vasp step2-launch Step2_300K Step2_450K Step2_600K
iface vasp step2-launch Step2_300K Step2_450K Step2_600K --execute
```

Reactive or proton-rich Step1 inputs can use the conservative preparation,
status, and rewind/repair path. See [VASP workflows](vasp.md) for the precise
WAVECAR policy, preconditioning, thermostat options, failure guards, and
archive behavior.

### Monitor MLIP committees

```bash
iface mlip-progress .
iface mlip-progress . --json
```

The command reads generated artifacts without touching running jobs. It rolls
up MACE epochs/RMSE, DeePMD steps/RMSE/checkpoint/freeze state, evaluation
completion, and comparison status. Scheduler state still comes from `squeue`
or `sacct`.

### Compare trained MACE and DeePMD committees

```bash
iface mlip-compare prepare --deepmd-arch dpa2 --force
sbatch audit/mlip_compare/run_mace_evaluate.slurm
iface mlip-compare status \
  --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
iface mlip-compare finalize \
  --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
```

Use a distinct `--output-root` for every architecture. The exact-frame proof,
metrics, calibration, and publication outputs are documented in
[Matched MLIP comparison](mlip-comparison.md).

### Build reactive magnetic surfaces

```bash
iface surface init -o surface_campaign.yaml
iface surface plan surface_campaign.yaml
iface surface build surface_campaign.yaml
iface surface audit generated_surface_campaign -o surface_audit.csv
```

See [Reactive surfaces](reactive-surfaces.md) before interpreting generated
terminations, AFM order, proton-transfer states, or docking geometries.

### Archive or package a trained committee

```bash
iface committee collect mace_committee stored_models/tin_sin_mace_v1.zip \
  --expected-members 4
iface committee collect models/deepmd/dpa2 stored_models/tin_sin_dpa2_v1 \
  --engine deepmd --expected-members 4
iface committee verify stored_models/tin_sin_mace_v1
```

Dataset backup archives and Hugging Face repository generation are covered in
[Packaging](packaging.md).

## Safety model

- Commands that submit jobs or apply supported in-place changes require an
  explicit `--execute`.
- Preparation manifests hash scientific inputs; launch commands recheck them.
- Existing output trees are not silently replaced.
- Recovery paths preserve or archive the preceding state.
- POTCAR comes only from an explicitly configured licensed source and is
  excluded from portable archives.

## Development check

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

Passing tests establishes encoded software behavior, not physical validity.
Consult [Verification and maturity](verification.md) before relying on a
feature in a production scientific campaign.
