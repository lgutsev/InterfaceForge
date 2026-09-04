# InterfaceForge

InterfaceForge is a research workflow toolkit for reproducible atomistic
interface calculations. It connects audited VASP and VASP-MLFF campaigns to
synchronized MACE/DeePMD datasets, committee training, model comparison, and
interface-specific validation.

The package is review-first: preparation and submission are separate,
submission is dry-run by default, generated inputs are audited and hashed, and
questionable outputs are flagged instead of silently repaired or relaunched.

> InterfaceForge is an alpha research code. Human-tested workflows are clearly
> distinguished from automated-test-only and experimental features in the
> [verification matrix](docs/verification.md).

## Capabilities

| Area | What InterfaceForge provides |
|---|---|
| VASP campaigns | Input preparation, OPT → Step1 → Step2 AIMD staging, guarded Slurm launch, live status, failed-run repair, VASP-MLFF audit/recovery, slab vacuum and band-alignment analysis |
| Datasets | Streaming OUTCAR collection, synchronized MACE extxyz and DeePMD NPY layouts, provenance, leakage-aware splitting, mapped multi-root campaigns |
| MLIPs | MACE and DeePMD committee generation/evaluation, live progress summaries, matched-frame cross-backend comparison, Allegro scaffolding, optional MACE-ROI training |
| Interface validation | Work of adhesion, rigid separation curves, separation energy, stratified errors, committee uncertainty and publication-oriented summaries |
| Surface chemistry | AFM-compatible reactive oxide cells, hydroxylation/proton-transfer states, phosphonate docking, VASP export and post-relaxation audits |
| Portability and adapters | Checksummed model/dataset archives, Hugging Face repository packaging, AI2-Kit active learning, InterMat and RegFGW adapters |

## Install

```bash
git clone https://github.com/lgutsev/InterfaceForge.git
cd InterfaceForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

Minimal and feature-specific installations are described in
[Getting started](docs/getting-started.md).

## Quick start

Create and inspect a campaign:

```bash
iface init my-interface
cd my-interface
# Edit campaign.yaml and the selected scheduler profile; add structures/inputs.
iface plan
iface prepare
iface submit                                      # dry-run
iface submit --system interface_ab_300k --stage train --execute
```

Once reference trajectories exist:

```bash
iface audit runs/vasp
iface collect
iface train mace
iface train deepmd
iface mlip-progress .
```

`--execute` is required for commands that submit jobs or apply supported
in-place changes. Inspect every generated input and begin with a small smoke
run for each engine, version, chemistry, and cluster environment.

## Common workflows

| Goal | Start here |
|---|---|
| Install, configure, or choose a workflow | [Getting started](docs/getting-started.md) |
| Prepare, audit, launch, monitor, or repair VASP runs | [VASP workflows](docs/vasp.md) |
| Configure and evaluate DeePMD/DPA committees | [DeePMD campaigns](docs/deepmd.md) |
| Compare MACE and DeePMD on identical frames | [MLIP comparison](docs/mlip-comparison.md) |
| Archive committees/datasets or package for Hugging Face | [Packaging](docs/packaging.md) |
| Build reactive magnetic oxide campaigns | [Reactive surfaces](docs/reactive-surfaces.md) |
| Validate adhesion or separation energetics | [Interface energy](docs/interface-energy.md) and [separation energy](docs/separation-energy.md) |

The complete topic index is in **[Documentation](docs/README.md)**.

## Core scientific and safety rules

- Submission is a dry-run unless `--execute` is present.
- Canonical labels are immutable; derived data and imported active-learning
  labels remain staged until reviewed.
- Reference forces retain frozen atoms; mobility is stored separately.
- Dataset splits guard against trajectory leakage and preserve source identity.
- INCAR settings such as ENCUT, spin, dispersion, convergence, and Hubbard
  terms are inherited or explicitly configured rather than silently guessed.
- POTCAR is assembled only from an explicit licensed local source and is never
  included in portable archives.
- Recovery operations archive the preceding state and never treat a generated
  suggestion as scientific validation.

## Verification status

Human-tested paths currently include VASP-MLFF campaign operations, VASP data
generation, four-member MACE training/evaluation, PyTorch DPA-2
training/continuation/freezing/evaluation, and live `iface mlip-progress`
monitoring on the periodic SiN/TiN/TiO campaign.

That evidence is specific to the tested chemistry, architecture, runtime, and
LONI environment. Other DPA architectures, foundation-model fine-tuning,
MACE-ROI, Allegro, LAMMPS deployment, active-learning loops, and optional
interface-generation adapters retain their documented narrower status. See
[Verification and maturity](docs/verification.md).

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations.
