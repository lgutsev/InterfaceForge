# InterfaceForge

InterfaceForge is a campaign operating system for heterogeneous-interface
potentials. It turns VASP and VASP-MLFF trajectories into one traceable dataset,
generates restartable MACE and DeePMD committees, selects new DFT labels, and
compares interface properties rather than stopping at a global force RMSE.

The project distills the reusable ideas in the original campaign templates into
a portable Python CLI. It does **not** contain structures, trajectories,
checkpoints, containers, cluster-specific private paths, or licensed VASP
pseudopotentials.

## What it covers

| Layer | InterfaceForge functionality |
|---|---|
| VASP preparation | static, relaxation, DFT-MD, DOS and line-band setup; geometry conversion, supercells, slabs, duplicate checks and Selective Dynamics |
| VASP-MLFF | train → refit → stability scaffolding, continuation/capacity recovery, mode-aware audits |
| Dataset | streaming OUTCAR collection, leakage-resistant splits, synchronized extxyz and DeePMD NPY layouts |
| MACE | two-stage energy/force training with restartable Slurm jobs |
| DeePMD | DPA-1, DPA-2, DPA-3 and experimental DPA-4 committees; TensorFlow/PyTorch backends; preflight → smoke → full → evaluation |
| Active learning | thermodynamic exploration matrix and uncertainty-plus-diversity labeling queue |
| Validation | parity metrics, work of adhesion with uncertainty propagation, rigid-separation curves |
| Provenance | manifests, hashes, append-only events, JSON/CSV/Markdown audits and a self-contained HTML report |

## Install

```bash
git clone https://github.com/lgutsev/InterfaceForge.git
cd InterfaceForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

For configuration, audits and validation only, `pip install -e .` is enough.
ASE-backed geometry and OUTCAR collection require `.[vasp]`.

## Ten-minute start

```bash
iface init my-interface
cd my-interface
# Edit campaign.yaml and profiles/loni.yaml; add structures and VASP inputs.
iface plan
iface prepare
iface submit                     # dry-run: lists jobs
iface submit --system interface_ab_300k --stage train --execute
```

After reference trajectories exist:

```bash
iface audit runs/vasp
iface collect
iface train mace
iface train deepmd
```

The generated DeePMD jobs have an intentional order:

```bash
sbatch models/deepmd/run_preflight.slurm
sbatch models/deepmd/run_smoke.slurm
# Inspect every smoke task before continuing.
sbatch models/deepmd/run_ensemble.slurm
sbatch models/deepmd/run_evaluate.slurm
```

DPA-4 stays labeled experimental. A trained checkpoint is not considered
deployable until freeze/export and the target LAMMPS path are independently
verified.

## Scientific invariants

- Reference forces are read with constraints unapplied. Frozen atoms remain
  represented in `REF_forces`; mobility is stored separately as `move_mask`.
- The default split assigns whole trajectories, so nearby MD frames do not leak
  across train/validation/test. A guarded contiguous-block strategy is also
  available.
- Generated INCAR presets do not guess ENCUT, k-point density, spin, dispersion,
  convergence thresholds or chemistry-specific settings.
- POTCAR is assembled only from an explicit licensed local tree and is excluded
  from portable archives.
- Submission is a dry-run unless `--execute` is present. Mutating recovery
  operations archive the current run first.

## Useful VASP commands

```bash
iface vasp incar md INCAR --temperature 600 --nsw 5000 --potim 1
iface vasp geom convert structure.xyz POSCAR --cell-from trusted.vasp
iface vasp geom slab bulk.vasp slab.vasp --miller 1 1 1 --layers 8
iface vasp geom freeze slab.vasp POSCAR --axis z --upper 5 --region inside
iface vasp restart run/ --clean-electronic
iface vasp recover continue run/ --temperature 450 --nsw 3000
iface vasp recover expand run/ --ml-mb 12000
iface vasp band scf/ bands/ --kpoints KPOINTS.line
iface vasp workfunction LOCPOT OUTCAR --plot-output workfunction.png
iface vasp pack campaign_outputs.zip --root runs/vasp
```

`recover expand` requires a recognized capacity failure in OUTCAR unless
`--force-expand` is explicitly supplied.

## Modern DeePMD configuration

```yaml
models:
  deepmd:
    enabled: true
    profile: deepmd_gpu
    backend: pytorch          # tensorflow, pytorch, or pt_expt
    architectures: [dpa2, dpa3]
    committee: 4
    seeds: [11, 23, 37, 53]
    numb_steps: 500000
    batch_atoms: 1024
    max_concurrent: 2
    container_image: /path/to/deepmd-kit.sif
```

Use `backend: tensorflow` for established DPA-1/`se_atten` workflows. DPA-2,
DPA-3 and DPA-4 require a PyTorch backend. `pt_expt` records that the
experimental PyTorch implementation is intended, while the generated DeePMD CLI
commands correctly use the public `--pt` switch.

See [the campaign format](docs/campaign.md),
[the VASP guide](docs/vasp.md), and
[the DeePMD guide](docs/deepmd.md). A complete editable configuration is in
[examples/interface-campaign/campaign.yaml](examples/interface-campaign/campaign.yaml).

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

InterfaceForge is an early research tool. Inspect generated inputs and smoke
test each engine/version combination before spending a large allocation.
