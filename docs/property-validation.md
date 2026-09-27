# Bulk-property validation (MatCalc)

`iface properties run` checks whether a **trained** MACE or DeePMD/DPA model
reproduces physical bulk properties, not just DFT energies and forces. It runs
[MatCalc](https://github.com/materialsvirtuallab/matcalc) calculators on the
local trained model, one committee member at a time, and writes a single
machine-readable `properties.json` with full provenance.

This is a post-training validation layer. It does not train, modify, or
download any model, and it never submits jobs.

## Where it sits

```text
DFT labels
    ↓
energy / force / stress error          (iface mlip-compare, stratified validation)
    ↓
derivative-sensitive probes            (committee spread, separation curves, ...)
    ↓
bulk physical-property validation      (iface properties run)   ← this page
```

A low force RMSE does not guarantee:

- a good equation of state (equilibrium volume, bulk modulus, B');
- the correct elastic response (C_ij, shear and Young's moduli);
- dynamically stable phonons (no spurious imaginary modes);
- transferable material behaviour away from the training distribution.

These properties depend on first and second derivatives of the PES over
strains and displacements that force-matched training may never have sampled.
Checking them on the bulk end members (e.g. TiN and β-Si₃N₄ for the SiN/TiN
campaign) is a cheap check to run before trusting a model at an interface.

## Install

MatCalc is optional and **requires Python ≥ 3.11**. The InterfaceForge core,
and every other command, still supports Python 3.10.

```bash
pip install -e '.[properties]'
```

The `properties` extra is deliberately **not** part of `[all]`, so
`pip install -e '.[all]'` keeps working on Python 3.10. Install the MLIP
backend itself separately (`mace-torch`, or `deepmd-kit` from your cluster
module). On Python 3.10, requesting the extra fails at install time. Running
`iface properties run` without MatCalc raises a `DependencyError` that tells
you what to install. Importing `interfaceforge` never imports MatCalc,
pymatgen, phonopy, MACE, DeePMD, or torch.

> **Known packaging issue (Sept 2026):** phonopy 4.5.0 with its Rust backend
> `phonors` 0.5.0 fails inside phonopy (`module 'phonors' has no attribute
> 'grid_index_from_address'`). `pip install 'phonors<0.5'` fixes it. The
> failure is confined to the phonon property of each member and is recorded
> in the JSON.

## Usage

MACE committee, all four properties (SiN/TiN example):

```bash
iface properties run TiN.vasp \
    --engine mace \
    --model seed_11.model \
    --model seed_23.model \
    --model seed_37.model \
    --model seed_53.model \
    --property relax \
    --property eos \
    --property elasticity \
    --property phonon \
    --output properties/TiN_mace
```

The same committee from a bundle collected by `iface committee collect`
(verified against its checksums before use; pass the extracted directory,
not the `.zip`):

```bash
iface properties run TiN.vasp --engine mace --committee committees/mace \
    --property eos --property elasticity --output properties/TiN_mace
```

A frozen DeePMD/DPA model:

```bash
iface properties run TiN.vasp \
    --engine deepmd \
    --model models/deepmd/dpa2/model_000/frozen_model.pth \
    --property relax \
    --property eos \
    --property elasticity \
    --property phonon \
    --output properties/TiN_dpa2
```

| Option | Meaning |
|---|---|
| `--engine {mace,deepmd}` | Backend used to load each model as an ASE calculator |
| `--model PATH` | Trained model file; repeat once per committee member |
| `--committee DIR` | Collected committee bundle (alternative to `--model`) |
| `--property NAME` | Repeatable: `relax`, `eos`, `elasticity`, `phonon` |
| `--output DIR` | Must be empty or absent; `--force` replaces only a prior result directory (one containing `properties.json`) |
| `--device` | MACE torch device (default `cpu`) |
| `--dtype {float32,float64}` | MACE precision (default `float64`); rejected for DeePMD |
| `--fmax` | Relaxation criterion in eV/Å (default 0.01) |
| `--max-steps` | Relaxation step cap (default 500) |
| `--optimizer {FIRE,BFGS,LBFGS}` | ASE optimizer (default FIRE) |
| `--phonon-min-length` | Minimum phonon supercell edge in Å (MatCalc default 20) |

Exit status is 0 when every member succeeded, 1 when at least one member
failed (the JSON is still written), and 2 for usage/safety errors.

### Precision

MACE runs in the requested `--dtype`; float64 is the default because
elasticity and phonons are derivative-sensitive. The record also stores the
checkpoint's own `model_native_dtype` and `dtype_converted`: MACE silently
upcasts a float32-trained model to float64, which changes the arithmetic but
not the precision the weights were trained at. A frozen DeePMD model's
precision is fixed at training/freeze time, and the ASE calculator cannot
change it, so `--dtype` is refused for `--engine deepmd` instead of being
silently ignored. `deepmd.calculator.DP` also takes no device argument. It
uses CUDA when visible, which the record notes.

## Scientific behaviour

**Member by member.** Every property is computed independently for every
committee member. There is no averaged-force calculator:
`property(mean PES) ≠ mean(property(member PES))` for nonlinear properties.
Members are reported first and summarized second.

**Bulk only.** All four properties require a fully 3D-periodic cell. Inputs
with any non-periodic direction are rejected with a `SafetyError`, as are
3D-periodic cells with an empty gap wider than 6 Å along any lattice vector
(a slab with vacuum). A "bulk modulus" of a slab is meaningless. `relax` is
held to the same guard in this first version, because it relaxes the cell.

**Stress is checked first.** Every property here relaxes or strains the cell.
Each member's calculator must declare `stress` and return a finite stress
tensor on the input structure before anything runs; otherwise that member
fails with a clear error.

**One convergence-checked relaxation feeds everything.** Each member first
runs a full atoms+cell relaxation (`RelaxCalc`, `FrechetCellFilter`). It
counts as converged only if max|F| ≤ `fmax` **and** the optimizer stopped
before `max_steps`. MatCalc's own `is_converged` checks atomic forces only,
so a run that hits the step cap with an unconverged cell would otherwise
pass. An unconverged relaxation fails that member, and no EOS, elastic, or
phonon numbers are produced. EOS, elasticity, and phonons then run with
`relax_structure=False` on that relaxed structure, which is written to
`members/<label>/relaxed.vasp` with its SHA-256.

**EOS.** MatCalc's Birch–Murnaghan scan (11 points, ±10 % linear strain,
shape relaxed at fixed volume). MatCalc reports `bulk_modulus_bm`. E₀, V₀,
and B′ come from the same fit via pymatgen. E₀ is a fit value, not the
relaxed energy (compare `relax.energy`). The strained-point relaxations
inside the scan are not convergence-checked by MatCalc, and the record says
so (`strained_relaxations_convergence_checked: false`).

**Elasticity.** Full 6×6 Voigt tensor in GPa (pymatgen order xx, yy, zz,
yz, xz, xy), plus VRH bulk and shear moduli and Young's modulus.

**Phonons.** phonopy finite displacements through MatCalc. The record keeps
mesh frequencies (THz), the minimum frequency, the number of modes below
−0.01 THz, a `dynamically_stable` flag, the supercell matrix, thermal
properties, and `members/<label>/phonon.yaml`. MatCalc writes `phonon.yaml`
into the working directory by default; here it always goes to the member
directory.

## Output

```text
properties/TiN_mace/
├── properties.json
└── members/
    └── member_000_seed_11/
        ├── relax.traj        # relaxation trajectory (step count source)
        ├── relaxed.vasp      # the structure every downstream property used
        └── phonon.yaml       # when phonons were requested
```

`properties.json` (schema version 1, `artifact_type:
interfaceforge_property_validation`) contains:

- `structure`: source path, SHA-256, formula, atom count, cell, PBC, and
  the bulk-guard result;
- `engine`, `properties_requested`, `settings`, and `packages` (Python,
  InterfaceForge, MatCalc, ASE, pymatgen, NumPy, the MLIP backend and torch,
  and phonopy when phonons were run);
- `models[]`, one per member: model path and SHA-256, source (`--model` or
  `committee:<dir>`), seed when recoverable (from `seed_N` in the path, or
  from the committee manifest), device, dtype, native dtype, calculator
  class, `status`, `error`, `failed_property`, per-property `results` with an
  explicit `units` map, and package versions;
- `summary`: `success_fraction` (e.g. `3/4`), `failed_members` with their
  errors, and per-property statistics over successful members only;
- `inputs_unchanged`: the structure and model files are re-hashed after the
  run. Any change is a `SafetyError`.

The output is built in a temporary sibling directory and renamed into place
only when complete.

### Committee summary

- Scalars (E₀, V₀, B, B′, R², lattice parameters, VRH moduli, minimum phonon
  frequency): mean, sample standard deviation (ddof = 1; `null` for one
  member), min, max, n.
- Elastic tensor: every member's full tensor is kept. The elementwise
  mean/std is given only when all shapes and conventions match; otherwise the
  entry says `aggregated: false` and why.
- Phonons: member frequencies are always kept. Mean/std arrays are given only
  when the arrays align exactly; otherwise `aggregated: false`. There is no
  scalar "phonon score". The summary reports how many members are
  dynamically stable.
- Failed members are never hidden: a 3/4 committee reports `3/4`, names the
  failed member, and records its error.

Because MACE and DeePMD runs produce the same normalized schema, a later
`iface properties compare` can put a MACE committee, DPA-2, a fine-tuned
DPA-2, and DPA-3 side by side on the same structure. That comparison command
is not implemented yet.

## Deliberately out of scope

- **Interfaces.** MatCalc's generic `InterfaceCalc` does **not** replace
  InterfaceForge's interface energetics ([interface energy](interface-energy.md),
  [separation energy](separation-energy.md)), finite-temperature treatment,
  or work of adhesion. For polar Si₃N₄/TiN interfaces, bulk-referenced
  interface energies are invalid anyway. Likewise, `OrderCalc` is not a
  substitute for [swap Monte Carlo](swap-mc.md).
- **Thermal conductivity.** `Phonon3Calc` / phono3py lattice thermal
  conductivity is deferred to the next milestone. No interface thermal
  conductance claims are made here.
- `SurfaceCalc`, `AdsorptionCalc`, `NEBCalc`, and MatCalc MD are not wired in.
- `matcalc.load_fp()` / universal foundation models are never used. Only the
  local trained model you pass is loaded.
