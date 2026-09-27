# Plan: virial fine-tuning and a bulk strain and rattle training campaign

**Status (2026-09-27):** in progress on branch `strain-training-campaign`,
worktree `.claude/worktrees/strain-training-campaign`. **Owner:** lgutsev.

This document is for anyone working on this: people and AI agents. Update
the checklist below when you finish a step. The scientific results that
motivated the plan are kept in the owner's private report repository. This
public repository deliberately gives only the engineering plan.

## Why

Bulk-property validation (`iface properties run`, see
[property-validation.md](../property-validation.md)) of a trained SiN/TiN/TiO
MACE committee showed that elastic constants are wrong in a way that does
not depend on the seed: off-diagonal C_ij are too small, shear is too
stiff, and the EOS degrades under large compression. The committee spread
did not flag any of this. Two causes were established:

1. **The training data has no strained cells.** It is fixed-cell AIMD only:
   `ISIF = 2`, and the lattice vectors never change. Second strain
   derivatives, which is what C_ij are, are therefore never sampled.
2. **Stress labels were collected but not trained on.** The extxyz files
   carry `REF_virial`. However, every MACE launcher
   (`launch_scripts/mace_train_committee.sh`,
   `mace_finetune_committee.sh`, `run_mace_gpu2_nomask_v2.sh`) uses
   `--loss weighted` with only energy and force keys. The DeePMD template in
   `src/interfaceforge/training.py` sets `start_pref_v = limit_pref_v = 0`.
   A spot check found that the model's frame-to-frame stress error is as
   large as the DFT stress variation.

## Approach: three independent tracks

The tracks don't compete for resources. The laptop builds and validates,
cluster GPUs fine-tune, and cluster CPUs run DFT. Models are **fine-tuned
from the existing seeds**, not retrained from scratch.

### Track A: fine-tuning with virials, on the existing data

- [ ] `launch_scripts/mace_finetune_committee.sh`: add opt-in virial
      training.
  - New env vars: `MACE_LOSS` (default `weighted`, so current behavior is
    unchanged), `MACE_VIRIALS_KEY` / `MACE_STRESS_KEY`, and their weights.
  - Add flags only after probing `mace_run_train --help`, using the
    existing grep pattern, and fail loudly if a requested flag is missing.
  - Add a `MACE_PT_VALID_FILE` pass-through for replay data.
- [ ] Add the same opt-in to `launch_scripts/mace_train_committee.sh`.
- [ ] Warn when the training data has `REF_virial` but no virial loss is
      requested. Do this in both launchers and in the MACE command builder
      in `training.py`.
- [ ] DeePMD: take `start_pref_v` / `limit_pref_v` from
      `models.deepmd.loss` (default 0) and record them in the manifest. The
      fine-tune path (`models.deepmd.finetune`, `dpa2_ft` / `dpa3_ft`)
      already exists.
- [ ] **Run A1 (cluster):** fine-tune each seed from its own `seed_N.model`
      (`MACE_FOUNDATION_MODEL`) with virials on, float64, about 20–30
      epochs at a lower learning rate.

### Track B: bulk strain and rattle DFT campaign

- [x] Branch `strain-training-campaign` off `main`, merging
      `feat/derivative-probe`, `origin/claude/project-thread-k6pncx` (neural
      density init) and `matcalc-property-validation`. Conflicts resolved:
      `cli.py` imports and one README table row.
- [ ] `src/interfaceforge/derivative_probe.py`: add `strain_mode = "voigt"`.
  - Add `--voigt-components xx yy zz yz xz xy`.
  - Build a symmetric ε, with shear equal to half the engineering strain,
    and apply it as `set_cell(cell @ (I + ε).T, scale_atoms=True)`.
  - Add `strain_component` to the case IDs, the manifest rows and csv, and
    the response keys.
  - Store `strain_matrix` in place of `linear_scale`.
- [ ] Define the campaign for the bulk phases (TiN, β-Si₃N₄, TiO ideal,
      TiO real):
  - Voigt strains: 6 components × ±0.5 / 1 / 2 %.
  - Isotropic strains: ±0.5 / 1 / 2 %.
  - Compression: −5 / −10 / −15 % in volume.
  - Paired 0.03 Å rattles on the 1 % centres.
  - That is about 60–70 static single points per phase.
  - Use small cells (8-atom TiN, 14-atom Si₃N₄). TiO real keeps its
    training supercell, because of its vacancies.
- [ ] Build the static INCAR from the campaign's Step2 `INCAR_FINAL` with
      the probe's `_static_incar`. Launch with **`vasp_std`**:
      `launch_scripts/runvasp.sh` uses `vasp_gam`, which is wrong for
      metals with k-points.
- [ ] Neural CHGCAR (`iface vasp initialize-density --backend neural-paw`)
      has only been tested against a mock backend so far. Pilot it on
      about 10 runs with `iface vasp density-init-bench` before relying on
      it. Its main payoff is the interface round.
- [ ] Collect with **`include_virial: true`**. The default is false in
      `templates/campaign.yaml` and `config.py`.
- [ ] **Run A2:** fine-tune from the A1 models on the original data plus
      the strain frames, with virials on.

### Track C: validation

- [ ] Stress-skill check on held-out frames, before and after A1 and A2:
      stress RMSE and the shear-component correlation.
- [ ] Acceptance: run `iface properties run` on each fine-tuned committee
      (TiN primitive, Si₃N₄ 14-atom). **Pass criteria:** C₁₂ and G within
      about 10–15 % of the committee's own DFT reference, and the full-scan
      EOS B′ between 3.5 and 5.

## Round 2 (later)

- Strained and rattled interface cells: Ideal/Real × N/Ti termination × O
  fraction 0–1. This is where the neural CHGCAR initialization pays off.
- Optionally, the phase-diagram phases that were never in training.

## Rules for contributors

- Keep default behavior unchanged. Every new training behavior is opt-in.
- Never submit Slurm jobs or push branches without the owner's explicit
  go-ahead.
- Don't modify `interface_energy.py`, `adhesion.py` or `swap_mc.py` for
  this work.
- The known Windows test baseline is 43 pre-existing failures (`/tmp`
  paths and symlinks). A change should not add to it.
