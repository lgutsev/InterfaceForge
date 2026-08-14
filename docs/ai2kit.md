# AI2-Kit active learning

> **Verification note:** both adapters are code-only. Generated files and
> failure guards have automated coverage, but no human-supervised external
> AI2-Kit/oh-my-batch campaign has completed through labeling and retraining.

InterfaceForge provides two process-isolated AI2-Kit 1.0.9 workflows:

| `workflow` | Path | Status |
|---|---|---|
| `tesla_mace` | Existing MACE committee → OpenMM NVT → committee force deviation → VASP labels; oh-my-batch handles Slurm expansion, concurrency, retry and recovery | Recommended for new MACE campaigns |
| `cll_deepmd` | DeepMD → LAMMPS → VASP through AI2-Kit's deprecated config-driven CLL interface | Retained for backward compatibility |

The TESLA implementation follows AI2-Kit's public
[MACE/OpenMM/VASP example](https://github.com/chenggroup/ai2-kit/tree/main/example/supplement/skill-demo),
with InterfaceForge configuration validation, hashing, label-key normalization
and LONI profiles added around it.

## TESLA with four ready MACE models

Start from [the editable campaign](../examples/ai2kit/campaign.yaml) and
[LONI profile](../examples/ai2kit/profile_loni.yaml). Set the four paths in
`committee_models`, the canonical training/validation extxyz files, exploration
structures, VASP inputs and POTCAR locations.

The first iteration uses the supplied committee without retraining it. For each
configured structure, in-plane strain, temperature and replica, the generated
workflow:

1. uses oh-my-batch to create and submit an OpenMM job;
2. propagates MD with the first committee member;
3. evaluates every saved frame with all committee members;
4. writes a DeePMD-compatible `model_devi.out` containing the maximum, minimum
   and mean per-atom force standard deviation;
5. uses AI2-Kit to write `good.xyz`, `decent.xyz`, `poor.xyz` and `stats.tsv`;
6. samples `decent.xyz`, generates VASP single-point jobs with oh-my-batch and
   collects completed OUTCAR labels into `new-dataset/dataset.xyz`.

When `max_iterations` exceeds one, later rounds train a new MACE committee from
the base training data plus all earlier `new-dataset` files. More than one
iteration still requires `--allow-multiple-iterations` at execution time.

Committee deviation is an uncertainty proxy, not a DFT error. Do not copy the
example thresholds blindly. Calibrate `trust_force_low` and
`trust_force_high` against held-out DFT force errors for the target system.
Frames below the low threshold are considered covered; frames between the
thresholds are labeling candidates; frames above the high threshold are
normally rejected as unsafe or badly extrapolative. `use_poor_frames` can admit
a small reviewed sample.

## Installation

Install InterfaceForge, AI2-Kit and oh-my-batch in the controller environment:

```bash
python -m pip install -e '.[ai2kit]'
```

The GPU environment referenced by `profile.ai2kit.commands.python` must also
provide `mace-torch`, `openmm`, and OpenMM-ML (`import openmmml`). The command
paths are explicit so the controller does not depend on interactive Conda shell
initialization. Edit the Slurm profile to match the LONI allocation and modules.

## Operation

Run the controller from a LONI login or other permitted service host, not from
inside a Slurm allocation:

```bash
iface active-learning ai2kit export -c campaign.yaml
iface active-learning ai2kit preflight -c campaign.yaml
iface active-learning ai2kit run -c campaign.yaml
iface active-learning ai2kit run -c campaign.yaml --execute
iface active-learning ai2kit status -c campaign.yaml
```

`run` is a dry-run unless `--execute` is supplied. The generated controller is
`runs/active_learning/ai2kit/generated/run.sh`. It calls `omb job slurm submit
--wait --recovery ...`; interrupted or failed submission state remains under
the corresponding iteration directory. Resume only after inspection:

```bash
iface active-learning ai2kit run -c campaign.yaml --execute --resume
```

The main first-round outputs are:

```text
runs/active_learning/ai2kit/work/iter-000/
  openmm/job-*/traj.xyz
  openmm/job-*/model_devi.out
  screening/good.xyz
  screening/decent.xyz
  screening/poor.xyz
  screening/stats.tsv
  vasp/job-*/OUTCAR
  new-dataset/dataset.xyz
```

The OpenMM driver is fixed-cell Langevin NVT. InterfaceForge creates explicitly
strained starting structures for cell/strain coverage; it does not claim NPT
sampling. It stops a trajectory when the driving model produces non-finite
values or a force above `max_force_ev_ang`, but committee screening itself is
post-processing rather than an online stopping criterion.

## Import and approval

Label collection does not mutate the canonical dataset. Stage and review one
round explicitly:

```bash
iface active-learning ai2kit import -c campaign.yaml \
  --round 0 \
  --result-root runs/active_learning/ai2kit/work/iter-000/new-dataset
iface active-learning ai2kit approve -c campaign.yaml --round 0
```

Import validates species, cells, coordinates, energies, raw force shapes,
finiteness, source hashes, VASP completion markers and duplicate structure
identities. Approval records review state; promotion into the canonical dataset
remains an explicit maintainer action.

## Legacy config-driven CLL

Existing configurations without `workflow`, or with `workflow: cll_deepmd`,
retain the pinned DeepMD/LAMMPS/VASP adapter. It expects the earlier profile
keys (`ssh`, `work_dir`, `python_cmd`, DeepMD/LAMMPS commands) and uses AI2-Kit's
deprecated but backward-compatible `workflow cll-mlp-training` command. This
path is maintained to avoid breaking existing campaigns, not recommended for a
new MACE committee.
