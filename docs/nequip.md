# NequIP committees

> **Verification note:** automated-test-only. Config generation, the dataset and
> split handoff, committee/seed handling, Slurm rendering, the member driver
> (failure → resume → package/compile, config-change refusal), checkpoint and
> final-model discovery, Lightning log parsing, the ASE evaluator, committee
> evaluation, `mlip-compare`, `mlip-progress` and packaging are regression
> tested with fake `nequip-*` executables and a stub calculator. **No NequIP
> model has been trained, compiled or evaluated through InterfaceForge yet**,
> and nothing here claims NequIP is validated for NiO. See
> [Verification and maturity](verification.md).

`iface train nequip` generates a seeded committee of NequIP message-passing
models (`nequip.model.NequIPGNNModel`, NequIP ≥ 0.7 Hydra/Lightning framework)
from the canonical dataset and split shared with MACE and DeePMD/DPA.

NequIP and Allegro share the NequIP framework (`nequip-train`,
`nequip-package`, `nequip-compile`) but are **different architectures**: NequIP
is a message-passing GNN, Allegro a strictly local model with its own
hyperparameters and LAMMPS path. Allegro stays in [its own guide](allegro.md)
and `models.allegro`; nothing in `models.nequip` configures Allegro.

## Configuration

```yaml
models:
  nequip:
    enabled: true
    profile: nequip_gpu            # scheduler profile job (nequip_cpu for CPU-only debugging)
    device: cuda                   # cuda | cpu; must agree with the profile's GPU request
    dataset: datasets/canonical    # canonical dataset (iface dataset export) -- preferred
    seeds: [11, 23, 37, 53]        # committee members; any unique non-negative integers
    max_concurrent: 2              # Slurm array throttle
    r_max: 5.0                     # REQUIRED: InterfaceForge never picks the cutoff
    model_dtype: float32           # float32 | float64
    # trainer_precision: "32-true" # optional Lightning precision (exclusive with tf32)
    # tf32: false                  # optional TF32 for float32 training
    # max_time: "02:23:00:00"      # stop cleanly before the walltime; resume picks up last.ckpt
    # data_seed: 1                 # one shuffle seed for all members (default: each member's seed)
    # num_layers / l_max / parity / num_features / radial_mlp_* / num_bessels /
    # polynomial_cutoff_p / batch_size / val_batch_size / num_workers / max_epochs /
    # learning_rate / ema_decay / loss_energy_weight / loss_forces_weight /
    # early_stopping_patience / lr_factor / lr_patience / lr_threshold / lr_min /
    # zbl / compile_training / compile_mode (aotinductor|torchscript) / compile_target
    # extra_model: {}              # extra NequIPGNNModel keys, passed through and recorded
```

`r_max` has no default: `iface train nequip` and `load_campaign` refuse an
enabled NequIP block without it. Every other hyperparameter falls back to a
documented starting point taken from the NequIP tutorial configuration; each
fallback is listed under `defaults_applied` in
`models/nequip/training_manifest.json` (and in packaged model cards) so a
default is never mistaken for a tuned NiO value. Unknown keys are errors.

`dataset:` points at a canonical dataset manifest; the train/valid/test files
must still match the manifest's hashes, and the dataset hash, split hash and
type map are recorded. `type_names` defaults to the manifest's `type_map` and,
if set explicitly, must contain the same elements. Explicit
`train_file`/`valid_file`/`test_file` are accepted for other datasets.

## What is generated

```text
models/nequip/
  training_manifest.json        commit, versions, dataset/split identity, hyperparameters,
                                defaults_applied, runtime profile, launchers, restart policy
  seed_<seed>/config.yaml       NequIP config (explicit train/val/test paths, per-member seed)
  seed_<seed>/config.sha256     the driver refuses a config edited after generation
  seed_<seed>/member.json
  seed_<seed>/train_member.sh   train (or resume) -> nequip-package -> nequip-compile
  smoke/config.yaml             first seed, a few epochs/batches, no early stopping
  run_preflight.slurm           nvidia-smi, nequip-* on PATH, versions.json, dataset sha256 -c
  run_smoke.slurm               smoke train + compile + 20-frame evaluation in smoke/job_<id>/
  run_committee.slurm           Slurm array: one member per task (array 0-N%max_concurrent)
  run_finalize.slurm            re-package/compile from best.ckpt (retry only)
  run_evaluate.slurm            Slurm array: canonical test-set predictions per member
  evaluate_nequip.py            standalone ASE evaluator (run where nequip is installed)
```

All scheduler settings (partition, account, walltime, GPUs, modules, the
activation preamble, environment) come from the campaign's scheduler profile.
The LONI profile provides `nequip_gpu` and `nequip_cpu`, which source
`$NEQUIP_ACTIVATE_SCRIPT`; set it to an environment providing `nequip-train`,
`nequip-package`, `nequip-compile` and (for GPU) a CUDA PyTorch build. With a
`local` profile the array launchers become sequential loops. Generation never
submits anything.

Each member directory gains at runtime:

```text
outputs/best.ckpt outputs/last.ckpt       Lightning checkpoints (hydra.run.dir is fixed)
logs/csv/version_<n>/metrics.csv          CSV logger (a new version per restart)
status.json                               state/stage/exit code/job/restarts, written by the driver
versions.json                             python/torch/nequip/lightning/e3nn/ase + CUDA device
restarts.count restarts.log               every resume from last.ckpt
final/model.nequip.zip                    portable nequip-package archive (from best.ckpt)
final/model.nequip.pt2                    compiled model (aotinductor, target ase) for this GPU type
final/checksums.sha256 best_ckpt.sha256
evaluation/predictions.npz(.json)         canonical test-set predictions, keyed by frame_id
```

## Restart and failure handling

Resubmitting `run_committee.slurm` is the restart procedure:

- a member with `outputs/last.ckpt` resumes with `++ckpt_path=<last.ckpt>`
  (NequIP restores the model/optimizer and continues the interrupted run stage);
- a completed member (`final/model.nequip.{zip,pt2}` present) exits immediately;
- a member whose `config.yaml` no longer matches `config.sha256` is refused
  (exit 3, `status.json` stage `config_check`). NequIP documents that the
  config must not change between restarts; regenerating with a *different*
  config while checkpoints exist is refused by `iface train nequip --force`
  too. Move the member directory aside to start it over;
- any failing command records `state: failed` with the stage (`train`,
  `package`, `compile`, ...) and exit code; `run_finalize.slurm` retries
  packaging/compilation from an existing `best.ckpt`.

## Status, evaluation and comparison

```bash
iface nequip status                 # read-only member table (also in iface mlip-progress)
iface nequip status --json
iface nequip evaluate               # committee metrics from evaluation/predictions.npz
iface mlip-compare prepare --nequip-root models/nequip   # + MACE and DeePMD on the same frames
```

`iface nequip status` / `iface mlip-progress` report per seed: state
(`not-started`, `running`, `stalled?`, `incomplete`, `failed`, `complete`),
stage, epoch/step and the latest `val0_epoch/*` metrics from `metrics.csv`,
checkpoint / package / compiled-model availability, test-set predictions and
committee evaluation. Neither command submits, repairs or restarts anything.

`iface nequip evaluate` aligns every member's predictions to the canonical test
frames by `frame_id` (refusing missing, extra or mismatched frames, and a test
file whose hash changed since generation) and writes `per_model.csv`,
`per_system.csv`, `per_frame.csv`, `predictions_aligned.npz` and
`summary.json`: per-member and committee-mean energy/force MAE and RMSE
(all atoms and mobile atoms), and committee **disagreement** (member standard
deviation). Disagreement is not a calibrated uncertainty and is never labelled
as one. See [MLIP comparison](mlip-comparison.md) for the three-backend
matched-frame comparison.

## Packaging

```bash
iface committee collect models/nequip stored_models/nio_nequip_v1 --engine nequip --expected-members 4
iface committee verify stored_models/nio_nequip_v1.zip
iface package huggingface stored_models/nio_nequip_v1 hf/nio_nequip \
    --metrics models/nequip/evaluation/summary.json
iface package campaign            # includes models/nequip when present
```

The bundle stores each member's portable `models/seed_<seed>.nequip.zip`
plus checksummed extras: the compiled model (GPU/PyTorch specific), the exact
`configs/seed_<seed>.config.yaml` and `provenance/seed_<seed>.json` (config and
best-checkpoint hashes, runtime versions, status, final epoch, validation and
test metrics). The bundle manifest records the InterfaceForge commit (with a
dirty-tree flag), architecture, type map, hyperparameters, `defaults_applied`,
dataset hash and split hash, and the committee evaluation. Checkpoints are
bundled only with `--include-checkpoints`; VASP files are never bundled.

## First LONI smoke test

```bash
# On LONI, from the campaign directory, with this InterfaceForge branch installed
# (pip install -e '.[vasp]') and profiles/loni.yaml holding your account.
iface dataset readiness /path/to/NiO_head --output audit/nio_readiness
iface dataset export    /path/to/NiO_head --output datasets/canonical
iface dataset readiness /path/to/NiO_head --output audit/nio_readiness \
    --dataset datasets/canonical -c campaign.yaml        # expect ready_for_gpu_smoke_tests: true

# campaign.yaml: models.nequip.enabled: true, dataset: datasets/canonical, r_max: <explicit>
iface train nequip
less models/nequip/training_manifest.json models/nequip/seed_11/config.yaml models/nequip/run_smoke.slurm

export NEQUIP_ACTIVATE_SCRIPT=/path/to/nequip-env/bin/activate
iface nequip submit preflight            # prints the sbatch command only
iface nequip submit preflight --execute
iface nequip submit smoke --execute      # 2 epochs x 5 batches, compile, 20 test frames
iface nequip status
```

The smoke job's output (`models/nequip/smoke/job_<id>/`) is **not** a trained
model. Only after the preflight and smoke logs look right should
`iface nequip submit committee --execute` be used.
