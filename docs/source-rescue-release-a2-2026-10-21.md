# SiN/TiN release rules for the first A2 (2026-10-21)

This note covers the release tool `iface-rescue-release` (`src/interfaceforge/rescue_release.py`) and the
policy `examples/source-rescue/release_policy.yaml`. The tool can now express the PI's recorded decisions,
and the policy encodes them. The decisions come from SiN-TiN_Dispatch `dispatch/desk_events.jsonl` and
`STATUS.md`.

No release was assembled and no training was started. Encoding these rules does not authorize the launch:
the A2 launch needs the PI's separate approval.

## What changed

**Label masks.** A source rule may set `label_use`:

| `label_use` | Trained terms |
|---|---|
| `all` (the default) | energy, forces and virial (the old behaviour) |
| `energy_forces` | energy and forces; the virial is not trained |
| `forces_virial` | forces and virial; the energy is not trained |

`assemble` carries the mask into every output:

- **Release ledger.** `release_ledger.tsv` has a new `label_use` column.
- **extxyz.** Each frame carries `label_use=`. The masked term is absent from `{train,valid,test}.extxyz`: no
  `REF_virial` for `energy_forces`, no `REF_energy` for `forces_virial`.
- **DeePMD.** Systems are grouped as `deepmd/<split>/<label_use>/<leaf>`. That layout changes for every
  system, `all` included.
  - A masked system has no `virial.npy` or no `energy.npy`.
  - Each system has a `release_labels.json` that lists its trained terms and absent files.
  - `deepmd/systems.json` indexes every system.
  - After writing each system, the tool checks that the label files on disk match its `label_use`.
- **Originals.** Original labels are never edited. The full values stay in the deck and in the relabel files,
  which the policy pins by sha256.

**Collector relabels.** Replacements can now be collector output. A collector `labeled.extxyz` has no
`source_run`, and its `source_frame` is the ionic step. So a replacement entry may name its package
`manifest` (with `manifest_sha256`), and each frame's `case_id` is mapped to its deck frame through the
manifest's `source_structure`. Both manifest formats in use are parsed.

These cross-checks are errors:

- a frame whose stamped `manifest_sha256` differs from the pinned one;
- a manifest split that differs from the frozen ledger's split;
- a frame count that differs from `expected_frames`.

**Added frames.** A new section, `added_frames`, lists labelled frames that are not in the deck. Each entry
pins an extxyz by sha256 and gives:

- `split`: train, valid, test or holdout;
- `label_use` and `reason`, both required;
- optionally `select` (an info-key filter, such as `role: train`), `expected_frames` and `leaf`.

Rules:

- Added training frames are written to their split's extxyz and to one DeePMD system per species order
  (`<leaf>/<formula>_<natoms>`).
- **Holdouts are written only to `holdout.extxyz`**, with all their labels, and never to a DeePMD system.
- A file may be listed twice with different `select` filters, but a frame selected by two entries is an error.
- An added frame whose geometry repeats a release frame is an error.

**Leakage checks.** A new section, `leakage_references`, lists geometry files that must not appear in the
splits named by `forbid`. Every added holdout is also a reference, forbidden in all three splits. A match
is either an equal geometry digest (1e-6 Å) or a match within `LEAKAGE_TOLERANCE_A` = 1e-3 Å (minimum
image, same species order and cell), so a rounded or rewrapped copy is still caught.

**`plan` subcommand.** `iface-rescue-release plan --ledger … --policy … [--deck …] [--report …]` is the dry
run. It writes no release. It runs every check that needs no deck arrays: rules, replacements, manifests,
added frames, holdouts and digest-level leakage. With `--deck` it also runs the tolerance-level leakage
check against the deck geometries, and only frames whose species order matches a reference are parsed.

**Existing checks.** All of them are kept:

- frozen splits by (source_run, source_frame);
- same-geometry replacement at 1e-6 Å;
- ledger geometry and label digests;
- duplicate, unknown and wrong-source replacements;
- `replace_all` completeness;
- releases are never overwritten.

**Shared writer.** `data._write_deepmd_system` takes `include_energy=False`. It now refuses a system in which
only some frames have a virial.

**Tests.** The new tests are in `tests/test_rescue_release.py`, which passes 47 of 47. The same Windows
failures as on the base commit remain; see the commit message.

## How DeePMD-kit treats a missing `energy.npy` or `virial.npy`

A1 is DeePMD 3.2.0 with the PyTorch backend, used with the LONI module
`deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu`. I read the v3.2.0b0 source from GitHub; I did not run it.

1. **The labels are optional.** The `ener` loss declares `energy`, `force` and `virial` as
   `DataRequirementItem(..., must=False)` (`deepmd/pt/loss/ener.py`, `label_requirement`).
2. **A missing file becomes a zero-filled label.** In that case `DeepmdData._load_data` returns
   `find_<key> = 0.0` with an array of zeros (`deepmd/utils/data.py`, the `elif must: … else:` branch).
   A present file gives `find_<key> = 1.0`.
3. **The loss term is zeroed.** The loss multiplies each prefactor by its flag:
   `pref_e = pref_e * find_energy` and `pref_v = pref_v * find_virial` (`ener.py`, about lines 262 and 445).
   A PyTorch batch is drawn from a single system, so the flag applies per system. A masked term contributes
   exactly 0 to the loss and the gradient.
4. **The energy bias skips masked systems.** Output statistics use only systems with `find_energy > 0`
   (`deepmd/pt/utils/stat.py`, the `global_sampled_idx` split). Energy-masked strain systems therefore do not
   bias the atomic energy shift.
5. **Monitoring shows NaN.** `display_if_exist` returns NaN when the flag is 0 (`deepmd/pt/loss/loss.py`),
   and DeePMD's lcurve header says so. Validation averages its batches with a natoms weight, so `rmse_v_val`
   reads NaN whenever a sampled validation batch comes from a virial-masked system. The policy creates such
   systems: TiN 300 K and Ideal Ti-term valid. `rmse_e_trn` behaves the same on strain batches.
   - This affects the display only, not training.
   - Judge A2 with per-system `dp test`, not with lcurve's validation columns.

So in 3.2.0b0, leaving the file out is the clean per-system exclusion. That is why the focused-A2 plan
forbids deleting `virial.npy` by hand while the virial loss stays nonzero: the term disappears silently. Here
the absence is a policy rule, recorded per frame and per system, and checked on disk after writing.

**Not verified:**

- **Not run.** I did not run `dp train` or `dp test` on a masked system. Before the launch, a 10-step
  `dp train` on two tiny systems (one without `virial.npy`, one without `energy.npy`) on the LONI module
  would confirm the behaviour.
- **Version.** I read the v3.2.0b0 tag, not the exact build in the container.
- **TensorFlow backend.** Not checked. A1 is PyTorch.
- **Fine-tuning from A1.** If A2 fine-tunes from A1 with a bias change (`change_bias` or finetune stats),
  confirm that path also filters on `find_energy`. Only `compute_output_stats` was read.

## Policy as encoded

| Source | Rule | `label_use` | Basis |
|---|---|---|---|
| `bulk_SiN-Bulk_300K` | `replace_subset`, unreplaced `hold`; SD004's 150 | all | SD004 accept (2026-10-16); SD016 3×3×7 converged |
| `bulk_TiN-Bulk_300K` | `replace_subset`, unreplaced `hold`; SD005's 48 + SD011's 102 | energy_forces | decision 8 as revisited after SD014 |
| `interface_*K_Ideal_Ti_Term_*` (10 sources) | keep | energy_forces | decision 4 (SD003 frame_0002 is a logged exception) |
| `interface_450K_Ideal_Ti_Term_SiN-TiN-Ti-term` | under the Ideal Ti-term rule (HOLD lifted 2026-10-21) | energy_forces | item 1 below |
| other `interface_*` | keep | all | B0 interface criteria |
| 450 K bulk, TiO-Ideal | exclude, as before | – | |
| TiO-Real 300/450 K | `replace_subset` hold, no relabels (as before): all 1,200 held | – | |
| Package 15, 24 strain cells | added, train | forces_virial | decision 9 |
| SD006, 20 training strain frames | added, train (`select: {role: train}`) | forces_virial | decision 9 |
| SD006, 16 holdouts | added, holdout (`select: {role: holdout}`) | – | never trained |
| Package 14, 38 references | leakage reference, forbidden in train/valid/test | – | |
| SD015, 4 Ideal Ti-term test frames | leakage reference, forbidden in train/valid | – | |

The dry run used the frozen ledger (`split_ledger.tsv`, sha256 `ed7ff1b9…`, 28,800 frames) and the D: inputs.
It ran both at digest level and with `--deck D:\Traj_Storage\SiN-TiN_Cer_Interface_BULK`, and passed with no
leaks.

| | Frames |
|---|---:|
| Kept | 24,000 |
| Replaced | 300 |
| Held | 2,100 |
| Excluded | 2,400 |
| Added to train | 44 |
| Holdout | 16 |

Release splits: 19,484 train, 2,430 valid and 2,430 test. By `label_use`:

| Split | all | energy_forces | forces_virial |
|---|---:|---:|---:|
| train | 14,520 | 4,920 | 44 |
| valid | 1,815 | 615 | 0 |
| test | 1,815 | 615 | 0 |

## Open items for the PI

1. **Stale HOLD on `interface_450K_Ideal_Ti_Term_SiN-TiN-Ti-term` (lifted by the PI 2026-10-21).** It was set
   in the package 23 era because package 21's frame_0009 (the converged-k check of deck frame 2725) failed SCF
   under ALGO=Fast. That question is answered:
   - Package 23's Normal controls ran cleanly and the owner's control reproduced them.
   - Against them, the Γ label is within energy (0.60 meV/atom) and stress (0.066 GPa); the force excess on Ti
     is the family's bias, which decision 4 covers.
   - SD003 frame_0010 reproduced the stored label of frame 2725 exactly.
   - SD015 f2015, from the same source, meets decision 4.

   The exact rule is deleted, so the source falls under the Ideal Ti-term glob (keep, energy_forces). The glob
   must stay above `interface_*`; a test pins this order. 600 frames return (480/60/60), and SD015 f2015's source
   is now in distribution (it stays in the test split, which its leakage reference allows).
2. **Package 15's labels file (resolved 2026-10-21).** Package 15's labels are its OUTCARs. I produced the
   extxyz with the desk's own collector:

   ```
   sin-tin-a2-gen fa221a6: python campaigns/a2_pilot/a2_campaign.py collect D:\MLIP_Work_Folder\sin_tin_a2\15_pilot --output …
   ```

   - The result is 24/24 ok, taking the final ionic step for the two relaxations.
   - It is byte-for-byte reproducible, sha256 `75985ad7…`.
   - With the owner's go-ahead it is now written to
     `D:\SiN-TiN_Interface_Dispatch\results\pkg15_collect_2026-10-21\labeled.extxyz`, with the pinned sha256.
   - `plan --deck` on the committed policy, unmodified, passes with no leaks.
3. **Decision 4d against the 2026-10-20 review (confirmed by the PI 2026-10-21: 4d superseded).** 4d
   (2026-10-19) said the 10 SD003/SD009 converged-k frames become A2's Ideal Ti-term test set. All 10 are in
   the deck's train split, and their MD neighbours stay in train, so they cannot be an independent test. The
   2026-10-20 review made SD015's 4 frozen-test frames the test set. The 10 stay in training and are not
   leakage references; no policy change. A2's report may still show them as an in-sample diagnostic (A2
   against their converged-k labels), not as a test.
4. **Sampling weights.** The 44 strain frames are 8–16-atom cells in their own small systems. With A1's
   `auto_prob: prob_uniform`, each small system would be sampled as often as a 600-frame interface. This is a
   training-input choice, not a release rule.

## What the tool cannot express or does not check

- **Symmetry.** Leakage is geometric identity within 1e-3 Å in the same cell and species order.
  Symmetry-equivalent copies (permuted atoms, rotated cells, ±shear mirrors) are not detected. Package 14's
  references are checked by their input geometries (`frames.extxyz`), not by their relaxed final geometries.
- **Per-frame masks.** Masks are per source rule or per added-frame entry, not per frame. SD003 frame_0002, the
  decision-4 exception, is recorded only as text in the rule's reason. It stays in training, as decided.
- **Holdouts in DeePMD.** No DeePMD system is written for holdouts. Evaluating them with `dp test` needs a
  separate conversion of `holdout.extxyz`.
- **Other trainers.** The extxyz files omit the masked keys. A trainer that requires every key, rather than
  weighting missing ones to zero, must read `label_use`.
