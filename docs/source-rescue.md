# SiN/TiN source rescue and routine data qualification

## Objective and current decision

Continue training from the existing model checkpoints after cleaning and
correcting the data. A restart from foundation weights is not a prerequisite.
Keep the original raw calculations, old exports, checkpoints and benchmark
results immutable; create a new dataset release and a new training output tree.

Quarantine **both TiO-Bulk-Ideal_300K and TiO-Bulk-Ideal_450K** from train,
validation and test. The 1,200 retained frames include 960 training, 120 validation
and 120 test frames. Excluding them leaves 27,600 original frames, nominally
22,080/2,760/2,760, before other exclusions, subset replacements or A2 additions.
Do not confuse TiO-Ideal with TiO-Real. Retain the original files for diagnosis;
these temperature histories are not representative 300/450 K sampling. No new
TiO-Ideal AIMD is required for this rescue. Use validated targeted statics only
if coverage of that phase is needed.

The investigation has three distinct questions: source calculation quality,
conversion correctness, and model accuracy. A successful MACE/DeePMD membership
cross-audit answers only the second. No audit script can guarantee physical
accuracy without converged DFT reference checks.

## 1. Use the directory map; discover the entire origin tree

`examples/mapped-leaf-campaign/periodic_nitride.yaml` remains the source of truth
for the four interface MD roots (Step2_300K/450K × Real/Ideal), all eight bulk
origins, and the planned A2 train sources. It now records disabled sources with
reasons and requires source qualification before staging/export. It writes into
`MD_Period/Periodic_MLIPs_rescue`, leaving `Periodic_MLIPs` unchanged.

`examples/source-rescue/sintin.yaml` reads that same map and adds:

- `/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD_Vac/Step2_Bulk` in full,
  to find newly added or unmapped runs;
- `/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD_Period/Wadh`, as references;
- the resettled `MD_Period/A2_DFT/14_references`, `15_pilot`, `17_audit` and
  `21_stratified_audit` directories, with non-training roles.

Wadh must be audited even if it contributes no training frames. It does not
replace auditing the original interface MD roots. Inventory backups and disabled
branches as quarantined; do not silently treat them as new independent runs.
Resolve any missing resettled package against `dispatch/resettled.tsv` and
`dispatch/manifest.jsonl`. Update the map/policy to the verified permanent home;
do not copy it back into smoke. Package 21's root is optional only because its
move may not have happened yet. Its reviewed results are still required for the
A2 label-policy decision. Packages 14/17/21 and BBVO never enter training.

Morning command, from the LONI InterfaceForge checkout:

```bash
export CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface
sbatch launch_scripts/audit_vasp_sources_single.sbatch \
  "$PWD/examples/source-rescue/sintin.yaml" \
  "$CER_INTERFACE_BASE/MD_Period/audit/source_rescue_initial"
```

Use an allocated node, not the head node, for scans and recollection. Use the
existing `lgutsev_dev` environment; do not duplicate models or create a new
environment unnecessarily. The launcher follows QB4's existing single-node
convention without `--mem`. Run paths and time limit can be overridden normally.
Submit from the checkout, or export `INTERFACEFORGE_ROOT` to its absolute path:
Slurm executes a copied script whose location cannot identify the repository.
The initial audit is expected to exit **2** with reports: this means unresolved
REVIEW/FAILED records, not that every run failed to execute.

If the cluster checkout has diverged, inspect `git status --short --branch` and
`git log --oneline --left-right HEAD...origin/main` after fetching. When the
working tree is clean and the local commits belong on local `main`, use
`git switch main` followed by `git merge --no-edit origin/main`. A normal merge
preserves both histories. Resolve any reported conflicts before submitting;
do not reset the checkout or push its local commits without reviewing them.

Alternatively, a detached worktree allows immediate submission without merging.
It creates a small source-only checkout while keeping existing commits and
uncommitted work intact:

```bash
(
set -e
REPO=/project/lgutsev/git_develop/InterfaceForge
git -C "$REPO" fetch origin main
AUDIT_CHECKOUT=$(mktemp -d /project/lgutsev/git_develop/InterfaceForge-source-audit.XXXXXX)
git -C "$REPO" worktree add --detach "$AUDIT_CHECKOUT" origin/main
export INTERFACEFORGE_ROOT="$AUDIT_CHECKOUT"
export CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface
cd "$AUDIT_CHECKOUT"
sbatch "$AUDIT_CHECKOUT/launch_scripts/audit_vasp_sources_single.sbatch" \
  "$AUDIT_CHECKOUT/examples/source-rescue/sintin.yaml" \
  "$CER_INTERFACE_BASE/MD_Period/audit/source_rescue_initial"
)
```

Keep the worktree until the job has finished. If the report directory already
contains a prior audit, choose a new report name rather than overwriting it.

## 2. Audit every source before selecting frames

The scanner reads all OSZICAR ionic/electronic steps and all ASE OUTCAR labels:

- Completion, parse failures, missing inputs/outputs, frame-count disagreement,
  restart index resets, NELM exhaustion, high charge residuals and residual jumps.
  An energy stopping criterion can be met after a density-residual jump; inspect
  the SCF trace. Increasing NELM alone does not repair false convergence.
- Actual echoed versus saved INCAR settings, ordered POTCAR titles, VASP version,
  NKPTS, and file hashes. Shared INCAR/KPOINTS/POTCAR inputs are resolved only
  within the configured source root.
- Temperature target, mean, standard deviation and percentiles after an explicit
  burn-in window. A correct average can hide severe cold/hot oscillations.
  Ramps and small-cell fluctuations require protocol-specific interpretation.
- Nonfinite E/F/stress/geometry, atom identity changes, force/stress outliers,
  energy jumps, volume and sampled minimum-image contacts. E/F/stress and finite
  geometry checks cover every frame; contact distances are sampled every 50 by
  default to bound cost. Set `contact_stride: 1` for a full contact scan.
- Explicit k-convergence review for every source. A completed run, a dense mesh,
  or reproducing the original stored label is not evidence of convergence.

Outputs: source_audit.json/CSV, SUMMARY.md, per-run SCF JSON, per-frame label
JSONL. Fingerprints include input files and outputs, including OSZICAR. No raw
file is renamed, moved, edited or deleted. Source changes during scanning or
between audit and collection invalidate qualification. Logs-only scans are
useful for lightweight archives but cannot authorize collection.

Thresholds are review triggers, not universal physical bounds. Classify every
finding: accepted with documented evidence; quarantine; relabel; or targeted
rerun. Acceptance requires a source_id, exact fingerprint, reviewer, evidence
and acknowledgement of every review code in the YAML. Hard failures such as
nonfinite labels, parse/count errors or unfinished outputs cannot be waived.
OUTCAR can echo `LREAL=T` for a saved `LREAL=Auto` or `On`. This lossy logical
echo raises `LREAL_MODE_UNVERIFIED`, not a hard settings mismatch: the different
optimization modes must still be established from original provenance or
reproduction evidence. `Auto` and `.TRUE.` are not equivalent INCAR settings.
For `MALFORMED_SCF`, inspect the original electronic line before judging the
calculation. The scanner accepts signed numeric fields without separating
spaces and a spaced algorithm colon; remaining parse failures retain the line
and error text. Never waive nonfinite values or unexplained corrupt columns.
Update decisions in a campaign copy of the YAML and audit into a new directory.
The collector may admit the explicitly accepted train subset even while
reference/audit findings remain unresolved. Missing required roots block it.

## 3. Complete numerical repair before production A2

Use package 23's staged checks and the existing dispatch/resettlement workflow:

1. Re-run the interface outlier with independent electronic starts; inspect
   residual history. Check the previously unaudited interface families. Keep
   Gamma interface labels only for families supported by E/F/stress comparisons.
2. Confirm Si3N4 at both temperatures and denser k; then replace all 1,200
   bulk labels with same-geometry E/F/stress from qualified calculations.
3. Run the 48-frame TiN first stage and denser-k checks; expand to the remaining
   frames if it passes. Replace all channels rather than applying a mean stress
   offset. That offset does not correct frame-specific force errors.
4. Qualify the TiO-Real mesh and relabel the selected 240-frame subset first.
   Exclude superseded Gamma frames from active training; preserve source splits.
5. Decide package 15's PREC/LREAL bridge including **force differences**, not
   only energy/stress. An element-additive energy shift cannot repair forces.
   Finish the 20 remaining train + 16 holdout labels with compatible numerics.

Use VASP 6.6.1 for new jobs and test reproduction across the binary change.
Preserve original geometry, species/order, POTCARs, dispersion and provenance.
Record E_sigma->0 and free energy separately. The existing deck uses E_sigma->0;
forces/stresses follow the finite-smearing free energy. Taking all outputs from
one SCF avoids mixing runs but does not remove that energy-convention
approximation. Do not silently change energy fields halfway through a release.
Energy/stress/force convergence tolerances must be declared before evaluation.
Small-displacement force errors should be judged relative to the force response.
Thermal-energy estimates are sanity checks, not acceptance proofs.

## 4. Recollect one clean release for every model family

Freeze the original (source run, zero-based source frame, split) map before
replacing any labels. Replacements inherit their original split. Use exactly one
approved E/F/virial label set per geometry; never include original and relabelled
copies together. Keep each exclusion's source hash and reason in the release
ledger. For unchanged original MD sources the original seed, logical leaf names
and stride reproduce random-frame membership; verify against the frozen map.
If the old map cannot be recovered, declare a split migration explicitly rather
than claiming historical test comparability.

The mapper supports whole-source replacement paths. A relabelled subset that
needs per-frame historical membership requires assembling reviewed split-specific
exports using the frozen map; **do not point it at mixed-role relabel outputs and
randomly re-split them**. Keep the old exports until the complete new release
passes. Do not use --force-datasets on the original dataset.

A2 is a planned, disabled source in the YAML. Enable it only after creating a
curated `A2_DFT/training_sources` tree of exactly 44 eligible, single-ionic-frame
OUTCARs with matching inputs and a `roles.json` admission registry:

```json
{
  "schema_version": 1,
  "sources": {
    "train_frame_001": {
      "role": "train",
      "frames": 1,
      "outcar_sha256": "<exact full SHA256>"
    }
  }
}
```

Build this registry from the original package manifests' roles; preserve their
IDs, roles, input hashes, strain/displacements and parent/source provenance in a
separate release frame ledger. Do not infer role from folder names. A2 OUTCARs
with multiple ionic steps must first be normalized to the chosen endpoint with
source provenance, or exported through a role-aware adapter; never pretend a
whole relaxation is one label. Put the 16 holdouts in a separate evaluation tree.
Do not enable 14/17/21, raw 15, or audit/reference roots as mapped train sources.
The mapper rejects missing roles, holdouts, changed hashes and multi-frame A2
sources. Disable per-leaf balancing when adding single-frame A2 sources; otherwise
the minimum count would collapse every 600-frame MD source to one frame.

Set the final map (including A2 enablement/replacement paths) **before** the final
audit. Changing it afterward invalidates qualification. Keep the map's fixed
`source_audit` path and install a symlink named `source_rescue_current` pointing
to the final report directory. Do not edit the map merely to point at a different
report filename after scanning. Use a fresh report directory for every pass and
retain prior reports. Then:

```bash
sbatch launch_scripts/collect_mapped_leaf_single.sbatch \
  "$PWD/examples/mapped-leaf-campaign/periodic_nitride.yaml"
```

The mapper checks source qualification before writing, and rejects stale staged
OUTCARs absent from the current map. MACE/DeePMD exports are checked for identical
membership. Since A2 statics and MD have different frame counts, balanced-count
checking is disabled deliberately while exact cross-format membership stays
mandatory. Generic collectors accept `--source-audit` for the same source gate.
Older unrelated workflows remain compatible; new campaigns must opt into this
standard in their config.

Before training, check finite exported arrays; species/order; energy convention;
force units; virial sign and all nine components against source outputs;
virial.npy in every DeePMD system using a virial loss; exclusion/role counts;
duplicate geometries and split overlap; and cross-format label values as well as
frame identity. The existing membership audit does not compare all numeric
labels; add independent spot checks and failure tests to the campaign adapter.
Checksum the completed release and every model's train configuration.

## 5. Warm-start, diagnose and validate recovery

Continue from verified existing checkpoints in new output directories. Treat this
as a new fine-tune with a declared learning-rate/loss/replay schedule rather than
blindly resuming the old optimizer schedule. Preserve paired seeds and record the
actual replay probability of the A2 frames. Model files stay in project storage.

A small cleanup-only branch (remove TiO-Ideal, otherwise unchanged) can isolate
its effect without waiting for all relabels; label it diagnostic, not production
A2. Production A2 uses the cleaned, corrected release plus the qualified 44 train
frames. No claim that exclusions alone fix all phonons.

Evaluate A1 and the recovered model against **the same qualified targets**:
TiO-Real separately; independent TiO statics; old held-out sets with corrected
labels; 16 A2 holdouts; TiN C11/C12/C44 and q-half frequency; Si3N4 relaxed-ion
C11/G, cell and B'; broader dispersions including transverse modes. Compare at
matched cells and at each model's relaxed cell. Report per-seed errors, volumes,
stability and forgetting of good interface behavior. Keep reference benchmarks
used during design distinct from independent generalization tests. Random-frame
MD splits measure interpolation, not independent-trajectory transfer.

A lower aggregate error after dropping bad TiO-Ideal test frames is not by itself
model improvement. Restrict A1 and recovered comparisons to identical clean
sets and report the exclusion. Add displaced-structure labels if residual phonon
errors remain; thermal-frame relabeling is necessary repair, not a guarantee of
curvature recovery or experimental agreement. Consider fresh initialization only
if warm-start recovery fails scientifically.

## Routine prevention checklist

Every future campaign: inventory all source roots and roles; run source audit;
resolve findings with converged-k E/F/stress and SCF/thermostat/geometry evidence;
freeze splits; collect into a new release with the admission gate; cross-check
exports and source labels; publish hashes and provenance; then train and run the
property battery. Re-audit after any input/output/map change. Never auto-accept
an entire stratum from a few frames or change audit thresholds merely to pass.
Record campaign decisions in git; store bulky calculations and model weights in
project storage. Physical backup/archive retention is separate from admission.

## Worker completion criteria

Deliver the full run inventory, quarantine and relabel ledger, final reviewed
source report, frozen split map, release checksums/counts, A2 role registry,
LONI preflight and warm-start commands, and paired evaluation plan. State exact
remaining blockers. The repository implements the scanner/admission mechanism;
the full LONI audit, numerical repairs, recollection and training still need to
run on the user's machines.
