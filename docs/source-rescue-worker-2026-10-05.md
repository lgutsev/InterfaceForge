# Worker assignment: complete the SiN/TiN rescue before recovery training

## Starting point and deliverable

The full source scan is complete and package 21 is resettled. Use the
[2026-10-05 evidence review](source-rescue-review-2026-10-05.md), the existing
[rescue plan](source-rescue.md), and both campaign YAMLs.

**The laptop worker has no LONI access.** It operates on verified D: copies,
delivered reports and its local repositories. Lavrenty executes all LONI commands,
including submission, remote inspection, final audit, recollection and training.
Use the existing dispatcher for package transfer, result intake and ledger
updates. Numerical work and heavy collection run on allocated nodes; do not
poll scheduler clients in a loop.

| Responsibility | Owner |
| --- | --- |
| Local provenance, frame selection, generators, software tests, scientific review and ready-to-run job packages | Laptop worker |
| Package/result transfer, archive verification, intake and dispatch ledger | Existing dispatch workflow |
| All LONI commands, job submission, remote audit/collection and eventual training | Lavrenty |

Every handoff must state the package/version, local and remote destinations,
input hashes, exact operator command, expected outputs and how to return results.
Keep laptop paths and LONI paths distinct in the profile; do not make the worker
resolve `/ddnB` paths as local files. The owner is already handling the first
Normal/3x3x3 outlier rerun. Record its actual directory, job ID and inputs when
supplied; review the returned outputs before preparing another identical job.

Produce a reviewed, immutable dataset release, its provenance/split ledger and
ready-to-submit recovery jobs. Preserve raw calculations, old exports and model
checkpoints. No full AIMD campaign or foundation-model restart is required just
to complete this rescue. Do not launch production training while its train
sources or numerical replacements remain unqualified.

## 1. Intake and provenance, before new DFT

1. Update InterfaceForge without discarding local changes or commits. Check
   `git status --short --branch`; fetch and merge `origin/main` into the clean
   working branch, resolving conflicts before using the updated files.
2. Read the delivered `review_evidence.tar(1).gz` report bundle on the laptop.
   It came from `MD_Period/audit/source_rescue_resettled21`. Verify its
   report/policy/map hashes against the dated review. Preserve that snapshot.
   `CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface` is the
   operator's remote execution setting, not a local laptop mount.
3. Use `A2_DFT/{14_references,15_pilot,17_audit,21_stratified_audit}` as the
   recorded permanent remote homes. Inspect verified local copies under
   `D:\MLIP_Work_Folder\sin_tin_a2` and the supplied dispatch results/archive
   records. Verify manifests/input hashes and RESETTLE_AUDIT.tsv. If a required
   local member is absent, list the exact relative paths for dispatch to deliver;
   do not assume the earlier OUTCAR/XML-only export contains all inputs.
   Remote-only checks become explicit commands for Lavrenty. Do not restore
   obsolete smoke paths.
4. Resolve all 91 duplicated `runs/outputs` calculations by manifest identity
   and hashes. Retain one canonical record for comparison/export, preserving
   each calculation's original role. Do not alter the historical files.
5. Join every package-21 pair to the exact original source, zero-based frame,
   geometry, cell and atom order through its manifest. Recompute orig versus
   stored E/F/full stress, then kconv versus orig, from full outputs. Report
   E_sigma->0 and free energy separately. Include the EDIFF/NELM differences
   rather than asserting all `orig` tags equal the MD tags. Energy matching
   alone cannot establish identity.

Deliver an intake table and paired-comparison table with input/output hashes,
source IDs, units, sign conventions, SCF stopping data and full force RMSE.

## 2. Priority rerun: package 21 frame 0009

Hold its kconv comparison as unresolved. Also hold the corresponding original
interface family from full-source training acceptance until the discrepancy is
resolved or a qualified curated subset is defined. Do not automatically delete
all interface data or reclassify an audit frame as training.

For Lavrenty only: these lightweight LONI reads locate the evidence; they do not
submit jobs. The laptop worker uses the corresponding delivered local files.

```bash
export CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface
P21="$CER_INTERFACE_BASE/MD_Period/A2_DFT/21_stratified_audit"
tail -n 25 "$P21/runs/frame_0009/static/OSZICAR"
cat "$P21/runs/frame_0009/static/INCAR"
cat "$P21/runs/frame_0009/static/KPOINTS"
```

The owner supplied the exact frame-0009 OUTCAR. It confirms a Gamma-centered
3x3x3 mesh, 16 EDDRMM/ZHEGV warnings and a late 4.91 eV electronic-energy
increase at iterations 42/43. Its original MD temperature history is reasonable
after equilibration; no sampling quarantine is justified from that plot.
The static label remains unqualified. Read the detailed follow-up in the review.

Compare frame 0008 and 0009 geometry, POTCAR order, NELECT, occupations and the
complete SCF trace. Check that the independent-start jobs use no unintended
WAVECAR/CHGCAR. The worker generates control inputs, manifests and launch scripts
locally and passes the package through dispatch. Lavrenty stages/submits them in
new LONI directories and returns outputs through the same workflow:

- First reproduce the same fixed geometry, actual KPOINTS, POTCARs and other
  numerics. Verify KPOINTS against the echoed 3x3x3 mesh; NKPTS=14 is its
  irreducible count. Keep VASP-version effects separate with a 6.5.1-to-6.6.1
  bridge where the licensed binaries are available.
- Prioritize an independent-start `ALGO=Normal`, EDIFF=1e-6 control at the same
  geometry and 3x3x3 mesh to isolate the electronic algorithm. Then tighten
  EDIFF to 1e-7 and compare full E/F/stress. Inspect diagonalization warnings,
  both energy changes and density history. Record mixing/NELMIN interventions
  separately if still needed. NELM is already 150, so increasing the limit alone
  is not the proposed cure. Do not substitute an intermediate pre-jump SCF
  energy for a qualified final endpoint.
- Once the SCF state is reproducible, test the next denser mesh. Compare full
  forces/stress as well as both energies. Apply the same checks to the Gamma
  counterpart if the independent results disagree.

Deliver a pass/hold/relabel decision with reproducibility evidence. If unresolved,
keep the source family on hold and prepare an explicitly reduced qualified
release rather than hiding the anomalous energy with an arbitrary offset.

## 3. Qualify the retained data and perform staged bulk repair

Before evaluating controls, record the campaign's numerical tolerances for
energy/atom, force RMSE and maximum force difference, every stress component,
and phonon displacement force response. Keep these tolerances distinct from
the scanner's generic warning triggers; never choose them afterward to pass.
Reuse package-23 controls where their geometry, settings and hashes match.
Inventory package 23's delivered/local state first. The worker prepares the
missing controls and staged relabel packages, reviews returned E/F/stress, and
records decisions; Lavrenty executes each DFT stage. No worker step requires
SSH, module access or Slurm access.

Use fixed geometries and isolate EDIFF, k mesh, PREC/LREAL and binary effects.
Start with a typical retained frame and the highest retained charge-residual
frame in each of the eight interface temperature/morphology/termination groups
and the four active bulk trajectories. Candidate worst indices are in the dated
earlier review; omit SiN/TiN 450 K from eligible train controls. This first stage
is 24 probe geometries before overlaps with existing controls. It is a triage
set, not a certificate for all 40 interface trajectories. Extend controls to
the remaining source leaves before admitting them; expand temporal/geometry
coverage or relabel where results vary.

| Source | Required action |
| --- | --- |
| TiO-Ideal 300/450 K, SiN 450 K, TiN 450 K | Keep excluded from all active splits; diagnostics only |
| SiN 300 K | Verify denser-k convergence, then replace all 600 retained geometries with qualified E/F/stress |
| TiN 300 K | Run the existing 48-frame first stage, qualify k mesh, then replace the remaining active retained labels if the stage passes |
| TiO-Real 300/450 K | Qualify mesh and SCF/force errors; relabel the planned 240-frame subset first with its historical splits preserved |
| Interfaces | Resolve the ideal-Ti outlier; qualify Gamma use per source, expanding beyond the six package-21 samples |
| A2 pilot and remaining plan | Bridge PREC/LREAL and version including full forces; finish 20 train and 16 holdout labels |
| Wadh | Review final endpoints independently; do not export relaxation intermediates as training |

A staged replacement does not qualify the unreplaced Gamma frames. In each
release explicitly record whether they remain qualified, are held out entirely,
or await further relabeling. Never include both old and replacement labels for
one geometry. The 450 K sampling exclusions persist even if static controls pass.

## 4. Curate, recollect and verify one release

1. On the laptop, recover and freeze the original source/frame/split map from
   existing verified exports, including the deck at
   `D:\Traj_Storage\SiN-TiN_Cer_Interface_BULK` where applicable. Survivors and
   replacement labels retain their splits; report any unrecoverable split
   history explicitly. List any missing provenance files for dispatch delivery.
2. Implement and verify a campaign adapter for partial replacements with exact
   frozen membership. The existing mapper's whole-source replacement support
   does not implement this operation; randomly re-splitting a relabel subset
   is incorrect. Check atom order/cell/positions and replace all E/F/virial
   channels together. Fail on missing, duplicate or wrong-source replacements.
3. Prepare the curated tree locally for eventual placement at the remote
   `A2_DFT/training_sources`: exactly 44 eligible A2 train endpoints with the
   required single-frame inputs and hash-bound `roles.json`. Separate
   all 16 holdouts for evaluation. Multi-step relaxations require an explicit
   endpoint adapter/provenance; their intermediate steps are not extra A2 labels.
   Keep 14, 17, 21, Wadh and BBVO outside the train map.
4. Set the final replacement/curated paths and A2 enablement before the final
   audit. Record scientific decisions against each source fingerprint and
   applicable review code. Keep unresolved train sources excluded or blocked.
5. The worker prepares and tests the final audit/collection launchers and exact
   operator commands. Lavrenty places the curated sources, submits the final
   source audit into a new directory and returns its reports. Bind final
   decisions to remote source IDs/fingerprints; a laptop audit against another
   profile/map is not a substitute for the LONI admission report. After intended
   train sources qualify, Lavrenty points `source_rescue_current` at that report
   without changing the already hashed map and launches collection into a fresh
   rescue release. Use the existing InterfaceForge launchers; do not reuse the
   earlier `source_rescue_initial` output or force-write old datasets.
6. The worker verifies the returned export checks and reports. Generate an
   allocated-node validation job if complete exports remain remote; Lavrenty
   runs it and returns the numeric checks and hashes. Verify exclusions and role
   counts, unique geometry/source IDs, zero split
   overlap, historical membership, finite arrays, MACE/DeePMD membership and
   numeric labels. Independently verify energy convention, force units and
   all nine virial components/sign against source stress and cell volume.
   Preserve `virial.npy` in every system training with virial loss.
7. Checksum the release, role registry, split ledger, source qualification and
   training inputs. Nominal original count is 26,400 before additional holds or
   subset decisions, plus 44 A2 train and 16 separate A2 holdouts. Report actual
   train/validation/test/A2 counts; do not silently force the nominal total.

## 5. Prepare and validate recovery training

Prepare new training outputs using existing verified checkpoints, a declared
fine-tune learning-rate/loss schedule and an explicit A2 replay sampling ratio.
The worker writes/tests configs and launch scripts using the recorded remote
checkpoint paths. Keep model weights on project storage; local preparation does
not require copying weights onto the laptop. Lavrenty runs preflight, submits
training after the release gates pass, and returns evaluation outputs for review.
Retain paired seeds. Compare A1 and recovery on identical qualified clean E/F/
stress targets, the 16 A2 holdouts, TiN C11/C12/C44 and q-half sign/frequency,
and SiN relaxed-ion C11/G and equilibrium cell. Report TiO-Real separately,
interface retention and broader phonon modes where available. Removing bad
test frames does not by itself demonstrate model improvement.

Deliver the final reviewed report, decisions and relabel ledger; frozen splits;
cross-format/source-label checks; release hashes/counts; exact LONI preflight
and training commands; replay schedule; paired evaluation plan; and any remaining
holds. The worker may proceed with authorized generation, repair, checks and
commits on the laptop; it supplies executable handoffs instead of claiming
remote execution. Production training starts only when the concrete release
passes its recorded gates. Do not send another generic request for the same
audit bundle.
