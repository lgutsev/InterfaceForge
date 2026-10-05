# Worker assignment: complete the SiN/TiN rescue before recovery training

## Starting point and deliverable

The full source scan is complete and package 21 is resettled. Use the
[2026-10-05 evidence review](source-rescue-review-2026-10-05.md), the existing
[rescue plan](source-rescue.md), and both campaign YAMLs. Work on the laptop for
generation, provenance and software checks; submit numerical work and heavy
recollection on allocated LONI nodes. Do not poll scheduler clients in a loop.

Produce a reviewed, immutable dataset release, its provenance/split ledger and
ready-to-submit recovery jobs. Preserve raw calculations, old exports and model
checkpoints. No full AIMD campaign or foundation-model restart is required just
to complete this rescue. Do not launch production training while its train
sources or numerical replacements remain unqualified.

## 1. Intake and provenance, before new DFT

1. Update InterfaceForge without discarding local changes or commits. Check
   `git status --short --branch`; fetch and merge `origin/main` into the clean
   working branch, resolving conflicts before using the updated files.
2. Set `CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface`.
   Read the existing report at
   `$CER_INTERFACE_BASE/MD_Period/audit/source_rescue_resettled21`. Verify its
   report/policy/map hashes against the dated review. Do not overwrite it.
3. Use `A2_DFT/{14_references,15_pilot,17_audit,21_stratified_audit}` as the
   permanent homes. Verify manifests/input hashes, RESETTLE_AUDIT.tsv and
   dispatch archive/resettle records. Do not restore obsolete smoke paths.
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

On LONI, these lightweight reads locate the evidence; they do not submit jobs:

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
WAVECAR/CHGCAR. Generate controls in new directories through the campaign and
dispatch workflow:

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

1. Recover and freeze the original source/frame/split map from existing exports
   before replacement. Survivors and replacement labels retain their splits;
   report any unrecoverable split history explicitly.
2. Implement and verify a campaign adapter for partial replacements with exact
   frozen membership. The existing mapper's whole-source replacement support
   does not implement this operation; randomly re-splitting a relabel subset
   is incorrect. Check atom order/cell/positions and replace all E/F/virial
   channels together. Fail on missing, duplicate or wrong-source replacements.
3. Curate exactly 44 eligible A2 train endpoints into `A2_DFT/training_sources`
   with the required single-frame inputs and hash-bound `roles.json`. Separate
   all 16 holdouts for evaluation. Multi-step relaxations require an explicit
   endpoint adapter/provenance; their intermediate steps are not extra A2 labels.
   Keep 14, 17, 21, Wadh and BBVO outside the train map.
4. Set the final replacement/curated paths and A2 enablement before the final
   audit. Record scientific decisions against each source fingerprint and
   applicable review code. Keep unresolved train sources excluded or blocked.
5. Submit the final source audit into a new directory. After it qualifies the
   intended train sources, point `source_rescue_current` at that report without
   changing the already hashed map, and collect into a fresh rescue release.
   Use the existing launchers from the InterfaceForge checkout; do not reuse
   the earlier `source_rescue_initial` output or force-write old datasets.
6. Verify exclusions and role counts, unique geometry/source IDs, zero split
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
Retain paired seeds. Compare A1 and recovery on identical qualified clean E/F/
stress targets, the 16 A2 holdouts, TiN C11/C12/C44 and q-half sign/frequency,
and SiN relaxed-ion C11/G and equilibrium cell. Report TiO-Real separately,
interface retention and broader phonon modes where available. Removing bad
test frames does not by itself demonstrate model improvement.

Deliver the final reviewed report, decisions and relabel ledger; frozen splits;
cross-format/source-label checks; release hashes/counts; exact LONI preflight
and training commands; replay schedule; paired evaluation plan; and any remaining
holds. The worker may proceed with authorized generation, repair, checks and
commits; production training starts only when the concrete release passes its
recorded gates. Do not send another generic request for the same audit bundle.
