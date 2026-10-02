# SiN/TiN source rescue: review of the corrected LONI audit

## Evidence and decision

Reviewed the uploaded `review_evidence.tar.gz` from
`MD_Period/audit/source_rescue_parser_fixed` after job 1065756.
The report's SHA256 is
`09837fa503762fae442ec1222fd86f804e78f2887baec0e5a7c7bbce04999d30`.
Policy SHA256:
`5cea4774bfd82268caa3d0db06206064db5f4056aa2c9d7ea15be744e670a70e`.
Map SHA256:
`88de455658e69e81a78a842b322f71dd3ee79f0ecd640ee305cfdf9d8dc3843e`.
These identify the reviewed snapshot; this document does not grant admission to
changed sources or a new policy.

Keep both TiO-Ideal MD runs quarantined. Preserve all raw data. No further
training source is automatically quarantined or accepted from this bundle.
Continue the staged label repair in [the rescue plan](source-rescue.md), then
recollect into a new release and warm-start recovery from existing checkpoints.
The audit provides no guarantee that experimental phonons will be reproduced.

## What was actually scanned

| Origin | Report rows | Distinct OUTCAR byte hashes | Role |
| --- | ---: | ---: | --- |
| Mapped original MD | 48 | 48 | 46 review, 2 quarantined train sources |
| Lightweight bulk backup | 8 | 0 | Quarantined inventory |
| Wadh | 19 | 19 | Reference |
| A2 package 14 | 76 | 38 | Reference |
| A2 package 15 | 48 | 24 | Audit; selected train endpoints still need qualification |
| A2 package 17 | 10 | 5 | Audit |
| A2 package 21 | 0 | 0 | **Absent from this scan** |

There are 209 rows, 199 REVIEW, 10 QUARANTINED, no FAILED and no root errors.
The 67 A2 calculations occur twice: under `runs/` and `outputs/`. Both OUTCAR
and OSZICAR hashes match within each pair. All 67 missing-input records belong
to the `outputs/` copies, whose INCAR/POSCAR/KPOINTS/POTCAR are absent there;
their `runs/` partners contain the inputs. This is evidence of duplicated
packaging, not evidence that those calculations ran without inputs. Keep each
fingerprint and role independent; never waive a missing input by finding a
similarly named run. Use the manifest/input hashes and verified archive to
establish a canonical calculation before curated export. Do not train both
copies or admit a reference/audit copy through a train counterpart.

Package 21 was optional in the policy because its resettlement was uncertain.
Its absence was therefore not reported as a required-root error. Resolve its
current location through the dispatch ledger, verify its archive and intake,
and add that verified root to the campaign policy. Require it for this rescue's
final scientific review. Do not assume the previously delivered 24 tasks were
covered by this audit, or move them back into the smoke tree.

## Active MD findings

The 46 active trajectories each contain 3,000 frames: 138,000 source frames,
27,600 retained at the configured stride 5, before further exclusions or A2
additions. All 138,000 OSZICAR/label frame indices and counts align. E0 agrees
with the ASE label energy to at most 0.0000501 eV per cell, consistent with the
coarser printed OSZICAR precision. This validates this alignment check, not the
previous exported deck or all force/stress conversions.

No active trajectory has a reported SCF-limit, residual-jump, index-reset,
nonfinite-label, frame-count, energy-jump, large-force, large-stress or sampled
short-contact finding. Contacts were sampled every 50 frames; absence of a
finding is not an exhaustive geometry certification.

All 46 use echoed EDIFF = 1e-4 eV, Gamma sampling and saved LREAL = Auto.
Each has the same three review codes: KPOINT_CONVERGENCE_UNVERIFIED,
LREAL_MODE_UNVERIFIED and SCF_RESIDUAL.

| Family | Source frames | Charge residual median / p95 / maximum | Maximum SCF iterations | Maximum force (eV/A) | Maximum absolute stress component (GPa) |
| --- | ---: | --- | ---: | ---: | ---: |
| Interfaces | 120,000 | 0.00838 / 0.0119 / 0.0285 | 32 | 10.32 | 16.36 |
| Si3N4 | 6,000 | 0.00337 / 0.00600 / 0.00925 | 14 | 8.53 | 18.40 |
| TiN | 6,000 | 0.00523 / 0.00645 / 0.0104 | 32 | 4.73 | 6.09 |
| TiO-Real | 6,000 | 0.00254 / 0.00426 / 0.00647 | 34 | 3.95 | 4.45 |

137,982 of 138,000 active frames exceed the generic 0.001 charge-residual
trigger. The report retains the last available rms(c), which can precede the
last electronic iteration. It did not retain that iteration's dE/d eps, so the
bundle cannot establish the energy stopping criterion or quantify SCF error.
VASP defines EDIFF using both free-energy and band-energy changes, and defines
rms(c) as the input/output charge-density difference:
[EDIFF](https://vasp.at/wiki/EDIFF), [OSZICAR](https://vasp.at/wiki/OSZICAR).
Do not convert this broadly triggered review threshold into a blanket failure,
silently increase it to make the report green, or treat energy convergence as
proof of converged E/F/stress. Compare fixed geometries at tighter EDIFF.

Separately reparsed all 24,000 ionic summaries in the eight previously supplied
bulk OSZICAR files using the new energy-change diagnostics. Each local file's
SHA256 matches the corresponding mapped original in this uploaded report.
Every printed final dE and d eps satisfies EDIFF = 1e-4 eV, including the
quarantined TiO-Ideal trajectories. Thus the active bulk residual flag is not
evidence of failure to meet the original energy stopping criterion, and meeting
that criterion does not repair TiO-Ideal's sampling pathology or the bulk
k-point errors. The interface electronic lines were not in the uploaded bundle;
their energy stopping criterion still needs the additional diagnostics.

The saved Auto / echoed T ambiguity can be reviewed through package 21's
reported exact-settings reproduction and original input provenance. First
inspect the actual paired evidence and its hashes. Do not equate Auto with
literal .TRUE. or accept unaudited source families from another family's test.

## Temperature findings below the default warning threshold

Statistics below use the last 2,000 frames after the declared burn-in.

| Trajectory | Mean +/- std (K) | p05 / median / p95 (K) | Action |
| --- | --- | --- | --- |
| Si3N4 450 K | 450.8 +/- 162.5 | 247 / 427 / 731 | Review thermostat cycles, equilibration and geometry coverage |
| TiN 450 K | 445.2 +/- 81.2 | 329 / 444.5 / 570.1 | Review cycling and coverage with the relabel campaign |
| TiO-Ideal 300 K | 308.3 +/- 571.8 | 2 / 24 / 1884.85 | Keep quarantined |
| TiO-Ideal 450 K | 486.1 +/- 763.7 | 9 / 71 / 2130.05 | Keep quarantined |

The Si3N4 and TiN runs do not trip the configured std/target threshold 0.75.
Their broad distributions still deserve scientific review. Small-cell Nose
cycles can broaden a distribution; this evidence alone does not show broken
SCF labels or require discarding the structures. Plot the histories and compare
geometry/force distributions across temperature blocks. Do not silently change
the historical retained frames or splits while reviewing.
TiO-Ideal reaches 100.62 GPa in absolute stress component across the two runs,
reinforcing its exclusion for this campaign.

## Wadh reference outlier

The only LARGE_FORCE and ENERGY_JUMP reference is `wadh/N2_gas`, fingerprint
`5984948e0d93b074f3f01c210778dd1d8da8b621a87ba13ce1c08f09426c0669`.
This is a nine-step molecular relaxation (IBRION=2), not training MD.
Its zero-based frame 2 reaches 97.65 eV/A; its final frame 8 has energy
-16.65201094 eV and maximum force 0.001322 eV/A, below the saved 0.02 eV/A
ionic stopping target. The outliers occur along relaxation, not at the endpoint.
Retain it as reference-only. Verify final N-N geometry, singlet state, final SCF
and molecular box/basis convergence before accepting the endpoint as an energy
reference. Do not export the intermediate relaxation frames into training.

Four other Wadh references have residual warnings: Si_mp149, Ti2O3_mp458,
Ti5Si4_mp505527 and TiO2_mp390. Inspect their final endpoints separately;
intermediate-step warnings are not automatically endpoint failures.

## Concrete next worker assignment

1. Locate and qualify package 21; link its actual paired results and manifest
   to this source review. Preserve the already verified permanent 14/15/17
   homes. Resolve A2 `runs/outputs` copies through manifests and input hashes.
2. Add tighter-SCF controls to the pending staged numerical checks. Use the same
   fixed geometry, POTCARs, k mesh, smearing, PREC/LREAL and energy convention;
   change EDIFF from 1e-4 to 1e-6 eV, then to 1e-7 where the difference matters.
   Keep the VASP-version bridge explicit. Set a sufficient NELM and inspect
   actual stopping behavior. Compare E per atom, force RMSE/maximum difference
   and all stress components against declared application tolerances. Reuse
   pending package-23 geometries where possible; route jobs through dispatch.
3. Cover all eight temperature/morphology/termination interface groups and all
   six active bulk trajectories with a worst retained-residual frame and a
   typical retained frame. These are targeted statics, not fresh AIMD. The
   table below supplies the worst retained-residual candidates; indices are
   zero-based original OUTCAR indices and satisfy index modulo 5 = 0. This is
   a review probe set, not a new training split or a convergence certificate.
4. Review the Si3N4/TiN 450 K thermal histories and the N2 endpoint. Keep the
   TiO-Ideal exclusions. Proceed with same-geometry Si3N4/TiN label repair and
   qualified TiO-Real/interface checks from the existing rescue plan.
5. Record full-source acceptance only after all applicable review codes have
   supporting evidence. Otherwise keep REVIEW, quarantine, or relabel. For
   subsets, create curated sources and bind decisions to their own fingerprints;
   a run-level accept entry cannot authorize only selected frames in that run.
6. Freeze original source/frame/split identities, exclude superseded labels,
   curate only A2 train roles, perform cross-format E/F/full-virial checks, and
   publish the clean release before warm-start recovery and paired-seed tests.

| Source ID (under mapped/) | Zero-based original frame | Last available rms(c) |
| --- | ---: | ---: |
| interface/300K/Ideal/N_Term/SiN_TiN_N-term | 740 | 0.0170 |
| interface/300K/Ideal/Ti_Term/SiN-TiN-Ti-term_O_x0.25 | 2810 | 0.0127 |
| interface/300K/Real/N_Term/SiN_TiN_N-term_O_x1.00 | 2210 | 0.0285 |
| interface/300K/Real/Ti_Term/SiN-TiN-Ti-term_O_x0.25 | 2670 | 0.0117 |
| interface/450K/Ideal/N_Term/SiN_TiN_N-term | 1565 | 0.0202 |
| interface/450K/Ideal/Ti_Term/SiN-TiN-Ti-term | 235 | 0.0132 |
| interface/450K/Real/N_Term/SiN_TiN_N-term | 1510 | 0.0206 |
| interface/450K/Real/Ti_Term/SiN-TiN-Ti-term_O_x1.0 | 2900 | 0.0177 |
| bulk/SiN-Bulk_300K/. | 0 | 0.00724 |
| bulk/SiN-Bulk_450K/. | 2735 | 0.00855 |
| bulk/TiN-Bulk_300K/. | 2670 | 0.00786 |
| bulk/TiN-Bulk_450K/. | 2650 | 0.00838 |
| bulk/TiO-Bulk-Real_300K/. | 195 | 0.00622 |
| bulk/TiO-Bulk-Real_450K/. | 935 | 0.00507 |

## Audit reporting changes after this review

Subsequent reports list missing optional roots and byte-identical output
groups. They retain the last dE, d eps, wavefunction rms, last available charge
residual's iteration, final-step charge residual (possibly absent), effective
EDIFF and whether the printed energy changes satisfy that threshold.
These additions do not alter roles, merge duplicates, waive residual warnings,
or grant admission. The energy criterion is diagnostic and can be affected by
printed precision; EDIFF=0 or unknown produces no inferred criterion result.
The uploaded audit remains immutable; use a fresh directory for a new scan.

Validation after these reporting changes: 697 pytest tests passed (including
the 48 focused audit/launcher tests); repository Ruff and both launchers' shell
syntax checks passed. The 24,000 original bulk ionic summaries were reparsed
without hard failures, and all 67 duplicated A2 OUTCAR/OSZICAR pairs were
checked against the uploaded hashes. Software checks do not replace the pending
scientific controls above.
