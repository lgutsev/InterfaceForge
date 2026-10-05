# SiN/TiN rescue: review after package 21 resettlement

## Decision and scope

The completed `MD_Period/audit/source_rescue_resettled21` report covers package
21 at its permanent A2 home. Do not repeat this scan just to reproduce the same
REVIEW counts. Resolve the numerical findings, curate the replacements and A2
roles, then run the final admission audit against the finished source map.

Keep the four existing original-source exclusions: both TiO-Ideal temperatures,
SiN 450 K and TiN 450 K. TiO-Real remains eligible for qualification. No additional
original source is declared defective or accepted for training by this review.
The nominal surviving deck is 26,400 retained frames from 44 MD trajectories;
this is a planned count, not a completed release. Recover from existing model
checkpoints after clean export. Experimental phonon agreement remains a
validation target, not a consequence guaranteed by this cleanup.

Follow the [worker assignment](source-rescue-worker-2026-10-05.md). Raw data,
historical reports, existing exports and models stay unchanged. This document
records evidence and repair decisions; it does not waive the admission gate.

## Reviewed snapshot

| Item | Value |
| --- | --- |
| Report | `MD_Period/audit/source_rescue_resettled21/source_audit.json` |
| Report SHA256 | `5f3d110edf9a1e7b3d8ad5dc6c76ae124da7b24e708c25781e8e056c07a2f35f` |
| Policy SHA256 | `16b63bc75332e177efc8553e8d086aebc26566792cfa4fc5d7a62639a1dc5933` |
| Directory map SHA256 | `ac711703c00311d64454456750a77ad70115e42969751fd4ad98da85f7e14bfe` |
| Status | BLOCKED: 245 REVIEW, 12 QUARANTINED, no FAILED |
| Required-root errors | None |
| Missing optional root | Disabled/unbuilt `A2_DFT/training_sources` |

The optional A2 train root is expected until role-safe curation is complete.
Its absence is not a reason to enable raw package 15 or any audit/reference tree.

| Origin | Report rows | Distinct output-bearing calculations | Role/status |
| --- | ---: | ---: | --- |
| Mapped original MD | 48 | 48 | 44 train REVIEW; 4 train QUARANTINED |
| Lightweight bulk backup | 8 | 0 | QUARANTINED inventory |
| Wadh | 19 | 19 | Reference REVIEW |
| Package 14 | 76 | 38 | Reference REVIEW |
| Package 15 | 48 | 24 | Audit REVIEW until eligible endpoints are curated |
| Package 17 | 10 | 5 | Audit REVIEW |
| Package 21 | 48 | 24 | Audit REVIEW |

All 91 A2 `runs/outputs` duplicate pairs match in OUTCAR and OSZICAR hashes,
as well as the generated SCF and label diagnostics. The 91 missing-input rows
are `outputs/` copies; their corresponding `runs/` copies have inputs. Use the
manifest and input hashes to establish canonical calculations. Do not merge
roles, train both copies, or approve a missing-input row by name alone.

## Active MD stopping diagnostics

The 44 active trajectories contain 132,000 ionic frames: 120,000 interface,
3,000 SiN 300 K, 3,000 TiN 300 K and 6,000 TiO-Real. They have no reported
charge-residual jumps. In every frame the last available charge residual is
printed one electronic iteration before the final step; final-step rms(c) is
absent. The widespread SCF_RESIDUAL flag therefore does not establish failure
of the final energy stopping criterion.

131,999 frames have both printed final energy changes strictly below EDIFF.
One unretained frame, zero-based 2946 in
`mapped/interface/450K/Real/Ti_Term/SiN-TiN-Ti-term`, prints d eps = -0.0001 eV,
exactly the EDIFF = 0.0001 eV boundary. Printed rounding can explain that equality;
do not label it a proven SCF failure. It is absent from the original stride-5
selection. All 26,400 nominal retained active frames pass this printed check.

VASP's EDIFF criterion concerns both energy changes; rms(c) measures an
input/output density difference. See [EDIFF](https://vasp.at/wiki/EDIFF) and
[OSZICAR](https://vasp.at/wiki/OSZICAR). Neither a low energy change nor a generic
residual threshold replaces fixed-geometry E/F/stress convergence controls.
The MD EDIFF = 1e-4 eV and Gamma sampling still need qualification. No blanket
threshold increase or automatic acceptance is justified.

## Package 21: paired energies and stress

These differences are recomputed from the supplied per-frame diagnostics:
`kconv - orig`, E_sigma->0 per atom and maximum absolute difference over all
nine stress components. Package indices identify calculations, not original MD
frame indices. Full force vectors and geometry provenance are absent from this
bundle, so force RMSE and stored-deck identity must be independently recomputed
from the actual manifests and outputs before admission.

| Pair | Orig / kconv folder | dE (meV/atom) | Max abs stress difference (GPa) | Scope |
| --- | --- | ---: | ---: | --- |
| TiN 300 K high | 0000 / 0001 | -3.9362 | 5.2040 | Active-source repair evidence |
| TiN 450 K median | 0002 / 0003 | -2.5296 | 4.8399 | Quarantined-source diagnostic |
| TiN 450 K high | 0004 / 0005 | -2.1762 | 4.9303 | Quarantined-source diagnostic |
| Interface 300 K ideal N, O=0 | 0006 / 0007 | +0.1906 | 0.0226 | Sampled interface |
| Interface 450 K ideal Ti, O=0 | 0008 / 0009 | **+16.5424** | 0.1192 | **Hold comparison; rerun** |
| Interface 300 K ideal N, O=1 | 0010 / 0011 | -0.3074 | 0.0282 | Sampled interface |
| Interface 450 K real N, O=0 | 0012 / 0013 | -0.1975 | 0.0130 | Sampled interface |
| Interface 300 K real Ti, O=1 | 0014 / 0015 | -0.0675 | 0.0228 | Sampled interface |
| Interface 450 K real Ti, O=0.5 | 0016 / 0017 | +0.4214 | 0.0698 | Sampled interface |
| TiO-Real 300 K | 0018 / 0019 | -4.2594 | 0.1699 | Qualify mesh and forces |
| TiO-Real 450 K | 0020 / 0021 | -4.1519 | 0.2423 | Qualify mesh and forces |
| SiN 300 K | 0022 / 0023 | **-137.4910** | **8.2270** | Prioritize relabeling |

The TiN stress differences agree with the earlier systematic anisotropy. The
SiN error remains large in the active 300 K source. A constant stress correction
would not repair the separately reported force errors. Correct all channels at
the same geometry rather than applying an unsupported mean offset.

Six sampled interfaces do not qualify all 40 trajectories. Likewise, one bulk
pair does not certify a mesh across all strained/thermal geometries.

## New finding in the interface energy outlier

The suspect calculation is **package 21, `runs/frame_0009/static`** at:

```text
/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD_Period/A2_DFT/21_stratified_audit/runs/frame_0009/static
```

It is the 308-atom Ti-terminated, oxygen-free ideal interface at 450 K,
`a2s_ifc450_ideal_ti_o0_kconv`. It is not TiO-Ideal bulk.
Its run fingerprint is
`d0dcd1bff370c997875b073b700c5abe0ea1d6213c9fdf35bb65493df08f9927`.

The SCF record reports one residual jump and last available rms(c) = 0.0321 at
iteration 57, followed by stopping at iteration 58. The final printed dE is
-6.1031e-8 eV and d eps is -9.9272e-7 eV, both below EDIFF = 1e-6 eV.
Final-step rms(c) is absent. Its Gamma counterpart, frame 0008, reports no jump
and last available rms(c) = 0.000234. The paired E_sigma->0 difference is
5.09507048 eV per cell. This association warrants an independent-start SCF
control; it does not prove that the residual jump caused the energy difference.

Both calculations already use NELM = 150, not 60. Their 58 steps did not reach
that limit. Merely raising NELM is not an established repair. First preserve
geometry and the actual audited k mesh, inspect the raw electronic history and
state/occupations, and test reproducibility with tighter SCF and independent
starts. A denser-mesh test follows once the SCF state is reproducible.

All package-21 `orig` calculations use EDIFF = 1e-6 and NELM = 150, whereas the
original active MD uses 1e-4 and 60. Thus `orig` should not be described as
literally identical numerical settings. The earlier dispatch report of close
stored-label reproduction can still be correct, but verify it through manifest
source IDs, geometry/order and full E/F/stress comparisons. Never infer source
identity from a nearest energy match.

## Remaining gates

1. Resolve frame 0009 and validate package-21 source provenance and force errors.
2. Complete representative SCF/k/PREC/LREAL/version controls; extend evidence
   to every source admitted for training. Apply the existing staged bulk repair.
3. Curate the 44 A2 train endpoints and separate 16 holdouts with compatible
   numerics; keep 14/17/21 and Wadh outside training.
4. Freeze historical splits, implement same-geometry subset replacement without
   re-splitting, and verify source-to-export E/F/full virial conversions.
5. Run the final fingerprint-bound admission audit on the finished map and
   publish one clean release before launching the recovery fine-tune.

Wadh endpoint review remains as recorded in the earlier review: N2 intermediate
relaxation outliers are not training frames, and its final molecular endpoint
needs geometry/state/convergence checks. Four other Wadh residual warnings need
endpoint review. They do not require blocking an independently qualified train
subset when the collector's required-root and train-source gates are satisfied.
