# SiN/TiN rescue: laptop worker progress, 2026-10-05

This records what the laptop worker completed against the
[worker assignment](source-rescue-worker-2026-10-05.md), with its evidence and the
exact operator handoffs. All raw calculations, earlier reports, exports and
checkpoints are unchanged. Nothing was run on LONI and nothing was pushed.

## Status at a glance

| Step | State | Evidence |
| --- | --- | --- |
| 1.1 Update InterfaceForge | Done: branch `source-rescue-2026-10-05` from `origin/main` (289c21f) in a separate worktree; the uncommitted work on `matcalc-property-validation` is untouched | `git worktree list` |
| 1.2 Delivered resettled21 bundle | Done: report, policy and map hashes match the review; all 910 A2 files it fingerprints match the D: copies | Section 1, "Resettled21 bundle" |
| 1.3 Local A2 copies and RESETTLE_AUDIT | Done: 3,259/3,259 audited members of 14, 15, 17, 21 verified by size and SHA256; nothing missing or unlisted | `intake.tsv` |
| 1.4 91 `runs/outputs` duplicates | Done: 91/91 OUTCAR and OSZICAR identical; one canonical record each (`runs/`, which holds the inputs); roles kept from the manifests | `intake.tsv` |
| 1.5 Package-21 source join and paired comparison | Done from full outputs | `pairs.tsv`, table below |
| 2 Frame 0009 | Comparison held. Control package 23 written and preflighted; the owner's ALGO=Normal/EDIFF=1E-6 run is not duplicated | Below |
| 3 Tolerances | Recorded before any new control result: `examples/source-rescue/numerical_tolerances.yaml`, status **proposed** | Owner sign-off pending |
| 3 Triage probe set | 24 geometries selected (`probe_selection.tsv`); package not yet written | Below |
| 4.1 Frozen split map | Done: 28,800 frames frozen, DeePMD cross-checked frame by frame | `split_ledger.tsv` |
| 4.2 Partial-replacement adapter | Done and tested; diagnostic cleanup release verified end to end | `iface-rescue-release` |
| 4.3–4.7, 5 | Not started: they need the relabels, the 36 remaining A2 labels and the final LONI audit | Blockers |

New tools (both standalone, standard InterfaceForge dependencies):

- `iface-rescue-intake` (`src/interfaceforge/rescue_intake.py`): `intake` (resettle/duplicate/
  canonical verification), `pairs` (package-21 orig/kconv/stored joined through the manifest),
  `compare` (numerics and complete SCF traces of two statics, optionally judged against the
  declared tolerances).
- `iface-rescue-release` (`src/interfaceforge/rescue_release.py`): `freeze` (historical split
  ledger plus DeePMD cross-check) and `assemble` (same-geometry partial replacement under
  `examples/source-rescue/release_policy.yaml`).

Outputs are on `D:\MLIP_Work_Folder\sin_tin_a2\rescue_2026-10-05\`; the small tables are also in
[`docs/source-rescue-2026-10-05/`](source-rescue-2026-10-05/).

| File | SHA256 |
| --- | --- |
| `intake/intake.tsv` | `1abec0865e9ef86bfc38f44c263c22f171d77bb5ad03a18ab2169e4103eac089` |
| `pairs/pairs.tsv` | `247f39429c62521c470bc21559059b22e6b001316685202d6dda7b00d91e8751` |
| `pairs/pairs.json` | `815ec3e045ce0d2ec6cada57b538cbf3d67372a59fb90035f5c779500fe897bd` |
| `cmp0809/scf_traces.tsv` | `34c9ceb206fd57086738d4d2b035df24e2d725121b6657deb65f5234758160b8` |
| `freeze/split_ledger.tsv` | `ed7ff1b96c7d706fd26b7f2ac385b6960a0988d72b593cb2acc0c848aaefab56` |
| `freeze/freeze_report.json` | `5711b67af827cc47082315c63dc5e034b5ed48542709ac717933eb0d9a0440d9` |
| `probe_selection.tsv` | `4aba055e572007a203b71a9f660faee5d692f900a5b470ee9c268d2f956bf441` |

## 1. Intake

### Resettled21 bundle

`D:\MLIP_Work_Folder\sin_tin_a2\REVIEW\review_evidence.tar.gz` (sha256
`8ac4b43ee2b5b11a8e09a69be2cea5434402657d57a2ac8b1cad918841235589`, 16,237,135 bytes) was
delivered on 2026-10-05. It replaced the 2026-10-02 snapshot at the same path, whose recorded hashes
were archive `e3367fab…` and `source_audit.json` `09837fa5…`. An immutable copy of the new bundle is
`D:\MLIP_Work_Folder\sin_tin_a2\rescue_2026-10-05\review_evidence_resettled21_8ac4b43e.tar.gz`.

| Item | In the bundle | Review / repository |
| --- | --- | --- |
| `source_audit.json` | `5f3d110edf9a1e7b3d8ad5dc6c76ae124da7b24e708c25781e8e056c07a2f35f` | equal to the review |
| Policy (`examples/source-rescue/sintin.yaml`) | `16b63bc75332e177efc8553e8d086aebc26566792cfa4fc5d7a62639a1dc5933` | equal to the review and to `origin/main` |
| Map (`periodic_nitride.yaml`) | `ac711703c00311d64454456750a77ad70115e42969751fd4ad98da85f7e14bfe` | equal to the review and to `origin/main` |
| Status | BLOCKED, 245 REVIEW, 12 QUARANTINED, no root errors | equal to the review |

Its rows are:

| Rows | Status | Role |
| --- | --- | --- |
| 44 mapped | REVIEW | train |
| 4 mapped | QUARANTINED | train |
| 8 bulk_inventory | QUARANTINED | train |
| 19 Wadh | REVIEW | reference |
| 76 package 14 | REVIEW | reference |
| 48 package 15 | REVIEW | audit |
| 10 package 17 | REVIEW | audit |
| 48 package 21 | REVIEW | audit |

All 910 A2 input and output files it fingerprints (INCAR, KPOINTS, POSCAR, POTCAR, CONTCAR, OSZICAR,
OUTCAR) match the local D: copies by SHA256, with none missing. So the laptop intake below works on
exactly the files the LONI report audited.


Local copies are `D:\MLIP_Work_Folder\sin_tin_a2\{14_references,15_pilot,17_audit,21_stratified_audit}`.
The 21 archive `resettle_20261002_1736_21_vasp_a2_stratified_audit.tar.gz` matches the desk ledger
(`e4f032799636d2ee43cea199cf02f9ef1d45d51f80e6c235f212b655dbb1aa7e`).

| Package | Audited members verified | Calculations | runs = outputs | Manifest role(s) |
| --- | ---: | ---: | ---: | --- |
| 14_references | 1,223 / 1,223 | 38 | 38 | reference |
| 15_pilot | 775 / 775 | 24 | 24 | train (still gated; see below) |
| 17_audit | 173 / 173 | 5 | 5 | audit |
| 21_stratified_audit | 1,088 / 1,088 | 24 | 24 | audit |

Each calculation's executed INCAR equals its staged `INCAR.<level>` tag by tag. Its KPOINTS equals
the staged file. No calculation reached NELM. Package 15's manifest role is `train`, but it stays out
of the train map until the PREC/LREAL/version bridge and endpoint curation (step 4.3).

### Package 21 paired comparison

The deck frame is found through the manifest's `source_run`/`source_frame`/`split`
(zero-based OUTCAR index), never by energy. For every pair, the deck frame equals
`dataset_frames.extxyz` exactly (geometry, E, F, virial). The executed POSCAR equals the deck geometry
to 1.5e-13 A with the same atom order, and `kconv` uses the identical POSCAR.

Two settings differ from the original MD:

| Setting | Original MD | Package 21 |
| --- | --- | --- |
| EDIFF | 1E-4 (VASP default) | 1E-6 |
| NELM | 60 (VASP default) | 150 |

The rest of the electronic settings match: ALGO Fast, PREC Med, LREAL Auto, ISMEAR 0, SIGMA 0.1, IVDW 11.

Orig minus stored, E_sigma->0 (meV/atom) / force RMSE and max component (eV/A) / max of 9 stress
components (GPa), then kconv minus orig with the same channels. kconv's free energy differs from its
E_sigma->0 change by at most 0.29 meV/atom (TiN 450 K hi).

| Pair (orig/kconv) | Deck frame | Orig - stored | Kconv - orig | kconv SCF |
| --- | --- | --- | --- | --- |
| TiN 300 K hi (0000/0001) | TiN-Bulk_300K 295 train | -0.0006 / 0.002, 0.012 / 0.019 | -3.94 / 0.080, 0.26 / 5.20 | clean |
| TiN 450 K med (0002/0003) | TiN-Bulk_450K 2195 train | -0.0009 / 0.002, 0.004 / 0.013 | -2.53 / 0.107, 0.33 / 4.84 | clean |
| TiN 450 K hi (0004/0005) | TiN-Bulk_450K 2555 train | -0.0005 / 0.004, 0.009 / 0.006 | -2.18 / 0.106, 0.39 / 4.93 | clean |
| Ifc 300 K ideal N O0 (0006/0007) | 1430 train | -0.0006 / 0.002, 0.029 / 0.002 | +0.19 / 0.015, 0.09 / 0.023 | clean |
| **Ifc 450 K ideal Ti O0 (0008/0009)** | 2725 train | -0.0002 / 0.001, 0.006 / 0.002 | **+16.54** / 0.035, 0.20 / 0.119 | **jump; +2.79 eV rise after settling** |
| Ifc 300 K ideal N O1 (0010/0011) | 575 train | +0.0002 / 0.002, 0.013 / 0.004 | -0.31 / 0.016, 0.07 / 0.028 | clean |
| Ifc 450 K real N O0 (0012/0013) | 730 train | +0.0002 / 0.001, 0.005 / 0.004 | -0.20 / 0.005, 0.02 / 0.013 | clean |
| Ifc 300 K real Ti O1 (0014/0015) | 2670 train | -0.0005 / 0.002, 0.014 / 0.003 | -0.07 / 0.026, 0.13 / 0.023 | clean |
| Ifc 450 K real Ti O0.5 (0016/0017) | 1275 train | -0.0004 / 0.002, 0.007 / 0.001 | +0.42 / 0.034, 0.15 / 0.070 | clean |
| TiO-Real 300 K (0018/0019) | 1060 train | -0.0007 / 0.005, 0.009 / 0.001 | -4.26 / 0.081, 0.34 / 0.170 | clean |
| TiO-Real 450 K (0020/0021) | 1010 train | -0.0011 / 0.002, 0.013 / 0.003 | -4.15 / 0.082, 0.31 / 0.242 | clean |
| SiN 300 K med (0022/0023) | SiN-Bulk_300K 2390 train | +0.0005 / 0.002, 0.020 / 0.006 | **-137.49** / 0.192, 0.50 / **8.23** | clean |

"Clean" means: no residual jump, no energy rise after the SCF first settled (|dE| < 1 meV), and
NELM not reached. The largest rise in any other kconv run is 0.025 eV (TiN 300 K hi). Orig-vs-stored
agreement confirms source identity and the stored E/F/virial conversion at the 1e-3 meV/atom,
0.005 eV/A RMSE and 0.02 GPa level. It does not qualify the Gamma labels.

## 2. Frame 0009: held

The local `runs/frame_0009/static/OUTCAR` hashes to
`d5eb657d10083f871e7e206f575e64015931fefbfb1c13af5d7127449e298b8e`. That is the owner-supplied
`OUTCAR(4)` recorded in the review, so its OUTCAR evidence is on this laptop.

Frame 0008 vs 0009 (`cmp0809/compare.json`, full traces in `scf_traces.tsv`):

| Item | 0008 (Gamma) | 0009 (3x3x3) |
| --- | --- | --- |
| POSCAR, POTCAR | identical bytes | identical bytes |
| POTCAR titles | PAW_PBE N 08Apr2002, Si 05Jan2001, Ti_sv 26Sep2005 | same, same order |
| NELECT | 2056 | 2056 |
| NBANDS | 1248 (KPAR 1) | **1236** (KPAR 4; default rounding) |
| KPOINTS / NKPTS | 1x1x1 MP / 1 | Gamma 3x3x3 / 14 irreducible |
| ISTART / ICHARG, restart files | 0 / 2, none present | 0 / 2, none present |
| ALGO / NELMDL / AMIX / BMIX | Fast (IALGO 68) / -5 / 0.40 / 1.00 | same |
| PSCENC, TEWEN, EATOM | equal | equal |
| EENTRO (final) | -0.488 eV | -0.425 eV |
| EDDRMM ZHEGV warnings | 0 | 16 |
| SCF | settles at iteration 19; 58 steps; last rms(c) 2.3e-4 | settles at 25; jump at 42; 58 steps; last rms(c) 0.032 |

The difference 0009 - 0008 is +16.54 meV/atom in E_sigma->0 and +16.65 meV/atom in free energy. Forces
differ by 0.035 eV/A RMSE (0.20 eV/A max) and stress by up to 0.12 GPa. Against the proposed
`gamma_label_use` class every channel and the SCF gate fail, so the comparison is **HOLD**. That is
consistent with the review; it neither qualifies nor rejects Gamma for this family.
`release_policy.yaml` holds the whole `interface_450K_Ideal_Ti_Term_SiN-TiN-Ti-term` source.

**Owner's control (to record when supplied):** the ALGO=Normal / EDIFF=1E-6 / 3x3x3 independent-start
rerun needs its LONI directory, job ID, VASP module and input hashes recorded here. Review its
returned outputs with `compare` before any further identical job.

### Package 23 (batch 08): ready for submission

The generator is sin-tin-a2-gen `a2-pilot-generator` 9c3cb1b (`a2_campaign.py write --only 23`).
In the desk it is `loni_smoke_tests` 5af681c, folder `batch08_2026-10-05/23_vasp_a2_ifc_scf_controls`.
The route is 23 -> sin-tin-mlip-validation. Preflight PASSED.

Every frame copies 21/frame_0009's POSCAR, KPOINTS, POTCAR.names and TRAINING_RUN byte for byte,
hash-checked against 21's manifest, and adds explicit ISTART 0 / ICHARG 2. The job refuses a level
directory that already holds WAVECAR/CHGCAR/CHG. Resources are 2 nodes x 64 ranks, as in 21, so the
default NBANDS stays 1236.

| Frame | Case | Change vs 0009 | Module | INCAR sha256 (prefix) |
| --- | --- | --- | --- | --- |
| 0000 | `repro_fast_e6_v651` | none (reproduction) | vasp6/6.5.1-cpu | 401902d040d5 |
| 0001 | `repro_fast_e6_v661` | binary only | vasp6/6.6.1-cpu | 655287957ed6 |
| 0002 | `normal_e7_v651` | ALGO Normal, EDIFF 1E-7 (pairs with the owner's Normal/1E-6) | vasp6/6.5.1-cpu | fc6a420638f7 |

The shared inputs are POSCAR `4135f47f0b72…`, KPOINTS `e66e48a8c706…` and POTCAR.names
`6c6265f5264e…`. The package hashes are manifest.json
`7f47b7702b436b7c83aa33b29a91d650ab0395b376304d16c56780deac0ae261` and run_vasp.slurm
`5ac869a4ec088eecda501673c8eb8175f97409b01952286cd5692be9e0b22ff3`.

Operator steps for Lavrenty:

1. On the laptop, push the desk (`loni_smoke_tests` main, 2 commits ahead) and the generator branch
   when satisfied.
2. On LONI:

   ```bash
   cd /work/lgutsev/loni_smoke_tests && git pull
   bash submit_smokes.sh --dry-run 23
   bash submit_smokes.sh 23
   ```

The expected outputs are `outputs/frame_000{0,1,2}/static/{OUTCAR,OSZICAR,vasprun.xml,CONTCAR,vasp.out}`,
`vasp_binary.txt`, `training_settings.txt` and `POTCAR.used`. Each frame is one 308-atom static at
3x3x3; 21's frame 0009 took 58 SCF steps under the 24 h cap.

Return path: `bash export_results.sh` puts the batch08 archive into `D:\MLIP_Work_Folder\hpc_smoke_tests`,
then the desk runs `dispatch.py intake`.

To review, compare each returned frame against 21's frame 0009 and frame 0008:

```bash
python -m interfaceforge.rescue_intake compare <21>/runs/frame_0009/static <23>/runs/frame_0000/static --labels f0009 repro651 --output <dir> --tolerances examples/source-rescue/numerical_tolerances.yaml --class reproducibility
```

Use class `reproducibility` for 0000, `convergence` for the 6.6.1 bridge (0001) and for 0002
against the owner's Normal run.

Decision rules, written before any result exists:

- **Fast/6.5.1 reproduces the jump:** the Fast endpoint is unstable. Decide frame 0009 from the
  Normal runs, if the owner's run and 0002 agree within `convergence` and pass the SCF gate.
  Then run the 4x4x4 stage.
- **Fast/6.5.1 does not reproduce the jump:** the event is stochastic in RMM-DIIS. Hold all
  Fast-only 3x3x3 labels of this family until Normal confirms.
- **Normal runs disagree, or reach NELM:** keep the family on hold and propose a reduced
  qualified release. Never apply an energy offset.

## 3. Tolerances and the triage probe set

[`numerical_tolerances.yaml`](../examples/source-rescue/numerical_tolerances.yaml) defines four
classes, each covering energy per atom, force RMSE, max force component and every stress component:

| Class | Energy (meV/atom) | Force RMSE (eV/A) | Max force (eV/A) | Stress (GPa) | Other |
| --- | ---: | ---: | ---: | ---: | --- |
| `reproducibility` (independent start, same numerics) | 0.1 | 0.005 | 0.03 | 0.03 | |
| `convergence` (one numerical setting changed) | 1.0 | 0.02 | 0.10 | 0.10 | |
| `gamma_label_use` | 1.0 | 0.02 | 0.10 | 0.10 | |
| phonon displacement | | | | 0.10 | force-error/force-response ratio 0.05 |

There is also an SCF gate: NELM not reached, no residual jump, an energy rise after settling of at
most 1 meV, no EDDRMM/ZHEGV warnings, and the energy criterion met. These are separate from the
scanner's review triggers.

They were recorded before any package-23 or probe output exists. They are **proposed**: the owner
signs off or edits them before the first evaluation, never afterward.

The triage set (`probe_selection.tsv`) has 24 geometries. For each of the 8 interface groups, it takes
the worst retained-residual frame from the 2026-10-02 review table and a typical retained frame
(rms(c) nearest the source's retained median). It does the same for the 4 active bulk trajectories:
SiN 300 K, TiN 300 K and TiO-Real 300/450 K. Every worst pick is the per-source maximum, none
overlaps package 21, and all are present in the frozen ledger with their splits. SiN 300 K's worst
frame is OUTCAR index 0, the first MD step. Rechecked against the resettled21 report: for all 12
sources the per-frame SCF records are identical to the 2026-10-02 scan, and the selection is unchanged.

The rungs are proposed per geometry: the stored-label reproduction at EDIFF 1E-6, then kconv (3x3x3
interfaces/TiO, 3x3x5 TiN, 3x3x7 SiN), then kconv at EDIFF 1E-7, then PREC Accurate/LREAL .FALSE.,
each on 6.6.1 with a 6.5.1 bridge on one geometry per stratum. The package is not written yet: the
interface rungs alone are of order 50-100 node-h, and the owner should approve that cost and the
tolerances first. Package-23 results may also change the ALGO choice.

## 4. Frozen splits and the replacement adapter

`iface-rescue-release freeze D:\Traj_Storage\SiN-TiN_Cer_Interface_BULK --deepmd <deck>\deepmd` gives
28,800 frames from 48 runs, each 480/60/60. There are no duplicate (run, frame) keys and no duplicate
geometries, and no geometry sits in more than one split. All 144 DeePMD systems match the extxyz in
membership, species order, coordinates, cell, energy, forces and all nine virial components with
zero difference. The deck hashes are train `bbb72e31…`, valid `3ab1c08a…` and test `c0660847…`.

`assemble` builds a new release from the ledger, a policy and reviewed relabels:

- Frames are matched by (source_run, source_frame), so the input can be a fresh admission-gated
  recollection. Each frame's split and its geometry/label digests must equal the frozen ledger.
  Missing, extra or duplicated frames fail.
- Every kept or replaced frame keeps its historical split. A replacement must have identical
  species order, cell and positions (1e-6 A, minimum image), and its E, F and full virial replace
  the old ones together.
- These fail: unknown frames, duplicates, replacements for kept or excluded sources, `replace_all`
  sources with missing frames, nonfinite or virial-less labels, changed replacement files, unlisted
  sources and non-empty output directories.
- DeePMD systems are written with `virial.npy`, and every frame records its `label_source`.

The diagnostic cleanup run (`release_policy_cleanup_diagnostic.yaml`, exclusions only; never
production) gives 26,400 = 21,120 / 2,640 / 2,640 frames. Re-freezing that output matches the
original ledger in split, geometry and label for all 26,400 frames, and its DeePMD tree matches
its extxyz exactly. The production `release_policy.yaml` refuses to build, as intended, until the
SiN and TiN 300 K relabels exist.

Intended LONI order: final source audit, then mapped collection (`collect_mapped_leaf_single.sbatch`,
admission-gated), then `iface-rescue-release assemble --deck <that collection> --ledger <frozen>
--policy <final>`. The adapter does not replace the collector's source-admission gate. It enforces
historical membership and same-geometry replacement on top of it.

## Blockers and owner decisions

1. **Tolerances:** approve or edit `numerical_tolerances.yaml` before package 23 returns.
2. **Package 23:** push the desk and generator, then submit (commands above). Also report the owner's
   Normal-run directory and job ID.
3. **Probe package:** approve the cost of the 24-geometry stage, after the tolerances.
4. Then come the relabel stages (SiN 300 K 600 frames, TiN 300 K stage of 48, TiO-Real 240-frame
   subsets), the 20 train + 16 holdout A2 labels, endpoint curation of `A2_DFT/training_sources`
   (44 + 16) and the final LONI audit. None is started; each needs DFT that has not run.
