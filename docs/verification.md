# Verification and maturity

This document records what has actually been exercised, not merely what is
implemented. It is deliberately conservative so that code generation, mocked
tests, and scientific validation are not conflated.

## Status definitions

| Status | Meaning |
|---|---|
| Human-tested | A researcher ran the feature on real scientific inputs and inspected the produced artifacts or behavior. This is evidence of practical usability, not universal correctness. |
| Automated-test only | Unit, regression, schema, or mocked command tests cover part of the implementation, but no human has verified the complete path with the relevant external engine. |
| Unverified/code-only | Code and documentation exist, but there is no recorded end-to-end execution evidence. Some code-only features also have automated tests. |
| Scientifically validated | Predictions were compared with independent references and application-specific observables under a documented protocol. No InterfaceForge MLIP workflow currently has this status. |

## Current assessment

This assessment is based on the maintainer's actual use of the repository as of
September 2026.

| Area | Current status | Evidence and limits |
|---|---|---|
| VASP-MLFF preparation and campaign scaffolding | Human-tested | Used to prepare real VASP-MLFF work. Coverage is strongest for the maintained campaign path, not every generic VASP command. |
| VASP-MLFF audit, BEEF plotting, and restart/recovery | Human-tested | Used against real VASP outputs and recovery cases. New failure signatures and VASP versions remain unverified until encountered. |
| VASP OUTCAR → MACE extxyz generation | Human-tested | Real trajectory data have been converted and inspected. This does not verify MACE training. |
| VASP OUTCAR → DeePMD NPY generation | Human-tested | Real trajectory data have been converted and inspected. This does not verify DeePMD training or type-map correctness for every chemistry. |
| Generic static/relaxation/MD/DOS/band helpers and geometry utilities | Automated-test only or unverified | These are outside the specifically human-tested VASP-MLFF path unless a command is separately recorded as exercised. |
| MACE standard training, committee, and evaluation | Human-tested | A four-seed MACE committee was trained and evaluated on the real periodic SiN/TiN/TiO campaign using the synchronized extxyz dataset. This verifies the exercised launcher/runtime path and held-out metric generation for that campaign, not foundation-model fine-tuning, universal restart behavior, deployment, transferability, or scientific validity for other chemistries. |
| MACE-ROI | Unverified/code-only | The loss, sampler, derived-data, and evaluation code require real training, ablation, and scientific validation. |
| DeePMD DPA-2 training, checkpoint continuation, freezing, and evaluation | Human-tested | A real PyTorch DPA-2 committee was trained on LONI from the synchronized NPY dataset, continued from checkpoints, frozen, and evaluated across the canonical test systems with `dp test`. This status is specific to the exercised DPA-2/runtime path; DPA-1/3/4, DPA-2 fine-tuning, cross-version model loading and LAMMPS deployment remain separate gates. |
| Allegro training and LAMMPS integration | Unverified/code-only | No real training or LAMMPS run has established compatibility. |
| AI2-Kit active-learning adapters | Unverified/code-only | Generated TESLA MACE → OpenMM → VASP/oh-my-batch and legacy DeepMD → LAMMPS → VASP paths have automated coverage, but neither has completed a real external-engine loop. |
| InterMat adapter | Unverified/code-only | Generated interfaces still require a real dependency-version test and human review of termination, strain, registry, and atom overlap. |
| Reactive magnetic surface campaigns (`iface surface`) | Automated-test only | Exercised with the bundled 200-atom NiO(110) slab and Me4PACz: exposed-site inventory, AFM-II graph, freezer, water-balanced coverage grid, direct/H-bond docking, VASP export, and the primitive-to-200-atom cell optimization are regression tested. No generated VASP relaxation or post-relaxation classifier result has yet been scientifically reviewed. |
| RegFGW registry pre-screening adapter | Unverified/code-only, experimental | Wraps a brand-new (1-star), externally unvalidated tool. `run_regfgw_optimize` deliberately does not parse its output (never inspected in this repo); `compare_registry_selection`'s comparison math has automated coverage but has not been checked against a real RegFGW trial. See `docs/regfgw.md` for the trial this is meant to support before any reliance on it. |
| Bulk MLFF training grid (`iface mlff-interfaces`) and `ML_LHEAT` heat-flux recovery | Unverified/code-only | Built on the human-tested `prepare_campaign`/`submit_campaign`/`stage_tags`/`iface audit` path, but the grid-specific pieces (source discovery, campaign generation, throttled array launch, grid rollup) and the new `heat` recovery operation have automated coverage only -- no real training/ML_LHEAT run has been reviewed. See `docs/mlff-interfaces.md`. |
| Work-of-adhesion preparation, audit, and summary | Unverified/code-only | Preparation is ported from a script the maintainer already used successfully outside InterfaceForge; the audit step reuses the existing, unit-tested `iface audit`/`iface validate` math. Both have automated coverage, but no adhesion campaign has been run end to end (prepare -> VASP -> audit) through this integration. The optional literature comparison (`-c campaign.yaml`, matching `validation.references` / bundled `reference_profiles` against the computed J/m²) and the `iface vasp adhesion summary` roll-up (per-interface re-audit, short-label disambiguation, pending handling, CSV/markdown/JSON, and the two-panel publication figure) have automated coverage for the table assembly and matching arithmetic only. |
| Literature reference profiles (`iface reference`, `validation.reference_profiles`) | Unverified/code-only | Bundled profiles (currently Sharifi et al. 2026 Si₃N₄/TiN work of adhesion and surface energies) are transcribed from the paper's tables by hand; `load_campaign` expansion, name/path resolution, hand-written-entry precedence, the profile schema guard, and the comment-preserving `iface reference activate` campaign edit (append / insert-under-block / flow- and block-list extension, idempotency, ambiguous-shape refusal, reload verification) have automated coverage. The transcribed numbers themselves have not been independently checked, and no computed value has yet been compared against them on real runs. |
| Active-learning selection and exploration orchestration | Unverified/code-only | Mathematical and file-level behavior may be tested, but no complete label/train/explore/relabel cycle has been run. |
| Interface-property validation and HTML reporting | Unverified/code-only | Reporting calculations are not evidence that a trained model reproduces DFT or experiment. |
| Geometry-stratified validation (kind/temperature/coordination classification and per-class error reporting) | Unverified/code-only | Classification (`iface collect`) and stratified reporting (`iface validate stratified`) have automated coverage but have not been run against a real trained committee's predictions. Per-class physical tests are not yet built. |
| Bulk-referenced interfacial energy (`iface validate interface-energy`) | Automated-test only | Computes γ_int(T) in J/m² from the canonical dataset by referencing each unoxidized interface leaf to its same-temperature bulk phases, with gcd-based formula-unit matching, nitrogen-balance checks, campaign-driven or longest-axis area / `n_interfaces`, and block-average error bars. Synthetic-dataset tests verify the excess arithmetic, the unit conversion, missing-reference handling, the oxidized-leaf skip, the `validation.interfaces` `polar_termination` skip, and metadata-supplied stacking axis / `n_interfaces`. It is a potential-energy excess, not a rigorous free energy; it deliberately does not attempt polar-terminated interfaces (see the `polar_termination` skip and `docs/interface-energy.md`), and has not been checked against a literature TiN/SiN interfacial energy. |
| MLIP progress rollup (`iface mlip-progress`) | Human-tested | Used while real MACE and DeePMD committee jobs were running. It correctly rolled up MACE epochs/RMSE, DeePMD steps/RMSE/checkpoint/freeze state, per-system evaluation completion and `mlip_compare` artifacts from live filesystem outputs. It is deliberately read-only and does not query or replace Slurm accounting. Synthetic tests additionally cover parsing and completion flags. |
| Matched MACE vs DPA-2 comparison audit (`iface mlip-compare`) | Automated-test only | A repository-owned synthetic `prepare -> status -> finalize` test runs under CI together with metric-definition and evaluator-syntax tests plus `SafetyError` guards for frame-identity drift, model arity, overwrite protection, incomplete finalization, and corrupted `dp test` reference columns. No real MACE/DPA-2 committee has yet completed the workflow, and the cross-backend numbers have not been checked against the standalone MACE test table or DeePMD `rmse_overall.csv`. The present legacy committee launcher and comparison evaluator both use float32. See `docs/mlip-comparison.md`. |
| Campaign-wide provenance, restartability, and portability | Unverified as a complete system | Individual mechanisms exist, but the full lifecycle has not been demonstrated across interruption, relocation, and restart. |

## What the automated tests establish

The test suite is useful for detecting regressions in parsing, configuration,
file generation, data splitting, numerical helpers, CLI behavior, and selected
failure guards. Depending on installed extras, it also exercises ASE-backed
code paths.

It does not run licensed VASP, GPU MACE training, DeePMD, Allegro, LAMMPS,
AI2-Kit/oh-my-batch execution, OpenMM-ML exploration, or a real InterMat campaign. A green CI badge must
therefore be read as **software regression checks passed**, not **the scientific
workflow is verified**.

## Promotion checklist

A feature should move from code-only to human-tested only when a dated test note
records:

1. the InterfaceForge commit, dependency versions, machine/profile, and input;
2. the exact generated commands and whether they were edited manually;
3. successful external-engine completion and preserved logs/artifacts;
4. human inspection of units, species ordering, atom counts, splits, and output
   semantics;
5. at least one intentional failure/restart check where restartability is part
   of the claim.

Scientific validation requires more: held-out DFT comparisons, leakage checks,
property-level tests, stability tests in the intended regime, and comparison to
an appropriate baseline. Until those checks are recorded, documentation should
say that a feature is implemented or generated, not that it works, is validated,
or is production-ready.
