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
August 2026.

| Area | Current status | Evidence and limits |
|---|---|---|
| VASP-MLFF preparation and campaign scaffolding | Human-tested | Used to prepare real VASP-MLFF work. Coverage is strongest for the maintained campaign path, not every generic VASP command. |
| VASP-MLFF audit, BEEF plotting, and restart/recovery | Human-tested | Used against real VASP outputs and recovery cases. New failure signatures and VASP versions remain unverified until encountered. |
| VASP OUTCAR → MACE extxyz generation | Human-tested | Real trajectory data have been converted and inspected. This does not verify MACE training. |
| VASP OUTCAR → DeePMD NPY generation | Human-tested | Real trajectory data have been converted and inspected. This does not verify DeePMD training or type-map correctness for every chemistry. |
| Generic static/relaxation/MD/DOS/band helpers and geometry utilities | Automated-test only or unverified | These are outside the specifically human-tested VASP-MLFF path unless a command is separately recorded as exercised. |
| MACE standard training, committee, restart, and evaluation | Unverified/code-only | Scripts and CLI generation exist, but a successful generated end-to-end training workflow is not yet claimed. Hand-written launcher backups do not verify package-generated jobs. |
| MACE-ROI | Unverified/code-only | The loss, sampler, derived-data, and evaluation code require real training, ablation, and scientific validation. |
| DeePMD training, smoke/full/evaluation, freeze/export, and LAMMPS use | Unverified/code-only | Dataset generation is the only human-tested DeePMD-related part. DPA/backend/version combinations must be tested separately. |
| Allegro training and LAMMPS integration | Unverified/code-only | No real training or LAMMPS run has established compatibility. |
| AI2-Kit active-learning adapters | Unverified/code-only | Generated TESLA MACE → OpenMM → VASP/oh-my-batch and legacy DeepMD → LAMMPS → VASP paths have automated coverage, but neither has completed a real external-engine loop. |
| InterMat adapter | Unverified/code-only | Generated interfaces still require a real dependency-version test and human review of termination, strain, registry, and atom overlap. |
| Work-of-adhesion preparation and audit | Unverified/code-only | Preparation is ported from a script the maintainer already used successfully outside InterfaceForge; the audit step reuses the existing, unit-tested `iface audit`/`iface validate` math. Both have automated coverage, but no adhesion campaign has been run end to end (prepare -> VASP -> audit) through this integration. |
| Active-learning selection and exploration orchestration | Unverified/code-only | Mathematical and file-level behavior may be tested, but no complete label/train/explore/relabel cycle has been run. |
| Interface-property validation and HTML reporting | Unverified/code-only | Reporting calculations are not evidence that a trained model reproduces DFT or experiment. |
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
