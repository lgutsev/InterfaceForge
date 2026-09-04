# InterfaceForge documentation

This directory contains the authoritative workflow guides. The root
[README](../README.md) is intentionally limited to project scope, a minimal
quick start, and maturity boundaries.

## Start here

| Guide | Use it for |
|---|---|
| [Getting started](getting-started.md) | Installation choices, first campaign, and routing to the appropriate workflow |
| [Campaign format](campaign.md) | `campaign.yaml`, systems, stages, scheduler profiles, and reproducibility state |
| [Verification and maturity](verification.md) | What has run on real scientific data versus automated-test-only or code-only features |
| [LONI environments](loni-environments.md) | Cluster environments and runtime separation |

## VASP and reference calculations

| Guide | Use it for |
|---|---|
| [VASP workflows](vasp.md) | Input utilities, OPT/Step1/Step2, launch/status/repair, VASP-MLFF audit/recovery, vacuum analysis, plotting and archives |
| [MLFF interface grid](mlff-interfaces.md) | Bulk/interface VASP-MLFF training grids and heat-flux recovery |
| [Reference profiles](reference-profiles.md) | Bundled literature/reference configurations |

Slab-alignment and work-function examples are under
[`examples/vasp/`](../examples/vasp/), including the calculation-family
configuration and Slurm launch pattern.

## Datasets and machine-learned potentials

| Guide | Use it for |
|---|---|
| [DeePMD campaigns](deepmd.md) | DPA architectures, backends, fine-tuning, preflight/smoke/training/evaluation, and runtime gates |
| [MACE committees](mace-committee.md) | Committee collection, checksums, verification, and restoration |
| [MACE-ROI](mace-roi.md) | Interface-local force weighting and thermodynamic-cycle loss |
| [Allegro](allegro.md) | Allegro job generation and LAMMPS-oriented preflight |
| [Matched MLIP comparison](mlip-comparison.md) | Exact-frame MACE/DeePMD comparison, metrics, calibration, figures, and live progress |
| [Packaging](packaging.md) | Committee archives, dataset backups, and upload-ready Hugging Face repositories |

For mapped collections spanning unrelated VASP roots, see the
[`mapped-leaf-campaign` example](../examples/mapped-leaf-campaign/README.md).

## Interfaces and validation

| Guide | Use it for |
|---|---|
| [Interface energy](interface-energy.md) | Work of adhesion and literature/reference comparison |
| [Separation energy](separation-energy.md) | Cleavage/separation calculations and DFT-versus-MLIP comparison |
| [Stratified validation](stratified-validation.md) | Errors by geometry class, temperature, coordination, and other physical groups |
| [Reactive surfaces](reactive-surfaces.md) | AFM-compatible cells, hydroxylation, proton transfer, phosphonate docking, export, and audit |

## Optional adapters

| Guide | Use it for |
|---|---|
| [AI2-Kit](ai2kit.md) | MACE/OpenMM/VASP TESLA and legacy DeepMD/LAMMPS/VASP active-learning adapters |
| [InterMat](intermat.md) | Commensurate crystalline film/substrate generation |
| [RegFGW](regfgw.md) | Experimental registry prescreening and comparison against exhaustive grids |

Optional adapters remain under explicit InterfaceForge control and inherit the
maturity limits recorded in [Verification and maturity](verification.md).

## Examples and launchers

- [`examples/interface-campaign/`](../examples/interface-campaign/) provides an
  editable campaign configuration and validation inputs.
- [`examples/ai2kit/`](../examples/ai2kit/) provides the active-learning
  controller/runtime split and smoke test.
- [`notebooks/nio_m110_hydroxylation/`](../notebooks/nio_m110_hydroxylation/)
  contains the NiO(110) reference notebook, inputs, and declarative campaign.
- [`launch_scripts/`](../launch_scripts/) contains maintained cluster launchers
  and their own [launcher index](../launch_scripts/README.md).

## Documentation policy

- Put general project identity and the shortest viable quick start in the root
  README.
- Put operational commands, outputs, failure modes, and debugging sequences in
  the relevant guide here.
- Record evidence changes in `verification.md`; implementation alone does not
  promote a feature to human-tested status.
- Link to one authoritative explanation instead of copying the same recipe
  into several files.
