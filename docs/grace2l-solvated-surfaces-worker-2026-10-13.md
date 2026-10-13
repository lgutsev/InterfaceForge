# Claude worker: InterfaceForge GRACE-2L solvated-surface branch

## Objective and branch

Repository: https://github.com/lgutsev/InterfaceForge
Create branch feat/grace2l-solvated-surfaces from freshly fetched main, in an isolated worktree. Do not overwrite another worker's edits. Read repository instructions first.
Audit snapshot inspected on 2026-10-13: main 380c14d4a5c37e230fa63d649cae1461396a50cd.
This assignment is to implement and test a usable optional integration, not just write a proposal. Keep integration correctness, DFT accuracy and production readiness as separate reported outcomes.

Current main has reactive oxide states, hydroxylation/proton-transfer construction, VASP export/audit and matched-frame MACE/DeePMD evaluation. No GRACE implementation was found in this audit. Relevant entry points: src/interfaceforge/vasp.py, vasp_provenance.py, mlip_compare.py, validation.py, surface/, scheduler/config/CLI modules; docs/reactive-surfaces.md, docs/vasp.md, docs/mlip-comparison.md, docs/verification.md. Inspect these before choosing module boundaries. Reuse an existing calculator abstraction if present in your checkout.

## 1. Preflight and model provenance

Use an isolated GRACE environment; do not disturb working MACE, UMA or DeePMD environments. Prepare downloads and environment manifests separately from production compute jobs. Heavy inference/training must run through the normal scheduler on compute nodes.

Verify the installed VASP executable, not just its module name. The GRACE interface requires VASP 6.6.0+ built with -Dlibvaspml and -DVASPML_ENABLE_GRACE. Prefer the user's VASP 6.6.1 route. Official documented tested pairings are 6.6.1 with tensorpotential 0.5.9/0.6.0; 6.6.0 is incompatible with tensorpotential models 0.5.8+. Pin a known working pairing for the first test, then evaluate newer versions separately. A 6.6.1 executable without GRACE build support must report BLOCKED, not silently fall back.

Check metadata.json, supported elements, model architecture and artifact format. Save full model name, release/revision, artifact hashes, download source, cutoff, dtype, training-data/DFT provenance, package versions, executable/build provenance and hardware. Resolve aliases to immutable artifacts.

Start with GRACE-2L-OMAT as the official integration smoke model. For scientific evaluation compare a small, bounded set of documented compatible 2L variants; investigate SMAX-OMAT where available, but do not assume every new checkpoint is readable by VASP's pinned interface. Element coverage is not evidence of solvent/interface training coverage.

## 2. Two inference paths and parity

Add optional GRACE ASE inference for energy, unmasked forces and stress where supported, plus a VASP prediction-only preparation/launch/status adapter. Keep TensorFlow/GRACE imports lazy and dependencies optional.

Official VASP tags:
    ML_LMLFF = .TRUE.
    ML_LIB = .TRUE.
    ML_MODE = run
    ML_TYPE = grace
    ML_GRACE_MODEL = GRACE-2L-OMAT

Verify model path/name resolution against the official ML_GRACE_MODEL documentation. Current VASP GRACE interface supports one MPI rank only; generate a dedicated one-rank scheduler profile with GPU allocation when available. Do not reuse multi-rank DFT launch settings. Tensorpotential is useful for model retrieval/training/ASE and is not itself required as a VASP runtime dependency.

Run identical coordinates, cell, atom ordering and model artifacts through ASE and VASP. Compare energy, raw per-atom forces and stress with explicit unit/sign conversions. Check orthogonal and skewed cells, periodic images, constrained bottom slab atoms and restart behavior. Compare unconstrained forces before any mobility mask; constraints must affect dynamics without erasing reference labels. Establish dtype-aware numerical tolerances from the first small test and record them before expanding the suite.

Implement preparation and status tests without requiring licensed binaries or model downloads in CI. Add separately marked real-runtime integration tests. A mocked test is not runtime evidence.

Do not implement GRACE training by setting VASP ML_MODE=train. This documented bridge is prediction-only. Fine-tuning is external through gracemaker/tensorpotential, with separate DFT labeling jobs.

## 3. Solvated-interface structures and benchmark

Begin with existing neutral, fixed-electron-state surface systems. Reuse approved structures/data where available. Include bulk solvent, dry slab, wetted slab and slab+adsorbate+solvent, with representative molecular orientations, adsorption distances, hydroxylation and proton-transfer configurations. The motivating use cases include SiO2/amino-acid or peptide/water and oxide/passivation surfaces; only benchmark systems for which consistent DFT labels are available.

Investigate the historical mirrored SOLVATED workflow and POSCAR_solv/Packmol artifacts before duplicating a builder. Packing in an axis-aligned box followed by wrapping is not sufficient validation for a skewed slab. Audit true cell geometry, solvent accessible volume/density, slab-normal region, periodic contacts, molecule self-images, frozen layers and atom/molecule identity. Multiple packing seeds should be reproducible.

Use audited, converged DFT labels with explicit functional, dispersion, Hubbard U, spin, POTCAR identity and k-point convergence provenance. Matching EDIFF is not proof of k-point accuracy. Preserve source/trajectory identities and immutable canonical labels. Do not mix incompatible DFT label conventions into one benchmark without explicitly separating them.

First evaluate foundation models without fine-tuning. Compare against available appropriate MACE/DeePMD baselines on identical frames; do not use the unrelated SiN/TiN model as a solvent baseline. Report:
- force errors separately for substrate, solvent, adsorbate, and interfacial atoms, including outliers and force-scale baselines;
- energy differences within matched composition/state and adsorption/proton-transfer energetics where reference data permits;
- orthogonal/skewed-cell parity, throughput and memory on the same hardware, with model-load versus steady-state timing separated;
- short MD stability, minimum distances, solvent density/orientation profiles, hydrogen bonds and relevant coordination changes;
- NVE drift after appropriate equilibration, with a timestep check. Stable thermostatted MD is not accuracy validation.

Slab-volume-normalized stress depends on vacuum; retain full cell provenance and avoid presenting it as a vacuum-independent interfacial observable. Do not infer finite-temperature adsorption free energies from raw total-energy subtraction.

For an initial bounded pilot use tens of diverse labeled frames and short trajectories, reusing existing data. Predetermine task-specific acceptance criteria and compute budget in the campaign configuration. Do not invent success thresholds after seeing results.

## 4. Targeted adaptation and active learning

If foundation-model accuracy is inadequate, add a documented fine-tuning preparation path using canonical energy/force labels and stress only where meaningful. Keep training/validation/test separated by parent trajectory, packing seed and reactive state; correlated snapshots must not straddle splits. Preserve an independent challenge set.

Reuse uncertainty-plus-diversity selection and reactive-state quotas. Evaluate whether per-atom GRACE uncertainty is actually available through each pinned runtime. Extrapolation scores and committee spread need error calibration; do not claim VASP automatically exposes uncertainty or calls DFT on demand.

Export selected full small-cell frames into the existing audited VASP labeling queue. Local extraction from a large hydrated interface is future research unless electrostatics/boundary fidelity is demonstrated. Retraining, export and VASP/ASE parity must repeat for each adapted artifact.

Never add D3 blindly to ML predictions: check whether the checkpoint already includes dispersion. Likewise NELECT, MAGMOM, dipole corrections, VASPsol/LSOL or applied bias in an INCAR do not prove GRACE predictions depend on those variables. Flag charged, spin-changing and constant-potential cases as scientifically unvalidated until explicitly modeled and tested. Keep the user's implicit-solvation DFT route available; do not claim GRACE can be combined with VASPsol physically just because both tags parse.

## 5. Delivery and completion evidence

Deliver optional backend code, dedicated scheduler profile, model registry/manifest, executable examples, benchmark config, tests, docs/grace-solvated-surfaces.md and an update to docs/verification.md. Use existing naming patterns; any new CLI names are proposals until implemented.

Report a compatibility matrix with separate states for code tests, real ASE execution, real VASP execution, cross-engine parity, DFT agreement and MD validation. Missing hardware/models/binary/data must be BLOCKED or NOT RUN, not PASS.

Return branch name, base and final commit, changed files, test results, exact pilot commands, logs/results and remaining blockers. Keep model weights, POTCARs and large trajectories out of git. Leave the feature branch reviewable; do not merge into main in this assignment. If git publication is available and authorized in your session, push the feature branch and provide its link.

## Primary references to inspect before implementation

- VASP integration/build/runtime constraints:
  https://vasp.at/wiki/Running_GRACE_force_fields_in_VASP
- Model selection and training:
  https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/
  https://github.com/ICAMS/grace-tensorpotential
  https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models
- GRACE foundation architecture/benchmarks/fine-tuning:
  https://arxiv.org/abs/2508.17936
- Direct solvated-surface evidence, custom-fitted Cu-water potentials:
  https://arxiv.org/abs/2509.17833
  https://doi.org/10.1021/acsmaterialsau.5c00174

The Cu-water study supports GRACE-2L as an architecture candidate; it does not validate a generic checkpoint for all solvated surfaces. Its historical memory/atom-count limits are runtime-specific, not permanent model limits.


## 6. Kresse Au/water/ion study: source investigation and electrostatics pilot

### Motivation and evidence boundaries

Lavrenty attended Georg Kresse's Materials Design presentation on October 13,
2026, “The Free Solvation Energy of Ions in Water and Their Interaction with
Surfaces.” The observed geometry was Au–(water + Na or F)–vacuum, with biased
ion sampling. Slide 34, “Understanding free solvation energies at surfaces,”
compares the group's Na+/F− profiles with Roncoroni et al., J. Chem. Phys. 164,
084702 (2026), and emphasizes charged-cell calculations for the macroscopic
point-of-zero-charge (PZC) limit and a finite-cell capacitor issue.

The screenshot shows reference minima roughly −0.11 eV for F− near 2.8 Å and
−0.12 eV for Na+ near 5 Å, versus substantially shallower comparison profiles.
These are approximate visual readings, not digitized reference data or
acceptance targets. The curves do not establish that GRACE architecture alone
causes the difference. Different DFT labels, charge treatment, sampling or
model errors could contribute. Do not equate a PMF minimum with a bond energy.

The slide calls the comparison “MACE OMat”; the indexed comparison-paper
abstract describes UMA. Resolve this discrepancy against the full paper and
supporting information before attributing the curves to a specific model.

### Live clarification from Kresse, relayed by Lavrenty

After being asked whether vacuum provides a desolvated reference and whether
ions cross the water–vacuum interface, Kresse replied that this sounds suitable
for neutral species but not charged species because there are two surfaces to
contend with instead of one. This is a user-relayed paraphrase, not a transcript.

Interpretation to investigate: taking a charged species through the free water
surface introduces the electrostatic contribution of the water–vacuum interface
in addition to the electrode–water interface. Do not treat such a trajectory as
an uncomplicated electrode adsorption or single-ion solvation reference.
Neutral species avoid the direct net-charge coupling to the interface potential,
but still experience the free surface and changes in hydration.

This response rejects our proposed simple desolvated-endpoint rationale for
charged ions; it does not establish the actual reason vacuum was retained,
whether the demonstrated trajectory crosses that boundary, or the exact
correction scheme. Prioritize an electrode-to-interior-water coordinate for an
adsorption pilot, with adequate separation from the free surface. Any study
that crosses the free surface must separately define and validate its charged
transfer observable and electrostatic reference.

### Additional live warning: electron localization in vacuum

Lavrenty also relayed Kresse's warning that charged surfaces with vacuum can
produce pockets of localized electrons in the vacuum. Record this as a
user-relayed statement; the cause and affected settings require source checks.
Do not assume Coulomb truncation alone proves that such states are absent.

Before accepting charged-slab DFT labels, inspect the electrostatic potential
along the slab normal and its spatial variation, electron density in the vacuum,
and spatial character/occupations of suspect near-Fermi states. Use planar
averages for screening, but retain spatially resolved diagnostics because a
localized pocket can be obscured by averaging. Separate normal decaying density
tails from occupied vacuum-localized states; do not reject solely on nonzero
vacuum density or impose an arbitrary universal threshold.

Check sensitivity to cell charge, vacuum thickness and the selected boundary
correction/truncation settings. Determine whether observed vacuum localization
is a numerical artifact, an inconsistent electronic boundary condition or a
physical electron-emission limit of the selected state. SCF convergence alone
does not qualify a label. Flag suspect frames for review and exclude them from
qualified training exports until resolved. Preserve diagnostic provenance in
the charge/boundary audit and report any missing diagnostics as NOT CHECKED.

A coordinate/species MLIP cannot represent electrons detached into vacuum as
an independent degree of freedom. Fitting energies and forces contaminated by
unintended vacuum electron states risks learning an unintended energy surface.

### Where to look, in order

1. Presentation abstract and recording access:
   https://www.materialsdesign.com/webinars/confirm/the-free-solvation-energy-of-ions-in-water-and-their-interaction-with-surfaces
   https://www.ugm.materialsdesign.com/sessions
   The public abstract describes approximately 100 water molecules, charged
   and uncharged Au(111) slabs, on-the-fly VASP MLFF training followed by
   GRACE-2L refitting, WHAM and metadynamics. Recording access may require
   registration. Do not claim to have read unavailable slides or recordings.
2. Direct group conference abstract:
   https://www.nanoge.org/proceedings/MATSUSFall26/6a6b422c39bcf250e25d7fcc
   Angela Rittsteuer, Ferenc Karsai, Michael Sahre and Georg Kresse,
   “Simulating Charged Electrode–Water Interfaces with Solvated Atoms:
   A DFT-Trained, ML-Accelerated Molecular Dynamics Workflow.” Search these
   authors for subsequent papers, preprints, supporting information and data.
   The abstract distinguishes single-ion adsorption profiles from combined
   Na+/F− ion-pairing/dissociation metadynamics; preserve that distinction.
3. Electrostatics method:
   Vijay et al., Physical Review B 112, 045409 (2025), “Efficient periodic
   density functional theory calculations of charged molecules and surfaces
   using Coulomb kernel truncation.”
   https://doi.org/10.1103/cd6s-cdkf
   https://arxiv.org/abs/2501.02435
   Follow the data-availability statement and supporting material, including
   the linked dataset https://doi.org/10.5281/zenodo.15494936 . Distinguish the
   stepped Au(211) method demonstration from the Au(111) ion study.
4. Official VASP charge/boundary documentation:
   https://vasp.at/wiki/KERNEL_TRUNCATION/LTRUNCATE
   https://vasp.at/wiki/KERNEL_TRUNCATION/IPAD
   https://vasp.at/wiki/Charged_systems_with_density_functional_theory
   Follow dimensionality/coarsening links and check applicability to the
   actual installed VASP version. Vacuum alone is not a charged-slab
   electrostatic correction.
5. Comparison paper:
   Roncoroni et al., “Deriving effective electrode–ion interactions from
   free-energy profiles at electrochemical interfaces,” J. Chem. Phys. 164,
   084702 (2026).
   https://escholarship.org/uc/item/0tr5953v
   Read full methods and supporting information for the exact model,
   electrostatic treatment, electrode state, geometry and sampling.

Create a source-backed methods table, with source locations and CONFIRMED,
INFERRED or UNKNOWN status for every setting. If methods/inputs are unavailable,
provide specific questions for Lavrenty to ask Kresse; do not contact authors.

### Required physical setup audit

- Record total charge/NELECT and its convention, ion composition, slab area,
  metal thickness, solvent thickness, vacuum, boundary settings, electron
  state and any electrode-potential reference. Distinguish cell charge from
  electrode charge. Do not assume a neutral finite ion-containing cell is
  equivalent to a macroscopic electrode at PZC; derive the applicable charge
  prescription from the sources rather than inventing one.
- Support or prepare configurations for both a free water film with vacuum
  and a fully filled periodic geometry, reusing existing builders. Explain
  that their interfaces differ: fully filled z-periodicity introduces another
  metal–water contact. They are not automatically matched physical controls.
- Establish solvent density explicitly. In a filled cell, document molecule
  count, accessible volume and equilibration/pressure protocol. For a film,
  measure density versus slab-normal position, thickness and whether a
  bulk-like region exists. A free surface permits thickness relaxation but
  does not guarantee a correct bulk density. A naive whole-cell barostat
  acting on vacuum is not an acceptable density-control solution.
- Vacuum purpose remains UNKNOWN: isolation, solvent density relaxation,
  and electrostatic reference are hypotheses. The simple desolvated-endpoint
  rationale for charged ions was challenged by Kresse in the live clarification. Confirm
  whether the ion actually crosses the water–vacuum boundary. Do not encode
  the hypotheses as established facts about Kresse's method.
- Check convergence with lateral area and water/slab thickness; vacuum and
  truncation/padding convergence where applicable. A second free surface
  must not silently contaminate the intended bulk-solution reference.

### Sampling and matched validation

Prepare a bounded ion–surface-distance/hydration-coordination sampling pilot
only after the source audit and compatibility preflight. Include uncommon
hydration states and relevant adsorbed/nonadsorbed configurations. Preserve
collective-variable definitions, walls, restraints, bias history, temperature
and parent-window/trajectory identity. Do not force vacuum crossing unless
that is part of the chosen physical observable.

Keep bias energies and forces separate from the canonical training labels.
Train on unbiased DFT energies and forces; verify what biased VASP outputs
contain before exporting labels. Free-energy estimates must document the
WHAM/reweighting method, reference zero, uncertainty, overlap and convergence.

Compare GRACE-2L and an appropriate existing baseline on identical labels,
geometries and electrostatic conditions. Separate (a) DFT boundary-condition
error, (b) MLIP approximation error and (c) sampling error. Report ion forces
projected onto the adsorption coordinate, hydration coordination, solvent
profiles and, only if sampling supports it, adsorption PMFs. Different
reference Hamiltonians must not be presented as an architecture comparison.

Standard coordinate/species-only inference does not become charge-aware by
changing NELECT in its launch input. Keep electron/charge states in separate
models/campaigns unless explicit charge conditioning is implemented and
validated. Charged/PZC production is an extension requiring evidence, not a
prerequisite for the neutral integration smoke test in sections 1–3.

### Deliverables and sequencing

First deliver the source/methods table, unresolved questions, and a small
pilot configuration with predeclared budget and acceptance criteria. Continue
the authorized optional integration and inexpensive tests. Do not commission
a large new DFT/training campaign merely to fill missing source information.

Include this investigation in docs/grace-solvated-surfaces.md and the final
worker report, with separate readiness states for neutral solvated surfaces
and charged electrode/PZC systems. Missing recording, DFT labels, suitable
binary, model support or charge prescription must remain explicit blockers.
