# Derivative-sensitive MLIP validation

> **Verification note:** automated-test only. Structure generation, input
> provenance, overwrite guards, cross-probe input-consistency guards, per-metric
> summary availability, the force/energy array export and analytic curvature
> recovery are covered by tests, including synthetic OUTCARs parsed through ASE
> and DFT–MLIP comparison summaries. Those tests are synthetic or mocked
> throughout: analytic ASE calculators stand in for MLIPs, hand-written OUTCARs
> stand in for VASP, and MACE/DeePMD calculator construction is patched rather
> than installed. **No real Si-Ti-N-O DFT/MACE/DeePMD probe campaign has yet
> been demonstrated**, so none of this is evidence of scientific validity.

Energy and force RMSE on ordinary held-out MD frames do not establish that an
MLIP has learned the local curvature of the potential-energy surface. This
workflow creates a small, purpose-designed set around selected relaxed bulk and
interface structures and evaluates DFT and MLIPs on exactly the same geometries.

## Scientific source

The defaults are motivated by:

> B. Póta, P. Ahlawat, G. Csányi, and M. Simoncelli, "Thermal conductivity
> predictions with foundation atomistic models," *Nature Communications*
> (2026), [doi:10.1038/s41467-026-76391-w](https://doi.org/10.1038/s41467-026-76391-w).

Póta *et al.* study finite-displacement amplitudes and use about 0.03 Å
rattling together with equilibrium and ±1% volume perturbations in their small
fine-tuning sets. InterfaceForge therefore defaults to a Cartesian
per-component displacement standard deviation of 0.03 Å and homogeneous volume
strains of -0.01, 0, and +0.01.

What comes from the paper is the **motivation for the defaults**: the Gaussian
rattle amplitude and the volume-perturbation magnitude. Everything below is an
**InterfaceForge extension** and must not be described as the paper's protocol:

- the explicit un-rattled center at each strain, kept as a structure in its own
  right so it can serve as an energy reference and a curvature midpoint;
- the exact negative of each random displacement vector, written as its
  opposite partner so central differences are available;
- the central-pair diagnostic `k_F-k_E`, which compares the two estimators
  built from that pair against each other.

The default nine structures per source are a deliberately small InterfaceForge
probe set. They are **not** the paper's exact fine-tuning dataset, neither in
size nor in composition, and results from them should not be compared with the
paper's fine-tuning results as if they were the same data.

## Prepare the probe

Start with fully relaxed 0 K structures. A VASP run directory may be supplied;
a non-empty `CONTCAR` is preferred over its `POSCAR`.

These are **full-coordinate PES probes**: preparation removes source constraints
and displaces all atoms. Evaluation uses raw forces and stress, including for
older probe trees retaining selective-dynamics flags. It does not test only the
free-atom subspace of a constrained relaxation.

```bash
iface validate derivative-probe prepare derivative_probe \
  bulk_sin=/path/to/bulk_SiN/CONTCAR \
  bulk_tin=/path/to/bulk_TiN/CONTCAR \
  o25_iface=/path/to/O25_interface/CONTCAR
```

For each source, the default set contains nine structures:

- three un-rattled centers at -1%, 0, and +1% volume strain;
- one deterministic +0.03 Å rattle and its exact negative around each center.

Here "0.03 Å" is the standard deviation of every Cartesian displacement
component, matching ASE's usual rattle convention. The realized RMS and full
3N-vector norm are recorded rather than assumed.

The output includes:

- one run directory and `POSCAR` per probe;
- `derivative_probe.extxyz` containing the same ordered structures for MLIP use;
- `manifest.json` and `manifest.csv` with source/input hashes and protocol;
- `runs.txt`, suitable for a scheduler-array wrapper.

Use more independent directions or a different amplitude only deliberately:

```bash
iface validate derivative-probe prepare derivative_probe \
  representative_iface=CONTCAR \
  --rattles 3 --displacement 0.02 --seed 53
```

`--strain-mode volume` is the default. Thus +0.01 means V'=1.01V, and
each lattice vector is scaled by 1.01^(1/3). Use
`--strain-mode linear` only if ±1% lattice-vector strain is actually intended.

## Make identical VASP single-point runs

Pass a trusted converged VASP run as the template:

```bash
iface validate derivative-probe prepare derivative_probe \
  representative_iface=CONTCAR \
  --vasp-template /path/to/reference_static_inputs
```

The template must contain non-empty `INCAR`, `KPOINTS`, and `POTCAR`.
One prepared tree can use one template; prepare separate trees when the sources
need different species blocks, POTCAR datasets, or k-point meshes.
InterfaceForge first verifies that POTCAR's VRHFIN order matches the source
POSCAR species blocks. It then copies the same KPOINTS/POTCAR into every probe and converts the
INCAR to a static protocol:

- `IBRION=-1`, `NSW=0`, and `ISYM=0`;
- `LWAVE=.FALSE.` and `LCHARG=.FALSE.`;
- MD-only, ionic-relaxation, and `ML_*` tags removed;
- electronic settings such as ENCUT, precision, spin, DFT+U, dispersion, and
  convergence retained from the template.

INCAR assignments separated by semicolons are parsed individually; comments
are removed and the last active assignment wins. Unparseable assignments fail
rather than silently changing the electronic protocol.

The generated hashes are checked again during evaluation. If any POSCAR or
shared VASP input has changed, evaluation stops instead of mixing protocols.

POTCAR is copied only into the local run tree; it is not added to the
InterfaceForge repository or portable archives.

## Evaluate DFT and MLIPs

After the VASP single points have written their `OUTCAR` files:

```bash
iface validate derivative-probe evaluate derivative_probe \
  --mace-model /models/mace/seed11.model \
  --mace-model /models/mace/seed23.model \
  --device cuda --output-stem mace
```

MACE defaults to `--mace-dtype float64` to reduce cancellation in energy
second differences. To assess precision sensitivity, repeat with
`--mace-dtype float32 --output-stem mace_float32`. This does not recover
precision already lost during training. `--device` controls MACE; DeePMD device
selection and precision remain backend-managed.

Run DeePMD/DPA models in their own compatible environment if necessary:

```bash
iface validate derivative-probe evaluate derivative_probe \
  --deepmd-model /models/dpa3/000/frozen_model.pth \
  --deepmd-model /models/dpa3/001/frozen_model.pth \
  --output-stem dpa3
```

The evaluator writes:

- `<stem>_predictions.csv`: energy, force norms, and stress (when implemented)
  for every backend/structure;
- `<stem>_responses.csv`: energy- and force-derived directional curvatures
  for every symmetric pair;
- `<stem>_results.json`: per-model comparison with DFT;
- `<stem>_arrays.npz`: the full per-structure force arrays (plus energies and
  stress) for every backend, so the reported errors can be recomputed without
  rerunning inference.

DFT references must match the indexed species/order, cell and positions
(with a 1e-5 Å absolute tolerance and periodic image equivalence), contain
exactly one force frame, show normal termination and the explicit EDIFF-reached
marker, and demonstrate `IBRION=-1`, `NSW=0`, `ISYM=0`. Contradictory geometry,
tracked executed settings, POTCAR titles or inputs within a source stop evaluation.
Other VASP convergence-marker formats are not currently accepted.

Shared VASP inputs are compared per file name, not per probe. For each source
label, the first probe that actually provides `INCAR`, `KPOINTS` or `POTCAR`
becomes that file's reference, and every later probe supplying the same file is
compared against it. A probe that is missing a file only contributes an explicit
warning, so it can no longer suppress a later disagreement: two probes with
conflicting `KPOINTS` stop evaluation even when the first probe had none, and
regardless of the order the probes appear in the manifest. Different source
labels keep independent references, because separate sources may legitimately
need different meshes or potentials.

`COMPLETE` means every DFT point was **collected and accepted as evidence**
without provenance warnings; `INCOMPLETE` means points are missing/unparseable;
`CHECK` means all points were read but provenance needs review. `COMPLETE` is a
statement about DFT collection, **not** scientific validation, and none of the
three is a pass/fail threshold on model quality.
Missing executed settings or potential/k-point identity remain explicit warnings.
POTCAR checks establish title identity, not binary equivalence to the potential
used by VASP; NKPTS is not a full executed k-point mesh comparison. Hubbard
arrays need manual review. Magnetic-state continuity is not established.

The JSON records manifest and implementation hashes, model paths/hashes,
OUTCAR/input hashes, inspected executed settings, package/Python/platform
versions and requested MACE dtype/device. Rerunning with the same output stem
replaces that stem's results; use distinct stems for comparisons.

The default stem is `derivative_probe`. Set a different `--output-stem` for
each backend-isolated run so a DeePMD environment does not overwrite MACE
results from the same probe tree.

### What each comparison needs

Each metric depends only on its own prerequisites, and a missing prerequisite
removes only the metrics that need it:

| Metric | Requires | Count field |
|---|---|---|
| `force_ev_a` | every structure with both a DFT and an MLIP result | `force_matched_structures` (and `force_components`) |
| `stress_ev_a3` | those structures that also carry stress on **both** sides | `stress_matched_structures` |
| `relative_energy_mev_atom` | that source's zero-strain center, present for both sides | `relative_energy_matched_structures` |
| `energy_curvature_ev_a2`, `force_curvature_ev_a2` | a complete minus/center/plus triplet for both sides | `matched_energy_curvatures`, `matched_force_curvatures` |

`matched_structures` remains the number of structures matched between DFT and
the model. A metric that cannot be computed is left out of the summary and
explained in `omitted_metrics`; a metric that is computed from only part of the
data keeps its value and lists what was excluded in `metric_notes`. So a probe
tree whose zero-strain center has not finished still reports force and stress
errors, and one source lagging behind does not discard the sources that are
complete.

### Directional curvature

Energy comparison uses each source's unstrained center as its zero, then
reports relative-energy errors in meV/atom. This removes arbitrary
composition-dependent offsets without mixing different source compositions.
Forces are compared component by component. Stress is compared when both
backends provide it, in ASE's Voigt order (xx, yy, zz, yz, xz, xy) and
eV/Angstrom^3.

Directional curvature is estimated two ways for a displacement vector d, with
q=||d|| and unit direction u=d/q:

k_E = [E(+d) - 2E(0) + E(-d)] / q^2

k_F = -[F(+d) - F(-d)] dot d / (2 q^2)

Both approximate u^T H u, the projection of the 3N x 3N Hessian H onto the
direction u, in eV/Angstrom^2. They are **not** mass-weighted phonon
frequencies: no dynamical matrix is built or diagonalized, and no mass enters
either expression. For a purely harmonic model both are exact; for an
anharmonic one both carry a leading error proportional to q^2, and k_F carries
twice the leading error of k_E.

Both estimators appear per symmetric pair in `<stem>_responses.csv` and are now
summarized separately in `<stem>_results.json`:

- `energy_curvature_ev_a2`: DFT-vs-model errors of the energy-derived estimator;
- `force_curvature_ev_a2`: the same for the force-derived estimator;
- `directional_curvature_ev_a2`: retained as a **compatibility alias** of
  `force_curvature_ev_a2` for readers of earlier `schema_version 1` results.

`curvature_definitions` in the results JSON records these formulas, their units
and their limits. The internal difference `k_F-k_E` is also reported per pair.
It is a useful finite-difference consistency diagnostic for one model, not a
replacement for comparison with DFT.

### Reproducing the reported errors

`<stem>_arrays.npz` holds every per-structure force array that entered the
summaries, together with the energies and, where available, the stress. It is a
plain NumPy archive, so `np.load(path, allow_pickle=False)` is enough; no
pickle is involved.

```python
import json
import numpy as np

results = json.loads(open("derivative_probe_results.json").read())
export = results["force_export"]
with np.load(export["path"], allow_pickle=False) as data:
    forces = {
        (entry["model"], entry["structure_id"]): data[entry["forces_key"]]
        for entry in export["entries"]
    }
```

Each entry names its model label, structure id, atom count, array shape and key,
so the mapping is unambiguous. `force_export` also records the units of each
quantity, the array layout (`(natoms, 3)` Cartesian in POSCAR atom order, raw
and unconstrained), and the SHA-256 of the archive itself. The archive follows
`--output-stem`, so backend-isolated runs do not overwrite one another.

## Interpretation and limits

This is a compact local-PES test, not a phonon or thermal-conductivity
calculation. Passing it supports the claim that a model reproduces selected
first- and directional second-derivative responses near the chosen minima.
It does not establish:

- complete harmonic force constants, a full Hessian, or phonon dispersions;
- third-order force constants, linewidths, or thermal conductivity;
- transferability far from these minima;
- correct energy ordering among chemically distinct N/O configurations.

In particular, `energy_curvature_ev_a2` and `force_curvature_ev_a2` are
directional projections u^T H u in eV/Angstrom^2 along a handful of sampled
directions. They are not mass-weighted phonon frequencies and do not make this
workflow a phonon, thermal-conductivity, or N/O-ordering-accuracy calculation.

For the current SiN/TiN/TiO work, use this probe to rank finalists for MD and
future thermal-transport work. Continue to validate the swap-MC model primarily
through DFT ordering energies at each oxygen composition.

One random direction is reused at each strain by default. Use multiple directions
and an amplitude sweep before interpreting model rankings. The energy diagnostic
uses ASE's default VASP energy (extrapolated to zero smearing); finite-smearing
forces need not be exact derivatives of that energy. Converge smearing and
consider this when interpreting `k_F-k_E`. Neither numerical amplitude convergence
nor a real MACE/DeePMD/DFT pilot has been established by the automated tests.

## Checklist for a future pilot

The following has **not** been run. It is the minimum a first real campaign
should record before any derivative-probe number is quoted scientifically:

1. **Representative structures** — fully relaxed bulk SiN, bulk TiN and at least
   one relaxed interface, rather than a single convenience cell.
2. **Multiple directions** — `--rattles` well above 1, so curvature is not read
   from one arbitrary random direction per strain.
3. **Amplitude sweep** — at least three `--displacement` values (for example
   0.04, 0.03, 0.02 Angstrom) in separate trees, checking that k_E and k_F
   converge toward each other and toward an amplitude-independent limit.
4. **Converged DFT settings** — ENCUT, k-point mesh and smearing converged for
   the second difference specifically, with the template recorded and reused.
5. **Actual backend runs** — real VASP single points plus real MACE and DeePMD
   inference, each in its own environment and its own `--output-stem`.
6. **Archived environment and results** — the `*_results.json` provenance block,
   the `*_arrays.npz` export, package versions, model hashes and the probe tree
   preserved together, so the reported errors can be recomputed later.

Until such a pilot is recorded and reviewed, this workflow is implemented and
regression-tested software, not a validated scientific measurement.
