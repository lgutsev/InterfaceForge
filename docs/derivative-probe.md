# Derivative-sensitive MLIP validation

> **Verification note:** automated-test only. Structure generation, input
> provenance, overwrite guards, and analytic harmonic-curvature recovery are
> covered by tests. No real Si-Ti-N-O DFT/MLIP probe campaign has yet been run.

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

InterfaceForge also writes the negative of each random displacement vector.
These symmetric pairs enable central-difference directional-curvature checks.
That pairing is an InterfaceForge extension and must not be described as the
paper's exact data-generation protocol.

## Prepare the probe

Start with fully relaxed 0 K structures. A VASP run directory may be supplied;
a non-empty `CONTCAR` is preferred over its `POSCAR`.

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
InterfaceForge first verifies that POTCAR's VRHFIN order matches the source
POSCAR species blocks. It then copies the same KPOINTS/POTCAR into every probe and converts the
INCAR to a static protocol:

- `IBRION=-1`, `NSW=0`, and `ISYM=0`;
- `LWAVE=.FALSE.` and `LCHARG=.FALSE.`;
- MD-only, ionic-relaxation, and `ML_*` tags removed;
- electronic settings such as ENCUT, precision, spin, DFT+U, dispersion, and
  convergence retained from the template.

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
  --device cuda
```

Run DeePMD/DPA models in their own compatible environment if necessary:

```bash
iface validate derivative-probe evaluate derivative_probe \
  --deepmd-model /models/dpa3/000/frozen_model.pth \
  --deepmd-model /models/dpa3/001/frozen_model.pth
```

The evaluator writes:

- `predictions.csv`: energy, force norms, and stress (when implemented) for
  every backend/structure;
- `responses.csv`: energy- and force-derived directional curvatures for every
  symmetric pair;
- `derivative_probe_results.json`: per-model comparison with DFT.

Energy comparison uses each source's unstrained center as its zero, then
reports relative-energy errors in meV/atom. This removes arbitrary
composition-dependent offsets without mixing different source compositions.
Forces are compared component by component. Stress is compared when both
backends provide it. Directional curvature is estimated two ways for a
displacement vector d, with q=||d|| and unit direction d/q:

k_E = [E(+d) - 2E(0) + E(-d)] / q^2

k_F = -[F(+d) - F(-d)] dot (d/q) / (2q)

The internal difference `k_F-k_E` is also reported. It is a useful
finite-difference consistency diagnostic, not a replacement for comparison
with DFT.

## Interpretation and limits

This is a compact local-PES test, not a phonon or thermal-conductivity
calculation. Passing it supports the claim that a model reproduces selected
first- and directional second-derivative responses near the chosen minima.
It does not establish:

- complete harmonic force constants or phonon dispersions;
- third-order force constants, linewidths, or thermal conductivity;
- transferability far from these minima;
- correct energy ordering among chemically distinct N/O configurations.

For the current SiN/TiN/TiO work, use this probe to rank finalists for MD and
future thermal-transport work. Continue to validate the swap-MC model primarily
through DFT ordering energies at each oxygen composition.
