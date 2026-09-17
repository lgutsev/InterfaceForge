# Finite-temperature gas reservoirs and the 2D chemical-potential map

The 0 K workflow in [interface-mu](interface-mu.md) measures every chemical
potential from an elemental reference computed at 0 K. For nitrogen that
reference is `E(N2)/2` — one N2 molecule in a box, at rest, divided by its two
atoms. No experiment provides that reservoir. A furnace provides a temperature
and a partial pressure.

This guide covers the layer that bridges the two, and the 2D stability region
that comes with it once oxygen is in play.

**The 0 K workflow is unchanged.** `iface phases hull` and `iface validate
interface-mu` behave exactly as before; neither grew a temperature flag. The
finite-`T,p` treatment lives in two new subcommands you have to ask for. A
regression test (`tests/test_chempot_map.py::ZeroKelvinPreservationTest`) pins
the old numbers and asserts that no temperature or pressure term appears
anywhere in the 0 K payload.

## The bookkeeping

Per X2 molecule:

```
mu_X2(T, p) = E_X2_dft [+ zpe] + dmu_thermal(T, p0) + dmu_pressure(T, p)

dmu_thermal(T, p0) = [H(T) - H(0)] - T * S(T, p0)
dmu_pressure(T, p) = kB * T * ln(p / p0)
```

and per **atom**, which is what the phase-diagram solver consumes:

```
mu_X(T, p)  = mu_X2(T, p) / 2
dmu_X(T, p) = mu_X(T, p) - E_X2_dft / 2
            = ( [zpe] + dmu_thermal + dmu_pressure ) / 2
```

### The factor of two, and where it is not

The last line is the one to read twice. `E_X2_dft` **cancels exactly** out of
`dmu`. The 0 K scale is already measured from `E_X2_dft/2`, so adding the DFT
energy back into a *delta* counts it twice. In the output, `mu_molecular_ev`
and `mu_atomic_ev` are absolute and do contain it; `dmu_atomic_ev` is the one
the hull consumes and does not.

The halving happens once per quantity and is pinned on its own in
`tests/test_gas_reference.py::FactorOfTwoTest`, including a test that feeding
three different `E(N2)` values leaves `dmu` untouched.

This matters because `pymatgen` already normalises per atom: for a PDEntry
built from `Composition({"N": 2})` with total energy `E`,
`PhaseDiagram.el_refs[N].energy_per_atom` is `E/2`, not `E`. A finite-`T` layer
that helpfully divided by two again would move the N-poor edge by an eV with no
symptom other than a wrong answer.

## Where the numbers come from

| Quantity | Source |
|---|---|
| Cp(T), S(T), H(T) − H(298.15) | Shomate coefficients from the [NIST Chemistry WebBook](https://webbook.nist.gov), reproducing Chase, M. W., Jr., *NIST-JANAF Thermochemical Tables, Fourth Edition*, J. Phys. Chem. Ref. Data, Monograph 9 (1998) |
| H(298.15) − H(0) | CODATA Key Values (Cox, Wagman and Medvedev, 1989): 8.670 ± 0.001 kJ/mol for N2, 8.680 ± 0.002 for O2 |
| Zero-point energy | `hc*we_/2` with `we_` from Huber and Herzberg's diatomic constants via the same WebBook: 2358.57 cm⁻¹ (N2, X¹Σg⁺), 1580.193 cm⁻¹ (O2, X³Σg⁻) |
| kB, N_A, e, h, c | Exact by SI definition since 2019, so the unit conversions carry no uncertainty |

Coefficients were transcribed on 2026-09-16. Nothing is hard-coded without a
check: `tests/test_gas_reference.py::ShomateTest` reproduces the thirty Cp, S
and H values NIST prints in the table beside the coefficients, across both
temperature intervals of both species. A mistyped digit fails the suite instead
of quietly biasing every chemical potential downstream.

### The 298.15 K problem

NIST's Shomate enthalpy is referenced to **298.15 K**. A DFT total energy is a
**0 K** quantity. Using the tabulated increment directly would be wrong by
`H(298.15) - H(0)` — 8.67 kJ/mol, which is 0.045 eV per N *atom*. Not a
rounding error.

That offset is the one number in the module that is not itself a Shomate
coefficient, so it gets an independent check. Below 100 K a diatomic ideal gas
has `Cp = 7R/2` exactly (rotation fully active, vibration frozen; NIST
tabulates 29.10 J/mol/K at 100 K, which is `7R/2` to four figures). Integrating
that to 100 K and adding the Shomate increment from there reproduces the CODATA
values to 0.01 kJ/mol. The test is
`test_the_codata_0k_offset_is_reproduced_by_integrating_the_heat_capacity`.

### Zero-point energy is opt-in

JANAF's `H(T) - H(0)` excludes the ZPE by construction: the ZPE sits inside
`H(0)` and cancels in the difference. A static DFT energy also excludes it. So
the two are consistent *without* it, and adding it to the gas alone — while
every solid here is a bare 0 K electronic energy — would be a one-sided
correction of about 0.073 eV/atom for N and 0.049 for O.

`--include-zpe` turns it on if you want it. Either way the available value is
reported, so you can see its size without rerunning.

### No empirical correction is applied

The raw DFT energies are used unmodified. This repository applies no O2
overbinding correction, no fitted anion shift, no `MaterialsProject2020Compatibility`
— nothing. `reference_scheme` in every output records this as `raw-dft`.

`interface-mu` already notes that PBE overbinds N2 by roughly 0.5 eV and offers
substituting the measured formation enthalpy as the remedy. That remains a
choice you make deliberately, not something this layer does behind your back.
If you do want a correction, pass `empirical_correction_ev` **and** a
`reference_scheme` naming where the number came from; supplying the shift
without naming the scheme is refused, because an unattributed shift to a
reference energy is exactly what makes a chemical potential unreproducible.

## What "anoxic" does not mean

A partial pressure asserts an *equilibrated* O2 reservoir. That is a much
stronger statement than the label on a gas bottle.

| Atmosphere | What it actually fixes |
|---|---|
| N2 | `p_N2` directly. Its oxygen content is a leak-and-getter number, not a thermodynamic one |
| Ar or other inert | Nothing about N or O. Both are residual impurities set by kinetics |
| Vacuum | A total pressure. The O2 partial pressure depends on outgassing and pump speed |
| H2/H2O forming gas | `mu_O` through the `H2 + 1/2 O2 <-> H2O` equilibrium — *not* through a p_O2 you can read off a gauge |
| Equilibrated O2 | `p_O2`, which is what this layer models |

Quoting `p_O2 = 1e-20 atm` for "an argon glovebox" asserts an equilibrium
nobody measured. Use a value you can defend — an oxygen-analyser reading, or a
bracketing range — and treat the rest as a sensitivity scan. The demonstration
pressures in the example are exactly that: a scan, not a measurement.

For a buffered reducing atmosphere the right treatment is the gas equilibrium,
and the seam is already there: `gas_reference.RESERVOIR_SCHEMES` carries
`h2-h2o` and `co-co2` as named schemes that currently refuse with the equation
they would need —

```
mu_O(T) = mu_H2O(T, p_H2O) - mu_H2(T, p_H2)
dmu_O   = dG_f(H2O, T) + kB*T*ln(p_H2O / p_H2)
```

— so the next person implements the right equation rather than guessing one.

## Solids stay at 0 K

Only the gases carry temperature and pressure. Every solid keeps its 0 K DFT
electronic energy: no phonons, no thermal expansion, no configurational
entropy. This is the standard ab-initio-thermodynamics approximation, and it is
good *because* the gas term dominates — roughly 0.6 eV per O2 at 600 K against
a few tens of meV for a solid.

But it means these are **not** full finite-temperature Gibbs free energies, and
a boundary between two solids moves less than this treatment can resolve. Every
payload carries `solid_approximation` saying so.

This is deliberately *not* the same thing `interface-mu` refuses. That guard
rejects an interface energy measured from an MD snapshot against 0 K
references, which is an inconsistency in the *state the DFT was run in*. Here
every DFT number is still a 0 K static; the temperature enters only as a
property of the reservoir the solids are in equilibrium with. Reservoir
temperature and calculation temperature are different things, and mixing them
up is the error that guard exists to catch.

## The 2D region

With four elements, the feasible set lives in 4D and a `(dmu_N, dmu_O)` picture
needs two dimensions eliminated. Holding the interface's own constituents in
equilibrium does exactly that:

```
TiN   in equilibrium:   dmu_Ti + dmu_N     = dHf(TiN)
Si3N4 in equilibrium:  3 dmu_Si + 4 dmu_N  = dHf(Si3N4)
```

Two equalities in four unknowns leave two free coordinates, taken to be `dmu_N`
and `dmu_O`. So `dmu_Ti` and `dmu_Si` are not assumed away — they are
*determined* by `dmu_N`. Substituting them into any other phase's stability
condition leaves a straight line in the plane. For composition
`(n_Ti, n_Si, n_N, n_O)`:

```
(n_N - n_Ti - (4/3) n_Si) dmu_N + n_O dmu_O
    <= dHf(phase) - n_Ti dHf(TiN) - (n_Si/3) dHf(Si3N4)
```

That `dmu_N` coefficient deserves a second look: `n_N - n_Ti - (4/3) n_Si` is
the same excess-nitrogen combination `dN_N` that sets the slope of `gamma(mu_N)`
in interface-mu. The interface energy and the phase boundaries keep the same
books — which is the check that the two halves of the framework agree.

The reduction is done numerically and in general, not hard-coded for TiN and
Si3N4. `reduce_to_plane` builds the null space of the equality system and
refuses, with a diagnosis, if the named compounds do not leave exactly two
degrees of freedom, or if the requested axes are not independent coordinates on
what is left.

### Active and inactive constraints

A line that is drawn is not a line that binds. Every constraint is tested for
redundancy by linear programming — maximise its own left-hand side over all the
*other* constraints and see whether it can ever be reached. Inactive phases are
listed, not plotted, and the report gives the slack by which each one misses.

This is the part that catches the failure mode that motivated the module: a
boundary plotted from one phase's individual stability while the joint
constraint is set by another. Both now come out of the same polytope.

The consistency check is explicit: the region's `dmu_N` extent must equal the
1D window from `iface phases hull`, computed by a completely different route.
`test_the_nitrogen_extent_reproduces_the_existing_1d_window` asserts it.

On the toy quaternary fixture the two routes disagree in an instructive way for
oxygen. The 1D `--anion O` path takes the `open_element_limit` bisection and
returns `-5.250034` attributed to **TiN**; the 2D route returns exactly `-5.25`
attributed to **TiO2**. Both are honest answers to different questions — "which
named compound falls over first" versus "which phase's stability condition
draws the line" — but only the second names the phase you would have to
suppress. The bisection is also only converged to its tolerance.

### Unbounded regions are reported, not clipped away

Nothing forces oxygen into the cell, so `dmu_O` usually runs to negative
infinity and the region is unbounded. The report says `"bounded": false`, names
the unbounded direction, and the polygon handed to the plotter is clipped to an
explicit `clip_box_ev` that appears in the JSON. The clip is a drawing choice
and carries no thermodynamic meaning.

## Commands

### `iface phases gas-chempot`

Gas thermochemistry on its own — no phases, no hull.

```bash
iface phases gas-chempot --temperature 600 --p-n2 1 --p-o2 1e-15 --pressure-unit atm
```

Reports the thermal and pressure terms separately, per molecule and per atom,
with the source of every number. Add `--e-n2-ev`/`--e-o2-ev` to get absolute
potentials as well as deltas. `--pressure-unit` accepts `bar`, `mbar`, `Pa`,
`kPa`, `MPa`, `atm`, `torr`. `--standard-pressure-bar` defaults to 1 bar,
which is what NIST-JANAF tabulates; setting it to 1.01325 would silently bias
every entropy term, so it is explicit.

`T <= 0` and `p <= 0` are refused. A temperature outside the fitted Shomate
range is refused too, rather than extrapolated — the polynomial would still
return a plausible number.

### `iface phases map`

The 2D region, from the same phase directories `iface phases hull` reads.

```bash
iface phases map \
    --phase TiN=runs/TiN --phase Si3N4=runs/Si3N4 \
    --phase TiO2=runs/TiO2 --phase SiO2=runs/SiO2 \
    --phase Ti=runs/Ti --phase Si=runs/Si \
    --phase N2=runs/N2 --phase O2=runs/O2 \
    --compound TiN --compound Si3N4 \
    --axes N,O \
    --temperature 300 --temperature 450 --temperature 600 \
    --p-n2 1 --p-o2 1e-15 --pressure-unit atm \
    --output phase_map_600K
```

`--temperature` is repeatable: one marker per temperature. Omit it entirely for
the 0 K region alone. `--fixed-dmu ELEMENT=EV` supplies a further equality if
the constituents do not leave exactly two degrees of freedom. `--clip` sets the
drawing window without touching the computed region.

Outputs: `chempot_map.json`, `chempot_map_constraints.csv`,
`chempot_map_conditions.csv`, `chempot_map.md`, and the figure as png/svg/pdf.

## Worked example

[`examples/chempot-map/`](../examples/chempot-map/) runs the whole thing on
Ti-Si-N-O, using experimental formation enthalpies with citations because this
repository holds no finished VASP runs. It prints the 0 K baseline, the 2D
region, the anoxic scan, and the TiSi2 handover threshold.

## Limitations

- Solids are 0 K electronic energies. No phonons, no thermal expansion, no
  configurational entropy.
- The gas is ideal. At the pressures of interest it is; at hundreds of bar it
  would not be.
- The Shomate fits cover 100–6000 K. Outside that the code refuses.
- `H(298.15) - H(0)` is tabulated for N2 and O2 only. Adding a species means
  transcribing its intervals and its CODATA offset, and pinning both against
  the published table.
- The hull is only as complete as the phases you supply. An omitted stable
  phase can only make the region look **too large**, never too small.
- Nothing here corrects the DFT error on the O2 or N2 molecule itself. If your
  reference is overbound, every `mu` derived from it inherits that.
