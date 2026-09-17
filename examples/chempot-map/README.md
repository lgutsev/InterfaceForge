# Ti-Si-N-O chemical-potential map

Runs the whole finite-`T,p` workflow end to end and prints the comparison:

```bash
python examples/chempot-map/tisino_chempot_map.py --output /tmp/chempot-demo
```

See [docs/gas-chempot.md](../../docs/gas-chempot.md) for the conventions.

## Where the energies come from

This repository holds no finished VASP runs, and mixing energy sources into one
hull is exactly what `iface phases hull` warns against. So every solid here is
entered at its **experimental standard formation enthalpy**, with the elemental
references set to zero energy. Each number is then traceable to a citation
rather than to a calculation nobody can reproduce, and the arithmetic is
visible: a compound's "total energy" simply *is* its formation enthalpy per
formula unit.

| Phase | ΔfH° (kJ/mol) | Source |
|---|---:|---|
| TiN | −337.65 | Chase 1998 (NIST-JANAF 4th ed.) |
| Si3N4 (α) | −744.75 | Chase 1998. The campaign uses β, within ~0.03 eV/f.u. |
| TiO2 (rutile) | −944.0 ± 0.8 | CODATA (Cox, Wagman, Medvedev 1984) |
| SiO2 (α-quartz) | −910.86 | Chase 1998 |

**TiSi2 is deliberately absent.** NIST publishes no condensed-phase
thermochemistry for it, and this example does not invent one. It is handled as
an explicit parameter instead — see below.

These values demonstrate the machinery. They are **not** a substitute for a
consistent DFT set: replace `PHASES` in the script with one entry per finished
run, or call `iface phases map --phase NAME=DIR ...`, which reads the runs
directly.

## What it shows

**The 0 K baseline is untouched.** `dmu(N)` runs from −1.9297 to 0 eV/atom,
lower bound set by elemental Si, upper by N2. The 2D region projected onto the
`dmu_N` axis reproduces that interval to nine decimal places by a completely
different route.

**SiO2, not TiO2, controls the O-rich edge.** With experimental energies the
TiO2 boundary never binds anywhere in the window — it is reported as inactive
with its slack. On the toy test fixture TiO2 *does* bind, so this is a genuine
consequence of the energies, not of the method. It is the kind of thing the
redundancy test exists to make visible.

**Oxidation is thermodynamically favoured at every atmosphere tested.** This is
the headline. Down to `p_O2 = 1e-20 atm` at 600 K, `dmu_O` only reaches
−1.80 eV/atom while the SiO2 boundary sits at −3.81. The required pressures:

| T (K) | dmu_O needed | p_O2 needed |
|---:|---:|---|
| 300 | −3.60 eV/atom | 10⁻¹¹² bar |
| 450 | −3.70 | 10⁻⁷³ bar |
| 600 | −3.81 | 10⁻⁵⁴ bar |

None of those is reachable. Read literally, a Si3N4/TiN interface is **never**
thermodynamically safe from SiO2 formation in any attainable atmosphere; what
protects it is kinetics — a dense oxide layer, slow transport, or simply not
enough time. That is a statement worth making explicitly rather than implying
that some sufficiently good glovebox would do it.

(The required pressure *rises* with temperature because the gas entropy term
already pushes `mu_O` down at higher `T`, so less help is needed from the
pressure term.)

**The pressure effect is large.** Per oxygen atom, `0.5 kB T ln(p/p0)` is
−0.15 eV at 300 K and 1e-5 atm, and −1.19 eV at 600 K and 1e-20 atm. These are
eV-scale shifts, not the 0.05 eV one might guess.

## The TiSi2 question

The script derives, rather than asserts, which phase sets the N-poor edge.

With TiN and Si3N4 held in equilibrium, `dmu_Ti` and `dmu_Si` are fixed
functions of `dmu_N`, so *every* Ti-Si phase reduces to a straight lower bound
on `dmu_N`. Elemental Si gives one; a silicide gives another. Which binds is
decided by a single inequality, and `silicide_handover()` returns the crossover:

> With this phase set the N-poor edge sits at −1.9297 eV/atom, set by Si. A
> TiSi2 with ΔHf more negative than **−1.5698 eV/f.u. (−151.5 kJ/mol)** would
> take that edge over and tighten the window; a weaker one leaves it alone.

That threshold is the whole of the −2.068 versus −1.973 eV/N question. Check
your own DFT TiSi2 against it.

Two caveats worth stating plainly:

- These experimental numbers are **not** the campaign's DFT numbers, so the
  −1.9297 here is not the −1.973 from the cluster run. It is close because the
  experimental ΔfH°(Si3N4) is close to the DFT value, not because they are the
  same calculation.
- The −2.068 and −1.973 values themselves appear nowhere in this repository or
  its git history, so they could not be re-derived here. The threshold above is
  the reproducible part: it says exactly what would have to be true of TiSi2
  for the tighter value to be the right one.
