# NiO AIMD policy

NiO slab and reactive-surface AIMD is a documented exception to InterfaceForge's generic Step1 MD defaults.

## Production rule

For **NiO-containing surface/slab systems**, including hydroxylated NiO, dissociated-water states, and phosphonate-decorated NiO, prepare Step1 with the conservative electronic/ionic settings unless a specific campaign has been separately validated and the override is documented.

Canonical preparation:

```bash
iface vasp step1-prepare OPT \
    --protocol training \
    --temperature 300 \
    --conservative \
    --precondition \
    --ramp-from 100
```

The NiO baseline is therefore:

- `POTIM = 0.5 fs`
- `ALGO = Normal`
- `EDIFF = 1E-5`
- `NELM = 120`
- `NELMIN = 6`
- one static magnetic DFT+U preconditioning SCF before MD (`--precondition`)
- `TEBEG = 100 K` ramping to the requested Step1 target (`--ramp-from 100`)
- keep `NBLOCK = 4` unless there is a separate physical reason to change the thermostat cadence

Do **not** use the generic Step1 recipe (`POTIM=1.0 fs`, `ALGO=Fast`, `EDIFF=1E-4`, `NELM=60`) for a new NiO campaign by default.

## Why NiO is treated conservatively

Production OH50 testing showed that the generic Step1 recipe can produce severe SCF saturation and subsequent force/temperature instability on magnetic DFT+U NiO surfaces, especially for proton-rich and dissociated configurations. The failure mode is electronic as well as ionic: many bad trajectories repeatedly reached the SCF iteration ceiling before the geometry became visibly unstable. A smaller ionic step alone is therefore not the full remedy; the magnetic/DFT+U state should be preconditioned and the MD SCF settings tightened at the same time.

The conservative policy is intentionally broader than OH50. OH25 was easier, but using one robust NiO baseline avoids silently reintroducing the fragile generic settings when a new hydroxylation level, adsorbate, coverage, or surface motif is generated.

## Recovery policy

For an existing unstable NiO Step1 run, use `step1-repair` or the conservative rescue wrapper. Recovery should preferentially rewind to a verified safe `XDATCAR` prefix; otherwise restart from the original structure. The same conservative electronic settings and preconditioning apply.

```bash
bash launch_scripts/rescue_step1_conservative.sh Step1
bash launch_scripts/rescue_step1_conservative.sh Step1 --execute
```

Langevin dynamics is **not** part of the default NiO baseline. Use it only as a second-stage stabilization measure when a case remains physically unstable after conservative SCF settings, preconditioning, the 100 K ramp, and `POTIM=0.5 fs`.

## Interpretation of status warnings

A trajectory should not be discarded solely because the fixed free-energy-reference heuristic reports an early `|F-Fref|` excursion. Check temperature history, SCF-ceiling fraction, trajectory continuity, and the actual geometry. A stable ~300 K trajectory with modest SCF-ceiling use can be a false positive under the current energy-drift detector.

Conversely, repeated SCF-ceiling saturation, multi-thousand-K temperature excursions, non-finite energies/forces, or obvious geometric runaway are grounds for stopping and repairing the run.
