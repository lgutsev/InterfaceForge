# Neural density initialization benchmark (example)

> **Illustrative only.** Generated from the synthetic OUTCAR/OSZICAR fixtures in `tests/test_density_init_bench.py` (two cases). No VASP or neural inference was run; the numbers demonstrate the report format, not a performance result.

Finished 2/2 cases; same solution in 2/2 compared; failures standard 0, neural 0.

## Acceptance criteria

- **1_same_electronic_solution**: True
- **2_forces_and_stress_unchanged**: True
- **3_no_increase_in_failure_rate**: True
- **4_afm_ii_preserved**: True
- **5_scf_reduced_in_some_cases**: True
- **6_end_to_end_wall_time_saved**: True

> SCF acceleration and end-to-end acceleration are reported separately; inference, model loading, CHGCAR writing and any grid dry run count against the neural arm. No criterion is evaluated until every case has finished.

## Cases

### sto — SAME_SOLUTION

Category: nonmagnetic-perovskite · ions: 5 · SCF speedup: 1.47× · end-to-end speedup: 1.33×

| Metric | Standard start | Neural start | Difference |
| --- | ---: | ---: | ---: |
| SCF iterations | 22 | 15 | -7 |
| VASP wall time (s) | 600.000 | 420.000 | -180.000 |
| Inference time (s) | 0 | 30.000 | 30.000 |
| Total wall time (s) | 600.000 | 450.000 | -150.000 |
| Final energy (eV) | -38.500000 | -38.499999 | 1.00e-06 |
| Max force difference (eV/Å) | ref |  | 0.000 |
| Stress difference (kB, max comp.) | ref |  | 0.000 |
| Magnetic state | non-magnetic | non-magnetic | — |
| Electronic convergence | CONVERGED | CONVERGED |  |

### nio_afm2 — SAME_SOLUTION

Category: NiO AFM-II · ions: 4 · SCF speedup: 1.14× · end-to-end speedup: 1.08×

| Metric | Standard start | Neural start | Difference |
| --- | ---: | ---: | ---: |
| SCF iterations | 40 | 35 | -5 |
| VASP wall time (s) | 1000.000 | 900.000 | -100.000 |
| Inference time (s) | 0 | 30.000 | 30.000 |
| Total wall time (s) | 1000.000 | 930.000 | -70.000 |
| Final energy (eV) | -40.000000 | -40.000000 | 0.000000 |
| Max force difference (eV/Å) | ref |  | 0.000 |
| Stress difference (kB, max comp.) | ref |  | 0.000 |
| Magnetic state | PRESERVED (moments 1.68–1.68 μB) | PRESERVED (moments 1.679–1.68 μB) | max Δm 1.00e-03 μB |
| Electronic convergence | CONVERGED | CONVERGED |  |
