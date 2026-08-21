# RegFGW registry pre-screening (optional, unvalidated)

> **Verification note:** code-only. `regfgw_coherent`'s output has not been
> run or inspected in this repo; `run_regfgw_optimize` deliberately does not
> parse it. `compare_registry_selection`'s comparison math has automated
> coverage but has not yet been checked against a real RegFGW trial. Do not
> treat this as validated until a trial has actually been run and reviewed.

[RegFGW](https://github.com/YuxuanTang2002/RegFGW) uses a fused
Gromov-Wasserstein distance plus Bayesian optimization to cheaply rank
candidate interface **registries** (lateral stacking offsets) without
relaxing every one — the same registry space InterMat's
`--displacement-interval` already sweeps exhaustively (see
[docs/intermat.md](intermat.md)). This is a small, isolated adapter plus a
comparison report to check whether RegFGW's cheap top-k picks actually
preserve the true low-energy registries before trusting it for anything.

It is intentionally optional (`pip install 'interfaceforge[regfgw]'`) and
brand new upstream (1 GitHub star at the time this adapter was written) —
treat it as an experiment, not a dependency InterfaceForge relies on.

## Why not parse RegFGW's output automatically

RegFGW's exact output file format is not documented in enough detail to
parse reliably without running it once and inspecting the result, and no
real run has happened in this repo. `run_regfgw_optimize` therefore just
runs `regfgw_coherent --mode optimize` as a subprocess and returns its
`output_dir`, stdout, and stderr — inspect `output_dir` by hand and adapt
whatever RegFGW actually wrote into the small top-k CSV
`compare_registry_selection` expects (one `registry_id` column, ranked
best-first). This avoids writing confidently-wrong parsing code for a tool
whose output has never actually been inspected here.

```bash
iface regfgw status
iface regfgw optimize substrate.cif film.cif runs/regfgw --budget 3
```

## Running the trial

The trial the maintainer asked for: for two or three existing TiN/SiN
orientation-termination pairs, compare RegFGW's top-k registry picks
against the exhaustive InterMat grid's relaxed work of adhesion, and only
rely on RegFGW going forward if it reliably preserves the low-energy
candidates.

1. Generate the exhaustive registry grid with InterMat
   (`iface intermat generate ... --displacement-interval ...`), then relax
   each candidate and get its work of adhesion (e.g. `iface vasp adhesion
   prepare`/`audit` per registry, or however the campaign already computes
   it). Assemble one CSV: `registry_id`, `work_of_adhesion_ev_a2`.
2. Run `iface regfgw optimize` on the same substrate/film pair, inspect its
   `output_dir`, and write the ranked top-k picks into a second CSV:
   `registry_id` only, best pick first.
3. Compare:

   ```bash
   iface regfgw compare topk.csv exhaustive.csv comparison.csv
   ```

   Reports, per `k` in `{1, 3, 5}` by default:

   - `recall_at_k`: fraction of the true best-`k` exhaustive-grid registries
     that RegFGW's top-`k` also picked;
   - `best_preserved`: whether the single true best registry was proposed
     at all;
   - `energy_regret`: gap between the best work of adhesion among RegFGW's
     picks and the true best — `0` means no regret;
   - `proposed_ids_missing_from_grid`: a proposed id absent from the
     exhaustive CSV (a mismatch to fix before trusting the rest of the row).

   `--lower-energy-is-better` flips the ranking direction for an
   `energy_column` where lower is better (e.g. a formation energy);
   the default assumes higher is better, matching work of adhesion.

4. Repeat for each of the two or three orientation-termination pairs.
   Integrate RegFGW into the real workflow only if `best_preserved` holds
   and `energy_regret` stays small across all of them — a single good trial
   is not enough to trust a 1-star, unvalidated tool.

`compare_registry_selection` is intentionally not RegFGW-specific: it takes
any ranked top-k CSV, so the same comparison works for any other cheap
registry-screening method later.
