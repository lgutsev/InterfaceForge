# Matched MACE versus DeePMD audit

This workflow compares existing MACE and DPA-2 committees on the exact same
canonical test configurations. It does not retrain either model.

Install the reporting dependencies before finalization so the audit can render
the per-system committee heatmaps:

```bash
pip install -e '.[vasp,report]'
```

For the current periodic SiN/TiN/TiO committee, the legacy training launcher
and the generated comparison evaluator both use float32. The inference dtype is
recorded in `comparison_manifest.json` and `comparison.json`.

The comparison is explicitly an **in-distribution interpolation benchmark**:
every source trajectory contributed frames to training, validation, and test.
Use an independent trajectory or physical-regime challenge set before making
transferability claims.

## Checking progress

`iface mlip-progress` prints one table covering every committee at once —
DeePMD training (step vs `numb_steps`, `rmse_f_val`, checkpoint/frozen flags),
DeePMD evaluation (`by_system` completion, `rmse_overall.csv`), the from-scratch
and fine-tuned MACE committees (last epoch, `RMSE_F`, final model), and each
`audit/mlip_compare*` run (MACE prediction count, finalized or not). It reads
only generated files, so it is safe to run against live jobs. Add `--json` for
the machine-readable payload, or `--mace-committee-root` if the MACE committees
are not under `models/mace_committee_520eV/`.

This rollup has been used against the real running SiN/TiN/TiO MACE and DeePMD
committees. It reports progress inferred from artifacts on disk; use `squeue`
or `sacct` separately for scheduler state, allocation and failure reasons.

## Workflow

From the campaign root:

```bash
iface mlip-compare prepare --force
```

Preparation needs one usable MACE export per seed `11 23 37 53` in
`seed_<seed>/mace_model/`. A stage-two / SWA export (`*_stagetwo.model`,
`*_swa.model`, …) is preferred; a single-stage export (`<name>.model`, e.g. a
naive EMA-only foundation-model fine-tune) is accepted with a note in
`comparison_manifest.json.model_selection_notes`. `*_compiled.model` and empty
files are ignored; the newest is used when several exports share the top tier
(same rule as the separation-energy launcher). It then proves that the MACE
extxyz and DeePMD NPY test data have identical `(IF_leaf, source_frame)`
membership, atom order, coordinates, cells, energies, and forces. The tolerance
is `1e-7`.

Submit the generated MACE inference array:

```bash
sbatch audit/mlip_compare/run_mace_evaluate.slurm
```

The array evaluates four models with at most two running concurrently. It is
restartable: completed non-empty per-system prediction files are skipped.

Check both backends:

```bash
iface mlip-compare status \
  --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
```

Finalize only after every model has 48/48 systems:

```bash
iface mlip-compare finalize \
  --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
```

### Comparing against DPA-3, DPA-4, or a fine-tuned DPA-2 / DPA-3

`prepare` takes `--deepmd-arch {dpa2,dpa2_ft,dpa3,dpa3_ft,dpa4}` (default
`dpa2`). It selects the `models/deepmd/evaluation/<arch>` tree and records the
choice in the manifest; `status` and `finalize` read it back, and every rendered
label (`comparison.md`, `comparison.svg`, the three summary figures, the
heatmaps) shows the real architecture. The metrics schema is unchanged — the
internal engine key stays `DPA2`.

Run once per architecture into its own output directory, against the same MACE
committee:

```bash
for arch in dpa2 dpa2_ft dpa3 dpa3_ft dpa4; do
  iface mlip-compare prepare --deepmd-arch "$arch" \
    --output-root "audit/mlip_compare_$arch" --force
  sbatch "audit/mlip_compare_$arch/run_mace_evaluate.slurm"
done
# the fine-tuned MACE committee is its own run (any --deepmd-arch will do; only
# its MACE rows are used downstream). --mace-models-root takes a bare name that
# resolves under models/mace_committee_520eV/:
iface mlip-compare prepare --deepmd-arch dpa2 \
  --mace-models-root mace_finetune_committee \
  --output-root audit/mlip_compare_mace_ft --force
sbatch audit/mlip_compare_mace_ft/run_mace_evaluate.slurm

# after every MACE array and every DeePMD evaluation finishes:
for arch in dpa2 dpa2_ft dpa3 dpa3_ft dpa4; do
  iface mlip-compare finalize --output-root "audit/mlip_compare_$arch" \
    --deepmd-eval-root "models/deepmd/evaluation/$arch/job_<jobid>"
done
iface mlip-compare finalize --output-root audit/mlip_compare_mace_ft \
  --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
```

MACE inference re-runs per architecture (cheap at float32).

### `combine`: one figure for every family

`iface mlip-compare combine` overlays the pooled per-group RMSE CSVs from
several finalized runs into one figure set — no committee is re-evaluated:

```bash
iface mlip-compare combine audit/mlip_compare_all \
  --run "MACE=audit/mlip_compare_dpa2" \
  --run "MACE (fine-tuned)=audit/mlip_compare_mace_ft" \
  --run "DPA-2=audit/mlip_compare_dpa2" \
  --run "DPA-2 (fine-tuned)=audit/mlip_compare_dpa2_ft" \
  --run "DPA-3=audit/mlip_compare_dpa3" \
  --run "DPA-3 (fine-tuned)=audit/mlip_compare_dpa3_ft" \
  --run "DPA-4=audit/mlip_compare_dpa4"
```

Each `--run LABEL[:ENGINE]=DIR` names a finalized output directory and the
family label to draw it as. `ENGINE` is `MACE` or `DPA2` and defaults to `MACE`
when the label starts with `mace` (case-insensitive), else `DPA2` — so a MACE
row is pulled from any run and each DeePMD arch from its own run. Output is
`audit/mlip_compare_all/{publication,temperature,oxidation}_rmse_summary.{png,svg,pdf}`
plus the merged CSVs and `combined_manifest.json`. Views whose CSV is missing
from a run are skipped with a note. `--members` forces the per-member open
circles (default: shown only for ≤ 4 families); `--no-members` forces the
compact ensemble-only form; families beyond four get alternating row bands.

Finalization independently checks that the reference columns written by
`dp test -d` still match the canonical MACE labels. It then writes:

- `metrics_by_system.csv`: every committee member and ensemble mean;
- `metrics_overall.csv`: micro and equal-trajectory macro metrics;
- `metrics_by_group.csv`: bulk/interface, temperature, family, termination,
  and oxidation breakdowns. Oxidation is `NA` for bulk heritage, `0` for an
  interface leaf with no `O_x` token, and `O_x1.0`/`O_x1.00` are the same `1`
  group;
- `uncertainty_calibration.csv`: committee-spread/error correlation,
  scale factor, and empirical one/two-sigma coverage;
- `comparison.json`, `comparison.md`, and `comparison.svg`;
- `publication_rmse_by_group.csv`: exact pooled energy and force RMSE for the
  overall test set and seven physical-system bins;
- `publication_rmse_summary.{png,svg,pdf}`: compact two-panel main figure with
  energy and force RMSE, four committee members, and the committee-averaged
  prediction;
- `temperature_rmse_by_group.csv`: pooled RMSE for Overall, 300 K, and 450 K;
- `temperature_rmse_summary.{png,svg,pdf}`: separate publication figure using
  the same energy/force and committee visual encoding for temperature;
- `oxidation_rmse_by_group.csv`: pooled RMSE for Overall, a single `Bulk` bin,
  and each interface oxygen coverage `O = 0`, `0.25`, `0.5`, `0.75`, `1`;
- `oxidation_rmse_summary.{png,svg,pdf}`: same encoding again, resolving whether
  accuracy is coverage-dependent (e.g. worse at full oxidation) for either
  engine;
- `force_rmse_heatmap_mace.{png,svg}` and
  `force_rmse_heatmap_dpa2.{png,svg}`: annotated member-by-system heatmaps;
- `force_rmse_heatmaps.{png,svg}`: both committees side by side with identical
  system order and one shared color scale, for a visually honest comparison.

The heatmaps report force-component RMSE in eV/Å, matching the conventional
unit used in MACE and DeePMD evaluation figures. Their underlying CSV values
remain in meV/Å. They deliberately exclude the ensemble-mean column: each of
the four cells shows one independently trained committee member.

The publication summary is deliberately less granular. It pools trajectories
into Bulk SiN, Bulk TiN, Bulk TiO, and the four interface combinations formed
by Ideal/Real structure and N/Ti termination. Temperature and oxidation are
therefore replicated conditions within those interface bins, while their full
resolution remains available in `metrics_by_group.csv` and the diagnostic
heatmaps. The plotted RMSE is recomputed by observation weighting of squared
per-system RMSEs, not by averaging per-system RMSE values. Open circles are
the four independently trained models, the connecting segment is their range,
and the filled diamond is the RMSE of the committee-averaged prediction. The
diamond is not the arithmetic mean of the four member RMSEs and may lie below
their range when member errors cancel.

The oxidation summary pools the 8 bulk trajectories into one `Bulk` bin (they
have no oxygen-coverage coordinate) and the 40 interface trajectories into
eight-system bins by coverage. Its Overall row is again identical to the other
two figures.

The temperature summary separately pools all 24 test systems at 300 K and all
24 at 450 K. Its Overall row is identical to the Overall row in the physical-
system figure, providing a direct visual reference without mixing temperature
and chemistry into one overloaded panel.

Energy is evaluated per atom. Force RMSE is over Cartesian components; vector
RMSE and force RMSE normalized by the reference-force standard deviation are
also reported. Centered energy RMSE removes one constant residual offset per
source trajectory. Virials are not included in the cross-backend headline
because this MACE committee was not trained on virials.

## Short debugging assignment

1. Pull and install current `main`, then run:

   ```bash
   pytest -q tests/test_mlip_compare.py
   iface mlip-compare prepare --force
   ```

2. Verify these preparation gates:

   ```bash
   jq '.validation | {
     exact_membership, systems, frames, duplicate_frame_ids, max_absolute_delta
   }' audit/mlip_compare/comparison_manifest.json
   ```

   Expected: `true`, 48 systems, 2,880 frames, zero duplicates, and every
   maximum delta at or below `1e-7`.

3. Submit the generated MACE array. After it finishes, run `status` with the
   explicit DPA-2 evaluation job directory. Expected: models `000..003` are
   all `48` for both MACE and DeePMD and status is `READY_TO_FINALIZE`.

4. Run `finalize`. Confirm `comparison.json` says `OK`, its DeePMD
   reference deltas are at most `1e-7`, and there are ten overall rows
   (four members plus ensemble, each with micro and macro averaging) per
   engine.

5. Confirm `publication_rmse_by_group.csv` contains the eight ordered bins
   `Overall`, three bulk chemistries, and four interface family/termination
   combinations for each member and ensemble. Open
   `publication_rmse_summary.pdf`; both panels must show four open member
   circles and one filled committee-prediction diamond per engine and bin.

6. Confirm `temperature_rmse_by_group.csv` contains `Overall`, `300 K`, and
   `450 K`, with 48, 24, and 24 systems respectively. Open
   `temperature_rmse_summary.pdf` and confirm its Overall values match the
   physical-system figure exactly.

7. Debugging cross-check: the individual MACE micro force RMSE values should
   reproduce the existing MACE test table near 62 meV/A within ordinary
   numerical/inference rounding. The DPA-2 individual micro values should
   exactly reproduce its existing `rmse_overall.csv`. Investigate any larger
   discrepancy before interpreting which architecture is better.

The unit test deliberately perturbs a copied DeePMD coordinate and confirms
that the workflow fails instead of silently comparing different frames.
