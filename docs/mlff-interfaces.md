# Bulk MLFF training grid + ML_LHEAT thermal conductivity

> **Verification note:** code-only. Reuses the human-tested VASP-MLFF
> preparation/audit/recovery path (`prepare_campaign`, `submit_campaign`,
> `stage_tags`, `iface audit`), but the grid tooling itself (discovery,
> campaign generation, throttled array launch, rollup audit) and the new
> `heat` recovery operation have not been run against a real campaign yet.

For a family x termination x composition grid of interface structures —
e.g. `Real`/`Ideal` x `N_Term`/`Ti_Term` x `x=0,0.25,0.5,0.75,1.0`, 20
independent, temperature-transferable MLFFs — training 20 MLFFs and later
running Green-Kubo thermal conductivity (`ML_LHEAT`) by hand does not scale.
This does not duplicate the existing campaign machinery: it turns a
reviewed source manifest into a `campaign.yaml` that machinery already
knows how to prepare/submit/audit, and adds the two pieces that machinery
did not have — throttled mass submission, and a grid-shaped audit rollup.

For the LA Tech tree, `launch_scripts/periodic_interface_mlff.sh` supplies
the established paths and settings. Its workflow remains deliberately
review-gated:

```bash
export CER_INTERFACE_BASE=/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface

launch_scripts/periodic_interface_mlff.sh discover
# Review MD_Period/VASP_MLFF_Interfaces_source_manifest.csv.
launch_scripts/periodic_interface_mlff.sh build
launch_scripts/periodic_interface_mlff.sh prepare
# Review, then submit runs/vasp/_arrays/train_array.slurm with sbatch.
launch_scripts/periodic_interface_mlff.sh audit
```

`prepare` writes the throttled `%2` training array but does not submit it.
This prevents an unresolved manifest or unreviewed launcher from starting 20
calculations. Override `IFACE_MLFF_SOURCE`, `IFACE_MLFF_OUTPUT`,
`IFACE_MLFF_MANIFEST`, or `IFACE_MLFF_PROFILE` when the tree moves.

## 1. Discover sources (review before trusting)

```bash
iface mlff-interfaces discover /path/to/Step2_450K manifest.csv
```

Best-effort match of one `CONTCAR` per grid cell, by simple case-insensitive
token matching against each candidate's path — `real`/`ideal`,
`n_term`/`nterm`/`n-term` (and the `ti_term` equivalents), and any numeric
directory token equal to that cell's `x` (so `x1`, `x1.0`, and `x=1.00` all
match `x=1.0`; `x=0` cannot accidentally match inside `x=0.25` — both were
real bugs caught while building this). Also handles the real observed
LONI convention directly: the `x=0` (oxygen-free) baseline carries **no**
numeric suffix at all (`SiN_TiN_N-term`), unlike the oxygen-substituted
cells (`SiN_TiN_N-term_O_x0.25`) — a bare leaf under a matching family/term
is treated as `x=0` only when nothing else there carries any x-like
numeric token, so it can't accidentally swallow an unrelated file.
`N_Term`/`Ti_Term` may each use a different separator style in their own
leaf names (`SiN_TiN_N-term` vs `SiN-TiN-Ti-term`) without needing separate
configuration — matching keys off the `N_Term`/`Ti_Term` *parent* directory,
not the leaf's own naming. **Never guesses past reporting**:
`manifest.csv`'s `match_status` per row is `matched`, `missing`, or
`ambiguous` (candidates listed). It also resolves `INCAR`, `KPOINTS`, and
`POTCAR` from the leaf or an ancestor no higher than `source_root`, records
their SHA-256 hashes, and reports `inputs_status`. Review the manifest before
the next step, which refuses anything unresolved or incomplete.
Point `source_root` at the temperature-specific folder itself (e.g.
`Step2_300K/`, not its `MD_Period` parent) so a temperature like `300` in
an ancestor directory name can't be mistaken for an `x` value.

## 2. Build the campaign

```bash
iface mlff-interfaces build manifest.csv VASP_MLFF_Interfaces \
  --profile examples/mlff-interfaces/profile_loni.yaml --tebeg 300 --teend 600
```

[`examples/mlff-interfaces/profile_loni.yaml`](../examples/mlff-interfaces/profile_loni.yaml)
defines `vasp_train`/`vasp_train_array`, matching the maintainer's
production `runvasp.sh` exactly (`workq`, 2 nodes / 128 MPI tasks,
`vasp6/6.5.1-cpu`, `vasp_std`, 72h). Copy and adjust it for a different
account/partition/binary rather than editing it as the source of truth.

Writes `VASP_MLFF_Interfaces/campaign.yaml` with one system per grid cell
(`kind: interface`, `tags: {family, term, x}`). For maximum reproducibility,
it snapshots each reviewed leaf's exact `CONTCAR`, `INCAR`, `KPOINTS`, and
chemistry-specific `POTCAR` below `inputs/systems/<system-id>/` and writes
`inputs/source_provenance.json` with source/snapshot paths and SHA-256 hashes.
It does not invent a Gamma mesh, replace a POTCAR, or discard other converged
INCAR settings. The build refuses a source INCAR unless `ENCUT`, `IVDW`, and
`POTIM` match the explicitly requested values (defaults: 520, 11, and 1 fs).

The 300→600 K training ramp (`TEBEG`/`TEEND`) is new support in
`stage_tags`/`prepare_campaign` (`teend` in a stage's settings), consistent
with VASP's MLFF best-practices guidance to train somewhat above the
highest application temperature. One transferable model per structure
across that range, rather than separate fixed-temperature models, is what
`--tebeg`/`--teend` produce.

Then the existing pipeline takes over unchanged:

```bash
iface prepare -c VASP_MLFF_Interfaces/campaign.yaml
iface submit -c VASP_MLFF_Interfaces/campaign.yaml --stage train --execute
```

## 3. Throttled mass launch

`iface submit --stage train --execute` submits every leaf's `run.slurm`
immediately with no concurrency limit — fine for a handful of jobs, not for
20 competing for LONI's fairshare queue at once. Instead:

```bash
iface mlff-interfaces array-launch -c VASP_MLFF_Interfaces/campaign.yaml \
  --stage train --concurrency 4
```

Writes **one** Slurm array job (`--array=0-19%4`, Slurm's native throttling
syntax — not a custom polling loop) at up to 4 concurrent tasks. Each task
changes into its manifest-selected run directory, verifies nonempty
`INCAR/KPOINTS/POSCAR/POTCAR`, and invokes the stage profile's VASP command
directly. It deliberately does not execute the leaf `run.slurm`: inside an
array, that nested script's `SLURM_SUBMIT_DIR` belongs to the parent array
and can redirect VASP to the wrong directory. The stage's own profile is
used by default; `--array-profile-name` is optional. LONI profiles must not
set memory manually. This command only writes the launcher; review it and
`sbatch` it yourself.

## 4. Grid-aware audit

```bash
iface mlff-interfaces audit -c VASP_MLFF_Interfaces/campaign.yaml
```

Runs the existing `iface audit` engine, then rolls its flat per-run table
up by (family, term, x) — one row per grid cell aggregating its
train/refit/stability health, plus `train_health_by_family`/
`train_health_by_term` totals — instead of scrolling through 60 individual
run rows (20 cells x 3 stages) to see the shape of where the grid stands.
Family/term/x come from `campaign.systems[i].tags`, not by re-parsing the
generated system id string. The audit also rehashes original and snapshotted
inputs, verifies staged POSCAR/KPOINTS/POTCAR membership, reports key INCAR
settings, checks `ML_AB[N]`, fast `ML_FFN`, and `ML_HEAT` readiness, and
writes persistent `reports/mlff_interfaces/audit.json` and `audit.csv`.
The `refit_ready` and `heat_ready` gates are true only when every grid cell
has the required artifact; do not advance a partial 20-cell grid silently.

## 5. ML_LHEAT production (try it out)

Once a leaf's committee has been trained, audited, and validated through
the existing `stability` recovery operation (`iface vasp ml-recover
stability LEAF`), the new `heat` operation promotes the same validated
`ML_FFN` and adds `ML_LHEAT = .TRUE.`:

```bash
iface vasp ml-recover heat LEAF --temperature 450 --nsw 1000 --ml-outblock 1
```

Requires `ML_FFN` reporting `ML_LFAST = .TRUE.` (same requirement as
`stability`) and archives the run before mutating it, same as every other
recovery operation. `ML_LHEAT` writes the heat flux to `ML_HEAT`, while
`ML_OUTBLOCK=1` makes the first short test easy to verify before committing
to a long production trajectory. After confirming `ML_HEAT` is nonempty,
prepare the intended production length, for example `--nsw 200000`. See
[the VASP wiki](https://vasp.at/wiki/ML_LHEAT). This uses whatever
ensemble/thermostat settings the validated `stability` INCAR already
established (`MDALGO`, `SMASS`, ...) rather than asserting a new one here —
Green-Kubo heat-flux methodology has real, non-obvious ensemble
requirements, and this intentionally does not make that scientific
judgment call on your behalf.

`--temperature`/`--nsw` override the run length/temperature for a longer
production trajectory than the validation run used; omit them to keep the
validated run's own settings unchanged aside from `ML_LHEAT`.
