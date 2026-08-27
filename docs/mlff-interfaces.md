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
`ambiguous` (candidates listed) — review and hand-fix `structure_path`
before the next step, which refuses to proceed on anything not `matched`.
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
(`kind: interface`, `tags: {family, term, x}`) and a starter
`inputs/INCAR` containing `ENCUT = 520` / `IVDW = 11` (add convergence
settings — `EDIFF`, `ISMEAR`, ... — yourself; this deliberately does not
invent electronic-structure settings it wasn't given). **No shared
reference POTCAR is set** — each system's own POSCAR (from its own
structure) carries its own species, so `iface vasp submit` generates the
correct POTCAR per leaf automatically. This matters here specifically:
oxygen-free and oxygen-containing interfaces in a grid like this cannot
share one POTCAR.

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
syntax — not a custom polling loop) that runs each already-prepared leaf's
`run.slurm` as a plain shell script (its `#SBATCH` lines are inert executed
this way) at up to 4 concurrent array tasks. Requires an
`array_profile_name` job (default `vasp_train_array`) in the scheduler
profile, sized for one leaf — the array multiplies it by concurrency, not
by task count. This only writes the launcher; `sbatch` it yourself.

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
generated system id string, which is deliberately robust to how that id
happens to be slugified.

## 5. ML_LHEAT production (try it out)

Once a leaf's committee has been trained, audited, and validated through
the existing `stability` recovery operation (`iface vasp ml-recover
stability LEAF`), the new `heat` operation promotes the same validated
`ML_FFN` and adds `ML_LHEAT = .TRUE.`:

```bash
iface vasp ml-recover heat LEAF --temperature 450 --nsw 200000
```

Requires `ML_FFN` reporting `ML_LFAST = .TRUE.` (same requirement as
`stability`) and archives the run before mutating it, same as every other
recovery operation. `ML_LHEAT` writes the heat flux to `ML_HEAT` for
Green-Kubo postprocessing — see
[the VASP wiki](https://vasp.at/wiki/ML_LHEAT). This uses whatever
ensemble/thermostat settings the validated `stability` INCAR already
established (`MDALGO`, `SMASS`, ...) rather than asserting a new one here —
Green-Kubo heat-flux methodology has real, non-obvious ensemble
requirements, and this intentionally does not make that scientific
judgment call on your behalf.

`--temperature`/`--nsw` override the run length/temperature for a longer
production trajectory than the validation run used; omit them to keep the
validated run's own settings unchanged aside from `ML_LHEAT`.
