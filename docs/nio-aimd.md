# NiO AIMD policy

NiO slab and reactive-surface AIMD is a documented exception to InterfaceForge's generic Step1 MD defaults.

## Production rule

For **NiO-containing surface/slab systems**, including hydroxylated NiO, dissociated-water states, and phosphonate-decorated NiO, use the named `nio` Step1 profile unless a specific campaign has been separately validated and the override is documented.

Canonical preparation:

```bash
iface vasp step1-prepare OPT \
    --profile nio \
    --protocol training \
    --temperature 300
```

`--profile nio` expands to the reviewed NiO baseline:

- `POTIM = 0.5 fs`
- `ALGO = Normal`
- `EDIFF = 1E-5`
- `NELM = 120`
- `NELMIN = 6`
- one static magnetic DFT+U preconditioning SCF before MD
- `TEBEG = 100 K` ramping to the requested Step1 target
- keep `NBLOCK = 4` unless there is a separate physical reason to change the thermostat cadence
- Langevin dynamics is **not** enabled by the profile

The profile is an explicit policy choice, not a directory-name heuristic. InterfaceForge does not silently guess that a calculation is NiO from its folder or composition. The selected profile and its resolved settings are written to `step1_manifest.json` and `step1_audit.json` for provenance.

Do **not** use the generic Step1 recipe (`POTIM=1.0 fs`, `ALGO=Fast`, `EDIFF=1E-4`, `NELM=60`) for a new NiO campaign by default.

## Deliberate overrides

The profile supplies a safe baseline but does not lock the calculation. Explicit tuning remains available for a scientifically justified case. For example:

```bash
iface vasp step1-prepare OPT \
    --profile nio \
    --protocol training \
    --algo All \
    --ramp-from 150
```

For a case that remains unstable after the baseline treatment, Langevin can be added explicitly:

```bash
iface vasp step1-prepare OPT \
    --profile nio \
    --protocol training \
    --langevin \
    --langevin-gamma 10
```

The resolved overrides, rather than only the profile name, are recorded in the Step1 provenance files.

## Why NiO is treated conservatively

Production OH50 testing showed that the generic Step1 recipe can produce severe SCF saturation and subsequent force/temperature instability on magnetic DFT+U NiO surfaces, especially for proton-rich and dissociated configurations. The failure mode is electronic as well as ionic: many bad trajectories repeatedly reached the SCF iteration ceiling before the geometry became visibly unstable. A smaller ionic step alone is therefore not the full remedy; the magnetic/DFT+U state should be preconditioned and the MD SCF settings tightened at the same time.

The conservative policy is intentionally broader than OH50. OH25 was easier, but using one robust NiO baseline avoids silently reintroducing the fragile generic settings when a new hydroxylation level, adsorbate, coverage, or surface motif is generated.

## Recovery policy

A Step1 run evolves through *generations* in its own directory. Generation 0 is the run exactly as `step1-prepare` wrote it. Every resume or repair prepares generation N+1, which is then launched like any prepared run:

```text
prepare → launch → [done]
                 ↘ [interrupted/stable] → resume
                 ↘ [unstable] → repair
                                  ↘ fails again → repair generation N+1
```

The same applies to every later generation: an interrupted resume is resumed again, a resumed segment that becomes unstable is repaired, and the accepted prefix accumulates across all of them. No file ever has to be renamed by hand between generations.

Maturity: resume, recover, the generation-aware launch ledger, the Slurm guard and the refined diagnostic are covered by synthetic automated tests only; see [Verification and maturity](verification.md).

### Operator commands

The normal sequence after a batch of Step1 jobs has left the queue:

```bash
iface vasp step1-status Step1                         # read-only report with one recovery category per run
iface vasp step1-recover Step1                        # read-only recovery plan (the default)
iface vasp step1-recover Step1 --execute              # resume + repair + (re)launch the automatic entries
iface vasp step1-recover Step1 --execute --only repair          # restrict to resume, repair and/or launch
iface vasp step1-recover Step1 --execute --no-submit            # prepare resumes/repairs, submit nothing
```

`step1-status` and `step1-recover` classify every run with the same rules (`step1_status.recovery_category`); the first matching rule wins:

1. a Slurm job uses the run (or a directory inside it, e.g. `precondition/`) as its WorkDir → `active`;
2. an archive left `IN_PROGRESS` (interrupted recovery mutation), or an unreadable or conflicting segment record → `review`;
3. no `INCAR` → `review`;
4. nothing has run in the current generation: launchable and not yet submitted → `launch`; already submitted → `review` when Slurm is verified (`submitted (job X) but not queued and no output; inspect slurm log`), otherwise `active`; not launchable → `review`;
5. activity files (`OSZICAR`/`OUTCAR`/`CONTCAR`/`XDATCAR`) modified within the activity window → `active`;
6. hard-unstable → `repair`;
7. complete: Step2-ready with severity `ok` → `done`; Step2-ready with any warning (including a benign startup transient) or with a VASP error marker → `review` (`... confirm before Step2`); thermal tail below threshold → `review`;
8. incomplete: VASP error marker, a non-benign warning, or no completed ionic step in the current segment → `review`; otherwise → `resume`.

Only `resume`, `repair` and `launch` are ever acted on. `step1-recover` additionally attaches the dry resume/repair plan or the launch preflight to each of those entries and moves the entry to `review` whenever that planner disagrees (resume status other than `READY`, no repair needed under the requested thresholds, launch preflight refused), when a repair or resume would need a preconditioning wrapper that the launcher cannot take (it must contain exactly one line that invokes `vasp`), when an explicit `--launcher` would bypass that wrapper, when the run has no launcher to submit, or when a launch ledger that may mention the run is unreadable. `done`, `review` and `active` runs are never touched.

`--execute` plans afresh, then processes the selected entries one run at a time: repairs and resumes first, then launches, each group in relative-path order. For every run it re-checks Slurm and the planning fingerprint, prepares the new generation, and, unless `--no-submit`, submits exactly that run through `step1-launch` before moving on. `--no-submit` also leaves `launch` entries alone; a later plan lists the prepared runs under `launch`. The plan footer prints the execute command with the chosen `--only`/`--no-submit`; repeat any other option (`--scheduler`, `--stale-hours`, repair, resume or diagnostic options) yourself.

Example plan (synthetic test tree, `--scheduler none`, trimmed):

```text
Step1 recovery plan: /path/to/Step1
  scheduler: NOT verified - scheduler check disabled (--scheduler none)
  activity window: 6 h (Slurm not verified (scheduler check disabled (--scheduler none)): files modified within 6 h count as active)
  repair: POTIM 0.5 fs, ALGO Normal, precondition, ramp from 100 K, safety 8 steps
  resume: CONTCAR tolerance 1 A
  diagnostics: energy jump 50 eV, catastrophic 500 eV, startup grace 10 steps, T limit max(1200 K, 4*T_target)
  counts: active 1, done 1, review 2, launch 1, resume 2, repair 2
...
resume (2) -- acted on by --execute
  OH25/stalled         33/400  incomplete (33/400), trajectory healthy
      resume g1 from CONTCAR (+33 accepted -> 33/400), NSW=367 @ 0.5 fs, TEBEG 300 K, TEEND 300 K, electronic start fresh
  OH50/resume_ramp    162/400  incomplete (162/400), trajectory healthy
      resume g2 from CONTCAR (+150 accepted -> 162/400), NSW=238 @ 0.5 fs, TEBEG 100 -> 177.32 K, TEEND 300 K, electronic start fresh

repair (2) -- acted on by --execute
  OH25/runaway            40/400  hard-unstable: sustained post-grace energy departure: ... (first unsafe step 25)
      repair g1: rewind to segment step 16 (cumulative 16/400), NSW=384 @ 0.5 fs, ALGO=Normal, precondition, ramp 100->300 K
  OH50/repaired_again     86/400  hard-unstable: sustained post-grace energy departure: ... (first unsafe step 61)
      repair g2: rewind to segment step 52 (cumulative 68/400), NSW=332 @ 0.5 fs, ALGO=Normal, precondition, ramp 100->300 K

Dry run: nothing changed. Execute resume+repair+launch with: iface vasp step1-recover /path/to/Step1 --execute
```

The per-leaf commands do one step each and print a JSON plan (progress and `CHANGED`/`submitted` lines go to stderr). Each accepts a Step1 root or a single run directory:

```bash
iface vasp step1-resume Step1                              # plan
iface vasp step1-resume Step1 --execute [--submit]         # prepare resume segments (and submit exactly those)
iface vasp step1-repair Step1 --precondition --ramp-from 100            # plan, with recover's NiO settings
iface vasp step1-repair Step1 --precondition --ramp-from 100 --execute [--submit]
iface vasp step1-launch Step1 [--only-repaired] [--only-resumed]        # launch preflight
iface vasp step1-launch Step1 --only-repaired --only-resumed --execute  # submit prepared repairs and resumes
```

`--submit` is refused without `--execute`. The repair defaults differ between the two entry points: `step1-recover` repairs with the conservative NiO rescue (`--potim 0.5 --algo Normal`, preconditioning, `--ramp-from 100`; turn parts off with `--no-precondition` / `--no-ramp`), while `step1-repair` uses `POTIM=0.5 fs` and `ALGO=Normal` but preconditions and ramps only when `--precondition` / `--ramp-from` are given. `step1-repair --execute` also keeps its tree-level rule: it refuses the whole tree while any unstable run is active in Slurm, recently modified or needs review. `step1-recover` decides per run.

### Original Step1, repair and resume

| | Original Step1 (`--profile nio`) | Repair segment | Resume segment |
|---|---|---|---|
| When | `step1-prepare` | hard instability (H1–H6 or corroborated warnings) | healthy, incomplete, inactive run with at least one completed ionic step |
| Restart geometry | OPT `CONTCAR` | an `XDATCAR` frame `--safety-steps` (8) before the first unsafe step; segment start if there is none; never `CONTCAR` | trusted `CONTCAR` copied verbatim; else the latest `XDATCAR` frame; else the unchanged segment-start `POSCAR` |
| Velocities | — | none (VASP draws Maxwell–Boltzmann velocities at `TEBEG`) | kept from `CONTCAR`; none from an `XDATCAR` frame |
| INCAR changes | profile baseline | `POTIM` (0.5), `ALGO` (Normal), `EDIFF=1E-5`, `NELM=120`, `NELMIN=6`, `NSW` = remaining steps, `ISTART` (1 with preconditioning, else 0), `TEBEG`, `TEEND` when needed, `ICHARG` removed; `--langevin` adds `MDALGO=3` + `LANGEVIN_GAMMA` and removes `SMASS` | only `NSW` (exactly the remaining steps), `TEBEG`, `TEEND` when needed, `ISTART` (electronic start mode), `ICHARG` removed; every other tag is kept |
| Temperature | `TEBEG=100 K` ramping to the target | `--ramp-from` (recover default 100 K) ramping to the original target; recover skips its ramp when fewer than 100 steps remain (a 100 → 300 K ramp needs ~98 steps to reach the 250 K thermal-tail threshold) and says so in the action; without it, the rewound segment's schedule temperature at the rewind point | the schedule temperature already reached, so an interrupted ramp continues |
| Electronic start | preconditioning SCF, then MD | preconditioned (`INCAR.precondition` + wrapped launcher, recover default) or fresh | see *electronic-start modes* below |
| Record | `step1_manifest.json` | `step1_repair.json` (schema 2) | `step1_resume.json` (schema 1) |

A root or generic `INCAR` is never copied in by either recovery; both edit the run's own `INCAR`.

**Repair.** The rewind point is found in the *current* segment's own `OSZICAR`/`XDATCAR`. With `first_bad_step` from the diagnostic, the safe segment step is `first_bad_step - 1 - safety_steps`, rounded down to a multiple of `NBLOCK` and limited to the last complete `XDATCAR` frame (`XDATCAR`, else `XDATCAR_FINAL`). Frame `k` is segment ionic step `k·NBLOCK`. When only the SCF statistic is hard (H6, no unsafe row), or the rewind reaches step 0, the segment-start `POSCAR` is rewritten without its velocity and predictor-corrector blocks. After archiving, the repair removes `WAVECAR`, `CHG`, `CHGCAR`, `CONTCAR`, `XDATCAR`, `XDATCAR_FINAL`, `OSZICAR`, `OUTCAR`, `REPORT`, `vasprun.xml`, `vasp_md.dat`, `vasp_md_FINAL.dat`, `.vasp_md.dat` and `MD_TempPlot.png`, updates `INCAR`, and, when preconditioning, writes `INCAR.precondition` from the final MD `INCAR` and wraps the launcher. When repairing NiO runs with `step1-repair`, pass `--precondition`; `step1-recover` does it by default.

**Resume.** The completed ionic steps of the current segment, `n`, are the `OSZICAR` MD rows numbered 1, 2, 3, … without a gap; a torn final line (the job was killed while writing it) is dropped. `CONTCAR` is *trusted* when it is nonempty, has the same lattice (relative tolerance 1e-6) and species/count lines as the segment-start `POSCAR`, finite coordinates, a finite and complete velocity block if one is present, and agrees with the latest expected position: the reference is `XDATCAR` frame `k = min(frames, n // NBLOCK)` (the segment-start `POSCAR` when `k = 0`), the lag `n - k·NBLOCK` must not exceed `NBLOCK`, and the largest minimum-image displacement must be within `--contcar-tolerance` (default 1.0 Å) + 0.05 Å/fs × (`n + 1 - k·NBLOCK`) × `POTIM`. A trusted `CONTCAR` accepts all `n` steps; the `XDATCAR` fallback accepts `k·NBLOCK`. `NSW` becomes exactly `original_nsw - cumulative accepted`.

Electronic-start modes (`electronic_start.mode`, first match wins):

- `precondition` — the launcher is already wrapped and `INCAR.precondition` exists (the normal case for `--profile nio` runs): `WAVECAR` is removed and `ISTART=1`, so the wrapper reconverges the magnetic DFT+U state at the resumed geometry;
- `precondition` — `--precondition` (`step1-recover --resume-precondition`): the launcher is wrapped as for a repair, `ISTART=1`, `WAVECAR` removed;
- `fresh` — `--fresh-start`: `ISTART=0`, `WAVECAR` removed (`--precondition` and `--fresh-start` together are refused);
- `wavecar` — `ISTART >= 1` and a nonempty `WAVECAR`: it is kept (a hard-linked `WAVECAR` is first replaced by a private copy) and `ISTART` is unchanged;
- `fresh` — otherwise: `ISTART=0`, an empty `WAVECAR` is removed.

Temperature continuation, worked example. A repaired segment with 12 accepted steps before it runs `NSW=388` at `POTIM=0.5 fs` with `TEBEG=100`, `TEEND=300`, and is interrupted after 150 completed steps with a trusted `CONTCAR`. VASP ramps the thermostat target linearly in ionic step over `NSW`, so the resumed segment gets

- `TEBEG = 100 + (300 - 100) × 150 / 388 = 177.32` K (≈ 177.3 K; written rounded to 0.01 K);
- `TEEND = 300` (kept; written explicitly whenever the segment ramps or `TEEND` was explicit);
- `NSW = 400 - (12 + 150) = 238`;

and the ramp rate is unchanged (≈ 0.515 K/step) because the 238 remaining steps equal the 388 − 150 left in the old segment. Had `CONTCAR` not been trusted, the resume would restart from `XDATCAR` frame 37 (segment step 148): `TEBEG` 176.29 K, `NSW` 240. When the remaining steps and the steps left in the old segment differ (e.g. a hand-edited `NSW`), the endpoint is kept and `temperature_continuation.note` records the changed rate.

### Repeated repair and cumulative accounting

The rewind frame is always an index into the current segment's own `XDATCAR`; the progress figures are cumulative over all generations. Real-shaped example (synthetic test fixture, `NBLOCK=4`, target 400 steps):

1. Generation 0 runs at 1.0 fs and becomes hard-unstable from ionic step 25. The repair keeps `25 - 1 - 8 = 16` steps (`XDATCAR` frame 4): generation 1, accepted 16, `NSW = 400 - 16 = 384` at 0.5 fs.
2. The generation-1 segment fails again from its segment step 61. The next repair keeps `61 - 1 - 8 = 52` steps of that segment (frame 13 of the generation-1 `XDATCAR`): generation 2, accepted `16 + 52 = 68`, `NSW = 400 - 68 = 332`.

The generation-2 `step1_repair.json` records both views: `safe_segment_steps` 52 and `rewind_frame` 13 (current segment), `previous_safe_prefix_steps` 16, `safe_prefix_steps` = `accepted_prefix_steps` 68 (cumulative), `repair_nsw` = `segment_nsw` 332, and the accepted-segment ledger `accepted_segments` = [16 steps @ 1.0 fs (generation 0), 52 steps @ 0.5 fs (generation 1)], i.e. `accepted_ps` 0.016 + 0.026 = 0.042 ps. Without `--ramp-from` (`step1-recover --no-ramp`) the new `TEBEG` would be the generation-1 schedule temperature at its step 52, 100 + 200 × 52/384 = 127.08 K; recover's default restarts the ramp at 100 K.

`step1-status` shows the same accounting on its `lineage:` line, e.g. before the second repair:

```text
lineage: repair g1 (legacy-repair-g1-20260901T000000Z; legacy record; record PREPARED) · target 400 · accepted 16 + segment 70 = 86 · segment NSW 384 @ POTIM 0.5 fs · accepted 0.051 ps · T 100→300 K ramp (now 136 K) · current generation submitted (job 111)
```

### Thermal tail and Step2 readiness

A 100 → 300 K heating ramp naturally has a whole-run mean temperature near 200 K, so the whole-run mean (`Tmean`) says little about thermalization. The useful thermal diagnostic is the late-window mean, `Ttail{n}`: the mean temperature over the last `n` ≤ 50 MD steps of the current segment, compared with 5/6 of the target temperature (`TEEND`, else `TEBEG`), i.e. 250 K for a 300 K target. A completed synthetic 400-step ramp prints `Tmean=200+/-58 K  Ttail50=288 K thermal-ok; ready for Step2`.

Four flags are reported separately in `step1-status --json`:

- `thermal_tail_ok` — the tail reached 5/6 of the target (independent of stability; `null` when it cannot be judged);
- `trajectory_stable` — no hard instability;
- `complete` — the whole-run target is known, the cumulative accepted steps reach it, and the run is not active now;
- `ready_for_step2` = `complete` ∧ `trajectory_stable` ∧ `thermal_tail_ok`.

(`thermal_ready` keeps its legacy meaning, tail ok ∧ stable.) The human output prints the thermal result and then the readiness, so a warm tail is never reported as a failure: a stalled 33/400 run at 300 K prints `Ttail33=300 K thermal-ok; incomplete`, an unstable run with a warm tail prints `... thermal-ok; UNSTABLE`, and a resumed segment early in its ramp prints e.g. `Ttail50=165 K (<250 K); incomplete`, which is expected at that point of the ramp. `(<250 K)` appears only for a tail that did not pass.

### Warning vs hard instability

`diagnose_step1_run` judges the `OSZICAR` MD rows of the current segment. Definitions: `T_target = max(TEBEG, TEEND)` (`TEEND` defaults to `TEBEG`, `TEBEG` to 300 K); the grace window is the first `--startup-grace-steps` rows; `F_ref` is the median free energy of the 10 rows after the grace window (with fallbacks when there are too few rows or the energy rises inside that window); a row *departs* when `|F - F_ref|` exceeds `--energy-jump`.

| Signal | Rule | Default threshold | Flag |
|---|---|---|---|
| H1 hard | non-numeric temperature (`T= ******`) at any step | — | — |
| H2 hard | non-numeric free energy at any step | — | — |
| H3 hard | temperature above `T_limit` at any step | `T_limit = max(1200 K, 4·T_target)` | `--max-temperature` |
| H4 hard | `\|F - F_ref\|` above the catastrophic limit at any step, grace window included | 500 eV | `--catastrophic-energy` |
| H5 hard | sustained post-grace departure: ≥ 2 consecutive departing rows, or the last recorded row departs | 50 eV band after a 10-step grace window | `--energy-jump`, `--startup-grace-steps` |
| H6 hard | `NELM` reached on ≥ 50 % of post-grace rows | 50 % (fixed) | — |
| W1 `startup_energy_excursion` | departure only inside the grace window; also a downhill relaxation that departs contiguously from step 1 (every row above `F_ref`) and settles into the band inside the reference window, i.e. finishes a few steps after the grace window | first 10 rows (up to 20 for the late-settling case), 50 eV | `--startup-grace-steps`, `--energy-jump` |
| W2 `isolated_energy_spike` | one post-grace departing row whose previous and next rows are in band | 50 eV | `--energy-jump` |
| W3 `scf_elevated` | 20 % ≤ post-grace `NELM`-ceiling fraction < 50 % | 20 % (fixed) | — |
| W4 `temperature_elevated` | `T_warn < T ≤ T_limit` | `T_warn = max(2·T_target, T_target + 300 K)`, at most `T_limit` | `--max-temperature` (via `T_limit`) |
| Corroboration → hard | W2 with any of W1/W3/W4, or W4 with W1 or W3 (`corroborated anomalies: A + B`); W1 + W3 alone stays a warning (the magnetic DFT+U fresh-start pattern). A settled, downhill W1 never corroborates: that startup relaxation plus one unrelated later W2/W4 stays a warning (`review`), so it cannot anchor a rewind to segment step 0 | — | — |
| Benign startup transient | W1 alone, *settled* (an in-band row follows the excursion) and *downhill* (every excursion row lies above `F_ref`) | — | — |
| Torn final line | the last `OSZICAR` MD line has no newline and stops before `E0=`: dropped, `torn_final_line: true`, not read as a non-numeric energy | — | — |

Consequences: any hard signal → `repair`. A benign startup transient does not block `resume` of an incomplete run (the reason says so); on a complete run it sends the run to `review` ("complete and Step2-ready by hard criteria; … — confirm before Step2"), never to automatic repair. This is the case of a completed 400/400, ~300 K, clean-tailed run whose ionic step 1 lies ~77 eV above the relaxed level. Any other warning, including an unsettled or uphill W1, → `review`; `step1-resume --accept-warnings` resumes such a run after inspection (`step1-recover` never accepts warnings). With `--startup-grace-steps 0` a lone step-1 departure is W2 rather than W1.

The four flags exist on `step1-repair`, `step1-resume` and `step1-recover`; `step1-status` always uses the defaults. In `step1-recover` they apply to the planners and to the re-check of `done` entries, so non-default thresholds can move a run to `review` but cannot add repairs or resumes beyond what the default classification allows.

### Generation-aware launch provenance

Generation identifiers:

- `g0-prepare` — generation 0 (no record);
- `g<N>-repair-<YYYYmmddTHHMMSSZ>` / `g<N>-resume-<stamp>` — written by `step1-repair` / `step1-resume`; the stamp is that of the archive created for the mutation;
- `legacy-repair-g<N>-<stamp>` — a schema-1 `step1_repair.json` (see below).

At most one segment record is *current*, at the run top level: `step1_repair.json` for a repair generation or `step1_resume.json` for a resume generation. Creating a new generation archives the previous record and removes it from the top level. If both files exist (hand edits, mixed InterfaceForge versions), the larger `generation`, then the newer `prepared_at`, then the newer mtime wins; when the two contradict each other the run is reported `CONFLICT`, an unparsable record `UNREADABLE`, and either is routed to `review` and refused by launch.

`step1_launch.json` (schema 2) lives in the directory `step1-launch` was invoked on: the Step1 root, or a leaf when a leaf was launched as its own root (`step1-recover` launches from its root). It keeps the whole history:

```json
{"format": "interfaceforge-step1-launch", "schema_version": 2, "root": "...",
 "status": "SUBMITTED", "preflight": "PASS", "latest_batch_id": "b-20260925T151617Z",
 "batches": [{"batch_id": "...", "started_at": "...", "finished_at": "...", "planned": 2,
              "status": "SUBMITTED", "submitted": 2, "failed": 0}],
 "runs": ["every row ever written, oldest first"]}
```

Each row carries `status`, `job_id`, `kind` (`prepared` / `repair-prepared` / `resume-prepared`), `root`, `relative_path`, `directory`, `launcher`, `notes`, `detail`, `generation`, `generation_id`, `submitted_at` and `batch_id` (imported schema-1 rows add `legacy: true` and `legacy_recorded_at`; sealed rows add `superseded_by_generation_id` and `superseded_at`). `step1_launch.tsv` mirrors the full history; its first seven columns are the pre-lineage ones in their old positions (`status`, `job_id`, `kind`, `relative_path`, `directory`, `launcher`, `detail`), so `cut -f2` still gives the job id, followed by `batch_id`, `submitted_at`, `generation` and `generation_id`. A batch id is `b-<UTC stamp>`, suffixed `-2`, `-3`, … when several launches share one second. Each job is recorded the moment `sbatch` returns: the ledger row is written and the submission is appended to the current record's `submissions`, whose `status` becomes `SUBMITTED` (generation 0 has no record). A failure is recorded as a `FAILED` row and stops the batch.

The duplicate guard reads every ledger that may mention the run: the run's own, each root passed to launch, and every ancestor up to the nearest `step1_manifest.json` (at most 8 levels). A `SUBMITTED` row blocks only the generation it belongs to: it is *current* when its `generation_id` equals the current one, or, for a legacy row without `generation_id`, when the current generation is itself legacy and is generation 0 or was prepared no later than the row was recorded (a legacy row whose time cannot be established counts as current); a current record with status `SUBMITTED` also counts. Rows of older generations are *historical*: they never block the current generation, and they are listed as `N older submission(s)` on the status `lineage:` line and in the launch plan notes. This is why a leaf launched as its own root (a historical row with `relative_path "."`) no longer blocks the launch of its next repair. A launch ledger that exists but cannot be parsed fails closed: the current generation counts as submitted, launch refuses and recover routes the run to `review` until the file is repaired or moved aside; an executing launch that must append to such a file first copies it to `step1_launch.json.unreadable-<stamp>`.

*Sealing.* When `--execute` creates a new generation, every ledger found for the run gets `superseded_by_generation_id` / `superseded_at` added to the run's rows of the retired generation (and its legacy rows). This is informational provenance only; matching never depends on it. Ancestor ledgers are annotated in place (atomic rewrite, keys only added, schema version unchanged) without an archived copy; a ledger inside the run directory is archived with the run first.

Generation 0 is launchable only from the root whose `step1_manifest.json` lists it with matching `INCAR`/`POSCAR` hashes; when invoked on another directory, `step1-status`, `step1-launch` and `step1-recover` all name that root (`generation 0 is listed in <root>/step1_manifest.json; launch it from <root>`).

Historical launches are inspected in `<root>/step1_launch.json` (`runs`, `batches`) and `step1_launch.tsv`, in leaf-level ledgers, in the `submissions` list of the current record and of archived records (`<run>/.interfaceforge/archive/*/step1_repair.json` / `step1_resume.json`), and in `step1-status --json` under `lineage.submission`, `lineage.historical_submissions` and `lineage.submission_rule`.

### Safety invariants

- **Dry run is the default** for `step1-resume`, `step1-repair`, `step1-launch` and `step1-recover` and writes nothing: no `.interfaceforge/`, no journal, no ledger upgrade, no temporary files. It reads files and asks `squeue`.
- **Active Slurm WorkDirs are never mutated or submitted.** `--scheduler` is `auto` (use `squeue` when it is on `PATH`), `slurm` (refuse when `squeue` cannot be asked) or `none` (file age only). The query is `squeue -h -a -u $USER -o %i|%T|%Z`; a job counts for a run when its WorkDir is the run or lies inside it, not when it is an ancestor, and the recovery's own Slurm job is ignored. Immediately before each run is changed, and before each `sbatch`, the scheduler is queried again when the snapshot is older than 15 s, and the run's fingerprint (size and mtime of `INCAR`, `POSCAR`, `CONTCAR`, `OSZICAR`, `OUTCAR`, `XDATCAR`, `step1_repair.json`, `step1_resume.json`) must equal the one taken at planning, else the run is refused. `step1-status` never fails because of `squeue`; it reports the failure and continues unverified.
- **Scheduler state beats file age.** A listed job is active however old its files are. When `squeue` answered (verified) and does not list the run, only a 0.1 h (6 min) settle window for files written as a job leaves the queue applies; when Slurm is not verified, files modified within 6 h count as active. An explicit `--stale-hours` overrides both. Status, resume, repair and recover all apply the window to the newest of `OSZICAR`/`OUTCAR`/`CONTCAR`/`XDATCAR`. With `--scheduler none` the file-age window is the only protection against touching a running job.
- **Archive before mutation.** Each resume or repair first copies the run state to `<run>/.interfaceforge/archive/step1_<repair|resume>_g<N>_<stamp>/` (records, ledgers, launchers, trajectories, `precondition/` outputs) with an `ARCHIVE_MANIFEST.json` whose `status` is `IN_PROGRESS`; it becomes `COMPLETE`, with the new `generation_id`, only after the mutation finished. `WAVECAR`, `CHG` and `CHGCAR` are regenerable and are not copied (listed under `not_archived`). A manifest left `IN_PROGRESS` marks an interrupted mutation: status and recover report `review`, launch, resume and repair refuse the run. After the copy (which can take minutes for large `OUTCAR`/`XDATCAR`) the scheduler and the fingerprint are checked once more before the first write; if a job started or a file changed meanwhile, the archive is marked `ABANDONED` and the run is left untouched.
- **One recovery action per run at a time.** Resume, repair and each `sbatch` of launch/recover/`--submit` hold an exclusive lock file `<run>/.interfaceforge/step1.lock` (created with `O_EXCL`, naming the operation, PID, host and time) from the last check to the final record, so two concurrent InterfaceForge processes can never both mutate or submit the same run: the second is refused. The lock is taken only after the Slurm check (nothing is written into an active WorkDir) and is removed afterwards; one left by a killed process is never broken automatically and must be deleted by hand once no InterfaceForge process is running.
- **No write-through.** `POSCAR`, the wrapped launcher and `INCAR.precondition` are replaced via a temporary file and a rename, like `INCAR` and the records, so a file hard- or sym-linked from outside the run (a shared template `POSCAR`, a shared `runvasp.sh`) is never modified.
- **Stop at the first failure.** `step1-recover --execute` rewrites `<root>/step1_recover.json` atomically after every step. The journal (`format` `interfaceforge-step1-recover`) holds one entry per execution in `executions`, with `status` `RUNNING` → `COMPLETED` or `FAILED` and one row per selected run; each row carries, among others, `relative_path`, `directory`, `category`, `action`, `outcome` (`pending` / `running` / `prepared` / `submitted` / `failed` / `not attempted`), `prepared`, `archive`, `parent_generation_id`, `generation_id`, `submitted`, `job_id`, `batch_id` and `error`. On failure the error names the runs changed, submitted and not attempted. A dry run, or an execution with nothing selected, writes no journal.
- `review`, `done` and `active` runs are never touched by any recovery command.

### Backward compatibility

- **Schema-1 `step1_repair.json`** (written before generations, with or without `previous_safe_prefix_steps`) is read in place. The lineage is reconstructed by following each record's `archive` directory, which holds the previous `step1_repair.json` and the `INCAR` of the rewound segment; the chain gives an exact ledger. A record whose archive holds no earlier `step1_repair.json` is generation 1. A record written before the cumulative-prefix fix (no `previous_safe_prefix_steps` / `safe_segment_steps`) that does have a predecessor is a repair of a repair whose `safe_prefix_steps` counted only its own segment and whose `original_nsw` was the rewound segment's NSW; the cumulative prefix is rebuilt from the chain and the target NSW is accepted + remaining steps. A schema-1 launch row of kind `prepared` was written only for generation 0 and never counts as the submission of a legacy repair, whatever the ledger's mtime. When the chain is broken the earlier prefix is kept as one `unknown` ledger row and the ledger is marked inexact (`ledger inexact` on the status line). Status marks such a generation `legacy record`. When the new `step1-launch` submits it, the schema-1 record is marked `SUBMITTED` in place without gaining a `generation_id`; its derived identity is pinned as `legacy_generation_id` / `legacy_prepared_at`.
- **Schema-1 `step1_launch.json`** is read in place. Its rows are legacy rows with `legacy_recorded_at` = the file's mtime and are matched by time: a legacy `SUBMITTED` row is current for generation 0 or for a legacy repair prepared no later than the row was recorded, and never for a generation written by this version. The file is upgraded to schema 2 only when an executing launch appends to it (`step1-launch --execute`, or the submission step of `step1-recover --execute` and `--execute --submit`): its rows are imported verbatim as legacy rows and a `legacy_import` note is added. Sealing during a repair/resume `--execute` only adds keys and pins `legacy_recorded_at`; the schema version stays 1.
- **Newly written repair records** are schema 2 and keep every schema-1 key with its original meaning (`safe_prefix_steps` = cumulative accepted steps, `safe_segment_steps`, `previous_safe_prefix_steps`, `rewind_frame`, `repair_nsw`, `original_potim_fs`, `repair_potim_fs`, `repair_algo`, `repair_electronic`, `repair_langevin_gamma`, `repair_ramp_from_k`, `repair_precondition`, `source`, `diagnostic`, `age_hours`). Resume records are a separate format (`interfaceforge-step1-resume`, schema 1).
- Existing CLI flags and Python keyword arguments keep their meaning. The default `--stale-hours` of `step1-status` and `step1-repair` (and `prepare_step1_repair(stale_hours=...)`) is now resolved from the scheduler (0.1 h verified, 6 h unverified) instead of a fixed 6 h; pass `--stale-hours 6` for the previous window on a Slurm machine. The Python default of `step1_status(stale_hours=...)` stays 6 h; pass `None` for the resolved window.
- Downgrade caveats. An older `step1-launch` reads only the invoked root's ledger and refuses every run with a `SUBMITTED` row for its relative path; because `runs` stays cumulative it still sees every historical submission, so it cannot double-submit, but it again refuses to relaunch repaired runs. An older `step1-launch --execute` rewrites `step1_launch.json` with only its own batch as schema 1, discarding the cumulative history; copy the ledger first. Older versions do not know `step1_resume.json`: a resumed run looks like an unrepaired generation-0 run with a changed `INCAR` (an older `step1-launch` stops with `changed since step1-prepare` when the run is in the manifest, and older status output misreports its progress). If an older `step1-repair` writes a schema-1 `step1_repair.json` next to a current `step1_resume.json`, this version reports `CONFLICT` and routes the run to `review`.

### Conservative rescue wrapper

`launch_scripts/rescue_step1_conservative.sh` delegates to `step1-recover --only repair`:

```bash
bash launch_scripts/rescue_step1_conservative.sh Step1              # plan
bash launch_scripts/rescue_step1_conservative.sh Step1 --execute    # repair and submit exactly those runs
```

It runs `iface vasp step1-recover Step1 --only repair --scheduler slurm --potim 0.5 --algo Normal --ramp-from 100` (plus `--stale-hours`, `--langevin --langevin-gamma` and `--execute` when requested) and requires `iface` and `squeue` on `PATH`. Environment overrides and the single-partition `sbatch` launcher are described in the [launcher index](../launch_scripts/README.md#step1-rescue-conservative-repair).

Langevin dynamics is **not** part of the default NiO baseline. Use it only as a second-stage stabilization measure when a case remains physically unstable after conservative SCF settings, preconditioning, the 100 K ramp, and `POTIM=0.5 fs` (`step1-recover --langevin --langevin-gamma 10`, `step1-repair --langevin`).

## Interpretation of status warnings

A trajectory should not be discarded solely because the fixed free-energy-reference heuristic reports an early `|F-Fref|` excursion. The diagnostic now encodes this: a settled, downhill excursion inside the grace window is a warning, not an instability. It is still a heuristic. For every `review` entry, check the temperature history, SCF-ceiling fraction, trajectory continuity, and the actual geometry before resuming it or using it in Step2. A stable ~300 K trajectory with modest SCF-ceiling use can be a false positive.

Conversely, repeated SCF-ceiling saturation, multi-thousand-K temperature excursions, non-finite energies/forces, or obvious geometric runaway are grounds for stopping and repairing the run.
