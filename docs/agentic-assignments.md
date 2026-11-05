# Agentic assignments

`iface assignment` turns a due-diligence handoff into a pre-registered,
machine-checkable assignment. The rule behind it:

- work a script can check becomes a tested `iface audit …` command;
- DFT/MLIP diagnostics follow prepare → dry run → `--execute` → compare, and an
  assignment only ever asks for the first two;
- only judgement stays in prose;
- an agent's report is checked by code, not trusted.

```bash
iface assignment init   DD-2026-11-05 --set ID=DD-2026-11-05 --set TARGET_REPO=... --set TARGET_COMMIT=...
# edit DD-2026-11-05/assignment.yaml
iface assignment render DD-2026-11-05/assignment.yaml          # -> ASSIGNMENT.md (deterministic)
iface assignment freeze DD-2026-11-05                          # -> frozen.json, NOTES.md, ledger event
iface assignment git-snapshot <target repo>                    # agent: before and after the work
iface assignment validate DD-2026-11-05 --report report.json   # -> {"valid": ..., "failures": [...]}
```

Schemas: [`schemas/assignment.schema.json`](../schemas/assignment.schema.json)
(the spec) and
[`schemas/assignment_report.schema.json`](../schemas/assignment_report.schema.json)
(what the agent returns). Template:
`src/interfaceforge/assignments/templates/due_diligence_v1/assignment.yaml`.
Agents start from [`AGENTS.md`](../AGENTS.md).

## The three task kinds

| Kind | Who decides the verdict | Required fields | Never |
|---|---|---|---|
| `audit` | The command's output. The agent may only convert inputs (read-only, nulls where the records are silent) and copy the verdict. | `command` | `cost` |
| `diag_prepare` | The pre-registered `criterion`, applied later by the PI to the executed result. The agent writes the proposal, prepares inputs and dry-runs. | `command`, `criterion {metric, unit, threshold, rule}`, `cost` (`from_records` or `not_inferable`) | `--execute`, job submission, invented hours |
| `judgement` | The agent, with every claim labelled `[E]` established from a file, `[P]` recorded/probable, `[H]` hypothesis, `[C]` concern, `[S]` suggestion. | — | `command`, `cost` |

Every task carries its own verdict vocabulary (`verdicts`). The report must give
exactly one of them per task. `hypothesis` marks the `[H]` claim a task tests;
the renderer collects them in a "Hypotheses under test" table.

## Pre-registration by hash

`render` is a pure function of the YAML: the same file gives byte-identical
Markdown on any machine (LF line endings, no timestamps), so the rendered text
can be hashed. `freeze`:

1. refuses if `frozen.json` already exists — an assignment is frozen once;
2. refuses if `ASSIGNMENT.md` is not exactly the render of `assignment.yaml`;
3. writes `frozen.json` with the sha256 of both files, the UTC time and the
   InterfaceForge version;
4. creates `NOTES.md` (header only) if it is missing;
5. appends a `frozen` event to the directory's ledger
   (`.interfaceforge/state.json`, the same `StateStore` used by campaigns).

After freezing, `render` refuses to rewrite `ASSIGNMENT.md` and `init` refuses
to rewrite `assignment.yaml`. The report quotes `frozen_sha256` (the
`assignment_yaml_sha256` in `frozen.json`); `validate` also re-hashes both files
so an edit after freezing is caught even if the report quotes the old hash.

## Dated amendments

Criteria are never edited after freezing. If the PI changes scope, the change
goes into `NOTES.md` as a dated entry (`## YYYY-MM-DD — <amendment>`). An
amendment can add work or explain a deviation; it cannot move a threshold that
was registered before the result existed. A changed threshold needs a new
assignment id.

## Code-checked reports

`iface assignment validate <dir> --report report.json` prints
`{"valid": bool, "failures": [...], "tasks": {...}}` and exits 0 whether the
report is valid or not; it exits 2 only when an input cannot be read. It checks:

- the report conforms to `assignment_report.schema.json`;
- `frozen_sha256` equals `frozen.json`, and neither `assignment.yaml` nor
  `ASSIGNMENT.md` has changed since freezing;
- every task in the spec has exactly one verdict, from that task's vocabulary,
  and no report task is unknown;
- every finding value has a non-empty `source`;
- `git_state.before.head == after.head` and the porcelain hashes are equal; a
  changed `branch_header` (e.g. `[behind 1]` after another process fetched) is
  allowed only when a `deviations` entry mentions `branch_header`;
- `target.commit` is an ancestor of (or equal to) the HEAD the agent observed
  (`git_state.before.head`), checked in `target.repo` with read-only
  `rev-parse` and `log`. The output records `target_commit`, `observed_head` and
  `target_ancestry` (`ok`, `not_ancestor`, `unknown_commit`, or `unchecked` when
  the repository is not readable on the validating machine; `unchecked` is a
  warning, the other two are failures). An agent that worked on a HEAD later
  than the pinned commit is therefore visible, and one that worked on unrelated
  history fails;
- every `files_written` path matches a `permissions.write` glob, is outside every
  `permissions.forbid` location, exists, and has the stated sha256;
- `not_assessed` and `deviations` are present (they may be empty; write
  `["none"]` for no deviations).

`iface assignment git-snapshot <repo>` gives the `git_state` values in the
expected form, using only `git --no-optional-locks rev-parse HEAD` and
`status --porcelain=v1 -b`. `porcelain_sha256` hashes the porcelain lines
without the `## branch` header, each terminated by `\n`.

Glob matching uses `fnmatch`: `*` also matches path separators, `\` and `/` are
equivalent, and Windows-style paths compare case-insensitively. A `forbid`
entry without glob characters is a directory prefix.

A partial report is normal while work is in progress: `validate` then lists the
tasks without a verdict under `failures` and marks them `missing` in `tasks`.

## Default thresholds and why

The assignment format itself has no numeric defaults: a `diag_prepare`
criterion is always written per assignment, before any result exists. The
defaults below belong to the audits an assignment calls.

| Audit | Threshold | Default | Why this value |
|---|---|---|---|
| `excess-temperature` C1 | `--tol-ratio` | 0.10 | Real solids at a few hundred K store a few per cent more potential energy than the classical (3/2) k_B T per atom (anharmonicity), so ratios of 1.01–1.05 are physical. The errors C1 exists to catch are much larger: a reference at the wrong temperature (450 K vs 300 K gives 1.5), a static energy labelled as a mean (0), a per-atom/per-f.u. mix-up (1/n or n). |
| `excess-temperature` C2 | `--tol-t` | 1 K | Thermostat targets are set exactly. Any difference of a kelvin or more is a different MD protocol, not noise. For ~600 interface atoms over ~800 Å² a 1 K offset is ~0.002 J/m², well under `--tol-gamma`. |
| `excess-temperature` C2 | `--tol-gamma` | 0.01 J/m² | Model-versus-DFT agreement on balanced interface excesses is reported at the 0.01 J/m² level. A temperature bias of that size is as large as the agreement being claimed, so it must be removed or reported. |

See [Excess-temperature audit](excess-temperature.md) for the audit itself.

## Example

The first real use is the Si₃N₄–TiN follow-up `DD-2026-11-05`. Its staging
folder (outside every repository) holds the filled-in `assignment.yaml`, the
rendered and frozen files, a read-only converter for the excess-temperature
input, the audit outputs, a partial `report.json` and the `validate` output.
