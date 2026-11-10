# Data audit: order of operations

Written guidance for anyone, human or agent, asked to audit a dataset, a model
or a result from its records. It needs no tooling.

**Status: procedure, not a validated standard.** It comes from one audit of one
campaign (the SiN–TiN Dispatch context-graph pilot on the A2.1 chain,
2026-11-10, `lgutsev/SiN-TiN_Dispatch#4`), where it caught real record
problems. That campaign has unusually well-structured records. It changes no
acceptance criterion, gate or record. The mapping to InterfaceForge records in
step 2 was made from the guides and has not yet been exercised in an audit.

## What a data audit is, and is not

A data audit answers four questions from the records:

1. Where did it come from (which calculations, which extraction, which inputs)?
2. Is it what the records say it is (do the hashes, counts and memberships agree)?
3. What depends on it (which models, simulations and conclusions)?
4. What failed, what was repaired, and what is still open?

It does not decide whether the science is right. Five states stay separate
throughout: a job finished, a calculation converged, a parser passed, a
validation passed, the PI accepted. An audit reports which of these the records
support and never promotes one into another. This is the same separation as the
status definitions in [Verification and maturity](verification.md).

## Order of operations

Do the steps in this order. Each one narrows what the next has to trust.

### 0. Boundary first

- Read-only. No runs, no submissions, no edits to registers, ledgers, datasets
  or acceptance states.
- Record the starting commit and worktree status
  (`iface assignment git-snapshot <repo>` gives both in a fixed form); work on
  an isolated branch.
- Record problems you find. The owner fixes records, not the auditor.

### 1. Freeze and inventory

- Fix the snapshot: repository commit plus a list of every file you will rely
  on, each with its sha256.
- List precisely what you cannot reach (cluster state, policy text, files on
  another machine). These become "unknown" in the report, never guesses.
- Hash large or licensed outputs; do not read them. Never open POTCAR, weights
  or credentials.

### 2. Authoritative records before prose

For each kind of fact, find the record that is authoritative and
machine-readable, and use it first. The right-hand column lists where
InterfaceForge workflows write such records; confirm which ones exist for the
campaign in hand before relying on the list.

| Fact | Look for | In InterfaceForge |
|---|---|---|
| What was intended | package manifest, input files and their checksum list | `campaign.yaml`; the mapped-leaf map (e.g. `examples/mapped-leaf-campaign/periodic_nitride.yaml`); the source-rescue policy YAML; `roles.json` admission registry |
| What was executed | submission log (job, tasks, script hash, commit) | `.interfaceforge/state.json` events; scheduler logs kept in the run directory. Dispatch-side submission logs live outside this repository |
| What came out | per-task output markers, files a job wrote itself, logs | OUTCAR/OSZICAR and their fingerprints in `source_audit.json`; per-run SCF JSON; per-frame label JSONL |
| What is in a dataset | per-frame membership ledger, release report, pinned hashes | `manifest.json` / `leaf_manifest.json`, `frames.csv`, the release frame ledger and frozen split map ([Source rescue](source-rescue.md) §4), `checksums.sha256` |
| What a model is | model register: frozen and checkpoint hashes, parents, exclusions | committee `manifest.json` + `checksums.sha256` ([MACE committees](mace-committee.md), [Packaging](packaging.md)); `ensemble_manifest.json` |
| What was decided | event ledger: approvals and acceptances with their text | source acceptances in the policy YAML (source_id, fingerprint, reviewer, evidence); `frozen.json` and `NOTES.md` of an assignment; `.interfaceforge/state.json` |
| What was measured | audit JSON holding both the value and its threshold | `iface audit …` and `iface validate …` JSON outputs |

Reviews, status pages and notes come last, and anything taken only from them is
labelled as such (step 10).

### 3. Identity by content, not by name

- Identify things by existing ids plus content hashes. A path or a Slurm job
  number is a pointer, not an identity.
- Keep three things apart: the calculation as specified, each execution
  attempt, and each output artifact. A restarted task is one calculation with
  two attempts.

### 4. Verify pins before trusting names

- Hash every file a pin names and compare. Do this before using anything the
  pin protects. `iface committee verify` and `iface package verify` do this for
  bundles and archives.
- A pinned field's name can mislead. In the pilot, `ledger_sha256` was not the
  hash of the release's `release_ledger.tsv`; it was the frozen source ledger
  in another folder.
- **When a pinned hash matches nothing obvious, search the archive by hash
  before writing "unknown".**
- Text files hashed on Windows and in git can differ by line endings. Say which
  bytes a recorded hash refers to.

### 5. Membership and splits, over the whole ledger

- Recount statuses, splits and label uses from every ledger row and compare
  with the reported counts.
- Compare each relabel package's intended roles with the splits the ledger
  gives those frames, for every frame.
- Report separately: source runs with frames in more than one split (a proven
  relation); identical geometry digests in more than one split (proven
  overlap); similarity between different geometries (not decidable unless a
  tested similarity measure exists).
- Find the text of the grouping or leakage rule that applies to this dataset
  (for `iface collect` datasets, the `grouped` / `guarded` strategy in
  [Campaign format](campaign.md)). If only its hash is available, say the rule
  cannot be checked. Do not borrow a rule from another dataset.
- Say "sample" whenever fewer than all rows were examined.

### 6. Lineage by hash, in both directions

- Join outputs to the register and to their consumers by hash: the job's own
  hash line, the register entry, every pin file that names a model.
- **Include outputs that no register names.** In the pilot an unregistered
  committee had a downstream user that a register-only scan could not see.
- Scan all pin files, not only those of packages in scope, and report pinned
  hashes you could not identify.
- State the bound of every "none found": which files were covered, and that
  use recorded elsewhere is not excluded.

### 7. Validations: value and threshold from the same structured record

- A verdict is usable when the value and its threshold both come from a
  structured record. Then it is a derived fact with a named rule.
- A verdict that exists only in a sentence is a quotation, to be marked as
  unverified.
- A model can be promoted with a failed test. Record both facts; neither
  cancels the other.

### 8. Failures and repairs

- Attempts come from the submission log. Outcomes come from what each attempt
  left behind (output files, the environment file a job appends to, per-task
  logs) before any narrative.
- When two attempts cover a task and no record says which produced the output,
  say so. Then check the per-attempt files.
- A cause recorded as "suspected, unverified" stays unverified in the audit.

### 9. Conflicts and stale records stay visible

- When two records disagree, report both with their locations. Do not pick the
  newer file.
- A dated observation (an export audit, a queue paste) is true for its date
  only.
- Re-hash the sources before reusing an earlier audit; if anything changed, say
  which files.

### 10. Report

- Every fact carries its source and one of five labels: observed, derived (with
  the rule), proposed (quoted, not verified), unknown, conflicting.
- State coverage and what was not covered. State the bound of each negative
  result.
- Keep four judgements apart: code correctness, provenance completeness,
  scientific validity, production readiness.
- List record problems for the owner in their own section. Change nothing.

Inside an [agentic assignment](agentic-assignments.md) the assignment's own
claim labels (`[E]`, `[P]`, `[H]`, `[C]`, `[S]`) and report schema apply. The
closest correspondence is observed/derived → `[E]`, proposed → `[P]`; "unknown"
goes in `not_assessed`, and a conflict is reported as a `[C]` with both
locations.

### 11. Check the audit itself

- Have a second, independent pass answer a few questions from the originals,
  with the reference answers written and hashed beforehand.
- If the reference turns out wrong, record an erratum; do not edit it after
  seeing results.
- Have someone who does not know which pass is which do the scoring where
  possible.

## Minimum inputs to start

- The repository commit and the list of record files that fill the step-2
  table for that campaign.
- Where archives live, and which files may only be hashed.
- The text, not just the hash, of the dataset's split and grouping policy.
  Without it step 5 ends in "cannot be checked".
- Who owns each record, so that findings have somewhere to go.

## Tooling

None is required. The pilot compared a purpose-built index against plain
repository search twice, one run per arm, and both reached the same answers
with no dependable saving in tool calls. Two lessons carry over:

- An index is partial by construction. "Unknown" from an index means unknown to
  the index; search the originals.
- The value of tooling was in exhaustive recounts and hash joins (steps 4–6 and
  9) that someone answering a single question does not think to run. If such
  checks are added here, they belong as tested `iface audit …` commands run
  periodically, not as the route for every question.

## Limits

- The procedure checks agreement between records. It cannot tell that two
  agreeing records are both wrong.
- No geometric or statistical check is part of it. Structural and numerical
  claims still come from the tested audits (`iface audit`, `iface validate`,
  the source-rescue scanner).
- A campaign whose facts live mostly in notes will behave differently, and
  step 2 will take most of the effort.
