# Prepare the SiN/TiN rescue while package 23 runs

The controls do not block inventory, exclusion checks, deterministic probe
selection or checking a recovered historical split ledger. The offline tool
reads the delivered audit bundle and its exact policy/map YAMLs. It does not
need `/ddnB` mounts, a scheduler or the original OUTCARs.

The checked snapshot produced
[the October 6 preparation](rescue-preparation-2026-10-06/SUMMARY.md):
44 active original trajectories, 132,000 raw frames and 26,400 candidate
stride-5 frames. These remain REVIEW. Four mapped bulk trajectories are
excluded: SiN 450 K, TiN 450 K, TiO-Ideal 300 K and TiO-Ideal 450 K.
The report's other eight quarantined records are inventory copies in a disabled
archive branch. TiO-Real 300/450 K remain candidates awaiting qualification.

The saved [probe selection](rescue-preparation-2026-10-06/probe_selection.csv)
contains 24 zero-based source indices: a typical and a worst available printed
charge-residual frame in each of eight interface groups and four active bulk
trajectories. Typical means nearest to the pooled retained-frame median, with
deterministic tie breaking. Worst is a distinct frame with the largest residual.
This reproducible selection is a planning proposal, not an assertion that it
matches the laptop worker's unpublished selection or that its inputs have been
generated. Reconcile the worker's selection and existing controls before new DFT.
All frame-level evidence is hashed in the plan. The last printed charge residual
may precede the final electronic iteration. Missing/nonfinite residual evidence
holds all probe selection rather than silently substituting a smaller candidate.

## Run offline preparation

From an updated InterfaceForge checkout, with `iface`'s Python environment:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -m interfaceforge.rescue_prepare \
  /path/to/unpacked/review_evidence/source_audit.json \
  examples/mapped-leaf-campaign/periodic_nitride.yaml \
  examples/source-rescue/sintin.yaml \
  /path/to/fresh/rescue_preparation
```

Use the delivered report and its `frames/*.scf.json` siblings. Windows users can
run the same module with absolute paths in their local checkout. This command
only reads these local files, so unset remote environment variables are harmless.
It rejects changed policy/map hashes, incomplete scans, conflicting source IDs,
re-enabled quarantines and missing or reordered SCF frame evidence.
It creates `SUMMARY.md`, `source_inventory.csv`, `probe_selection.csv` when
selection is available, and `rebuild_plan.json`. The output directory must be
new. Installed packages also expose `iface-rescue-prepare` with the same args.

These outputs describe the historical audit snapshot. They do not verify files
that changed afterward on LONI, accept any source, merge `runs/outputs` copies,
extract probe geometries, export training data or launch calculations. The 91
duplicate output groups are retained for later manifest/input/role reconciliation.
The final source audit must use the curated live sources and a fresh directory.
A2 is still disabled in this snapshot's training map. Its package roots are
reference/audit evidence; the 44 train endpoints and 16 holdouts need separate
curation and numerical qualification.

## Check recovered historical split membership

Recover membership from the original exported frame maps or equivalent verified
provenance. The old random seed alone is not proof of original membership.
Use explicit source IDs from `source_inventory.csv`, source-frame indices and
original OUTCAR hashes. A normalized ledger has this schema:

```json
{
  "schema_version": 1,
  "report_sha256": "SHA256_OF_THIS_SOURCE_AUDIT_JSON",
  "frames": [
    {
      "source_id": "mapped/interface/300K/Real/N_Term/SiN_TiN_N-term",
      "source_frame": 0,
      "outcar_sha256": "ORIGINAL_OUTCAR_SHA256",
      "split": "train"
    }
  ]
}
```

Include every surviving retained frame exactly once, using `train`, `valid`
and `test`. Historical entries for excluded mapped sources may also be included;
they are counted separately and cannot re-enter the candidate deck. Add
`--historical-membership /path/to/historical.json` to the preparation command.
The checker rejects duplicate/cross-split frames, wrong OUTCAR/report hashes,
unknown/reference/audit sources, invalid indices and missing survivors. It
verifies consistency with the supplied ledger, not the independent authenticity
of an invented ledger; record which historical exports establish its membership.
The checked snapshot has no supplied historical ledger, so its status is MISSING.
No source/frame/split memberships have been invented.

## Work that still depends on results

Review all package-23 controls with full forces and stress against tolerances
set before evaluating results. Extend source qualification beyond the triage
probes; qualify the bulk replacements; implement and verify the partial label
replacement exporter with frozen splits; finish the A2 labels if included; then
run final admission and synchronized MACE/DeePMD export checks. Keep all nine
virial components and the virial loss data. This preparation command is
intentionally unable to mark training ready.
