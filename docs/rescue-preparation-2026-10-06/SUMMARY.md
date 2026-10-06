# Rescue preparation: training remains blocked

Snapshot report SHA256: `5f3d110edf9a1e7b3d8ad5dc6c76ae124da7b24e708c25781e8e056c07a2f35f`

Candidate original sources: 44; candidate frames at stride 5: 26400. These are inventory counts, not accepted labels.

Quarantined records: 12. Probe geometries: 24.

Historical split membership: MISSING.

This offline preparation did not read live raw sources, qualify labels, merge copies, stage a dataset or submit jobs. Frame indices are zero-based. Probe selection uses the median and worst available printed charge residual among stride-retained frames. A last printed residual is not necessarily the residual at the final iteration. Verify geometry/input hashes before generating probes; reconcile existing controls to avoid duplicates.

Remaining gates:

- Review package-23 full E/F/stress against predeclared tolerances
- Qualify every retained source and staged replacement; triage is not full-source acceptance
- Freeze historical splits and implement validated partial replacement export
- Finish and role-curate the A2 44 train / 16 holdout plan if included
- Run a fresh LONI admission audit and synchronized MACE/DeePMD export checks
