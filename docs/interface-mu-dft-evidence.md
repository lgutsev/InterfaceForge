# Interface-mu DFT evidence: review, implementation and verification task

## Task

Preserve and validate DFT evidence before interpreting interface-mu discrepancies.
Baseline reviewed: `17db581c4f4729752fce1e8142bce2641842eca0`.

The numerical attribution identities check arithmetic, but cannot establish that
an energy came from the intended composition or a compatible DFT calculation.
This task closes those reporting and input-consistency gaps while retaining
numerical results for inspection when provenance is incomplete.

## Review findings and implemented changes

| Finding at baseline | Fix location and behavior |
|---|---|
| `_dft_energy` retained only normal termination and final sigma-zero energy, dropping warnings, health and optimization status. | `separation_energy._dft_record` retains the audit fields and assigns a mode-aware status. Static and MD calculations do not need an optimization convergence marker. Incomplete/nonfinite energies remain unavailable. `_dft_energy` remains a compatibility wrapper. |
| Structure composition was used without comparison to the energy source. | `dft_evidence.structure_evidence` compares NIONS and ions-per-type totals, then species-resolved TITEL counts. Known contradictions raise a path-specific SafetyError; absent species/count evidence stays NOT_CHECKED. |
| References could mix DFT settings or potential variants. | `dft_evidence.audit_provenance` reuses `build_vasp_reference_record`, retains hashes and inputs, and compares executed ENCUT, GGA, METAGGA, LHFCALC, AEXX, HFSCREEN, LDAU and IVDW. Input/executed contradictions are reported. Shared-element TITEL comparison allows legitimately different phase compositions. |
| Aggregate NOT_CHECKED was converted into PASS. | `interface_mu.summarize_audit` now retains the worst status, includes DFT health/composition/provenance checks, and places the DFT-evidence qualification before MLIP interpretation. Numeric attribution remains available. |
| Equal-energy separation partials could lose incoming audit evidence. | `_merge_dft` adopts evidence from an incoming partial if absent and refuses conflicting evidence instead of silently selecting one record. |
| No interface-mu verification row existed. | `docs/verification.md` records automated coverage and its scientific limits. |

Evidence is attached to every interface and every reference phase, including
auxiliary hull phases and molecular references. The JSON retains full provenance;
the Markdown report includes per-structure run/composition status, health,
warnings, provenance conflicts and missing executed settings.

## Acceptance criteria

- [x] A finite energy with a warning or failed relaxation cannot yield a clean DFT health check.
- [x] Static and MD runs do not fail merely because `opt_converged` is false.
- [x] Same total atom count with different species counts raises SafetyError.
- [x] Missing species identity cannot pass merely because the atom total matches.
- [x] Executed settings take precedence; an INCAR alone does not prove executed provenance.
- [x] Numeric ENCUT equivalence and VASP boolean spelling equivalence are accepted.
- [x] A shared-element potential variant or functional/ENCUT mismatch yields CHECK.
- [x] Different complete POTCAR files for different compositions are permitted.
- [x] Missing checks remain visible and cannot aggregate to PASS.
- [x] Equal-energy partial merging cannot suppress conflicting warnings.
- [x] Existing interface-mu arithmetic and separation-energy workflows retain regression coverage.
- [ ] A researcher inspects a real campaign's complete evidence report.

## Verification commands

From the repository root with development dependencies installed:

```bash
python -m pytest tests/test_dft_evidence.py tests/test_interface_mu.py tests/test_separation_energy.py tests/test_vasp_audit.py tests/test_separation_energy_launchers.py -q
python -m ruff check src/interfaceforge/dft_evidence.py src/interfaceforge/interface_mu.py src/interfaceforge/separation_energy.py tests/test_dft_evidence.py
git diff --check
```

Recorded on 2026-09-13 with Python 3.12: **156 tests passed**; targeted Ruff
checks and `git diff --check` passed.

These checks exercise synthetic inputs and mocked model backends. They do not
run VASP, MACE or DeePMD. The existing campaign launch command can be rerun into
a fresh output directory for human review; no new DFT calculation is required
merely to inspect existing OUTCAR evidence.

## Ready-to-assign follow-up: real campaign evidence review

Run the current interface-mu command on the existing ideal N-terminated and
Ti-terminated cells with the exact bulk/molecular/auxiliary references used for
the comparison. Record commit, command, environment, source paths and output
hashes. Inspect every CHECK and NOT_CHECKED before interpreting absolute gamma
or assigning an offset to MLIP training coverage. Resolve unknowns from original
run artifacts, not reconstructed INCAR assumptions. Preserve unresolved findings.

Acceptance: a dated note lists each structure's count check, run-health check,
executed-setting coverage and potential identity; explains intentional reference
differences; and records an intentional wrong-composition failure using copied
fixtures. Promotion to human-tested requires inspection of the actual output.

Limits: compatible potential titles are weaker evidence than per-element dataset
hashes. Whole-file hashes are retained but cannot establish cross-composition
potential equality. Species-resolved Hubbard parameters remain NOT_CHECKED when
LDAU is enabled. Different k meshes and molecular spin states are permitted;
k-point/cutoff convergence, real-space geometry identity with the final energy,
and full custom-functional equivalence are separate scientific checks. Missing
executed tags are deliberately not replaced with assumed VASP defaults.
