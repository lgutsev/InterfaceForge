# SiN/TiN original 450 K bulk trajectories: quarantine decision

On 2026-10-02 the campaign owner reviewed the temperature plots and identified
both original SiN-Bulk_450K and TiN-Bulk_450K sampling as flawed. The rescue
directory map disables both sources, and the audit profile quarantines matches
to either name throughout the discovered origin trees. No raw files, existing
exports, models or historical audit reports are modified by this decision.
The owner's subsequent TiO clarification refers to TiO-Ideal, already excluded;
it does not add a TiO-Real quarantine.

## Evidence and scope

The supplied SiN plot shows persistent, large periodic temperature cycles over
the 3 ps trajectory. In the unchanged corrected audit, its post-burn-in mean
and standard deviation are 450.8 and 162.5 K, with p05/median/p95 of
247/427/731 K. TiN's corresponding values are 445.2 and 81.2 K, with
329/444.5/570.05 K. Its cycling was additionally reported by the campaign owner.
The amplitudes differ; the recorded common concern is unqualified sampling.

Nosé thermostat mass affects the frequency of temperature oscillations, and
poor coupling to ionic modes can compromise canonical sampling:
[VASP SMASS](https://vasp.at/wiki/SMASS). These plots alone do not identify a
unique cause or establish that each static E/F/stress label is incorrect. The
plot's "Total Energy" definition must also be verified against the generating
script and OSZICAR's E/F/EK/SP/SK columns before diagnosing conserved-energy
drift: [VASP OSZICAR](https://vasp.at/wiki/OSZICAR).

Conservative campaign decision: hold both entire original 450 K trajectories
out of train, validation and test until sampling is independently qualified.
Relabeling the same geometries at denser k or tighter SCF changes label accuracy
but does not repair their original sampling. Such controls can still diagnose
numerical issues while retaining a non-training role.

| Excluded source | Retained frames | Original train / validation / test |
| --- | ---: | --- |
| TiO-Bulk-Ideal_300K | 600 | 480 / 60 / 60 |
| TiO-Bulk-Ideal_450K | 600 | 480 / 60 / 60 |
| SiN-Bulk_450K | 600 | 480 / 60 / 60 |
| TiN-Bulk_450K | 600 | 480 / 60 / 60 |
| Total excluded | 2,400 | 1,920 / 240 / 240 |

The nominal remaining original deck is 26,400 frames, split
21,120/2,640/2,640. There are 44 active original MD sources before further
qualification, subset replacements and A2 additions. These are expected counts
from the original 600-frame-per-run plan, not an assertion that recollection
has already produced or qualified that deck. Keep historical source/frame/split
identities; do not randomly redistribute the survivors.

## Worker actions

1. Update the local InterfaceForge checkout and use the changed directory map
   and audit policy. Earlier audit reports remain valid historical evidence,
   but their policy/map hashes no longer authorize collection with this profile.
   Package 21's observed current location is
   `/work/lgutsev/loni_smoke_tests/batch06_2026-09-30/21_vasp_a2_stratified_audit`;
   the policy now requires that root. Resettlement is still pending. Inspect its
   manifests and qualification evidence and rerun the audit into a fresh output
   directory. After verified resettlement, update the policy to its permanent
   `MD_Period/A2_DFT/21_stratified_audit` home before the next scan.
2. Preserve the four quarantined original sources and exclude all matching
   frames from active train, validation and test. Check frozen membership IDs
   and both MACE/DeePMD exports, not only filenames. Keep diagnostic results
   separate from eligible training labels.
3. Continue SiN/TiN numerical repair on the active 300 K sources. Retain the
   existing 450 K probe selections only where useful for diagnosis. Decide
   whether qualified A2 strains/perturbations provide the needed coverage;
   additional full AIMD is not a prerequisite for this rescue. Any replacement
   sampling must have explicit provenance and qualification.
4. Retain SiN/TiN 300 K and TiO-Real sources pending their existing qualification
   gates. Do not infer a TiO-Real failure from TiO-Ideal or bulk 450 K failures.
5. Publish a new clean dataset release and recover existing checkpoints with
   the declared fine-tuning schedule and seed-matched validation. A full restart
   is not required solely because these sources were removed. Report performance
   against independently qualified targets; no model improvement is claimed
   from this exclusion alone.
