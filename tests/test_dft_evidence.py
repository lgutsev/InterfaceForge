from unittest.mock import patch

import pytest

from interfaceforge.dft_evidence import TAGS, audit_provenance, structure_evidence
from interfaceforge.errors import SafetyError
from interfaceforge.interface_mu import summarize_audit
from interfaceforge.separation_energy import _dft_record, _merge_dft


def write_outcar(path, *, counts="2 2", nions=4, titles=("PAW_PBE Ti 08Apr2002", "PAW_PBE N 08Apr2002"), encut="520"):
    path.mkdir(exist_ok=True)
    settings = dict.fromkeys(TAGS, "0")
    settings.update(ENCUT=encut, GGA="PE", METAGGA="--", LHFCALC="F", LDAU="F")
    (path / "OUTCAR").write_text(
        "vasp.6.5.1\n" + "\n".join(f" TITEL = {t}" for t in titles)
        + f"\n ions per type = {counts}\n NIONS = {nions}\n"
        + "\n".join(f" {k} = {v}" for k, v in settings.items())
    )


def test_species_mismatch_with_equal_total_is_refused(tmp_path):
    write_outcar(tmp_path)
    with pytest.raises(SafetyError, match="OUTCAR composition"):
        structure_evidence(tmp_path, {"Ti": 1, "N": 3})


@pytest.mark.parametrize("counts,nions", [("1 2", 4), ("2 2", 5)])
def test_total_count_mismatch_is_refused(tmp_path, counts, nions):
    write_outcar(tmp_path, counts=counts, nions=nions)
    with pytest.raises(SafetyError):
        structure_evidence(tmp_path, {"Ti": 2, "N": 2})


def test_missing_species_is_not_checked_even_if_total_matches(tmp_path):
    write_outcar(tmp_path, titles=())
    assert structure_evidence(tmp_path, {"Ti": 2, "N": 2})["composition"]["status"] == "NOT_CHECKED"


def test_cross_chemistry_titles_and_numeric_encut_equivalence(tmp_path):
    a, b = tmp_path / "TiN", tmp_path / "N2"
    write_outcar(a)
    write_outcar(b, counts="2", nions=2, titles=("PAW_PBE N 08Apr2002",), encut="520.0")
    evidence = {"TiN": structure_evidence(a, {"Ti": 2, "N": 2}), "N2": structure_evidence(b, {"N": 2})}
    assert audit_provenance(evidence)["status"] == "PASS"
    evidence["N2"]["provenance"]["potcar_titles"] = ["PAW_PBE N_h 08Apr2002"]
    result = audit_provenance(evidence)
    assert result["status"] == "CHECK"
    assert "N2: N POTCAR" in result["issues"][0]


@pytest.mark.parametrize("tag,value", [("ENCUT", "400"), ("GGA", "PS"), ("LHFCALC", "T"), ("IVDW", "11")])
def test_executed_setting_conflict(tmp_path, tag, value):
    write_outcar(tmp_path)
    a = structure_evidence(tmp_path, {"Ti": 2, "N": 2})
    import copy
    b = copy.deepcopy(a)
    b["provenance"]["outcar_executed_tags"][tag] = value
    assert audit_provenance({"a": a, "b": b})["status"] == "CHECK"


def test_input_is_not_substituted_for_missing_executed_evidence(tmp_path):
    write_outcar(tmp_path)
    evidence = structure_evidence(tmp_path, {"Ti": 2, "N": 2})
    record = evidence["provenance"]
    del record["outcar_executed_tags"]["ENCUT"]
    record["incar_tags"]["ENCUT"] = "520"
    assert audit_provenance({"a": evidence})["status"] == "NOT_CHECKED"


@pytest.mark.parametrize("kind,converged,warning,expected", [
    ("static", False, "", "PASS"), ("md", False, "", "PASS"),
    ("opt", False, "", "CHECK"), ("opt", True, "", "PASS"),
    ("static", False, "electronic convergence", "CHECK"),
])
def test_energy_retains_health_and_mode_appropriate_optimization(tmp_path, kind, converged, warning, expected):
    (tmp_path / "OUTCAR").touch()
    health = {"static": "static calculation finished", "opt": "converged",
              "md": "completed; average behavior reported"}[kind]
    row = dict(finished_normally=True, sigma0_energy_ev_last=-10, run_kind=kind,
               opt_converged=converged, warnings=warning, health=health)
    with patch("interfaceforge.separation_energy.audit_run", return_value=row):
        result = _dft_record(tmp_path)
    assert result["status"] == expected
    assert result["energy_ev"] == -10
    assert result["health"] == health
    assert result["warnings"] == warning
    assert result["opt_converged"] is converged


def test_unfinished_and_missing_energy(tmp_path):
    assert _dft_record(tmp_path)["status"] == "NOT_CHECKED"
    (tmp_path / "OUTCAR").touch()
    row = {"finished_normally": False, "sigma0_energy_ev_last": -10}
    with patch("interfaceforge.separation_energy.audit_run", return_value=row):
        assert _dft_record(tmp_path)["energy_ev"] is None


def test_unknown_evidence_does_not_become_overall_pass():
    assert summarize_audit({})["status"] == "NOT_CHECKED"


def test_separation_merge_preserves_evidence():
    incoming = {"ready": True, "evidence": {"interface": {"warnings": "SCF"}}}
    assert _merge_dft({"ready": False}, incoming, "sample")["evidence"] == incoming["evidence"]


def test_equal_energy_merge_cannot_hide_conflicting_warnings():
    base = {"ready": True, "energies_ev": dict.fromkeys(("interface", "slab_a", "slab_b"), -10),
            "evidence": {"interface": {"warnings": ""}}}
    incoming = {**base, "evidence": {"interface": {"warnings": "SCF"}}}
    with pytest.raises(SafetyError, match="audit evidence mismatch"):
        _merge_dft(base, incoming, "sample")


def test_interface_report_preserves_unknown_evidence(tmp_path):
    from test_interface_mu import _phases, _run

    from interfaceforge.interface_mu import interface_mu, write_reports
    phases = _phases(tmp_path)
    run = _run(tmp_path / "interface", [("Ti", 4), ("Si", 6), ("N", 12)], -190)
    payload = interface_mu([("interface", run)], phases=phases, n_interfaces=2)
    assert payload["audit"]["checks"]["outcar_composition"] == "NOT_CHECKED"
    assert payload["audit"]["checks"]["vasp_provenance"] == "NOT_CHECKED"
    assert payload["reference_phases"]["N2"]["dft_evidence"]["run"]["health"]
    outputs = write_reports(payload, tmp_path / "report")
    from pathlib import Path
    markdown = next(Path(v).read_text() for v in outputs.values() if str(v).endswith(".md"))
    assert "## DFT evidence" in markdown
    assert "executed ENCUT unknown" in markdown


def test_boolean_spelling_and_input_execution_disagreement(tmp_path):
    write_outcar(tmp_path)
    evidence = structure_evidence(tmp_path, {"Ti": 2, "N": 2})
    evidence["provenance"]["incar_tags"]["LHFCALC"] = ".FALSE."
    assert audit_provenance({"a": evidence})["status"] == "PASS"
    evidence["provenance"]["incar_tags"]["ENCUT"] = "400"
    audit = audit_provenance({"a": evidence})
    assert audit["status"] == "CHECK"
    assert "INCAR/OUTCAR ENCUT differs" in audit["issues"][0]
