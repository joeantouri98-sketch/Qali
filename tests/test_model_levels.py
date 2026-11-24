"""Model levels in the Rydberg stage: DHF/Brueckner energies for the series NIST has no level of (Fr nf)."""
from __future__ import annotations

import json
import math

import pytest

from helpers import load_json, run_pipeline
from qali.physics import model_levels as ML

RY = 13.6056
EV_CM = 8065.543937
FR_ION_CM = 32848.872


def _f_doc(delta=1.03, n_range=range(5, 11), ion_cm=FR_ION_CM):
    """A Brueckner-like output: an f series with quantum defect *delta*, J split by 0.001 cm⁻¹."""
    levels = [dict(label="7s1/2", n=7, l=0, j=0.5, binding_eV=ion_cm / EV_CM, energy_cm=0.0)]
    for n in n_range:
        b = RY / (n - delta) ** 2
        for j, shift in ((2.5, 0.0), (3.5, 0.001 / EV_CM)):
            levels.append(dict(label=f"{n}f{int(2 * j)}/2", n=n, l=3, j=j, binding_eV=b + shift,
                               energy_cm=ion_cm - (b + shift) * EV_CM))
    return dict(element="Fr", model="brueckner", ionization_cm=ion_cm, levels=levels,
                nist_comparison=[dict(state="8s1/2", nist_cm=19732.523, diff_cm=-10.0, fitted=False),
                                 dict(state="7p1/2", nist_cm=12237.409, diff_cm=0.0, fitted=True)])


def test_ritz_fit_recovers_the_series_defect():
    lev = [x for x in ML.model_levels(_f_doc(delta=1.03)) if x["l"] == "f"]
    fit = ML.ritz_fit(lev, RY)
    assert fit["delta0"] == pytest.approx(1.03, abs=2e-4) and abs(fit["delta2"]) < 0.05


def test_misfit_scale_uses_only_levels_the_model_did_not_fit():
    sc = ML.misfit_scale(_f_doc(), RY)
    ns = math.sqrt(RY / ((FR_ION_CM - 19732.523) / EV_CM))
    assert set(sc) == {"s", "*"} and sc["*"] == pytest.approx(10.0 * ns ** 3)


def test_apply_replaces_the_hydrogenic_series_and_adds_the_missing_lowest_member():
    """Fr: no NIST f level, so the stage's f series is hydrogenic and starts at 6f; the model has 5f and J."""
    ie = FR_ION_CM / EV_CM
    states = [dict(n=n, l="f", J=None, energy_eV=-RY / n ** 2, label=f"{n}f", source="QDT model")
              for n in range(6, 15)] + [dict(n=8, l="s", J="1/2", energy_eV=-1.0, label="8s", source="NIST")]
    out = ML.apply_model_levels(states, _f_doc(), "brueckner", ["f"], ie, RY, 50)
    labels = [s["label"] for s in states]
    assert "5f5/2" in labels and "5f7/2" in labels and "6f" not in labels and "10f" not in labels
    assert "11f" in labels and "8s" in labels                # above the model's n, and other series, kept
    model = [s for s in states if s["source"] == "Brueckner"]
    assert len(model) == out["levels"] == 12 and all(s["theoretical"] for s in model)
    assert all(s["uncertainty_eV"] > 0 for s in model)
    assert [s["state_number"] for s in states] == list(range(1, len(states) + 1))


def test_rydberg_fr_with_model_levels_end_to_end(workspace, tmp_path):
    model = tmp_path / "Fr_model.json"
    model.write_text(json.dumps(_f_doc(n_range=range(5, 14))), encoding="utf-8")
    result = run_pipeline(["rydberg", "Fr", "--model-levels", "brueckner", "--model-file", str(model)], workspace)
    assert result.returncode == 0, result.stderr
    d = load_json(workspace / ".qali" / "data" / "Fr_rydberg.json")
    assert d["model_levels"]["series"] == ["f"] and d["model_levels"]["levels"] == 18
    assert d["quantum_defect_sources"] == {"f": "Brueckner"}
    # the stage's reduced-mass Rydberg constant differs from the test's by 3e-6: δ shifts by ~1e-4 at n = 13
    assert d["quantum_defects_ritz"]["f"]["delta0"] == pytest.approx(1.03, abs=2e-3)
    f = {e["label"]: e for e in d["excitations"] if e["l"] == "f"}
    assert f["5f5/2"]["source"] == "Brueckner" and f["5f5/2"]["theoretical"]
    # above the model: the series continues with the model's quantum defect, not hydrogenic
    e20 = f["20f"]["energy_eV"]
    assert 20 - math.sqrt(d["rydberg_constant_eV"] / -e20) == pytest.approx(1.03, abs=2e-3)
    # measured series are untouched: NIST d levels stay NIST
    assert all(e["source"] == "NIST" for e in d["excitations"] if e["l"] == "d" and e["n"] <= 30)


def test_rydberg_without_a_model_file_says_which_stage_to_run(workspace):
    result = run_pipeline(["rydberg", "Fr", "--model-levels", "dhf"], workspace)
    assert result.returncode != 0 and "run the DHF stage" in (result.stderr + result.stdout)
