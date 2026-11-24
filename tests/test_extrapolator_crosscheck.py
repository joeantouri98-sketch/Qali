"""The Extrapolator against DHF / Brueckner: rows, model scatter, cancellation and pulls."""
from __future__ import annotations

import json

import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation.crosscheck import cross_check, model_f
from qali.physics.polarizability_core import load_nist_fvalues


@pytest.fixture(scope="module")
def rb():
    ryd = json.loads(reference_path("data_json", "Rb_rydberg.json").read_text(encoding="utf-8"))
    return ryd, load_nist_fvalues("Rb", verbose=False)


def _doc(rows):
    return {"transitions": [{"lower": "5s1/2", "upper": f"{n}p{j}", "f_absorption": f, "cancellation": c}
                            for n, j, f, c in rows]}


def test_model_f_reads_ground_np_lines_only():
    doc = _doc([(6, "1/2", 1e-3, 0.5), (6, "3/2", 2e-3, 0.5)])
    doc["transitions"].append({"lower": "5p1/2", "upper": "6s1/2", "f_absorption": 0.1})
    assert model_f(doc, "5s") == {(6, "1/2"): 1e-3, (6, "3/2"): 2e-3}


def test_a_model_off_by_3x_beyond_the_measured_lines_is_flagged(rb):
    ryd, nist = rb
    first = cross_check("Rb", nist, ryd, {"dhf": _doc([(25, "1/2", 1e-7, 0.5)])})   # rows up to n = 25
    ext = {(r["n"], r["J"]): r for r in first["rows"] if r["f_extrap"]}
    meas = {k: r["f_measured"] for k, r in ext.items() if r["f_measured"]}
    beyond = [k for k in ext if k not in meas and k[0] <= 25]
    assert beyond, "Rb needs lines beyond the measured ones"
    # a model equal to the measured lines and 3x the Extrapolator beyond them
    rows = [(n, j, f, 0.5) for (n, j), f in meas.items()] + [(n, j, 3 * ext[(n, j)]["f_extrap"], 0.5)
                                                            for n, j in beyond]
    res = cross_check("Rb", nist, ryd, {"dhf": _doc(rows)})
    s = res["summary"]["dhf"]
    assert s["sigma_vs_nist"]["1/2"] == pytest.approx(0.05) and s["compared"] == len(beyond)
    assert s["median_ratio"] == pytest.approx(1 / 3, rel=1e-6) and len(s["beyond_2sigma"]) == len(beyond)


def test_cancellation_widens_a_models_uncertainty(rb):
    ryd, nist = rb
    rows = [(9, "1/2", 1e-5, 0.5), (9, "3/2", 1e-5, 0.005)]
    res = cross_check("Rb", nist, ryd, {"brueckner": _doc(rows)})
    r = {x["label"]: x for x in res["rows"]}
    assert r["9p3/2"]["sigma_brueckner"] > 1.9 > r["9p1/2"]["sigma_brueckner"]     # 1 % / 0.005 = 2
