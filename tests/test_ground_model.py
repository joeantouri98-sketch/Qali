"""Ground ns->np beyond the measured values: the Extrapolator's calibrated model replaces QDT Numerov
(extrapolation.stage.GroundModel), in Lifetimes and Polarizability alike."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qali.physics.extrapolation.stage import GROUND_MODEL_SOURCE, apply_to_stage, ground_model
from qali.physics.polarizability_core import load_nist_fvalues


def _load(data_dir, el, stage):
    return json.loads(Path(data_dir, f"{el}_{stage}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def models(data_dir):
    out = {}
    for el in ("Li", "Na", "K", "Rb", "Cs"):
        ryd = _load(data_dir, el, "rydberg")
        nist = load_nist_fvalues(el, verbose=False, max_ek_eV=ryd["ionization_energy_eV"])
        out[el] = ground_model(el, ryd, nist)
    return out


def test_label_without_j_is_the_sum_of_both_components(models):
    m = models["Cs"]
    assert m.f("30p") == pytest.approx(m.f_j("30p", 0.5) + m.f_j("30p", 1.5), rel=1e-12)
    assert m.f("6s") is None and m.f("nonsense") is None
    assert ground_model("Mg_c1", {"excitations": []}) is None          # no model for ions


def test_uncertainty_follows_the_distance_to_nist(models):
    # Rb sits within 25 % of NIST at the top of the series (floor); K's p1/2 is about 2x off there
    assert models["Rb"].sigma("30p1/2") == pytest.approx(0.25)
    assert models["K"].sigma("30p1/2") > 0.5
    assert models["K"].sigma("30p") == max(models["K"].sigma("30p1/2"), models["K"].sigma("30p3/2"))


def test_no_ground_line_is_left_to_numerov(data_dir):
    for el in ("Li", "Na", "K", "Rb", "Cs", "Fr"):
        gs = _load(data_dir, el, "polarizability")
        srcs = {t["source"] for t in gs["transitions_gs"] if t["other_label"].endswith(("p", "p1/2", "p3/2"))}
        assert not srcs & {"Numerov", "Coulomb"}, (el, srcs)


def test_nist_values_are_kept(data_dir):
    rb = {t["other_label"]: t for t in _load(data_dir, "Rb", "polarizability")["transitions_gs"]}
    assert rb["20p3/2"]["source"] == "NIST"                            # the D+ outlier stays (Jo's call)
    cs = {t["other_label"]: t for t in _load(data_dir, "Cs", "polarizability")["transitions_gs"]}
    assert cs["25p1/2"]["source"] == "NIST" and cs["26p"]["source"] == GROUND_MODEL_SOURCE


@pytest.mark.parametrize("el,gs,cont_hi", [("K", "4s", 3.0), ("Cs", "6s", 3.0)])
def test_bound_series_meets_the_continuum(data_dir, el, gs, cont_hi):
    """f n*^3 of the highest np sits within x3 of df/de of the continuum at threshold (Numerov was 10-17x)."""
    from qali.figures import recipe as R

    r = R.new_recipe("oscillator_density", el)
    r["params"].update(merge_j=True, energy_unit="au")
    _h, rows = R.resolve(r, str(data_dir)).table
    bound = max((x for x in rows if x[0] != "continuum"), key=lambda x: x[1])
    cont = min((x for x in rows if x[0] == "continuum"), key=lambda x: x[1])
    assert 1 / cont_hi < bound[2] / cont[2] < cont_hi


def test_cs_high_np_keep_their_ground_decay(data_dir):
    lt = _load(data_dir, "Cs", "lifetimes")
    rows = [t for t in lt["transitions"] if t["lower_label"] == "6s" and t["upper_label"] == "30p"]
    assert len(rows) == 1 and rows[0]["fosc_source"] == GROUND_MODEL_SOURCE
    t = rows[0]
    assert t["A_s"] == pytest.approx(6.6703e13 / t["wavelength_nm"] ** 2 * 2 / 6 * t["fosc"], rel=1e-6)
    assert lt["ground_model"]["n_lines"] == 25


def test_off_and_missing_config(tmp_path, data_dir):
    ryd = _load(data_dir, "K", "rydberg")
    nist = load_nist_fvalues("K", verbose=False, max_ek_eV=ryd["ionization_energy_eV"])
    assert apply_to_stage("K", tmp_path / "none.json", nist, {}, ryd)[0] is None
    # the user's model parameters next to the config change the raw model
    (tmp_path / "model_settings.json").write_text(json.dumps({"model": {"K": {"lam": 0.2}}}), encoding="utf-8")
    mine = ground_model("K", ryd, nist, tmp_path / "K_extrapolation.json")
    base = ground_model("K", ryd, nist)
    assert not base.custom
    try:
        import arc  # noqa: F401
    except ImportError:                       # without ARC (and no rebuilt table): the shipped model, said so
        assert not mine.custom and "shipped" in mine.note and mine.f("20p3/2") == base.f("20p3/2")
        return
    assert mine.custom and mine.f("20p3/2") != pytest.approx(base.f("20p3/2"), rel=1e-3)


def test_active_config_path_passes_model_edits_when_off(tmp_path):
    pytest.importorskip("PySide6")
    from qali.gui.extrapolation_dialog import active_config_path, config_path

    path = config_path(tmp_path, "K")
    assert active_config_path(tmp_path, "K") == ""
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "model_settings.json").write_text(json.dumps({"model": {"K": {"lam": 0.2}}}), encoding="utf-8")
    assert active_config_path(tmp_path, "K") == str(path)
    assert active_config_path(tmp_path, "Cs") == ""
