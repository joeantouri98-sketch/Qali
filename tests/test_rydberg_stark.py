"""Rydberg Stark engine (physics/rydberg_stark): energies, radial integrals, polarisabilities, maps."""
import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation.exact import MARINESCU
from qali.physics.rydberg_stark import radial
from qali.physics.rydberg_stark.defects import RYDBERG_DEFECTS, RydbergEnergies, polarization_defect
from qali.physics.rydberg_stark.stark import (angular_z, avoided_crossings, build_basis, dynamic_polarizability,
                                              floquet_map, polarizability, scalar_tensor, stark_map)


def _energies(sp):
    from qali.pipelines.dhf import nist_inputs
    ip, nist = nist_inputs(sp, str(reference_path("NIST Levels")))
    return RydbergEnergies(sp, nist, ip, MARINESCU[sp]["alphaC"])


@pytest.mark.parametrize("l", [0, 5, 12, 18])
def test_hydrogen_radial_integrals_are_exact(l):
    # <n l|r|n l+1> = (3/2) n sqrt(n² - (l+1)²) for hydrogen
    n = 20
    x = radial.x_grid(n, 0.05)
    a = radial.solve(None, l, l + 0.5, [n], x)
    b = radial.solve(None, l + 1, l + 1.5, [n], x)
    assert radial.dipole(a, b, x)[0, 0] == pytest.approx(1.5 * n * np.sqrt(n * n - (l + 1) ** 2), rel=1e-6)


def test_angular_factor_sums_to_cos2():
    # s1/2, m = 1/2: sum over p1/2, p3/2 of |<s|cos θ|p j m>|² = <cos² θ> = 1/3
    assert sum(angular_z(0, 1, 1, jp, 1) ** 2 for jp in (1, 3)) == pytest.approx(1 / 3)
    # p3/2, m = 3/2 couples only to d5/2 and s is excluded by m: |<cos θ>|² sums to <cos² θ> = 1/5
    assert sum(angular_z(1, 3, 2, jp, 3) ** 2 for jp in (3, 5)) == pytest.approx(1 / 5)


def test_table_is_editable_and_matches_nist_where_both_are_good():
    from qali.core import user_tables
    assert "rydberg.defects" in user_tables.BY_ID
    assert {"Li", "Na", "K", "Rb", "Cs"} <= set(RYDBERG_DEFECTS)
    en = _energies("K")
    for n in range(8, 19):                                   # Peper et al. 2019 vs NIST, cm-1
        for l, j2 in ((0, 1), (1, 1), (1, 3)):
            lv = en.level(n, l, j2)
            assert abs(en.ip_cm - lv.binding_cm - en.nist[(n, l, j2)]) < 0.05
    assert en.level(5, 0, 1).source == "NIST" and en.level(30, 0, 1).source == "table"


def test_fr_comes_from_nist_and_its_fit():
    en = _energies("Fr")
    assert en.level(8, 0, 1).source == "NIST"
    assert en.level(40, 0, 1).source == "NIST fit"
    assert en.level(40, 6, 11).source == "polarisation"
    assert 5.0 < en.level(40, 0, 1).n - en.level(40, 0, 1).n_star < 5.2


def test_high_l_polarisation_defect_matches_rb_g():
    # Rb g: 0.0040 measured (Moore et al. 2020); first-order core polarisation gives 0.0039
    assert polarization_defect(9.076, 50, 4) == pytest.approx(0.0040, abs=2e-4)


@pytest.mark.parametrize("n, expt", [(9, 1.0377e-4), (10, 2.7254e-4)])
def test_rb_low_ns_polarisability_matches_measurement(n, expt):
    # Rb 9S, 10S: 103.77(9), 272.54(16) MHz/(kV/cm)² (van Wijngaarden group, 2000)
    b = build_basis(_energies("Rb"), "Rb", n, 0, 1, 1, 4, n + 5, n - 1)
    assert polarizability(b) == pytest.approx(expt, rel=5e-3)


def test_rb_8d_scalar_and_tensor_polarisability():
    # Rb 8D5/2: α0 222.68(14), α2 51.91(10) MHz/(kV/cm)²
    en = _energies("Rb")
    al = {m2: polarizability(build_basis(en, "Rb", 8, 2, 5, m2, 4, 14, 7)) for m2 in (1, 3, 5)}
    a0, a2 = scalar_tensor(al, 5)
    assert a0 == pytest.approx(2.2268e-4, rel=1e-2)
    assert a2 == pytest.approx(5.191e-5, rel=6e-2)


@pytest.mark.slow
def test_rb_30s_polarisability_and_first_avoided_crossing():
    # O'Sullivan & Stoicheff, PRA 31, 2718 (1985), 85Rb nS (n = 15-80):
    #   α0 = 2.202e-9 n*^6 + 5.53e-11 n*^7 MHz/(V/cm)², first crossing 4.638e8/n*^5 + 1.528e10/n*^7 V/cm
    b = build_basis(_energies("Rb"), "Rb", 30, 0, 1, 1, 26, 34, 33)
    ns = b.levels[b.target].n_star
    assert polarizability(b) == pytest.approx(2.202e-9 * ns ** 6 + 5.53e-11 * ns ** 7, rel=0.02)
    Ec = 4.638e8 / ns ** 5 + 1.528e10 / ns ** 7
    m = stark_map(b, np.linspace(0.5 * Ec, 1.5 * Ec, 400))
    assert avoided_crossings(m)[0][0] == pytest.approx(Ec, rel=0.01)


def test_stark_stage_writes_its_data_and_page(tmp_path):
    import json

    from helpers import make_workspace, run_pipeline
    root = make_workspace(tmp_path / "ws")
    r = run_pipeline(["stark", "Na", "--state", "15d5/2", "--mj", "3/2", "--n-range", "2", "--points", "12"], root)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((root / ".qali" / "data" / "Na_stark.json").read_text(encoding="utf-8"))
    assert d["state"] == "15d5/2" and d["mj"] == 1.5 and d["basis"]["size"] > 30
    assert len(d["fields_V_cm"]) == 12 == len(d["energies_ghz"]) == len(d["character"])
    assert set(d["polarizability"]["by_mj"]) == {"1/2", "3/2", "5/2"} and d["polarizability"]["alpha0"] > 0
    html = (root / ".qali" / "plots" / "Na" / "Na_stark.html").read_text(encoding="utf-8")
    assert "$!" not in html and '"state":"15d5/2"' in html
    bad = run_pipeline(["stark", "Na", "--state", "15d5/2", "--mj", "7/2"], root)
    assert bad.returncode != 0 and "above j" in bad.stderr


def test_stark_figure_from_a_computed_map(tmp_path):
    from qali.core.fileio import write_json
    from qali.figures import templates
    from qali.pipelines.stark import compute
    d = compute("Rb", 30, 0, 1, 1, energies=_energies("Rb"), points=20, n_range=2, progress=lambda m: None)
    write_json(tmp_path / "Rb_stark.json", d)
    r = templates.stark({"crossings": True}, "Rb", str(tmp_path), {})
    assert r.panels[0].colorbar == r"$|\langle 30S_{1/2}|\mu\rangle|^2$" and any("alpha0" in n for n in r.notes)
    assert templates.stark({"cbar_label": "weight"}, "Rb", str(tmp_path), {}).panels[0].colorbar == "weight"


# ---------------------------------------------------------------- AC (Floquet) maps

def _cs40s():
    return build_basis(_energies("Cs"), "Cs", 40, 0, 1, 1, 37, 43, 8)


def test_floquet_matches_arc_shirley_method():
    # ARC 3.x ShirleyMethod(Caesium): defineBasis(40, 0, 0.5, 0.5, 0, 37, 43, 8), fn = 3, 5 GHz;
    # -targetShifts at 1.5 and 3 V/cm peak amplitude, MHz
    m = floquet_map(_cs40s(), 5.0, [1.5, 3.0], sidebands=3)
    k = np.argmax(m.character, axis=1)
    shift = m.energies_ghz[[0, 1], k] * 1e3
    assert shift == pytest.approx([-6.0345, -24.1713], rel=3e-4)


def test_floquet_weak_field_is_the_dynamic_polarisability():
    b = _cs40s()
    assert dynamic_polarizability(b, 1e-6) == pytest.approx(polarizability(b), rel=1e-9)
    a = dynamic_polarizability(b, 5.0)
    m = floquet_map(b, 5.0, [0.3], sidebands=2, window_ghz=4.0)
    e = m.energies_ghz[0, np.argmax(m.character[0])]
    assert e * 1e3 == pytest.approx(-0.25 * a * 0.3 ** 2, rel=2e-3)


def test_floquet_parity_split_keeps_every_quasienergy():
    b = _cs40s()
    a = floquet_map(b, 5.0, [2.0], sidebands=2, window_ghz=20.0)
    c = floquet_map(b, 5.0, [2.0], sidebands=2, window_ghz=20.0, dc_V_cm=1e-14)    # no split
    assert np.allclose(a.energies_ghz, c.energies_ghz, atol=1e-9) and np.allclose(a.character, c.character, atol=1e-9)


def test_ac_stark_stage_and_figure(tmp_path):
    import json

    from helpers import make_workspace, run_pipeline

    from qali.figures import templates
    root = make_workspace(tmp_path / "ws")
    r = run_pipeline(["stark", "Na", "--state", "15s1/2", "--n-range", "2", "--points", "8",
                      "--ac-frequency", "20", "--field-max", "50"], root)
    assert r.returncode == 0, r.stdout + r.stderr
    data = root / ".qali" / "data"
    d = json.loads((data / "Na_stark.json").read_text(encoding="utf-8"))
    assert d["mode"] == "ac" and d["ac"]["frequency_ghz"] == 20 and d["ac"]["sidebands"] == 2
    assert d["window_ghz"] == pytest.approx(30.0) and len(d["quadratic_ghz"]) == 8
    assert d["ac"]["alpha_dynamic"]["alpha0"] != d["polarizability"]["alpha0"]
    res = templates.stark({}, "Na", str(data), {})
    assert "F_\\mathrm{ac}" in res.panels[0].xlabel and any("alpha0(omega)" in n for n in res.notes)
