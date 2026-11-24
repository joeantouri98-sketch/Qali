"""Pair potentials, C3 and C6 of two alkali Rydberg atoms (qali.physics.rydberg_stark.pair).

References from ARC 3.x (Šibalić et al., CPC 220, 319 (2017)), PairStateInteractions with
Rubidium85(): getC6perturbatively(theta, 0, 5, 25e9) and its degenerate version, and
defineBasis(0, 0, 4, 5, 25e9) + diagonalise.  ARC quotes -C6 (V = -C6/R⁶); here ΔE = C6/R⁶.
"""
import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation.exact import MARINESCU
from qali.physics.rydberg_stark.defects import RydbergEnergies
from qali.physics.rydberg_stark.pair import angular_weights, dispersion, pair_basis, pair_map, single_states
from qali.physics.rydberg_stark.stark import build_basis


def _energies(sp):
    from qali.pipelines.dhf import nist_inputs
    ip, nist = nist_inputs(sp, str(reference_path("NIST Levels")))
    return RydbergEnergies(sp, nist, ip, MARINESCU[sp]["alphaC"])


def _single(n, l, j2, n_range, l_max):
    b = build_basis(_energies("Rb"), "Rb", n, l, j2, 1, n - n_range, n + n_range, l_max)
    return b.target, single_states(b.levels, b.R_au)


def test_angular_weights_on_axis_and_sum_rule():
    w = angular_weights(0.0)
    assert w == pytest.approx({(0, 0): -2.0, (1, -1): -1.0, (-1, 1): -1.0})
    # the tensor (1 - 3 n n) is traceless for any axis: Σ_q (-1)^q W_{q,-q} = 0
    for th in (0.3, 1.1, np.pi / 2):
        w = angular_weights(th)
        assert sum((-1) ** abs(q) * w.get((q, -q), 0.0) for q in (-1, 0, 1)) == pytest.approx(0.0, abs=1e-12)


def test_rb_60s_c6_matches_arc():
    t, ss = _single(60, 0, 1, 5, 1)
    d = dispersion(ss, t, t, 1, 1, ecut_ghz=25.0)
    assert d.c6_chosen == pytest.approx(138.88275, rel=3e-4)
    assert sorted(d.c6_ghz_um6) == pytest.approx([137.37158, 138.88275, 138.88275, 143.41628], rel=3e-4)
    assert d.c3_chosen == 0.0
    top = d.channels[0]
    assert top["pair"] == "59p3/2 + 60p3/2" and 0.45 < top["share"] < 0.55 and top["forster_ghz"] < 0


def test_rb_60d_c6_angle_matches_arc():
    t, ss = _single(60, 2, 5, 5, 3)
    got = [dispersion(ss, t, t, 5, 5, th, ecut_ghz=25.0).c6_chosen for th in (0.0, np.pi / 4, np.pi / 2)]
    assert got == pytest.approx([-213.16953, -99.84282, -82.55858], rel=3e-4)


def test_rb_60s_pair_potential_matches_arc():
    t, ss = _single(60, 0, 1, 4, 5)
    pb = pair_basis(ss, t, t, 1, 1, 0.0, 25.0)
    assert len(pb.pairs) == 1617                                  # ARC's basis
    m = pair_map(pb, [4.0, 2.5], window_ghz=3.0)
    k = np.argmax(m.character, axis=1)
    assert m.energies_ghz[[0, 1], k] == pytest.approx([0.0335280, 0.4894219], rel=1e-3)
    assert m.character[[0, 1], k] == pytest.approx([0.97735, 0.75943], abs=1e-3)


def test_pair_stage_resonant_and_figure(tmp_path):
    import json

    from helpers import make_workspace, run_pipeline

    from qali.figures import templates
    root = make_workspace(tmp_path / "ws")
    r = run_pipeline(["pair", "Na", "--state", "20s1/2", "--state2", "20p3/2", "--mj2", "3/2",
                      "--n-range", "2", "--points", "12"], root)
    assert r.returncode == 0, r.stdout + r.stderr
    data = root / ".qali" / "data"
    d = json.loads((data / "Na_pair.json").read_text(encoding="utf-8"))
    assert d["state1"] == "20s1/2" and d["state2"] == "20p3/2" and d["c3"]["resonant"]
    assert len(d["R_um"]) == 12 == len(d["energies_ghz"]) and d["basis"]["symmetric"] > 0
    bl = d["blockade"]
    assert bl["from_c3"] and bl["R_perturbative_um"] == pytest.approx((abs(d["c3"]["chosen"]) * 1e3) ** (1 / 3))
    html = (root / ".qali" / "plots" / "Na" / "Na_pair.html").read_text(encoding="utf-8")
    assert "$!" not in html and '"state2":"20p3/2"' in html
    res = templates.pair({}, "Na", str(data), {})
    assert any("Resonant" in n for n in res.notes) and "20P" in res.panels[0].colorbar
    bad = run_pipeline(["pair", "Na", "--state", "20s1/2", "--mj", "3/2"], root)
    assert bad.returncode != 0 and "above j" in bad.stderr
