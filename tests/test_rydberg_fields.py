"""Two atoms in an electric field (Förster maps, Stark-dressed pair states), Stark maps with a
magnetic field (parallel and crossed), and the large-j 3j symbols they need."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics.atomic_structure.angular import _threej2_exact, threej2
from qali.physics.rydberg_stark import forster as FO
from qali.physics.rydberg_stark import pair as P
from qali.physics.rydberg_stark import stark as S
from qali.physics.rydberg_stark import zeeman as ZM
from qali.pipelines.stark import energies_for


@pytest.fixture(scope="module")
def rb():
    return energies_for("Rb", str(reference_path("NIST Levels")))


# ---------------------------------------------------------------- 3j symbols at large j

def test_threej_large_j_matches_closed_form():
    for j2 in (117, 125, 201):                         # l = 58 and above overflowed the float formula
        for m2 in (1, 51, j2):
            ref = (-1) ** ((j2 - m2) // 2) * (m2 / 2) / math.sqrt(j2 / 2 * (j2 / 2 + 1) * (j2 + 1))
            assert threej2(j2, 2, j2, -m2, 0, m2) == pytest.approx(ref, rel=1e-12)
    assert threej2(7, 5, 4, 3, -1, -2) == pytest.approx(_threej2_exact(7, 5, 4, 3, -1, -2), abs=1e-15)


# ---------------------------------------------------------------- Förster maps and dressed pairs

def test_dressed_states_reduce_to_field_free_and_give_permanent_dipoles(rb):
    full = S.build_basis(rb, "Rb", 20, 0, 1, 1, 18, 22, 21)
    atom = FO.Atom.from_levels(full.levels, full.R_au)
    keep = [i for i, lv in enumerate(full.levels) if lv.l <= 3]
    a = keep.index(full.target)
    plain = P.single_states([full.levels[i] for i in keep], full.R_au[np.ix_(keep, keep)])
    d0 = P.dispersion(plain, a, a, 1, 1)
    dz = P.dispersion(FO.dressed_states(atom, 0.0, keep), a, a, 1, 1)
    assert dz.c6_chosen == pytest.approx(d0.c6_chosen, rel=1e-10)
    # in a field: first-order C3 = μ² (1 - 3cos²θ), μ = -dE/dF of the state
    F, h = 5.0, 1e-3
    k = atom.block(1)
    r = int(np.flatnonzero(k == full.target)[0])

    def E(f):
        e, v = atom.dress(1, f)
        return e[int(np.argmax(v[r] ** 2))]
    mu = -(E(F + h) - E(F - h)) / (2 * h) / FO.F_UNIT
    d = P.dispersion(FO.dressed_states(atom, F, keep), a, a, 1, 1)
    assert d.c3_chosen == pytest.approx(-2 * mu ** 2 * P.C3_UNIT, rel=1e-4)


def test_forster_map_finds_the_rb_37p_resonance(rb):
    # Rb 37P3/2 + 37P3/2 -> 37S + 38S: ~1.79 V/cm (Ryabtsev et al., PRL 104, 073003 (2010); ARC 1.7897)
    b = S.build_basis(rb, "Rb", 37, 1, 3, 1, 35, 39, 12)
    atom = FO.Atom.from_levels(b.levels, b.R_au)
    fm = FO.forster_map(atom, b.target, 1, atom, b.target, 1, np.linspace(1.5, 2.1, 13), window_ghz=0.5)
    top = fm.resonances[0]
    assert top["pair"] == "37s1/2 +1/2 + 38s1/2 +1/2"
    assert top["field_V_cm"] == pytest.approx(1.79, abs=0.01)
    assert abs(top["c3_ghz_um3"]) == pytest.approx(1.04, abs=0.03)         # √2 <37S 38S|V R³|37P 37P>


def test_pair_stage_in_a_field(tmp_path):
    from qali.pipelines import pair as stage
    root = tmp_path
    rc = stage.main(["Na", "--state", "14s1/2", "--field", "200", "--forster-max", "300", "--forster-points", "11",
                     "--points", "8", "--n-range", "2", "--data-dir", str(root / "data"),
                     "--output-dir", str(root / "plots")])
    assert rc == 0
    d = json.loads((root / "data" / "Na_pair__14s1-2_14s1-2_mj1-2_1-2_F200Vcm_forster300Vcm.json").read_text("utf-8"))
    assert d["field_V_cm"] == 200 and d["dressing"]["character"][0] > 0.99
    assert d["c3"]["resonant"] and d["c3"]["chosen"] < 0                      # permanent dipoles, θ = 0
    assert len(d["forster"]["fields_V_cm"]) == 11 and len(d["forster"]["target_ghz"]) == 11
    assert (root / "plots" / "Na" / "Na_pair__14s1-2_14s1-2_mj1-2_1-2_F200Vcm_forster300Vcm.html").exists()


# ---------------------------------------------------------------- Stark maps with a magnetic field

def test_zeeman_matrices_are_consistent_with_the_dipole_phases(rb):
    base = S.build_basis(rb, "Rb", 10, 1, 3, 1, 9, 11, 3)
    mm = max(lv.j2 for lv in base.levels)
    zb = ZM.zeeman_basis(base, 1, 0.0, 90.0, mm)
    Jx = ZM.magnetic_matrix(zb.levels, zb.m2s, 0.0, 1.0, 1.0, 1.0)
    Jz = ZM.magnetic_matrix(zb.levels, zb.m2s, 1.0, 0.0, 1.0, 1.0)
    assert np.allclose(Jz, np.diag(zb.m2s / 2))
    Jp = 2 * np.where(zb.m2s[:, None] - zb.m2s[None, :] == 2, Jx, 0.0)
    Jy = (Jp - Jp.T) / 2                               # i J_y, real
    c = lambda A, B: A @ B - B @ A                     # noqa: E731
    # Σ_{x,y} [J_i, [J_i, z]] = 2 z for the vector operator z
    assert np.max(np.abs(c(Jx, c(Jx, zb.Z_au)) - c(Jy, c(Jy, zb.Z_au)) - 2 * zb.Z_au)) < 1e-9


def test_zeeman_weak_field_and_rotation(rb):
    base = S.build_basis(rb, "Rb", 10, 1, 3, 1, 9, 11, 4, r2=True)
    gl = ZM.g_l(rb.rm_cm)
    for m2 in (-3, -1, 1, 3):
        zb = ZM.zeeman_basis(base, m2, 1.0, 0.0, rydberg_cm=rb.rm_cm)
        w, v = np.linalg.eigh(zb.h0())
        k = int(np.argmax((zb.tvec() @ v) ** 2))
        gj = gl * (15 / 4 - 3 / 4 + 2) / (2 * 15 / 4) + ZM.G_S * (15 / 4 + 3 / 4 - 2) / (2 * 15 / 4)
        assert w[k] == pytest.approx(gj * m2 / 2 * ZM.MU_B_GHZ_PER_G, rel=1e-4)
    # zero electric field: the spectrum does not depend on B's direction (diamagnetic term included)
    mm = max(lv.j2 for lv in base.levels)
    spec = [np.sort(np.linalg.eigvalsh(ZM.zeeman_basis(base, 1, 2e5, a, mm, rb.rm_cm).h0())) for a in (30.0, 90.0, 147.0)]
    assert max(np.max(np.abs(s - spec[0])) for s in spec) < 1e-7
    # the chosen m_j is taken along B: an F = 0 eigenstate
    zb = ZM.zeeman_basis(base, 3, 50.0, 60.0, None, rb.rm_cm)
    assert np.max((zb.tvec() @ np.linalg.eigh(zb.h0())[1]) ** 2) > 0.9999


def test_diamagnetic_shift_of_an_s_state(rb):
    b = S.build_basis(rb, "Rb", 30, 0, 1, 1, 28, 32, 10, r2=True)
    b0 = S.build_basis(rb, "Rb", 30, 0, 1, 1, 28, 32, 10)
    for B in (100.0, 500.0):
        e = [np.linalg.eigh(ZM.zeeman_basis(x, 1, B, 0.0, rydberg_cm=rb.rm_cm).h0()) for x in (b, b0)]
        k = [int(np.argmax(v[x.target] ** 2)) for (w, v), x in zip(e, (b, b0))]
        dia = e[0][0][k[0]] - e[1][0][k[1]]
        first = ZM._dia_ghz(B) * (2 / 3) * b.R2_au[b.target, b.target]           # (B²/8)(2/3)<r²>
        assert dia == pytest.approx(first, rel=2e-3)


def test_parallel_field_without_j_mixing_reproduces_arc(rb):
    # ARC StarkMap, Rb85 30S1/2 m 1/2, n 26-34, l <= 20, Bz = 0.05 T: shift by B at 0 and 40 V/cm,
    # 0.700623 and 0.686583 GHz.  ARC keeps only the diagonal of the Zeeman term (no j mixing, no
    # diamagnetism); with that the map agrees.
    base = S.build_basis(rb, "Rb", 30, 0, 1, 1, 26, 34, 20)
    zb = ZM.zeeman_basis(base, 1, 500.0, 0.0)
    assert zb.H0_extra is not None
    zb.H0_extra = np.diag(np.diag(zb.H0_extra))
    z0 = ZM.zeeman_basis(base, 1, 0.0, 0.0)
    s1, s0 = S.stark_map(zb, [0.0, 40.0]), S.stark_map(z0, [0.0, 40.0])
    for k, ref in enumerate((0.700623, 0.686583)):
        i, j = int(np.argmax(s1.character[k])), int(np.argmax(s0.character[k]))
        assert s1.energies_ghz[k, i] - s0.energies_ghz[k, j] == pytest.approx(ref, abs=5e-6)


def test_stark_stage_crossed_fields(tmp_path):
    from qali.pipelines import stark as stage
    rc = stage.main(["Na", "--state", "12s1/2", "--field-max", "50", "--points", "6", "--n-range", "1",
                     "--b-field", "200", "--b-angle", "90", "--data-dir", str(tmp_path / "data"),
                     "--output-dir", str(tmp_path / "plots")])
    assert rc == 0
    d = json.loads((tmp_path / "data" / "Na_stark__12s1-2_mj1-2_B200G_a90.json").read_text("utf-8"))
    mg = d["magnetic"]
    assert mg["crossed"] and mg["m_max"] == 2.5 and mg["diamagnetic"]
    assert max(d["character"][0]) > 0.999                    # the Zeeman sublevel along B
    assert mg["zeeman_shift_ghz"] == pytest.approx(ZM.G_S / 2 * ZM.MU_B_GHZ_PER_G * 200, rel=1e-3)


# ---------------------------------------------------------------- pages kept for Generate plot from...

def test_stored_page_round_trip(tmp_path):
    from qali.pipelines.page_store import page_record, pages_only, page_stem
    rec = page_record("hyperfine_page.html", "test page", '{"x":1}', file="Rb_hf_page")
    p = tmp_path / "data" / "Rb_hyperfine.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"element": "Rb", "page": rec}), encoding="utf-8")
    assert page_stem(str(p)) == "Rb_hf_page" and page_stem(str(p), "mine.html") == "mine"
    assert pages_only("hyperfine", [str(p)], str(tmp_path / "plots")) == 0
    html = (tmp_path / "plots" / "Rb" / "Rb_hf_page.html").read_text(encoding="utf-8")
    assert '{"x":1}' in html and "test page" in html
    old = tmp_path / "data" / "Rb_old.json"
    old.write_text(json.dumps({"element": "Rb"}), encoding="utf-8")
    assert pages_only("hyperfine", [str(old)], str(tmp_path / "plots")) == 1


def test_replot_lists_runs_of_stages_that_keep_pages(tmp_path):
    from qali.gui.replot_dialog import list_runs
    from qali.gui.stages import STAGE_BY_ID
    from qali.pipelines.page_store import page_record
    d = tmp_path
    (d / "Rb_feshbach.json").write_text("{}", encoding="utf-8")                  # the stage's input, not a run
    (d / "Rb_feshbach_out.json").write_text(json.dumps({"element": "Rb", "page": page_record(
        "feshbach_page.html", "t", "{}", {"PHYS_JS": "feshbach_physics.js"}, file="Rb_feshbach")}), encoding="utf-8")
    (d / "Rb_blackbody.json").write_text(json.dumps({"element": "Rb"}), encoding="utf-8")
    runs = list_runs("feshbach_out", d, "Rb")
    assert [r.path.name for r in runs] == ["Rb_feshbach_out.json"] and runs[0].summary == "page kept"
    assert list_runs("blackbody", d, "Rb")[0].summary.startswith("no page kept")
    for sid, stem in (("blackbody", "blackbody"), ("hyperfine", "hyperfine"), ("feshbach", "feshbach_out")):
        assert STAGE_BY_ID[sid]["replot"]["stem"] == stem


def test_pair_states_in_a_magnetic_field(rb):
    b = S.build_basis(rb, "Rb", 37, 1, 3, 1, 35, 39, 12, r2=True)
    keep = [i for i, lv in enumerate(b.levels) if lv.l <= 3]
    a = keep.index(b.target)
    plain = P.single_states([b.levels[i] for i in keep], b.R_au[np.ix_(keep, keep)])
    tiny = FO.Atom.from_levels(b.levels, b.R_au, 1e-6, b.R2_au, rb.rm_cm)
    assert P.dispersion(FO.dressed_states(tiny, 0.0, keep), a, a, 1, 1, deg_tol_ghz=1e-3).c6_chosen == \
        pytest.approx(P.dispersion(plain, a, a, 1, 1).c6_chosen, rel=1e-6)
    # a pair of S states moves by g_S (m1 + m2) μ_B B plus the two diamagnetic shifts
    atom = FO.Atom.from_levels(b.levels, b.R_au, 10.0, b.R2_au, rb.rm_cm)
    a0 = FO.Atom.from_levels(b.levels, b.R_au)
    fm = FO.forster_map(atom, b.target, 1, atom, b.target, 1, [0.0], window_ghz=0.2, c3_rel=0.0)
    f0 = FO.forster_map(a0, b.target, 1, a0, b.target, 1, [0.0], window_ghz=0.2, c3_rel=0.0)
    lab = "37s1/2 +1/2 + 38s1/2 +1/2"
    de = fm.energies_ghz[0][fm.labels[0].index(lab)] - f0.energies_ghz[0][f0.labels[0].index(lab)]
    ns = {lv.n: lv.n_star for lv in b.levels if lv.l == 0}
    dia = ZM.diamagnetic_estimate_ghz(ns[37], 0, 10.0) + ZM.diamagnetic_estimate_ghz(ns[38], 0, 10.0)
    assert de == pytest.approx(ZM.G_S * ZM.MU_B_GHZ_PER_G * 10.0 + dia, rel=2e-4)
    # the Förster resonance moves with the Zeeman shifts of the defect (1.79 -> ~1.71 V/cm at 10 G)
    fr = FO.forster_map(atom, b.target, 1, atom, b.target, 1, np.linspace(1.5, 2.0, 21), window_ghz=0.5)
    assert fr.resonances[0]["pair"] == lab and fr.resonances[0]["field_V_cm"] == pytest.approx(1.706, abs=0.01)


def test_pair_stage_in_a_magnetic_field(tmp_path):
    from qali.pipelines import pair as stage
    rc = stage.main(["Na", "--state", "14s1/2", "--b-field", "20", "--points", "6", "--n-range", "2",
                     "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "plots")])
    assert rc == 0
    d = json.loads((tmp_path / "data" / "Na_pair__14s1-2_14s1-2_mj1-2_1-2_B20G.json").read_text("utf-8"))
    assert d["b_gauss"] == 20 and d["field_V_cm"] == 0 and len(d["c6"]["manifold"]) == 1


def test_forster_map_does_not_depend_on_eigenvector_signs(rb, monkeypatch):
    # LAPACK builds (Windows MKL/OpenBLAS, Linux) return eigenvectors with different signs
    import scipy.linalg as sl
    b = S.build_basis(rb, "Rb", 37, 1, 3, 1, 35, 39, 12)
    rng = np.random.default_rng(1)

    def flipped(*a, **k):
        E, V = sl.eigh(*a, **k)
        return E, V * rng.choice([-1.0, 1.0], V.shape[1])
    monkeypatch.setattr(FO, "eigh", flipped)
    for _ in range(3):
        atom = FO.Atom.from_levels(b.levels, b.R_au)
        r = FO.forster_map(atom, b.target, 1, atom, b.target, 1, np.linspace(1.5, 2.1, 13), window_ghz=0.5).resonances[0]
        assert r["field_V_cm"] == pytest.approx(1.79, abs=0.01) and abs(r["c3_ghz_um3"]) == pytest.approx(1.04, abs=0.03)


def test_figure_views_of_stark_and_pair_runs(tmp_path):
    from qali.figures import templates as T
    from qali.pipelines import pair as pstage
    from qali.pipelines import stark as sstage
    data = tmp_path / "data"
    assert sstage.main(["Na", "--state", "12s1/2", "--field-max", "200", "--points", "8", "--n-range", "1",
                        "--ramp-time", "1", "--data-dir", str(data), "--output-dir", str(tmp_path / "plots")]) == 0
    for view in ("map", "state", "ramp"):
        assert T.stark({"view": view}, "Na", str(data), {}).panels
    with pytest.raises(T.TemplateError):
        T.stark({"view": "ionisation"}, "Na", str(data), {})
    assert pstage.main(["Na", "--state", "14s1/2", "--field", "200", "--forster-max", "300", "--forster-points", "6",
                        "--points", "6", "--n-range", "2", "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "plots")]) == 0
    for view in ("potentials", "loglog", "blockade", "forster"):
        assert T.pair({"view": view}, "Na", str(data), {}).panels
    # in a field the asymptote is C3/R³ + C6/R⁶ (permanent dipoles), not ±C3/R³ alone
    d = json.loads((data / "Na_pair.json").read_text(encoding="utf-8"))
    laws = T._pair_law(d)
    R = np.array([3.0])
    assert len(laws) == 1 and laws[0][1](R) == pytest.approx(d["c3"]["chosen"] / 27 + d["c6"]["chosen"] / 729)
