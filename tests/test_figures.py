"""Publication figures: data catalogue, safe expressions, every template, save formats, scripts, recipes, CLI."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from qali.figures import catalog, recipe as R, spec as S, style, templates  # noqa: E402

SPECIES = ["Li", "Na", "K", "Rb", "Cs", "Fr"]


@pytest.fixture(scope="module")
def dd(data_dir) -> str:
    return str(data_dir)


def _recipe(template, species, **params):
    r = R.new_recipe(template, species)
    r["params"].update(params)
    return r


# ------------------------------------------------------------------ catalogue
def test_catalogue_finds_species_and_tables(dd):
    sp = catalog.available_species(dd)
    for s in SPECIES:
        assert s in sp
    tables = catalog.species_tables("Rb", dd)
    stages = {t.stage for t in tables}
    assert {"rydberg", "lifetimes", "polarizability"} <= stages
    for t in tables:
        lengths = {len(v) for v in t.columns.values()}
        assert len(lengths) == 1 and t.rows == lengths.pop()   # every column of a table is the same length
        assert t.numeric_columns()
    assert catalog.find_table("Rb", dd, tables[0].key) is tables[0] or \
        catalog.find_table("Rb", dd, tables[0].key).key == tables[0].key
    assert catalog.find_table("Rb", dd, "nope:nothing") is None


def test_units_and_labels():
    assert catalog.unit_of("energy_eV") == "eV"
    assert "eV" in catalog.label_for("energy_eV")


# ------------------------------------------------------------------ expressions
def test_expressions_compute():
    cols = {"a": np.array([1.0, 4.0, 9.0]), "b c": np.array([2.0, 2.0, 2.0])}
    assert np.allclose(catalog.evaluate("sqrt(a) + 1", cols), [2, 3, 4])
    assert np.allclose(catalog.evaluate('col("b c") * a', cols), [2, 8, 18])
    assert list(catalog.evaluate("(a > 2) & (a < 9)", cols)) == [False, True, False]
    assert np.allclose(catalog.evaluate("log10(a * 10) - pi + pi", cols), np.log10(cols["a"] * 10))


@pytest.mark.parametrize("bad", [
    "__import__('os')", "a.__class__", "open('x')", "[x for x in a]", "lambda: 1", "a[0]",
    "exec('1')", "nope + 1", "a if a else a", "globals()",
])
def test_expressions_reject_unsafe_or_unknown(bad):
    with pytest.raises(catalog.ExpressionError):
        catalog.evaluate(bad, {"a": np.array([1.0])})


# ------------------------------------------------------------------ style
def test_sizes_follow_the_journal():
    w, h = style.figure_size({"journal": "aps", "columns": 1, "aspect": None, "width_cm": None, "height_cm": None}, 0.75)
    assert w == pytest.approx(3.375) and h == pytest.approx(3.375 * 0.75, abs=1e-3)
    w2, _ = style.figure_size({"journal": "aps", "columns": 2, "aspect": 0.5, "width_cm": None, "height_cm": None})
    assert w2 == pytest.approx(7.0)
    w3, h3 = style.figure_size({"journal": "aps", "columns": 1, "aspect": None, "width_cm": 10.0, "height_cm": 5.0})
    assert w3 == pytest.approx(10 / 2.54, abs=1e-3) and h3 == pytest.approx(5 / 2.54, abs=1e-3)


def test_tex_safe_escapes_outside_math_only():
    assert style.tex_safe("a_b 5% $x_1$", True) == r"a\_b 5\% $x_1$"
    assert style.tex_safe("a_b", False) == "a_b"


# ------------------------------------------------------------------ templates
CASES = [
    ("grotrian", {}),
    ("grotrian", {"j_split": False, "wavelengths": False, "energy_unit": "cm-1"}),
    ("quantum_defects", {}),
    ("scaling", {}),
    ("theory_vs_expt", {}),
    ("alpha", {}),
]


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("template,params", CASES)
def test_every_template_resolves_and_draws(dd, species, template, params):
    spec = R.resolve(_recipe(template, species, **params), dd)
    assert spec.panels and all(p.ops for p in spec.panels)
    fig = S.draw(spec)
    assert fig.get_axes()
    import matplotlib.pyplot as plt
    plt.close("all")


def test_grotrian_honours_chosen_levels_and_lines(dd):
    lv, lines, IE = templates.grotrian_choices("Rb", dd)
    chosen = templates.default_grotrian_levels(lv, IE, count=8)
    between = templates.lines_between(lines, set(chosen))
    assert between and all(ln["upper_label"] in chosen and ln["lower_label"] in chosen for ln in between)
    pick = [f"{between[0]['upper_label']}>{between[0]['lower_label']}"]
    spec = R.resolve(_recipe("grotrian", "Rb", levels=chosen, lines=pick), dd)
    head, rows = S.data_table(spec)
    assert head[0] == "item"
    assert [r[1] for r in rows if r[0] == "level"] == chosen
    assert [(r[1], r[2]) for r in rows if r[0] == "line"] == [(between[0]["upper_label"], between[0]["lower_label"])]


def test_scaling_exponents_are_near_three(dd):
    spec = R.resolve(_recipe("scaling", "Rb", quantity="lifetime"), dd)
    labels = [op["kw"].get("label", "") for p in spec.panels for op in p.ops]
    import re
    exps = [float(m.group(1)) for s in labels for m in [re.search(r"([0-9]+\.[0-9]+)\s*\}?\$?\s*$", s)] if m]
    assert exps and all(2.7 < e < 3.2 for e in exps)


def test_builder_series_and_errors(dd):
    t = next(t for t in catalog.species_tables("Rb", dd) if len(t.numeric_columns()) >= 2)
    x, y = t.numeric_columns()[:2]
    r = _recipe("builder", "Rb", series=[{"table": t.key, "x": x, "y": y, "kind": "markers", "label": "s"}])
    spec = R.resolve(r, dd)
    assert spec.panels[0].legend
    with pytest.raises(templates.TemplateError):
        R.resolve(_recipe("builder", "Rb"), dd)
    bad = _recipe("builder", "Rb", series=[{"table": t.key, "x": "__import__('os')", "y": y}])
    with pytest.raises(templates.TemplateError):
        R.resolve(bad, dd)
    with pytest.raises(templates.TemplateError):
        R.resolve(_recipe("builder", "Rb", series=[{"table": "no:table", "x": "a", "y": "b"}]), dd)


def test_missing_stage_says_which(tmp_path):
    with pytest.raises(templates.TemplateError):
        R.resolve(_recipe("grotrian", "Rb"), str(tmp_path))


# ------------------------------------------------------------------ saving, scripts, recipes
@pytest.mark.parametrize("ext", S.SAVE_FORMATS)
def test_save_every_format(dd, tmp_path, ext):
    spec = R.resolve(_recipe("quantum_defects", "Na"), dd)
    out = tmp_path / f"f.{ext}"
    S.save(spec, str(out), dpi=150)
    assert out.stat().st_size > 1000


def test_axes_overrides_apply(dd):
    r = _recipe("scaling", "Rb")
    r["axes"].update({"ylabel": "custom y", "title": "T", "ymin": 1.0, "legend": "off"})
    spec = R.resolve(r, dd)
    p = spec.panels[0]
    assert p.ylabel == "custom y" and p.title == "T" and p.ylim[0] == 1.0
    assert all(q.legend is None for q in spec.panels)


def test_recipe_roundtrip(tmp_path):
    r = _recipe("grotrian", "Cs", levels=["6s", "6p3/2"])
    path = tmp_path / "x.qfig"
    R.save_recipe(r, str(path))
    back = R.load_recipe(str(path))
    assert back["params"]["levels"] == ["6s", "6p3/2"] and back["species"] == "Cs"
    path.write_text('{"qfig": 1, "template": "nope"}')
    with pytest.raises(ValueError):
        R.load_recipe(str(path))
    path.write_text('{"template": "grotrian"}')
    with pytest.raises(ValueError):
        R.load_recipe(str(path))


def test_old_recipe_gets_missing_keys():
    r = R.normalized({"qfig": 1, "template": "alpha", "species": "K", "axes": {"title": "x"}})
    assert r["axes"]["title"] == "x" and "legend" in r["axes"] and "palette" in r["style"]


@pytest.mark.parametrize("template", ["grotrian", "scaling", "alpha"])
def test_script_runs_standalone(dd, tmp_path, template):
    spec = R.resolve(_recipe(template, "Rb"), dd)
    spec.style["usetex"] = False
    script = tmp_path / "fig.py"
    script.write_text(S.to_script(spec, "fig.png"), encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    out = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, capture_output=True, text=True,
                         timeout=180, env=env)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "fig.png").stat().st_size > 1000


@pytest.mark.skipif(not style.latex_available(raster=False), reason="LaTeX is not installed")
def test_latex_pdf(dd, tmp_path):
    r = _recipe("quantum_defects", "Rb")
    r["style"]["usetex"] = True
    spec = R.resolve(r, dd)
    out = tmp_path / "tex.pdf"
    try:
        S.save(spec, str(out))
    except S.RenderError as exc:
        pytest.skip(str(exc))                       # LaTeX present but a package (cm-super) is missing
    assert out.stat().st_size > 1000


# ------------------------------------------------------------------ CLI
def test_cli_draws_a_recipe(dd, tmp_path):
    from qali.pipelines import figure

    path = tmp_path / "g.qfig"
    R.save_recipe(_recipe("grotrian", "Rb"), str(path))
    out = tmp_path / "g.png"
    assert figure.main([str(path), "--species", "Cs", "--data-dir", dd, "--out", str(out), "--dpi", "100",
                        "--script", str(tmp_path / "g.py")]) == 0
    assert out.exists() and (tmp_path / "g.py").exists()
    assert figure.main([str(tmp_path / "missing.qfig")]) == 1
    assert figure.main([str(path), "--data-dir", str(tmp_path), "--out", str(out)]) == 1


# ------------------------------------------------------------------ named runs and other files
@pytest.fixture()
def runs_dir(dd, tmp_path):
    """Na's outputs plus a lifetimes run saved as 'slow' (every lifetime doubled) and a copy elsewhere."""
    import json
    import shutil

    d = tmp_path / "data"
    d.mkdir()
    for stage in ("rydberg", "lifetimes", "polarizability"):
        shutil.copy(os.path.join(dd, f"Na_{stage}.json"), d / f"Na_{stage}.json")
    doc = json.loads((d / "Na_lifetimes.json").read_text(encoding="utf-8"))
    for r in doc["lifetimes"]:
        r["lifetime_ns"] *= 2
    (d / "Na_lifetimes__slow.json").write_text(json.dumps(doc), encoding="utf-8")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "Na_lifetimes.json").write_text(json.dumps(doc), encoding="utf-8")
    return d, other / "Na_lifetimes.json"


def _scaling_y(spec):
    return np.concatenate([np.asarray(op["args"][1], dtype=float) for op in spec.panels[0].ops
                           if op["op"] == "plot" and op["kw"].get("linestyle") == "none"])


def test_named_runs_are_listed_and_chosen(runs_dir):
    d, other = runs_dir
    assert catalog.runs("Na", "lifetimes", str(d)) == ["slow"]
    assert catalog.runs("Na", "rydberg", str(d)) == []
    assert "Na" in catalog.available_species(str(d))
    assert templates.inputs_of("scaling", {"quantity": "lifetime"}) == ["rydberg", "lifetimes"]
    assert templates.inputs_of("scaling", {"quantity": "bbr_lifetime"}) == ["rydberg", "blackbody"]
    base = _scaling_y(R.resolve(_recipe("scaling", "Na"), str(d)))
    r = _recipe("scaling", "Na")
    r["inputs"] = {"lifetimes": "slow"}
    spec = R.resolve(r, str(d))
    assert np.allclose(_scaling_y(spec), 2 * base)
    assert any("slow" in n for n in spec.notes)
    r["inputs"] = {"lifetimes": str(other)}                     # a file anywhere
    assert np.allclose(_scaling_y(R.resolve(r, str(d))), 2 * base)
    r["inputs"] = {"lifetimes": "nope"}
    with pytest.raises(templates.TemplateError, match="nope"):
        R.resolve(r, str(d))
    r["inputs"] = {"lifetimes": ""}                              # back to the usual output
    assert np.allclose(_scaling_y(R.resolve(r, str(d))), base)
    assert catalog.load_json("Na", "lifetimes", str(d))["lifetimes"][0]["lifetime_ns"] == pytest.approx(
        catalog.load_json("Na", "lifetimes", str(d), run="slow")["lifetimes"][0]["lifetime_ns"] / 2)


def test_builder_lists_named_runs(runs_dir):
    d, _ = runs_dir
    keys = [t.key for t in catalog.species_tables("Na", str(d))]
    assert "lifetimes:lifetimes" in keys and "lifetimes__slow:lifetimes" in keys
    t = catalog.find_table("Na", str(d), "lifetimes__slow:lifetimes")
    assert t.run == "slow" and "(slow)" in t.title


def test_cli_input_option(runs_dir, tmp_path):
    from qali.pipelines import figure

    d, other = runs_dir
    path = tmp_path / "s.qfig"
    R.save_recipe(_recipe("scaling", "Na"), str(path))
    out = tmp_path / "s.png"
    assert figure.main([str(path), "--data-dir", str(d), "--out", str(out), "--dpi", "80",
                        "--input", "lifetimes=slow"]) == 0
    assert figure.main([str(path), "--data-dir", str(d), "--out", str(out), "--dpi", "80",
                        "--input", f"lifetimes={other}"]) == 0
    assert figure.main([str(path), "--data-dir", str(d), "--out", str(out), "--input", "lifetimes=nope"]) == 1
    assert figure.main([str(path), "--data-dir", str(d), "--out", str(out), "--input", "slow"]) == 1


def test_view_limits_of_later_panels(dd):
    r = _recipe("quantum_defects", "Rb")
    r["axes"].update({"xmin": 10.0, "xmax": 30.0, "view": {"1": {"y": [2.6, 2.7]}}})
    spec = R.resolve(r, dd)
    assert spec.panels[0].xlim == [10.0, 30.0] and spec.panels[1].ylim == [2.6, 2.7]


# ------------------------------------------------------------------ second batch: physics checks
NEW_CASES = [
    ("contributions", {"quantity": "alpha0", "view": "channels"}),
    ("contributions", {"quantity": "C6", "view": "convergence"}),
    ("contributions", {"quantity": "alpha0", "view": "budget"}),
    ("contributions", {"quantity": "alpha0", "view": "convergence", "relative": True}),
    ("branching", {}),
    ("oscillator_density", {}),
    ("oscillator_density", {"merge_j": True, "energy_unit": "au"}),
    ("blackbody", {"view": "shifts"}),
    ("blackbody", {"view": "rates"}),
    ("blackbody", {"view": "temperature"}),
    ("blackbody", {"view": "temperature", "quantity": "gamma"}),
    ("breit_rabi", {}),
    ("radial", {}),
    ("radial", {"quantity": "density", "xscale": "linear"}),
]


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("template,params", NEW_CASES)
def test_new_templates_resolve_and_draw(dd, species, template, params):
    spec = R.resolve(_recipe(template, species, **params), dd)
    assert spec.panels and spec.panels[0].ops
    S.draw(spec)
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.mark.parametrize("species", ["Li", "Na", "K", "Rb", "Cs"])
def test_feshbach_resolves(dd, species):
    spec = R.resolve(_recipe("feshbach", species), dd)
    S.draw(spec)


def test_feshbach_k39_poles(dd):
    spec = R.resolve(_recipe("feshbach", "K", channels=["|1,+1> + |1,+1>"]), dd)
    poles = [op["args"][0] for op in spec.panels[0].ops if op["op"] == "axvline"]
    assert any(abs(b - 402.6) < 2 for b in poles)              # the broad 39K resonance


@pytest.mark.parametrize("species", ["Li", "Rb", "Cs"])
@pytest.mark.parametrize("qty", ["alpha0", "C6"])
def test_contributions_add_up_to_the_stage(dd, species, qty):
    from qali.figures.templates import _polarizability_terms

    _gs, rows, total, sig = _polarizability_terms(species, dd, qty)
    assert sum(r[1] for r in rows) == pytest.approx(total, rel=1e-5)
    assert math.sqrt(sum(r[2] ** 2 for r in rows)) == pytest.approx(sig, rel=2e-3)
    head, table = R.resolve(_recipe("contributions", species, quantity=qty, top=50), dd).table
    assert sum(r[1] for r in table) == pytest.approx(total, rel=1e-5)


def test_branching_adds_to_100(dd):
    spec = R.resolve(_recipe("branching", "Cs", state="7s", top=30), dd)
    _h, rows = spec.table
    assert sum(r[1] for r in rows) == pytest.approx(100, abs=0.05)
    assert {r[0] for r in rows} == {"6p_{1/2}", "6p_{3/2}"}


def test_df_de_meets_the_continuum_for_na(dd):
    """Na 3s: df/dε of the highest bound np and of the continuum at threshold agree (no Cooper minimum there)."""
    spec = R.resolve(_recipe("oscillator_density", "Na", merge_j=True, energy_unit="au"), dd)
    _h, rows = spec.table
    bound = sorted((r for r in rows if r[0] != "continuum"), key=lambda r: r[1])
    cont = sorted((r for r in rows if r[0] == "continuum"), key=lambda r: r[1])
    assert 0.5 < bound[-1][2] / cont[0][2] < 2.0


def test_breit_rabi_zero_field_splitting(dd):
    spec = R.resolve(_recipe("breit_rabi", "Rb", unit="MHz"), dd)
    import json
    from pathlib import Path
    d = json.loads(Path(dd, "Rb_hyperfine.json").read_text(encoding="utf-8"))
    st = next(s for s in d["states"] if s["label"] == "5s1/2")
    E0 = [op["args"][1][0] for op in spec.panels[0].ops if op["op"] == "plot"]
    assert max(E0) - min(E0) == pytest.approx(st["A_MHz"] * (d["I"] + 0.5), rel=1e-9)   # ΔE_hfs = A(I + 1/2)


def test_radial_expectation_matches_qdt(dd):
    spec = R.resolve(_recipe("radial", "Na", states=["20s"]), dd)
    r1 = float(spec.notes[0].split("⟨r⟩ = ")[1].split(" a0")[0])
    lv, _IE, Ry = templates.levels("Na", dd)
    ns = templates.n_star(next(x for x in lv if x.label == "20s"), Ry)
    assert r1 == pytest.approx(1.5 * ns ** 2, rel=0.01)                  # (3n*² − l(l+1))/2


def test_bbr_temperature_sweep_is_tau(dd):
    spec = R.resolve(_recipe("blackbody", "Rb", view="temperature", states=["7s"]), dd)
    TK, tau = spec.panels[0].ops[0]["args"][:2]
    t300 = float(np.interp(300, TK, tau))
    assert 80 < t300 < 95 and tau[0] >= tau[-1]                         # Rb 7s: 88 ns, shorter when hot


def test_builder_band_residual_and_file(dd, tmp_path):
    import shutil

    ext = tmp_path / "other.json"
    shutil.copy(os.path.join(dd, "Cs_lifetimes.json"), ext)
    assert catalog.find_file_table(str(ext), "file:lifetimes").rows > 10
    r = _recipe("builder", "Rb", residual="percent", series=[
        {"species": "Rb", "table": "lifetimes:lifetimes", "x": "n", "y": "lifetime_ns", "yerr": "sigma_tau_ns",
         "where": "l == 's'", "kind": "band"},
        {"file": str(ext), "table": "file:lifetimes", "x": "n", "y": "lifetime_ns", "where": "(l == 's') & (n > 6)"}])
    spec = R.resolve(r, dd)
    assert len(spec.panels) == 2 and spec.sharex
    assert any(op["op"] == "fill_between" for op in spec.panels[0].ops)
    assert any("other.json" in n for n in spec.notes)
    script = tmp_path / "b.py"
    script.write_text(S.to_script(spec, "b.png"), encoding="utf-8")
    out = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, capture_output=True, text=True,
                         timeout=180, env={**os.environ, "MPLBACKEND": "Agg"})
    assert out.returncode == 0, out.stderr


# ------------------------------------------------------------------ spectrum and trap depth
@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("params", [{"view": "absorption"}, {"view": "emission"}, {"view": "absorption", "quantity": "od"}])
def test_spectrum_draws(dd, species, params):
    if species == "Fr" and params.get("quantity"):
        with pytest.raises(templates.TemplateError, match="vapour"):
            R.resolve(_recipe("spectrum", species, **params), dd)
        return
    spec = R.resolve(_recipe("spectrum", species, **params), dd)
    S.draw(spec)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_spectrum_absorption_peaks_at_the_d_lines(dd):
    spec = R.resolve(_recipe("spectrum", "Rb", wl_min=770, wl_max=800, labels=0), dd)
    lam, sig = (np.asarray(a) for a in spec.panels[0].ops[0]["args"][:2])
    for line in (780.241, 794.979):                                     # D2, D1 (vacuum)
        near = np.abs(lam - line) < 0.01
        assert sig[near].max() > 1e3 * np.median(sig)
    d2 = sig[np.abs(lam - 780.241) < 0.01].max()
    d1 = sig[np.abs(lam - 794.979) < 0.01].max()
    assert 1.3 < d2 / d1 < 2.7                                          # f(D2) ≈ 2 f(D1), widths alike


def test_spectrum_rb_d2_transmission_dips(dd):
    spec = R.resolve(_recipe("spectrum", "Rb", view="dlines", line="D2", T_K=320.0), dd)
    x, t = (np.asarray(a) for a in spec.panels[0].ops[0]["args"][:2])
    assert t.max() == pytest.approx(1.0, abs=1e-3) and t.min() < 0.5    # a 5 cm cell at 320 K is thick on D2
    labels = [op["kw"].get("label", "") for op in spec.panels[0].ops if op["op"] == "vlines"]
    assert len(labels) == 2 and any("85" in s for s in labels) and any("87" in s for s in labels)
    # the ⁸⁷Rb ground hyperfine splitting (6.83 GHz) separates the outer dips
    dips = x[(t < 0.99) & (np.r_[t[1:], 1] > t) & (np.r_[1, t[:-1]] > t)]
    assert dips.max() - dips.min() == pytest.approx(6.83 + 0.27, abs=0.6)


def test_spectrum_fr_falls_back_to_sigma(dd):
    spec = R.resolve(_recipe("spectrum", "Fr"), dd)
    assert "cm$^2$" in spec.panels[0].ylabel


def test_trap_sign_changes_across_the_d_lines(dd):
    spec = R.resolve(_recipe("trap", "Rb", wl_min=500, wl_max=1100), dd)
    wl, U = (np.asarray(a) for a in spec.panels[0].ops[0]["args"][:2])
    R_sc = np.asarray(spec.panels[1].ops[0]["args"][1])
    assert np.interp(1064, wl, U) < 0 < np.interp(532, wl, U)          # red: attractive, blue: repulsive
    # scattering is strongest near the D lines
    assert R_sc[np.argmin(np.abs(wl - 785))] > 100 * np.interp(1064, wl, R_sc)


# ------------------------------------------------------------------ builder: right axis, inset, broken axis, maps
_S = {"species": "Rb", "table": "lifetimes:lifetimes", "x": "n", "y": "lifetime_ns", "where": "l == 's'",
      "kind": "line", "label": "s"}


def _draw_and_script(spec, tmp_path, name):
    fig = S.draw(spec)
    script = tmp_path / f"{name}.py"
    script.write_text(S.to_script(spec, f"{name}.png"), encoding="utf-8")
    out = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, capture_output=True, text=True,
                         timeout=180, env={**os.environ, "MPLBACKEND": "Agg"})
    assert out.returncode == 0, out.stderr
    return fig


def test_builder_right_axis(dd, tmp_path):
    r = _recipe("builder", "Rb", y2label="rate", y2scale="log",
                series=[_S, {**_S, "y": "A_total_s", "on": "y2", "label": "A"}])
    spec = R.resolve(r, dd)
    assert spec.panels[0].y2 == {"label": "rate", "scale": "log", "lim": None}
    fig = _draw_and_script(spec, tmp_path, "y2")
    main = fig._qali_panels[0]
    twin = next(a for a in fig.axes if a is not main and a.bbox.bounds == main.bbox.bounds)
    assert twin.get_yscale() == "log" and twin.get_ylabel() == "rate"
    assert len(twin.get_legend().get_texts()) == 2                    # one legend for both axes


def test_builder_inset(dd, tmp_path):
    r = _recipe("builder", "Rb", inset_xlim=[6, 12], inset_ylim=[0, 500], series=[_S, {**_S, "on": "inset"}])
    spec = R.resolve(r, dd)
    fig = _draw_and_script(spec, tmp_path, "inset")
    (ins,) = fig._qali_panels[0].child_axes
    assert ins.get_xlim() == (6.0, 12.0) and ins.get_ylim() == (0.0, 500.0)


@pytest.mark.parametrize("axis,at", [("y", [3000, 20000]), ("x", [20, 40])])
def test_builder_broken_axis(dd, tmp_path, axis, at):
    spec = R.resolve(_recipe("builder", "Rb", break_axis=axis, break_at=at, series=[_S]), dd)
    assert spec.panels[0].axis_break == {"axis": axis, "at": at}
    fig = _draw_and_script(spec, tmp_path, f"break_{axis}")
    assert fig._qali_panels == [None]                                  # no zoom on a broken axis
    lo, hi = fig.axes[:2] if axis == "x" else fig.axes[1::-1]
    get = (lambda a: a.get_ylim()) if axis == "y" else (lambda a: a.get_xlim())
    assert get(lo)[1] == at[0] and get(hi)[0] == at[1]
    # each side spans its own data: the low side starts just below the smallest value, not at the full range
    smallest = 6 if axis == "x" else 0.0
    assert smallest - 0.1 * (at[0] - smallest) < get(lo)[0] <= smallest


def test_builder_break_needs_a_range_and_no_extras(dd):
    with pytest.raises(templates.TemplateError, match="range"):
        R.resolve(_recipe("builder", "Rb", break_axis="y", break_at=[None, 5], series=[_S]), dd)
    with pytest.raises(templates.TemplateError, match="combined"):
        R.resolve(_recipe("builder", "Rb", break_axis="y", break_at=[1, 5], series=[_S, {**_S, "on": "y2"}]), dd)


def test_builder_maps(dd, tmp_path):
    import json

    nx, ny = np.meshgrid(np.arange(5), np.arange(4))
    rows = [{"x": int(a), "y": int(b), "z": float(a * b)} for a, b in zip(nx.ravel(), ny.ravel())]
    f = tmp_path / "grid.json"
    f.write_text(json.dumps({"points": rows}), encoding="utf-8")
    key = catalog.file_tables(str(f))[0].key
    for kind, op in (("heatmap", "pcolormesh"), ("contour", "contourf")):
        spec = R.resolve(_recipe("builder", "Rb", series=[{"file": str(f), "table": key, "x": "x", "y": "y",
                                                           "z": "z", "kind": kind}]), dd)
        ops = [o["op"] for o in spec.panels[0].ops]
        assert ops[0] == op and spec.panels[0].colorbar == "z"
        Z = np.asarray(spec.panels[0].ops[0]["args"][2])
        assert Z.shape == (4, 5) and Z[3, 4] == 12.0                      # on the grid: z[y, x]
        fig = _draw_and_script(spec, tmp_path, kind)
        assert len(fig.axes) == 2                                         # the plot and its colour bar
    # scattered points: triangulated
    tri = {"species": "Rb", "table": "lifetimes:transitions", "x": "lower_n", "y": "upper_n", "z": "log10(fosc)",
           "kind": "heatmap", "where": "(upper_l == 'p') & (lower_l == 's') & (upper_n < 15)"}
    spec = R.resolve(_recipe("builder", "Rb", series=[tri]), dd)
    assert spec.panels[0].ops[0]["op"] == "tricontourf"


# ------------------------------------------------------------------ decay cascade
@pytest.mark.parametrize("species", SPECIES)
def test_cascade_draws_and_conserves_atoms(dd, species):
    spec = R.resolve(_recipe("cascade", species), dd)
    S.draw(spec)
    import matplotlib.pyplot as plt
    plt.close("all")
    lt = catalog.load_json(species, "lifetimes", dd)
    rows = {r["state"]: r for r in lt["lifetimes"] if r.get("channels")}
    E, _IE = templates._level_energies(species, dd)
    start = spec.notes[0].split()[1]
    P, F = templates.cascade_flux(rows, start, E)
    gs = min(P, key=lambda k: templates._energy_of(k, E))
    assert P[gs] == pytest.approx(1.0, abs=2e-3)                       # every atom ends in the ground state
    for k, v in P.items():                                              # what reaches a level leaves it
        if k not in (start, gs) and k in rows:
            assert sum(f for (u, _lo), f in F.items() if u == k) == pytest.approx(v, rel=1e-3, abs=1e-6)


def test_cascade_li_3p_branches_follow_A(dd):
    """Li 3p1/2: through 3s with A(3p→3s) / (A(3p→3s) + A(3p→2s)), and the mean time adds τ(3s)."""
    lt = catalog.load_json("Li", "lifetimes", dd)
    rows = {r["state"]: r for r in lt["lifetimes"] if r.get("channels")}
    E, _IE = templates._level_energies("Li", dd)
    P, _F = templates.cascade_flux(rows, "3p1/2", E)
    A = {c["lower_label"]: c["A_s"] for c in rows["3p1/2"]["channels"]}
    assert P["3s"] == pytest.approx(A["3s"] / sum(A.values()), rel=1e-3)
    t = templates.cascade_time(rows, "3p1/2", E)
    tau = {s: rows[s]["lifetime_ns"] for s in ("3p1/2", "3s", "2p1/2", "2p3/2")}
    br_2p = {c["lower_label"]: c["branching_ratio"] for c in rows["3s"]["channels"]}
    t3s = tau["3s"] + sum(br * tau[lo] for lo, br in br_2p.items())
    assert t == pytest.approx(tau["3p1/2"] + P["3s"] * t3s, rel=1e-3)


def test_cascade_cm1_and_cut(dd):
    a = R.resolve(_recipe("cascade", "Na", state="5d5/2", min_pct=1.0), dd)
    b = R.resolve(_recipe("cascade", "Na", state="5d5/2", min_pct=10.0, energy_unit="cm-1"), dd)
    n_lines = [sum(op["op"] == "annotate" for op in s.panels[0].ops) for s in (a, b)]
    assert n_lines[0] > n_lines[1] > 0
    top = max(op["args"][0] for op in b.panels[0].ops if op["op"] == "hlines")
    assert top == pytest.approx(37036.8, abs=2)                         # Na 5d5/2: 37 036.8 cm⁻¹ (NIST)


def test_equal_range_ends_are_left_out(dd, tmp_path):
    import warnings

    r = _recipe("builder", "Rb", inset_xlim=[5, 5], inset_ylim=[0, 500], y2lim=[1, 1],
                series=[_S, {**_S, "on": "inset"}])
    r["axes"].update({"xmin": 10.0, "xmax": 10.0})
    spec = R.resolve(r, dd)
    assert any("inset x range was left out" in n for n in spec.notes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # no "identical low and high xlims"
        _draw_and_script(spec, tmp_path, "equal")


def test_builder_marker_type_fill_and_size(dd, tmp_path):
    s = {**_S, "kind": "markers", "marker": "h", "marker_fill": "open", "marker_size": 7.5}
    spec = R.resolve(_recipe("builder", "Rb", series=[s, {**_S, "kind": "markers"}]), dd)
    kw0, kw1 = (op["kw"] for op in spec.panels[0].ops)
    assert kw0["marker"] == "h" and kw0["markersize"] == 7.5 and kw0["markerfacecolor"] == "none"
    assert "markersize" not in kw1 and "markerfacecolor" not in kw1      # auto: the style's size, filled
    line = _draw_and_script(spec, tmp_path, "markers").axes[0].lines[0]
    assert line.get_marker() == "h" and line.get_markersize() == 7.5 and line.get_markerfacecolor() == "none"


def test_blackbody_map_matches_the_stage_sweep(dd):
    """The (n, T) map reproduces the Blackbody stage's own temperature sweep where they overlap."""
    import json
    from pathlib import Path

    bb = json.loads(Path(dd, "Rb_blackbody.json").read_text(encoding="utf-8"))
    ts = bb["rates"]["T_sweep"]
    T0 = ts["T_K"]
    spec = R.resolve(_recipe("blackbody", "Rb", view="map", quantity="gamma", l=["s"], T_min=T0[0], T_max=T0[-1],
                             T_points=len(T0)), dd)
    _h, rows = spec.table
    got = {(r[0], round(r[1], 6)): r[2] for r in rows}
    for state in ("7s", "29s", "50s"):
        for t_, g in zip(T0, ts["states"][state]):
            assert got[(state, round(t_, 6))] == pytest.approx(g, rel=1e-9)
    assert spec.panels[0].colorbar and spec.panels[0].ops[0]["op"] == "pcolormesh"
    assert templates.inputs_of("blackbody", {"view": "map"}) == ["lifetimes"]


@pytest.mark.parametrize("species", ["Rb", "Cs"])
def test_both_alpha_curves_pass_through_each_magic_marker(dd, species):
    """The curves break only on the poles, so they reach magic crossings a fraction of a nm from a resonance."""
    r = templates.polarizability_spectrum({}, species, dd, {})
    ground, excited, magic = (op["args"] for op in r.panels[0].ops[:3])
    wl = np.asarray(ground[0])
    assert len(magic[0]) > 5
    for x, y in zip(*magic):
        i = int(np.searchsorted(wl, x))
        assert wl[i] == pytest.approx(x)
        assert ground[1][i] == pytest.approx(y, rel=1e-3, abs=0.1)
        assert excited[1][i] == pytest.approx(y, rel=1e-2, abs=1.0)


def test_alpha_samples_and_magic_marker_look(dd):
    r = templates.polarizability_spectrum({"points": 1000, "magic_marker": "D", "magic_fill": "filled",
                                           "magic_size": 7, "magic_colour": "C3"}, "Rb", dd, {})
    kw = r.panels[0].ops[2]["kw"]
    assert (kw["marker"], kw["markersize"], kw["color"]) == ("D", 7.0, "C3")
    assert "markerfacecolor" not in kw
    fine = templates.polarizability_spectrum({"points": 20000}, "Rb", dd, {})
    assert len(fine.panels[0].ops[0]["args"][0]) > len(r.panels[0].ops[0]["args"][0])


def _magic_shown(r):
    plots = [op for op in r.panels[0].ops if op["op"] == "plot" and op["kw"].get("label") == "magic"]
    return list(plots[0]["args"][0]) if plots else []


def test_alpha_crossings_display_modes(tmp_path, dd):
    """all: every crossing; range: cut near lines except inside the window; practical: the stage's choice."""
    import json
    import shutil
    from qali.pipelines.polarizability import main as pol_main
    for f in ("Rb_rydberg.json", "Rb_transitions.json", "Rb_lifetimes.json"):
        shutil.copy(os.path.join(dd, f), tmp_path / f)
    pol_main(["Rb", "--data-dir", str(tmp_path)])
    d = json.loads((tmp_path / "Rb_polarizability.json").read_text(encoding="utf-8"))
    every = [m["wl_nm"] for m in d["magic_wavelengths"] if 400 <= m["wl_nm"] <= 1600]
    run = lambda **p: _magic_shown(templates.polarizability_spectrum(p, "Rb", str(tmp_path), {}))  # noqa: E731
    every_n = dict(magic_n_min=1, magic_n_max=99)
    assert len(run(magic_show="all", **every_n)) == len(every)
    lo_n, hi_n = d["magic_n_range"]
    assert (lo_n, hi_n) == (1, 10)                         # Rb: n_ground + 5
    in_n = [m["wl_nm"] for m in d["magic_wavelengths"] if 400 <= m["wl_nm"] <= 1600 and m["line_n"] <= hi_n]
    assert run(magic_show="all") == pytest.approx(in_n) and len(in_n) == 12
    clean = run(magic_show="range", **every_n)
    assert 0 < len(clean) < len(every) and not any(474.7 < w < 475.0 for w in clean)
    window = run(magic_show="range", range_min=474.0, range_max=476.0, **every_n)
    assert any(474.7 < w < 475.0 for w in window) and len(window) > len(clean)
    assert run(magic_show="range", range_min=474.0, **every_n) == run(magic_show="range", range_min=474.0,
                                                                      range_max=1600.0, **every_n)
    practical = [m["wl_nm"] for m in d["magic_wavelengths"] if m["practical"] and 400 <= m["wl_nm"] <= 1600]
    assert run(magic_show="practical", **every_n) == pytest.approx(practical)
    strict = run(magic_show="practical", window_min=1.0, **every_n)
    assert 0 < len(strict) < len(practical) and not any(474.7 < w < 475.0 for w in strict)


def test_alpha_range_with_one_end_keeps_the_other_default(dd):
    one = templates.polarizability_spectrum({"wl_min": 600}, "Rb", dd, {})
    wl = one.panels[0].ops[0]["args"][0]
    assert min(wl) == pytest.approx(600.0) and max(wl) == pytest.approx(1600.0)
    other = templates.polarizability_spectrum({"wl_max": 900}, "Rb", dd, {})
    wl = other.panels[0].ops[0]["args"][0]
    assert min(wl) == pytest.approx(400.0) and max(wl) == pytest.approx(900.0)


def test_one_ended_ranges_keep_the_other_end_automatic():
    assert templates._num_pair([None, 5]) == [None, 5.0]
    assert templates._num_pair(["2", ""]) == [2.0, None]
    assert templates._num_pair([None, None]) is None and templates._num_pair([3, 3]) is None
    assert templates._num_pair([None, 5], both=True) is None
