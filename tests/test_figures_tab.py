"""The Figures tab: every template previews, the builder adds series, the Grotrian dialog, recipes reopen."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.figures import recipe as R, templates  # noqa: E402
from qali.gui.figures_tab import FiguresTab, GrotrianDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(app, data_dir):
    t = FiguresTab()
    t.resize(1100, 700)
    t.set_context(str(data_dir), "Rb")
    t.show()
    app.processEvents()
    yield t
    t.close()


def _pick(tab, key):
    tab.template.setCurrentIndex(tab.template.findData(key))
    tab.redraw()


@pytest.mark.parametrize("key", [k for k in templates.TEMPLATES if k != "builder"])
def test_each_template_previews(tab, key):
    _pick(tab, key)
    assert tab.spec is not None, tab.message.text()
    assert not tab.preview.pixmap().isNull()
    assert tab.param_pages[key].isVisible()
    assert all(not p.isVisible() for k, p in tab.param_pages.items() if k != key)


def test_builder_needs_then_draws_a_series(tab):
    _pick(tab, "builder")
    assert tab.spec is None and "series" in tab.message.text().lower()
    tab.series_editor._add()
    tab.redraw()
    assert tab.spec is not None, tab.message.text()
    assert not tab.preview.pixmap().isNull()


def test_species_switch_redraws(tab):
    _pick(tab, "quantum_defects")
    tab.species_box.setCurrentIndex(tab.species_box.findText("Cs"))
    tab.redraw()
    assert tab.recipe["species"] == "Cs" and tab.spec is not None


def test_grotrian_dialog_choices(app, data_dir):
    lv, lines, IE = templates.grotrian_choices("Rb", str(data_dir))
    chosen = templates.default_grotrian_levels(lv, IE, count=10)
    dlg = GrotrianDialog(lv, lines, IE, chosen, None)
    assert sorted(dlg.chosen_levels()) == sorted(chosen)
    picked = dlg.chosen_lines()
    assert picked and all(k.split(">")[0] in chosen and k.split(">")[1] in chosen for k in picked)
    dlg._set_all(False)
    assert dlg.chosen_levels() == [] and dlg.chosen_lines() == []
    dlg.n_max.setValue(6)
    dlg._quick_pick()
    assert dlg.chosen_levels() and all(int("".join(c for c in s[:2] if c.isdigit())) <= 6
                                      for s in dlg.chosen_levels())
    for r in range(dlg.table.rowCount()):
        dlg.table.item(r, 0).setCheckState(Qt.Unchecked)
    assert dlg.chosen_lines() == []


def test_recipe_reopens_in_the_tab(tab, tmp_path):
    r = R.new_recipe("scaling", "Na")
    path = tmp_path / "s.qfig"
    R.save_recipe(r, str(path))
    tab.open_recipe(str(path))
    tab.redraw()
    assert tab.template.currentData() == "scaling" and tab.recipe["species"] == "Na"
    assert tab.spec is not None


def test_no_output_folder_says_so(app):
    t = FiguresTab()
    t.redraw()
    assert t.spec is None and t.message.text()


def _centre(tab, i=0):
    """A point on the preview inside panel *i*."""
    from PySide6.QtCore import QPoint

    ax = tab.figure.axes[i]
    pix = tab.preview.pixmap()
    r = pix.devicePixelRatio() or 1.0
    pw, ph = pix.width() / r, pix.height() / r
    ox, oy = (tab.preview.width() - pw) / 2, (tab.preview.height() - ph) / 2
    cx, cy = (ax.bbox.x0 + ax.bbox.x1) / 2 / r, (ax.bbox.y0 + ax.bbox.y1) / 2 / r
    return QPoint(int(ox + cx), int(oy + ph - cy))


def _wheel(tab, pos, steps, mods=Qt.NoModifier):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    ev = QWheelEvent(QPointF(pos), QPointF(tab.preview.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(tab.preview, ev)


from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402


def test_wheel_zoom_pan_box_and_reset(tab, app, tmp_path):
    _pick(tab, "scaling")
    ax = tab.figure.axes[0]
    x0, y0 = ax.get_xlim(), ax.get_ylim()
    pos = _centre(tab)
    _wheel(tab, pos, 2)                                           # zoom in
    a = tab.recipe["axes"]
    assert a["xmin"] is not None and a["ymin"] is not None
    assert (np.log10(a["xmax"]) - np.log10(a["xmin"])) < (np.log10(x0[1]) - np.log10(x0[0]))
    assert tab.xmin.text()                                        # the Axes fields follow
    spec = R.resolve(tab.recipe, tab.data_dir)                    # the export draws the same view
    assert spec.panels[0].xlim == [a["xmin"], a["xmax"]]
    _wheel(tab, pos, -1, Qt.ShiftModifier)                        # x only
    assert tab.recipe["axes"]["ymin"] == a["ymin"]
    before = (a["xmin"], a["xmax"])
    QTest.mousePress(tab.preview, Qt.LeftButton, Qt.NoModifier, pos)
    QTest.mouseMove(tab.preview, pos + QPoint(40, 0))
    QTest.mouseRelease(tab.preview, Qt.LeftButton, Qt.NoModifier, pos + QPoint(40, 0))
    after = (tab.recipe["axes"]["xmin"], tab.recipe["axes"]["xmax"])
    assert after[0] < before[0] and after[1] < before[1]          # dragged right: the view moves left
    tab.redraw()
    pos = _centre(tab)
    QTest.mousePress(tab.preview, Qt.RightButton, Qt.NoModifier, pos - QPoint(20, 20))
    QTest.mouseMove(tab.preview, pos + QPoint(20, 20))
    QTest.mouseRelease(tab.preview, Qt.RightButton, Qt.NoModifier, pos + QPoint(20, 20))
    b = tab.recipe["axes"]
    assert b["xmin"] < b["xmax"] and b["ymin"] < b["ymax"]
    QTest.mouseDClick(tab.preview, Qt.LeftButton, Qt.NoModifier, pos)
    assert all(tab.recipe["axes"][k] is None for k in ("xmin", "xmax", "ymin", "ymax"))
    tab.redraw()
    assert np.allclose(tab.figure.axes[0].get_xlim(), x0) and np.allclose(tab.figure.axes[0].get_ylim(), y0)


def test_zoom_on_a_lower_panel(tab):
    _pick(tab, "quantum_defects")
    _wheel(tab, _centre(tab, 1), 1, Qt.AltModifier)          # y only, panel (b)
    assert tab.recipe["axes"]["view"]["1"]["y"]
    assert tab.recipe["axes"]["ymin"] is None


def test_input_lists_follow_figure_and_runs(app, tmp_path, data_dir):
    import json
    import shutil

    d = tmp_path / "data"
    d.mkdir()
    for f in os.listdir(data_dir):
        if f.startswith("Na_"):
            shutil.copy(os.path.join(data_dir, f), d / f)
    doc = json.loads((d / "Na_lifetimes.json").read_text(encoding="utf-8"))
    (d / "Na_lifetimes__cold.json").write_text(json.dumps(doc), encoding="utf-8")
    t = FiguresTab()
    t.set_context(str(d), "Na")
    _pick(t, "grotrian")
    assert list(t.input_boxes) == ["rydberg", "lifetimes"]
    box = t.input_boxes["lifetimes"]
    assert box.findData("cold") >= 0 and box.itemText(box.count() - 1) == "Other file..."
    box.setCurrentIndex(box.findData("cold"))
    t._input_picked("lifetimes")
    t.redraw()
    assert t.recipe["inputs"]["lifetimes"] == "cold" and t.spec is not None
    assert any("cold" in n for n in t.spec.notes)
    _pick(t, "builder")
    assert t.inputs_box.isHidden()
    _pick(t, "scaling")
    t.sc_qty.setCurrentIndex(t.sc_qty.findData("bbr_lifetime"))
    assert list(t.input_boxes) == ["rydberg", "blackbody"]


def test_data_lists_follow_the_species(tab):
    _pick(tab, "branching")
    assert tab.br_state.count() > 10 and tab.br_state.itemData(0) == ""
    tab.br_state.setCurrentIndex(tab.br_state.findData("7s"))
    tab.redraw()
    assert tab.recipe["params"]["state"] == "7s" and tab.spec is not None
    _pick(tab, "feshbach")
    assert tab.fe_channels.count() >= 1
    tab.species_box.setCurrentIndex(tab.species_box.findText("K"))
    tab.redraw()
    assert tab.fe_channels.count() == 3 and tab.spec is not None
    _pick(tab, "breit_rabi")
    assert tab.bra_state.count() >= 2


def test_builder_file_source(tab, tmp_path, data_dir, monkeypatch):
    import shutil

    from qali.gui import figures_tab as FT

    ext = tmp_path / "x.json"
    shutil.copy(os.path.join(data_dir, "Cs_lifetimes.json"), ext)
    _pick(tab, "builder")
    ed = tab.series_editor
    ed._add()
    monkeypatch.setattr(FT.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(ext), "")))
    ed.species.setCurrentIndex(ed.species.findData(FT.OTHER_FILE))
    s = ed.series()[0]
    assert s.get("file") == str(ext) and s["table"].startswith("file:")
    tab.residual.setCurrentIndex(tab.residual.findData("ratio"))
    tab.redraw()
    assert tab.spec is not None


def test_changing_figure_keeps_the_shown_species(tab):
    _pick(tab, "quantum_defects")
    tab.species_box.setCurrentIndex(tab.species_box.findText("Cs"))
    tab.redraw()
    _pick(tab, "breit_rabi")                       # set_context said Rb; the box shows Cs
    assert tab.recipe["species"] == "Cs" and tab.species_box.currentText() == "Cs"
    assert tab.spec is not None and "Cs" in tab.spec.notes[0]
    assert all(tab.bra_state.itemText(i) for i in range(tab.bra_state.count()))
    _pick(tab, "blackbody")
    tab.bb_view.setCurrentIndex(tab.bb_view.findData("temperature"))
    assert tab.bb_sweep_names and "of" in tab.bb_states_lbl.text()


def test_builder_axes_controls(tab, app):
    _pick(tab, "builder")
    ed = tab.series_editor
    ed._add()
    ed._add()
    ed.on.setCurrentIndex(ed.on.findData("y2"))
    assert ed.series()[1]["on"] == "y2"
    tab.y2label.setText("right")
    tab.y2label.editingFinished.emit()
    tab.redraw()
    assert tab.spec.panels[0].y2["label"] == "right"
    ed.kind.setCurrentIndex(ed.kind.findData("heatmap"))
    assert ed._form.isRowVisible(ed.z) and not ed._form.isRowVisible(ed.yerr)
    ed.kind.setCurrentIndex(ed.kind.findData("line"))
    ed.on.setCurrentIndex(0)
    tab.break_axis.setCurrentIndex(tab.break_axis.findData("y"))
    tab.break_a.setText("3000")
    tab.break_b.setText("20000")
    tab.break_b.editingFinished.emit()
    tab.redraw()
    assert tab.spec.panels[0].axis_break == {"axis": "y", "at": [3000.0, 20000.0]}
    assert tab.recipe["params"]["break_at"] == [3000.0, 20000.0]
    assert tab._hit(QPoint(tab.preview.width() // 2, tab.preview.height() // 2)) is None


def test_series_marker_fill_and_size(tab, app):
    _pick(tab, "builder")
    ed = tab.series_editor
    ed._add()
    ed.marker.setCurrentIndex(ed.marker.findData("s"))
    ed.marker_fill.setCurrentIndex(ed.marker_fill.findData("open"))
    ed.marker_size.setValue(6.0)
    s = ed.series()[0]
    assert (s["marker"], s["marker_fill"], s["marker_size"]) == ("s", "open", 6.0)
    tab.redraw()
    line = tab.figure.axes[0].lines[0]
    assert line.get_markersize() == 6.0 and line.get_markerfacecolor() == "none"
    ed.marker_size.setValue(0.0)
    assert ed.marker_size.text() == "auto"


def test_controls_never_run_under_the_scroll_bar(tab, app):
    from PySide6.QtWidgets import QSplitter

    _pick(tab, "builder")
    tab.series_editor._add()
    for _ in range(3):
        app.processEvents()
    tab.findChildren(QSplitter)[0].setSizes([10, 2000])           # squeeze the controls panel
    app.processEvents()
    sc = tab.controls_scroll
    assert sc.viewport().width() >= sc.widget().minimumSizeHint().width()


def test_alpha_samples_and_marker_controls_round_trip(tab):
    _pick(tab, "alpha")
    tab.al_points.setValue(12000)
    tab.al_marker.setCurrentIndex(tab.al_marker.findData("s"))
    tab.al_fill.setCurrentIndex(tab.al_fill.findData("filled"))
    tab.al_msize.setValue(6.0)
    tab._controls_changed()
    p = tab.recipe["params"]
    assert (p["points"], p["magic_marker"], p["magic_fill"], p["magic_size"]) == (12000, "s", "filled", 6.0)
    tab._load_controls()
    assert tab.al_points.value() == 12000 and tab.al_marker.currentData() == "s"


def test_alpha_crossings_controls_show_per_mode_and_round_trip(tab):
    _pick(tab, "alpha")
    tab.al_show.setCurrentIndex(tab.al_show.findData("range"))
    assert tab.al_form.isRowVisible(tab.al_win_row) and not tab.al_form.isRowVisible(tab.al_rmax)
    tab.al_win0.setText("480")
    tab.al_win1.setText("520")
    tab.al_cut.setText("0.2")
    tab._controls_changed()
    p = tab.recipe["params"]
    assert (p["magic_show"], p["range_min"], p["range_max"], p["cutoff_nm"]) == ("range", 480.0, 520.0, 0.2)
    tab.al_show.setCurrentIndex(tab.al_show.findData("practical"))
    assert tab.al_form.isRowVisible(tab.al_rmax) and not tab.al_form.isRowVisible(tab.al_cut)
    tab.al_rmax.setText("50")
    tab._controls_changed()
    tab._load_controls()
    assert tab.al_show.currentData() == "practical" and tab.al_rmax.text() == "50"
    tab.redraw()
    assert tab.spec is not None, tab.message.text()


def test_alpha_line_n_range_round_trips(tab):
    _pick(tab, "alpha")
    tab.al_n0.setText("3")
    tab.al_n1.setText("12")
    tab._controls_changed()
    p = tab.recipe["params"]
    assert (p["magic_n_min"], p["magic_n_max"]) == (3, 12)
    tab._load_controls()
    assert (tab.al_n0.text(), tab.al_n1.text()) == ("3", "12")
