"""Edit > Preferences: custom colours layered on the theme, saved per theme."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import preferences_dialog as P  # noqa: E402
from qali.gui import theme  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    app = QApplication.instance() or QApplication([])
    for name in theme.COLOUR_THEMES:
        theme.set_overrides(name, {})
    yield app
    for name in theme.COLOUR_THEMES:
        theme.set_overrides(name, {})


def test_override_changes_tokens_and_palette_without_touching_defaults(_clean):
    default = theme.THEMES[theme.DARK]
    theme.set_overrides(theme.DARK, {"highlight": "#d6336c", "accent": "#4c8bd6"})
    t = theme.tokens_for(theme.DARK)
    assert t.highlight == "#d6336c" and t.accent == "#4c8bd6" and t.accent_ink == "#4c8bd6"
    assert t.highlight_text in ("#ffffff", "#10180a")
    assert theme.THEMES[theme.DARK] == default                      # shipped colours are never modified
    theme.apply_theme(_clean, theme.DARK)
    assert _clean.palette().color(_clean.palette().ColorRole.Highlight).name() == "#d6336c"


def test_invalid_colours_are_ignored():
    theme.set_overrides(theme.DARK, {"highlight": "not-a-colour"})
    assert theme.tokens_for(theme.DARK) == theme.THEMES[theme.DARK]


def test_save_and_load_roundtrip_is_per_theme(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    P.save_overrides(st, theme.DARK, {"highlight": "#d6336c"})
    P.load_overrides(st)
    assert theme.tokens_for(theme.DARK).highlight == "#d6336c"
    assert theme.tokens_for(theme.LIGHT).highlight == theme.THEMES[theme.LIGHT].highlight
    P.save_overrides(st, theme.DARK, {})                            # resetting removes the saved value
    P.load_overrides(st)
    assert theme.tokens_for(theme.DARK).highlight == theme.THEMES[theme.DARK].highlight


def test_cancel_restores_and_ok_saves(tmp_path, _clean):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    calls = []
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: calls.append(1))
    d.chosen[theme.DARK] = {"highlight": "#123456"}
    d._apply()
    assert theme.tokens_for(theme.DARK).highlight == "#123456"
    d._cancel()
    assert theme.tokens_for(theme.DARK).highlight == theme.THEMES[theme.DARK].highlight
    d2 = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d2.chosen[theme.DARK] = {"highlight": "#123456"}
    d2._ok()
    assert st.value("appearance/colors/dark/highlight") == "#123456"


def test_each_colour_is_separate_and_stage_dots_follow_them():
    tokens = {t for t, _ in theme.EDITABLE_COLOURS}
    assert {"highlight", "text_select", "menu_select", "stage_done", "stage_todo"} <= tokens
    theme.set_overrides(theme.DARK, {"stage_done": "#ff0000", "menu_select": "#223344"})
    t = theme.tokens_for(theme.DARK)
    assert t.stage_done == "#ff0000" and t.menu_select == "#223344"
    assert t.highlight == theme.THEMES[theme.DARK].highlight and t.stage_todo == theme.THEMES[theme.DARK].stage_todo
    assert "#223344" in theme.build_stylesheet(t)
    theme.set_overrides(theme.DARK, {"text_select": "#ffffff"})
    assert theme.tokens_for(theme.DARK).text_select_text != theme.THEMES[theme.DARK].text_select_text


def test_overview_table_copies_selection_and_has_units(tmp_path):
    from qali.gui.overview import CopyableTable, SpeciesOverviewWidget

    t = CopyableTable(2, 2)
    from PySide6.QtWidgets import QTableWidgetItem
    for r in range(2):
        for c in range(2):
            t.setItem(r, c, QTableWidgetItem(f"{r}{c}"))
    t.selectAll()
    assert t.selected_text() == "00\t01\n10\t11"
    assert not t.editTriggers() & t.EditTrigger.DoubleClicked
    assert SpeciesOverviewWidget._with_unit(1.5, "nm") == "1.5 nm"
    assert SpeciesOverviewWidget._with_unit("text", "nm") == "text"


def test_one_theme_is_shown_at_a_time_and_all_are_saved(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    assert (theme.DARK, "highlight") in d.swatches and (theme.LIGHT, "highlight") not in d.swatches
    d.chosen[theme.DARK]["highlight"] = "#111111"
    d._choose_theme(theme.LIGHT)
    assert (theme.LIGHT, "highlight") in d.swatches and (theme.DARK, "highlight") not in d.swatches
    d.chosen[theme.LIGHT]["highlight"] = "#eeeeee"
    d._ok()
    assert st.value("appearance/colors/dark/highlight") == "#111111"
    assert st.value("appearance/colors/light/highlight") == "#eeeeee"
    d3 = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d3._cancel()
    assert theme.tokens_for(theme.LIGHT).highlight == "#eeeeee"   # cancel restores what was open, not the defaults


def test_preview_shows_a_choice_before_it_is_applied(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d.chosen[theme.DARK]["accent"] = "#4c8bd6"
    d._refresh()
    assert d.preview.tokens.accent == "#4c8bd6"
    assert theme.tokens_for(theme.DARK).accent == theme.THEMES[theme.DARK].accent      # nothing applied yet
    assert d._marks["accent"].isVisibleTo(d) is not None and not d._resets["accent"].isHidden()
    assert d._resets["highlight"].isHidden()


def test_changed_count_and_search(tmp_path):
    from PySide6.QtCore import Qt

    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None, root=tmp_path)
    d.chosen[theme.DARK]["accent"] = "#4c8bd6"
    d.base_choice = "#203040"
    d._update_counts()
    rows = {d.nav.item(i).data(Qt.UserRole): d.nav.item(i) for i in range(d.nav.count())}
    assert rows["appearance"].data(P.BADGE_ROLE) == 2
    assert rows["extrapolator"].data(P.BADGE_ROLE) == 0
    d.extrapolator.spins["ridge.tau1"].setValue(1.0)
    d._update_counts()
    assert rows["extrapolator"].data(P.BADGE_ROLE) == 1
    d.search.setText("slope prior")
    assert d.nav.currentItem().data(Qt.UserRole) == "extrapolator"
    assert rows["appearance"].isHidden() and not rows["extrapolator"].isHidden()
    assert d.extrapolator.labels["ridge.tau1"].property("match")
    d.search.setText("stage done")
    assert d.nav.currentItem().data(Qt.UserRole) == "appearance"
    assert d.colour_labels["stage_done"].property("match")
    d.search.setText("")
    assert not any(d.nav.item(i).isHidden() for i in range(d.nav.count()))


def test_cancel_undoes_an_earlier_apply_on_every_page(tmp_path, monkeypatch):
    from qali.core import paths, user_tables

    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    from qali.gui.extrapolator_settings import settings_file
    from qali.physics.extrapolation import fit_settings as fs

    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    before = user_tables.load_overrides()
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None, root=tmp_path)
    d.extrapolator.spins["ridge.tau1"].setValue(1.0)
    tid = user_tables.TABLES[0].table_id
    key, packaged, _now = user_tables.packaged_rows(tid)[0]
    d.tables._overrides = {tid: {key: float(packaged) * 2 + 1}}
    d._apply()
    assert fs.load(settings_file(tmp_path)) == {"ridge.tau1": 1.0}
    assert user_tables.load_overrides().get(tid)
    d._cancel()
    assert fs.load(settings_file(tmp_path)) == {}
    assert user_tables.load_overrides() == before


def test_defaults_button_asks_first(tmp_path, monkeypatch):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d.chosen[theme.DARK]["accent"] = "#4c8bd6"
    monkeypatch.setattr(P.QMessageBox, "question", lambda *a, **k: P.QMessageBox.No)
    d._reset_all()
    assert d.chosen[theme.DARK] == {"accent": "#4c8bd6"}
    monkeypatch.setattr(P.QMessageBox, "question", lambda *a, **k: P.QMessageBox.Yes)
    d._reset_all()
    assert d.chosen[theme.DARK] == {}


def test_last_page_and_size_are_remembered(tmp_path):
    from PySide6.QtCore import QSize

    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None, root=tmp_path, start_tab="tables")
    d.resize(900, 640)
    d._cancel()
    d2 = P.PreferencesDialog(None, theme.DARK, st, lambda: None, root=tmp_path)
    assert d2.nav.currentItem().text() == "Tables and variables" and d2.size() == QSize(900, 640)


def test_card_labels_use_real_subscripts_and_copy_as_plain_text():
    from qali.gui.overview import CARD_LABELS, plain_label

    assert CARD_LABELS["C6_au"] == "C<sub>6</sub>" and CARD_LABELS["alpha_0"] == "α<sub>0</sub>"
    assert plain_label("C<sub>6</sub> source") == "C₆ source"
    assert plain_label("a<sub>bg</sub>") == "a_bg"
    from qali.physics.extrapolation import fit_settings as fs

    assert [f["label"] for f in fs.FIELDS[:3]] == ["Offset prior (τ₀)", "Slope prior (τ₁)", "Threshold term prior (τ₂)"]
