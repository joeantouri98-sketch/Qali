"""Preferences > Plots: defaults of new figures, shared with the Figures tab's "Use as default"."""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import plots_page as PP  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def settings(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)


def test_defaults_count_apply_undo(app, settings):
    page = PP.PlotsPage(settings)
    assert page.changed_count() == 0
    page.journal.setCurrentIndex(page.journal.findData("nature"))
    assert page.font_size.value() == 7.0 and page.font_box.currentData() == "sans"
    assert page.changed_count() == 1                      # the preset's own font is no extra change
    page.dpi.setCurrentIndex(page.dpi.findData(300))
    page.apply()
    saved = PP.load_defaults(settings)
    assert saved["size"]["journal"] == "nature" and saved["export"]["dpi"] == 300
    page.undo_apply()
    assert settings.value(PP.DEFAULTS_KEY) is None
    page.reset_all()
    assert page.changed_count() == 0


def test_tab_uses_and_writes_the_same_defaults(app, settings, monkeypatch):
    from qali.gui import figures_tab as FT

    monkeypatch.setattr(FT, "_settings", lambda: settings)
    PP.save_defaults(settings, {**PP.factory_defaults(), "export": {"format": "svg", "dpi": 1200},
                                "style": {**PP.factory_defaults()["style"], "palette": "tol-muted"}})
    tab = FT.FiguresTab()
    assert tab.recipe["style"]["palette"] == "tol-muted"
    tab.recipe["style"]["grid"] = True
    tab._save_defaults()
    d = PP.load_defaults(settings)
    assert d["style"]["grid"] and d["export"] == {"format": "svg", "dpi": 1200}   # export kept


def test_broken_value_gives_factory(app, settings):
    settings.setValue(PP.DEFAULTS_KEY, "{not json")
    assert PP.load_defaults(settings) == PP.factory_defaults()
    settings.setValue(PP.DEFAULTS_KEY, json.dumps({"export": {"format": "doc"}}))
    assert PP.load_defaults(settings)["export"]["format"] == "pdf"


def test_page_in_preferences(app, settings):
    from qali.gui.preferences_dialog import PreferencesDialog

    d = PreferencesDialog(None, "dark", settings, lambda: None)
    keys = [d.nav.item(i).data(0x0100) for i in range(d.nav.count())]
    assert "plots" in keys
    d.plots.columns.setCurrentIndex(1)
    d._ok()
    assert PP.load_defaults(settings)["size"]["columns"] == 2


def test_last_folder_not_exported():
    from qali.gui.advanced_page import LOCAL_KEYS

    assert "figures/last_dir" in LOCAL_KEYS
