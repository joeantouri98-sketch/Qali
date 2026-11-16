"""Preferences > Advanced: export/import of preferences and Reset all preferences."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from qali.gui import advanced_page, units  # noqa: E402
from qali.gui.preferences_dialog import PreferencesDialog  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    s = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    s.setValue("appearance/theme", "light")
    s.setValue("units/energy", "cm-1")
    s.setValue("root_path", "C:/work/qali")
    s.setValue("orca/executable", "C:/orca/orca.exe")
    s.setValue("preferences/size", "900x600")
    yield s
    units.set_current(units.DEFAULTS)


def test_export_leaves_out_this_computers_paths(settings, tmp_path):
    out = tmp_path / "out.ini"
    n = advanced_page.export_settings(settings, str(out))
    read = QSettings(str(out), QSettings.IniFormat)
    assert set(read.allKeys()) == {"appearance/theme", "units/energy"} and n == 2


def test_import_round_trip_keeps_local_paths(settings, tmp_path):
    out = tmp_path / "out.ini"
    advanced_page.export_settings(settings, str(out))
    other = QSettings(str(tmp_path / "other.ini"), QSettings.IniFormat)
    other.setValue("root_path", "D:/elsewhere")
    advanced_page.import_settings(other, str(out))
    assert other.value("units/energy") == "cm-1" and other.value("appearance/theme") == "light"
    assert other.value("root_path") == "D:/elsewhere"


def test_import_never_brings_local_keys(tmp_path):
    src = QSettings(str(tmp_path / "src.ini"), QSettings.IniFormat)
    src.setValue("root_path", "X:/other")
    src.setValue("units/sig_figs", 4)
    src.sync()
    dst = QSettings(str(tmp_path / "dst.ini"), QSettings.IniFormat)
    advanced_page.import_settings(dst, str(tmp_path / "src.ini"))
    assert dst.value("root_path") is None and int(dst.value("units/sig_figs")) == 4


def test_advanced_is_last_and_reset_stages_defaults(app, settings, monkeypatch):
    units.load(settings)
    dlg = PreferencesDialog(None, "light", settings, lambda: None)
    keys = [dlg.nav.item(i).data(0x0100) for i in range(dlg.nav.count())]
    assert keys[-1] == "advanced"
    assert dlg.units.changed_count() == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg.advanced._reset()
    assert dlg.units.changed_count() == 0
    assert units.current().energy == "cm-1"          # nothing in force until OK
    dlg.nav.setCurrentRow(keys.index("advanced"))
    assert dlg.defaults_btn.isHidden()
    dlg.reject()


def test_search_finds_export(app, settings):
    dlg = PreferencesDialog(None, "light", settings, lambda: None)
    assert dlg.advanced.highlight("export") >= 1
    dlg.reject()


def test_reset_everything_with_every_page(app, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = PreferencesDialog(None, "dark", store, lambda: None, root=tmp_path, species="Rb",
                            default_root=tmp_path)
    dlg.chosen["dark"]["accent"] = "#ff0000"
    dlg.tables._overrides["x"] = {"a": 1.0}
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg.advanced._reset()
    assert dlg.chosen["dark"] == {} and dlg.tables._overrides == {}
    assert dlg.tables.changed_count() == 0 and dlg.extrapolator.changed_count() == 0
    dlg.reject()
