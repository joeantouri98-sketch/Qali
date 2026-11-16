"""Edit > Preferences > General: output folder, ORCA program, reopening the last species."""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import general_page as G  # noqa: E402
from qali.gui import preferences_dialog as P  # noqa: E402
from qali.gui.stage_process import stage_env_values  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def fake_orca(tmp_path):
    """An 'orca' with ORCA's own modules beside it."""
    folder = tmp_path / "orca"
    folder.mkdir()
    exe = folder / ("orca.exe" if sys.platform == "win32" else "orca")
    exe.write_text("")
    (folder / "orca_scf").write_text("")
    return exe


def _settings(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)


def test_general_is_the_first_page_and_starts_at_the_defaults(app, tmp_path):
    d = P.PreferencesDialog(None, "dark", _settings(tmp_path), lambda: None, root=tmp_path,
                            default_root=tmp_path / "default")
    assert d.nav.item(0).text() == "General" and d.nav.currentRow() == 0
    assert d.general.changed_count() == 0
    assert not G.reopen_enabled(d.settings)


def test_apply_saves_the_orca_program_and_switches_the_folder(app, tmp_path, fake_orca):
    st = _settings(tmp_path)
    switched = []
    page = G.GeneralPage(st, tmp_path, tmp_path / "default", switched.append)
    page.orca.setText(str(fake_orca))
    page.folder.setText(str(tmp_path / "results"))
    page.reopen.setChecked(True)
    assert page.changed_count() == 3
    assert page.apply() == []
    assert G.orca_from_settings(st) == str(fake_orca)
    assert st.value(G.ROOT_SETTING) == str(tmp_path / "results") and (tmp_path / "results").is_dir()
    assert switched == [tmp_path / "results"]
    assert G.reopen_enabled(st)
    page.undo_apply()
    assert G.orca_from_settings(st) == "" and st.value(G.ROOT_SETTING, "") == ""
    assert switched[-1] is None and not G.reopen_enabled(st)


def test_a_wrong_orca_program_is_not_saved(app, tmp_path):
    st = _settings(tmp_path)
    page = G.GeneralPage(st, tmp_path, tmp_path / "default")
    (tmp_path / "orca").write_text("")                      # a lone file: no ORCA modules next to it
    page.orca.setText(str(tmp_path / "orca"))
    assert page.apply() and G.orca_from_settings(st) == ""


def test_the_chosen_orca_reaches_the_stages():
    assert stage_env_values(".", "/opt/orca/orca")["ORCA_EXE"] == "/opt/orca/orca"
    assert "ORCA_EXE" not in stage_env_values(".")


def test_detect_finds_orca_from_orca_exe(app, tmp_path, fake_orca, monkeypatch):
    monkeypatch.setenv("ORCA_EXE", str(fake_orca))
    page = G.GeneralPage(_settings(tmp_path), tmp_path, tmp_path / "default")
    page._detect()
    assert page.orca.text() == str(fake_orca)


def test_search_finds_general_settings(app, tmp_path):
    d = P.PreferencesDialog(None, "dark", _settings(tmp_path), lambda: None, root=tmp_path, start_tab="tables")
    d.search.setText("orca")
    assert d.nav.currentItem().text() == "General"
    assert d.general.orca_title.property("match")
