"""Preferences > Appearance additions: text sizes and the console position."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import theme  # noqa: E402
from qali.gui.appearance_extras import CONSOLE_ON_TOP_SETTING  # noqa: E402
from qali.gui.preferences_dialog import PreferencesDialog  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    s = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    yield s
    theme.set_fonts(theme.DEFAULT_FONTS)


def _dialog(settings, **kw):
    return PreferencesDialog(None, "dark", settings, lambda: None, **kw)


def test_stylesheet_points_follow_the_interface_size():
    css = theme.build_stylesheet(theme.tokens_for(theme.DARK))
    theme.set_fonts(theme.Fonts(ui_size=12))
    try:
        big = theme.build_stylesheet(theme.tokens_for(theme.DARK))
        assert "font-size: 15pt" in css and "font-size: 18pt" in big
        assert theme.ui_font().pointSize() == 12
    finally:
        theme.set_fonts(theme.DEFAULT_FONTS)


def test_fonts_round_trip_and_defaults_not_written(settings):
    theme.save_fonts(settings, theme.DEFAULT_FONTS)
    assert settings.allKeys() == []
    theme.save_fonts(settings, theme.Fonts(11, "DejaVu Sans Mono", 10))
    assert theme.load_fonts(settings) == theme.Fonts(11, "DejaVu Sans Mono", 10)
    settings.setValue(theme.UI_SIZE_SETTING, 99)
    assert theme.load_fonts(settings).ui_size == theme.UI_SIZE_RANGE[1]


def test_staged_until_apply_then_cancel_undoes(app, settings):
    seen = []
    dlg = _dialog(settings, look_changed=lambda: seen.append(1))
    x = dlg.extras
    x.ui_size.setValue(12)
    x.console_top.setChecked(True)
    assert x.changed_count() == 2 and theme.fonts().ui_size == theme.DEFAULT_FONTS.ui_size
    dlg._update_counts()
    dlg._apply()
    assert theme.fonts().ui_size == 12
    assert settings.value(CONSOLE_ON_TOP_SETTING, type=bool) and seen == [1]
    dlg.reject()
    assert theme.fonts() == theme.DEFAULT_FONTS
    assert not settings.value(CONSOLE_ON_TOP_SETTING, type=bool) and seen == [1, 1]


def test_page_defaults_and_search(app, settings):
    dlg = _dialog(settings)
    dlg.extras.mono_size.setValue(12)
    dlg._restore_page_defaults("appearance")
    assert dlg.extras.changed_count() == 0
    dlg.search.setText("console")
    keys = [dlg.nav.item(i).data(0x0100) for i in range(dlg.nav.count())]
    assert not dlg.nav.item(keys.index("appearance")).isHidden()
    dlg.reject()
