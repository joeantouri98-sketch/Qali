"""Edit > Preferences > Units and numbers: conversions, number format, the page and the views."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import units as U  # noqa: E402


@pytest.fixture(autouse=True)
def _defaults():
    app = QApplication.instance() or QApplication([])
    U.set_current(U.DEFAULTS)
    yield app
    U.set_current(U.DEFAULTS)


def test_defaults_show_the_stored_units():
    assert U.show(2.1044292, "energy") == "2.10443 eV"
    assert U.show(589.1583, "wavelength") == "589.158 nm"
    assert U.show(162.68, "polarizability") == "162.68 a.u."
    assert U.show(100.4, "length") == "100.4 a₀"


def test_conversions():
    u = U.Units(energy="cm-1", wavelength="air", polarizability="hHz", length="nm", sig_figs=7)
    assert U.convert(2.1044292, "energy", u)[0] == pytest.approx(16973.37, abs=0.01)
    assert U.convert(1.0, "energy", U.Units(energy="Eh"))[0] == pytest.approx(1 / 27.211386245988)
    assert U.convert(589.1583, "wavelength", u)[0] == pytest.approx(588.9950, abs=2e-4)     # NIST air D2
    assert U.convert(150.0, "wavelength", u) == (150.0, "nm")                            # vacuum UV stays
    assert U.convert(162.68, "polarizability", u)[0] == pytest.approx(40.48, abs=0.01)
    assert U.convert(100.4, "length", u)[0] == pytest.approx(5.3129, abs=1e-4)


def test_prefixes_and_number_format():
    assert U.show(16.24e-9, "time") == "16.24 ns"
    assert U.show(16.24e-9, "time", U.Units(time="base")) == "1.62400e-08 s"
    assert U.show(9192631770.0, "frequency") == "9.19263 GHz"
    assert U.number(123456.0) == "1.23456e+05"
    assert U.number(123456.0, U.Units(sci_above=1e6)) == "123456"
    assert U.number(3.14159265, U.Units(sig_figs=3)) == "3.14"


def test_saved_choices_load_and_defaults_are_not_written(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    U.save(st, U.Units(energy="cm-1", sig_figs=4))
    assert st.value("units/energy") == "cm-1" and st.value("units/time") is None
    U.set_current(U.DEFAULTS)
    assert U.load(st) == U.Units(energy="cm-1", sig_figs=4) == U.current()


def test_page_shows_examples_counts_and_cancel_undoes_apply(tmp_path):
    from qali.gui import preferences_dialog as P

    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    calls = []
    d = P.PreferencesDialog(None, "dark", st, lambda: None, units_changed=lambda: calls.append(1))
    page = d.units
    page.energy.setCurrentIndex(page.energy.findData("cm-1"))
    assert page.example_values[0].text().endswith("cm⁻¹")
    assert U.current().energy == "eV"                         # nothing in force before Apply
    assert page.changed_count() == 1
    d._apply()
    assert U.current().energy == "cm-1" and st.value("units/energy") == "cm-1" and calls == [1]
    d._cancel()
    assert U.current().energy == "eV" and st.value("units/energy") is None and calls == [1, 1]


def test_overview_and_comparison_table_follow_the_units(tmp_path):
    from qali.gui.compare_table_view import _header, _shown
    from qali.gui.overview import SpeciesOverviewWidget

    U.set_current(U.Units(energy="cm-1", wavelength="air"))
    assert SpeciesOverviewWidget._with_unit(2.1044292, "eV").endswith("cm⁻¹")
    assert SpeciesOverviewWidget._with_unit(1.5, "kW/cm²") == "1.5 kW/cm²"
    assert _header("IE_eV", "IE (eV)") == "IE (cm⁻¹)" and _header("D2_wl", "λ (nm)") == "λ air (nm)"
    assert _shown("D2_wl", 589.1583)[0] == pytest.approx(588.9950, abs=2e-4)
    assert _shown("soc", 2.1)[0] == pytest.approx(2.1e-3 * U.EV_TO_CM)
    U.set_current(U.DEFAULTS)
    assert _shown("IE_eV", 5.139) is None or _shown("IE_eV", 5.139)[0] == 5.139
