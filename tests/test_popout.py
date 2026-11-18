"""Plot windows: ↗ opens a new, independent window each press; whole-view zoom in the tab and the windows."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from qali.gui import popout as P  # noqa: E402
from qali.gui.figures_tab import FiguresTab  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(app, data_dir):
    t = FiguresTab()
    t.resize(1100, 700)
    t.set_context(str(data_dir), "Rb")
    t.template.setCurrentIndex(t.template.findData("scaling"))
    t.show()
    app.processEvents()
    t.redraw()
    yield t
    t.close()


def _wheel(widget, steps, mods=Qt.NoModifier):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    ev = QWheelEvent(pos, QPointF(widget.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def test_next_zoom_steps_and_clamps():
    assert P.next_zoom(1.0, 1) == 1.1
    assert P.next_zoom(1.0, -1) == 0.9
    assert P.next_zoom(1.05, 1) == 1.1 and P.next_zoom(1.05, -1) == 1.0
    assert P.next_zoom(P.ZOOM_STEPS[-1], 1) == P.ZOOM_STEPS[-1]
    assert P.next_zoom(P.ZOOM_STEPS[0], -1) == P.ZOOM_STEPS[0]


def test_ctrl_wheel_magnifies_the_whole_figure_not_the_axes(tab, app):
    assert tab.spec is not None
    lim = tab.figure.axes[0].get_xlim(), tab.figure.axes[0].get_ylim()
    w0 = tab.preview.width()
    _wheel(tab.preview, 2, Qt.ControlModifier)
    app.processEvents()
    assert tab.view_zoom > 1.0
    assert tab.preview.width() > w0
    assert (tab.figure.axes[0].get_xlim(), tab.figure.axes[0].get_ylim()) == lim
    tab._view_zoom_step(0)
    assert tab.view_zoom == 1.0


def test_figure_window_is_a_snapshot_that_zooms(tab, app):
    host = QWidget()
    win1 = P.figure_popout(host, tab.spec, "one")
    win2 = P.figure_popout(host, tab.spec, "two")
    app.processEvents()
    assert win1 is not win2 and win1.isVisible() and win2.isVisible()
    snap = win1.findChild(P.FigureSnapshot)
    snap.render()
    w0 = snap.label.width()
    _wheel(snap.viewport(), 1)
    assert snap.zoom > 1.0 and snap.label.width() > w0
    # the window keeps its own copy: changing the tab's figure leaves it alone
    tab.template.setCurrentIndex(tab.template.findData("quantum_defects"))
    tab.redraw()
    assert snap.spec is not tab.spec
    snap.step(0)
    assert snap.zoom == 1.0
    win1.close()
    win2.close()
    host.deleteLater()


def test_text_window(app):
    host = QWidget()
    win = P.text_popout(host, '{"a": 1}', "json")
    assert win.windowTitle() == "json" and win.isVisible()
    win.close()
    host.deleteLater()


def test_main_window_popout_buttons(app, monkeypatch, data_dir):
    from qali.gui import main_window as M

    monkeypatch.setattr(M, "HAS_WEBENGINE", False)
    mw = M.MainWindow()
    mw.show()
    app.processEvents()
    try:
        assert mw.popout_center_btn.text() == "↗" and mw.popout_bottom_btn.text() == "↗"
        figs = next(i for i in range(mw.bottom_tabs.count()) if mw.bottom_tabs.tabText(i) == "Figures")
        mw.bottom_tabs.setCurrentIndex(figs)
        mw.bottom_figures.set_context(str(data_dir), "Rb")
        mw.bottom_figures.redraw()
        app.processEvents()
        assert mw.bottom_figures.spec is not None
        assert mw.popout_bottom_btn.isEnabled()
        n = len(mw._popouts)
        mw.popout_bottom_btn.click()
        mw.popout_bottom_btn.click()
        app.processEvents()
        assert len(mw._popouts) == n + 2
        for w in list(mw._popouts):
            w.close()
        app.processEvents()
    finally:
        mw.close()
