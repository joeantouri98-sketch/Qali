"""Plot windows: a panel's current plot opened in a window of its own, and whole-view zoom.

Each pop-out is independent (pressing again opens another window): an interactive page loads its own copy, a
publication figure is a snapshot of the figure as it was, the JSON viewer a copy of its text. Zoom here magnifies
the whole view (Ctrl + wheel, Ctrl + = / − / 0); the Figures tab keeps the plain wheel for its axis ranges.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QScrollArea, QToolTip, QVBoxLayout, QWidget

ZOOM_STEPS = (0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0)


def next_zoom(current: float, step: int) -> float:
    """The next zoom level up (step > 0) or down (step < 0) from *current*."""
    if step > 0:
        return next((z for z in ZOOM_STEPS if z > current + 1e-6), ZOOM_STEPS[-1])
    return next((z for z in reversed(ZOOM_STEPS) if z < current - 1e-6), ZOOM_STEPS[0])


def show_zoom(widget: QWidget, factor: float) -> None:
    QToolTip.showText(widget.mapToGlobal(QPoint(widget.width() // 2, widget.height() // 2)),
                      f"{round(factor * 100)} %", widget, widget.rect(), 900)


def add_zoom_shortcuts(widget: QWidget, step) -> None:
    """Ctrl + = / + / − / 0 on *widget* and its children: step(+1), step(-1), step(0) (reset)."""
    for keys, s in ((QKeySequence.StandardKey.ZoomIn, 1), ("Ctrl+=", 1), (QKeySequence.StandardKey.ZoomOut, -1),
                    ("Ctrl+0", 0)):
        sc = QShortcut(QKeySequence(keys), widget)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda s_=s: step(s_))


class _WebWheel(QObject):
    """Ctrl + wheel on a web view (the event reaches Chromium's child widget, so the filter sits there)."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            web_zoom_step(self.view, 1 if event.angleDelta().y() > 0 else -1)
            return True
        if event.type() == QEvent.ChildAdded and isinstance(event.child(), QWidget):
            event.child().installEventFilter(self)
        return False


def web_zoom_step(view, step: int) -> None:
    factor = 1.0 if step == 0 else next_zoom(view.zoomFactor(), step)
    view.setZoomFactor(factor)
    show_zoom(view, factor)


def enable_web_zoom(view) -> None:
    """Whole-page zoom on a QWebEngineView: Ctrl + wheel and Ctrl + = / − / 0."""
    f = _WebWheel(view)
    view.installEventFilter(f)
    for child in view.findChildren(QWidget):
        child.installEventFilter(f)
    add_zoom_shortcuts(view, lambda s: web_zoom_step(view, s))


# ------------------------------------------------------------------------------------------------ windows
class PopoutWindow(QWidget):
    """A top-level window holding one plot; it closes with the main window and frees itself when closed."""

    def __init__(self, parent: QWidget, title: str, body: QWidget):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(body)
        self.body = body
        self.resize(900, 640)


def web_popout(parent: QWidget, source_view, title: str, on_load=None) -> PopoutWindow:
    """An independent copy of the page *source_view* shows, at its zoom."""
    from PySide6.QtWebEngineWidgets import QWebEngineView

    from .web_downloads import attach_web_downloads

    view = QWebEngineView()
    attach_web_downloads(view, parent)
    if on_load is not None:
        view.loadFinished.connect(lambda ok: ok and on_load(view))
    view.load(source_view.url())
    view.setZoomFactor(source_view.zoomFactor())
    enable_web_zoom(view)
    win = PopoutWindow(parent, title, view)
    win.show()
    return win


def text_popout(parent: QWidget, text: str, title: str, font=None) -> PopoutWindow:
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    edit.setPlainText(text)
    if font is not None:
        edit.setFont(font)
    win = PopoutWindow(parent, title, edit)          # QPlainTextEdit zooms with Ctrl + wheel itself
    win.show()
    return win


class FigureSnapshot(QScrollArea):
    """A publication figure drawn at the window's size times the zoom (redrawn, so it stays sharp).

    Wheel or Ctrl + wheel: zoom about the cursor. Drag: scroll. Double-click or Ctrl + 0: fit the window.
    """

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = copy.deepcopy(spec)
        self.zoom = 1.0
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.label)
        self.setAlignment(Qt.AlignCenter)
        self.setWidgetResizable(False)
        self._drag: QPoint | None = None
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(60)
        self._redraw.timeout.connect(self.render)
        self.viewport().installEventFilter(self)
        self.label.installEventFilter(self)
        add_zoom_shortcuts(self, self.step)

    def fit_dpi(self) -> float:
        vp = self.viewport().size()
        return max(30.0, min((vp.width() - 8) / self.spec.width_in, (vp.height() - 8) / self.spec.height_in))

    def render(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from qali.figures import spec as S

        ratio = self.devicePixelRatioF() or 1.0
        dpi = min(self.fit_dpi() * self.zoom, 1200.0)
        fig = S.draw(self.spec, raster=True)
        fig.set_dpi(dpi * ratio)
        agg = FigureCanvasAgg(fig)
        with S.style_context(self.spec.style, raster=True):
            agg.draw()
        w, h = agg.get_width_height()
        img = QImage(bytes(agg.buffer_rgba()), w, h, QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(img)
        pix.setDevicePixelRatio(ratio)
        self.label.setPixmap(pix)
        self.label.resize(int(w / ratio), int(h / ratio))

    def step(self, s: int, anchor: QPoint | None = None) -> None:
        old = self.zoom
        self.zoom = 1.0 if s == 0 else next_zoom(self.zoom, s)
        if self.zoom == old:
            return
        # keep the point under the cursor (or the centre) where it is
        anchor = anchor or QPoint(self.viewport().width() // 2, self.viewport().height() // 2)
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        fx = (h.value() + anchor.x()) / max(self.label.width(), 1)
        fy = (v.value() + anchor.y()) / max(self.label.height(), 1)
        self.render()
        h.setValue(int(fx * self.label.width() - anchor.x()))
        v.setValue(int(fy * self.label.height() - anchor.y()))
        show_zoom(self.viewport(), self.zoom)

    def eventFilter(self, obj, event) -> bool:
        t = event.type()
        if t == QEvent.Wheel:
            pos = self.viewport().mapFromGlobal(event.globalPosition().toPoint())
            self.step(1 if event.angleDelta().y() > 0 else -1, pos)
            return True
        if t == QEvent.MouseButtonDblClick:
            self.step(0)
            return True
        if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._drag = event.globalPosition().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return True
        if t == QEvent.MouseMove and self._drag is not None:
            p = event.globalPosition().toPoint()
            d = p - self._drag
            self._drag = p
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return True
        if t == QEvent.MouseButtonRelease and self._drag is not None:
            self._drag = None
            self.viewport().unsetCursor()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._redraw.start()


def figure_popout(parent: QWidget, spec, title: str) -> PopoutWindow:
    snap = FigureSnapshot(spec)
    win = PopoutWindow(parent, title, snap)
    ratio = spec.height_in / spec.width_in
    win.resize(900, int(min(max(900 * ratio, 300), 900)) + 20)
    win.show()
    return win
