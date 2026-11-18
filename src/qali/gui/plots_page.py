"""Preferences > Plots: the size, style and export settings new publication figures (Figures tab) start with.

The same stored value is written by the Figures tab's "Use as default"; figures already open keep their own style.
"""

from __future__ import annotations

import copy
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from qali.figures.spec import SAVE_FORMATS
from qali.figures.style import FONTS, JOURNALS, PALETTES, default_style, latex_problem

from .extrapolator_settings import set_match

DEFAULTS_KEY = "figures/default_style"
DPI_CHOICES = (150, 300, 600, 1200)


def factory_defaults() -> dict:
    return {"style": default_style("aps"), "size": {"journal": "aps", "columns": 1},
            "export": {"format": "pdf", "dpi": 600}}


def load_defaults(settings) -> dict:
    """The stored defaults merged over the factory ones (a missing or broken value gives the factory ones)."""
    out = factory_defaults()
    try:
        raw = settings.value(DEFAULTS_KEY, "")
        saved = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        saved = {}
    if isinstance(saved, dict):
        for block in ("style", "size", "export"):
            if isinstance(saved.get(block), dict):
                out[block].update(saved[block])
    if out["export"].get("format") not in SAVE_FORMATS:
        out["export"]["format"] = "pdf"
    return out


def save_defaults(settings, defaults: dict) -> None:
    settings.setValue(DEFAULTS_KEY, json.dumps(defaults))


def _card(title: str) -> tuple[QFrame, QVBoxLayout, QLabel]:
    card = QFrame()
    card.setObjectName("prefsCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 12, 16, 14)
    lay.setSpacing(8)
    head = QLabel(title)
    head.setObjectName("cardTitle")
    lay.addWidget(head)
    return card, lay, head


class PlotsPage(QWidget):
    """Edit the defaults of new figures; Apply saves them."""

    changed = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._at_open = settings.value(DEFAULTS_KEY, "")
        self.applied = False
        self._filling = False
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 12, 0)
        row.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(14)
        row.addLayout(left, 3)
        self.labels: dict[str, QLabel] = {}

        def form_card(title: str):
            card, lay, head = _card(title)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignLeft)
            form.setHorizontalSpacing(18)
            form.setVerticalSpacing(8)
            lay.addLayout(form)
            left.addWidget(card)
            return form, head

        def add(form, key: str, text: str, widget, tip: str = "") -> None:
            label = QLabel(text)
            widget.setToolTip(tip)
            form.addRow(label, widget)
            self.labels[key] = label

        form, self.size_title = form_card("Size")
        self.journal = QComboBox()
        for key, j in JOURNALS.items():
            self.journal.addItem(j.label, key)
        self.columns = QComboBox()
        self.columns.addItem("One column", 1)
        self.columns.addItem("Two columns", 2)
        add(form, "journal", "Preset", self.journal, "Journal widths and font size")
        add(form, "columns", "Width", self.columns)

        form, self.style_title = form_card("Style")
        self.palette = QComboBox()
        for key, (label, _c) in PALETTES.items():
            self.palette.addItem(label, key)
        self.font_box = QComboBox()
        for key, label in FONTS.items():
            self.font_box.addItem(label, key)
        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(5.0, 24.0)
        self.font_size.setSingleStep(0.5)
        self.font_size.setSuffix(" pt")
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.3, 4.0)
        self.line_width.setSingleStep(0.1)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1.0, 12.0)
        self.marker_size.setSingleStep(0.5)
        add(form, "palette", "Colours", self.palette)
        add(form, "font", "Font", self.font_box)
        add(form, "font_size", "Font size", self.font_size)
        add(form, "line_width", "Line width", self.line_width)
        add(form, "marker_size", "Marker size", self.marker_size)
        self.checks: dict[str, QCheckBox] = {}
        for key, text in (("usetex", "Typeset text with LaTeX"), ("ticks_in", "Ticks inside"),
                          ("mirror_ticks", "Ticks on all sides"), ("minor_ticks", "Minor ticks"),
                          ("grid", "Grid"), ("panel_labels", "Panel labels (a), (b)")):
            c = QCheckBox(text)
            form.addRow(c)
            self.checks[key] = c
        problem = latex_problem(raster=False)
        self.checks["usetex"].setToolTip(problem or "Slower; needs LaTeX with the cm-super fonts")

        form, self.export_title = form_card("Export")
        self.fmt = QComboBox()
        for ext, label in SAVE_FORMATS.items():
            self.fmt.addItem(label.split(" (")[0] + f" (.{ext})", ext)
        self.dpi = QComboBox()
        for d in DPI_CHOICES:
            self.dpi.addItem(f"{d} dpi", d)
        add(form, "format", "Format", self.fmt, "Suggested when exporting")
        add(form, "dpi", "Resolution", self.dpi, "PNG and TIFF")
        card, lay, self.controls_title = _card("Figure controls")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignLeft)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for action, effect in (("Mouse wheel", "Zoom the data (Shift: x only, Alt: y only)"), ("Drag", "Pan"),
                               ("Right-drag", "Zoom to a box"), ("Double-click", "Full view"),
                               ("Ctrl + wheel", "Magnify the whole figure (Ctrl + 0: fit)"),
                               ("Middle-drag", "Move a magnified figure"),
                               ("↗ button", "Open the panel's plot in its own window")):
            key = QLabel(action)
            self.labels[f"control:{action}"] = key
            val = QLabel(effect)
            val.setObjectName("muted")
            grid.addRow(key, val)
        lay.addLayout(grid)
        hint = QLabel("The view sets the axis ranges, so exports and saved recipes show it.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        right = QVBoxLayout()
        right.addWidget(card)
        right.addStretch(1)
        note = QLabel("New figures start with these; open figures keep their own.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        left.addWidget(note)
        left.addStretch(1)
        row.addLayout(right, 2)

        for w in (self.journal, self.columns, self.palette, self.font_box, self.fmt, self.dpi):
            w.currentIndexChanged.connect(self._edited)
        for w in (self.font_size, self.line_width, self.marker_size):
            w.valueChanged.connect(self._edited)
        for c in self.checks.values():
            c.toggled.connect(self._edited)
        self.journal.currentIndexChanged.connect(self._journal_changed)
        self.set_values(load_defaults(settings))

    # ---- values -------------------------------------------------------------------------------------
    def values(self) -> dict:
        style = {"palette": self.palette.currentData(), "font": self.font_box.currentData(),
                 "font_size": self.font_size.value(), "line_width": round(self.line_width.value(), 2),
                 "marker_size": self.marker_size.value()}
        style.update({k: c.isChecked() for k, c in self.checks.items()})
        return {"style": style, "size": {"journal": self.journal.currentData(), "columns": self.columns.currentData()},
                "export": {"format": self.fmt.currentData(), "dpi": self.dpi.currentData()}}

    def set_values(self, v: dict) -> None:
        self._filling = True
        st, sz, ex = v["style"], v["size"], v["export"]
        for box, value in ((self.journal, sz.get("journal")), (self.columns, sz.get("columns")),
                           (self.palette, st.get("palette")), (self.font_box, st.get("font")),
                           (self.fmt, ex.get("format")), (self.dpi, ex.get("dpi"))):
            box.setCurrentIndex(max(box.findData(value), 0))
        self.font_size.setValue(float(st.get("font_size", 8)))
        self.line_width.setValue(float(st.get("line_width", 1.0)))
        self.marker_size.setValue(float(st.get("marker_size", 3.5)))
        for k, c in self.checks.items():
            c.setChecked(bool(st.get(k)))
        self._filling = False
        self._edited()

    def _journal_changed(self, *_):
        if self._filling:
            return
        j = JOURNALS[self.journal.currentData()]
        self.font_size.setValue(j.font_pt)                       # a preset brings its font size and family
        self.font_box.setCurrentIndex(max(0, self.font_box.findData("sans" if j.family == "sans" else "cm")))

    def _differences(self) -> set[str]:
        """Keys whose value differs from the factory defaults (the font size of the chosen preset counts as
        default)."""
        v, f = self.values(), factory_defaults()
        base = default_style(v["size"]["journal"])
        out = {k for k in ("palette", "line_width", "marker_size", *self.checks) if v["style"][k] != f["style"][k]}
        out |= {k for k in ("font", "font_size") if v["style"][k] != base[k]}
        out |= {k for k in ("journal", "columns") if v["size"][k] != f["size"][k]}
        out |= {k for k in ("format", "dpi") if v["export"][k] != f["export"][k]}
        return out

    def _edited(self, *_):
        if self._filling:
            return
        diff = self._differences()
        for name, label in self.labels.items():
            font = label.font()
            font.setBold(name in diff)
            label.setFont(font)
        for name, c in self.checks.items():
            font = c.font()
            font.setBold(name in diff)
            c.setFont(font)
        self.changed.emit()

    # ---- Preferences hooks --------------------------------------------------------------------------
    def changed_count(self) -> int:
        return len(self._differences())

    def apply(self) -> list[str]:
        merged = copy.deepcopy(load_defaults(self.settings))
        for block, vals in self.values().items():
            merged[block].update(vals)
        save_defaults(self.settings, merged)
        self.applied = True
        return []

    def undo_apply(self) -> None:
        if self._at_open:
            self.settings.setValue(DEFAULTS_KEY, self._at_open)
        else:
            self.settings.remove(DEFAULTS_KEY)
        self.applied = False

    def reset_all(self) -> None:
        self.set_values(factory_defaults())

    def _search_labels(self) -> list:
        return [self.size_title, self.style_title, self.export_title, self.controls_title, *self.labels.values(),
                *self.checks.values()]

    def search_texts(self) -> list[str]:
        return [w.text() for w in self._search_labels()] + ["journal", "APS", "Nature", "LaTeX", "dpi", "PDF",
                                                           "figure", "matplotlib", "zoom", "pan"]

    def highlight(self, needle: str) -> int:
        needle = needle.strip().lower()
        for w in self._search_labels():
            set_match(w, bool(needle) and needle in w.text().lower())
        return sum(needle in t.lower() for t in self.search_texts()) if needle else 0
