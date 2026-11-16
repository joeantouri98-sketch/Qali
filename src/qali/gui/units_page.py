"""Preferences > Units and numbers: the units and number format the app shows values in, with examples."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import units as U
from .extrapolator_settings import set_match

SCI_ABOVE = (1e3, 1e4, 1e5, 1e6, 1e9)
SCI_BELOW = (1e-2, 1e-3, 1e-4, 1e-6, 1e-9)

# (label, value in the stored unit, kind) — values the app really shows, so the effect is plain
EXAMPLES = (
    ("Na D₂ photon energy", 2.1044292, "energy"),
    ("Na D₂ wavelength", 589.1583, "wavelength"),
    ("Na 3p₃/₂ lifetime", 16.24e-9, "time"),
    ("Cs clock frequency", 9192631770.0, "frequency"),
    ("Na ground-state α₀", 162.68, "polarizability"),
    ("⁸⁷Rb background scattering length", 100.4, "length"),
)


def _combo(options: dict[str, str]) -> QComboBox:
    box = QComboBox()
    for key, label in options.items():
        box.addItem(label, key)
    return box


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


class UnitsPage(QWidget):
    """Edit :class:`units.Units`; Apply makes them current and saves them."""

    changed = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._at_open = U.current()
        self.applied = False
        self._filling = False

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 12, 0)
        row.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(14)
        row.addLayout(left, 3)

        card, lay, self.units_title = _card("Units")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        self.energy = _combo(U.ENERGY_UNITS)
        self.wavelength = _combo(U.WAVELENGTH_MODES)
        self.time = _combo(U.SCALE_MODES)
        self.frequency = _combo(U.SCALE_MODES)
        self.polarizability = _combo(U.POLARIZABILITY_UNITS)
        self.length = _combo(U.LENGTH_UNITS)
        self.labels: dict[str, QLabel] = {}
        for name, box, text, tip in (
            ("energy", self.energy, "Energy", "Level and photon energies"),
            ("wavelength", self.wavelength, "Wavelength", "Air: standard air, from 200 nm up"),
            ("time", self.time, "Time", "Lifetimes"),
            ("frequency", self.frequency, "Frequency", "Shifts and splittings"),
            ("polarizability", self.polarizability, "Polarizability", "α; C₆ stays in a.u."),
            ("length", self.length, "Length", "Scattering lengths"),
        ):
            label = QLabel(text)
            box.setToolTip(tip)
            box.setMinimumWidth(220)
            box.currentIndexChanged.connect(self._edited)
            form.addRow(label, box)
            self.labels[name] = label
        lay.addLayout(form)
        left.addWidget(card)

        card, lay, self.numbers_title = _card("Numbers")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        self.sig = QSpinBox()
        self.sig.setRange(2, 12)
        self.sig.setFixedWidth(90)
        self.sig.valueChanged.connect(self._edited)
        self.above = QComboBox()
        for v in SCI_ABOVE:
            self.above.addItem(f"{v:g}", v)
        self.below = QComboBox()
        for v in SCI_BELOW:
            self.below.addItem(f"{v:g}", v)
        self.above.currentIndexChanged.connect(self._edited)
        self.below.currentIndexChanged.connect(self._edited)
        for r, (text, widget, key) in enumerate((("Significant figures", self.sig, "sig_figs"),
                                                 ("Scientific notation from", self.above, "sci_above"),
                                                 ("and below", self.below, "sci_below"))):
            label = QLabel(text)
            self.labels[key] = label
            grid.addWidget(label, r, 0)
            grid.addWidget(widget, r, 1, Qt.AlignLeft)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        left.addWidget(card)
        note = QLabel("Display only: results files, pages and exported CSV keep their own units.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        left.addWidget(note)
        left.addStretch(1)

        card, lay, self.examples_title = _card("Examples")
        ex = QGridLayout()
        ex.setHorizontalSpacing(14)
        ex.setVerticalSpacing(10)
        self.example_values: list[QLabel] = []
        for r, (text, _value, _kind) in enumerate(EXAMPLES):
            name = QLabel(text)
            name.setObjectName("muted")
            name.setWordWrap(True)
            shown = QLabel()
            shown.setTextInteractionFlags(Qt.TextSelectableByMouse)
            ex.addWidget(name, r, 0)
            ex.addWidget(shown, r, 1, Qt.AlignRight)
            self.example_values.append(shown)
        ex.setColumnStretch(0, 1)
        lay.addLayout(ex)
        right = QVBoxLayout()
        right.addWidget(card)
        right.addStretch(1)
        row.addLayout(right, 2)

        self.set_units(U.current())

    # ---- values -------------------------------------------------------------------------------------
    def units(self) -> U.Units:
        return U.Units(
            energy=self.energy.currentData(), wavelength=self.wavelength.currentData(),
            time=self.time.currentData(), frequency=self.frequency.currentData(),
            polarizability=self.polarizability.currentData(), length=self.length.currentData(),
            sig_figs=self.sig.value(), sci_above=self.above.currentData(), sci_below=self.below.currentData())

    def set_units(self, u: U.Units) -> None:
        self._filling = True
        for box, value in ((self.energy, u.energy), (self.wavelength, u.wavelength), (self.time, u.time),
                           (self.frequency, u.frequency), (self.polarizability, u.polarizability),
                           (self.length, u.length), (self.above, u.sci_above), (self.below, u.sci_below)):
            box.setCurrentIndex(max(box.findData(value), 0))
        self.sig.setValue(u.sig_figs)
        self._filling = False
        self._edited()

    def _edited(self, *_):
        if self._filling:
            return
        u = self.units()
        for (_text, value, kind), label in zip(EXAMPLES, self.example_values):
            label.setText(U.show(value, kind, u))
        for name, label in self.labels.items():
            font = label.font()
            font.setBold(getattr(u, name) != getattr(U.DEFAULTS, name))
            label.setFont(font)
        self.changed.emit()

    # ---- Preferences hooks --------------------------------------------------------------------------
    def changed_count(self) -> int:
        return U.changed(self.units())

    def apply(self) -> list[str]:
        U.set_current(self.units())
        U.save(self.settings, self.units())
        self.applied = True
        return []

    def undo_apply(self) -> None:
        U.set_current(self._at_open)
        U.save(self.settings, self._at_open)
        self.applied = False

    def reset_all(self) -> None:
        self.set_units(U.DEFAULTS)

    def _search_labels(self) -> list[QLabel]:
        return [self.units_title, self.numbers_title, *self.labels.values()]

    def search_texts(self) -> list[str]:
        return [lbl.text() for lbl in self._search_labels()] + ["eV", "cm-1", "air", "vacuum", "a.u.",
                                                                 "significant figures", "scientific"]

    def highlight(self, needle: str) -> int:
        needle = needle.strip().lower()
        for label in self._search_labels():
            set_match(label, bool(needle) and needle in label.text().lower())
        return sum(needle in t.lower() for t in self.search_texts()) if needle else 0

