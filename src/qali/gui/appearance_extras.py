"""Preferences > Appearance, lower half: text sizes and where the console panel sits."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .extrapolator_settings import set_match

CONSOLE_ON_TOP_SETTING = "layout/console_panel_on_top"
AUTOMATIC = "Automatic"


def _card(title: str) -> tuple[QFrame, QVBoxLayout, QLabel]:
    card = QFrame()
    card.setObjectName("prefsCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 10, 12, 12)
    lay.setSpacing(6)
    head = QLabel(title)
    head.setObjectName("cardTitle")
    lay.addWidget(head)
    return card, lay, head


def _mono_families() -> list[str]:
    """Installed fixed-pitch families, the app's own choices first."""
    installed = [f for f in QFontDatabase.families() if QFontDatabase.isFixedPitch(f)]
    first = [f for f in theme.MONO_FONT_FAMILIES if f in installed]
    return first + sorted(f for f in installed if f not in first)


def _spin(lo_hi: tuple[int, int]) -> QSpinBox:
    box = QSpinBox()
    box.setRange(*lo_hi)
    box.setSuffix(" pt")
    box.setFixedWidth(84)
    return box


class AppearanceExtras(QWidget):
    """Edits that Apply puts in force: text sizes (theme.Fonts) and the console position."""

    changed = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._fonts_at_open = theme.fonts()
        self._top_at_open = settings.value(CONSOLE_ON_TOP_SETTING, False, type=bool)
        self.applied = False
        self._filling = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        # ---- text ----
        card, tlay, self.text_title = _card("Text")
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.ui_size = _spin(theme.UI_SIZE_RANGE)
        self.mono_family = QComboBox()
        self.mono_family.addItem(AUTOMATIC, "")
        for family in _mono_families():
            self.mono_family.addItem(family, family)
        self.mono_family.setMinimumWidth(150)
        self.mono_size = _spin(theme.MONO_SIZE_RANGE)
        self.ui_size_label = QLabel("Interface size")
        self.mono_label = QLabel("Console font")
        form.addWidget(self.ui_size_label, 0, 0)
        form.addWidget(self.ui_size, 0, 1, 1, 2, Qt.AlignLeft)
        form.addWidget(self.mono_label, 1, 0)
        form.addWidget(self.mono_family, 1, 1)
        form.addWidget(self.mono_size, 1, 2)
        form.setColumnStretch(1, 1)
        tlay.addLayout(form)
        self.mono_sample = QLabel("E = 2.104429 eV   λ = 589.158 nm")
        self.mono_sample.setObjectName("muted")
        tlay.addWidget(self.mono_sample)
        for w in (self.ui_size, self.mono_size):
            w.valueChanged.connect(self._edited)
        self.mono_family.currentIndexChanged.connect(self._edited)
        self.ui_size.setToolTip("Text across the app")
        self.mono_family.setToolTip("Console, commands and raw data")
        lay.addWidget(card)

        # ---- layout ----
        card, llay, self.layout_title = _card("Layout")
        self.console_top = QCheckBox("Console panel above the viewer")
        self.console_top.setToolTip("Also View > Console panel on top")
        self.console_top.toggled.connect(self._edited)
        llay.addWidget(self.console_top)
        lay.addWidget(card)

        self._set_fonts(self._fonts_at_open, self._top_at_open)

    # ---- editing ------------------------------------------------------------------------------------
    def fonts(self) -> theme.Fonts:
        return theme.Fonts(self.ui_size.value(), self.mono_family.currentData() or "", self.mono_size.value())

    def _set_fonts(self, f: theme.Fonts, on_top: bool) -> None:
        self._filling = True
        self.ui_size.setValue(f.ui_size)
        self.mono_size.setValue(f.mono_size)
        i = self.mono_family.findData(f.mono_family)
        if i < 0:                                    # a family that is not installed here
            self.mono_family.addItem(f.mono_family, f.mono_family)
            i = self.mono_family.count() - 1
        self.mono_family.setCurrentIndex(i)
        self.console_top.setChecked(on_top)
        self._filling = False
        self._edited()

    def _edited(self, *_):
        if self._filling:
            return
        f = self.fonts()
        _bold(self.ui_size_label, f.ui_size != theme.DEFAULT_FONTS.ui_size)
        _bold(self.mono_label, (f.mono_family, f.mono_size) != (theme.DEFAULT_FONTS.mono_family,
                                                               theme.DEFAULT_FONTS.mono_size))
        sample = QFont([f.mono_family] if f.mono_family else list(theme.MONO_FONT_FAMILIES))
        sample.setStyleHint(QFont.Monospace)
        sample.setPointSize(f.mono_size)
        self.mono_sample.setFont(sample)
        self.changed.emit()

    # ---- Preferences hooks --------------------------------------------------------------------------
    def changed_count(self) -> int:
        f, d = self.fonts(), theme.DEFAULT_FONTS
        n = (f.ui_size != d.ui_size) + (f.mono_family != d.mono_family) + (f.mono_size != d.mono_size)
        return n + int(self.console_top.isChecked())

    def is_different(self) -> bool:
        """Whether Apply would change anything now in force."""
        return (self.fonts() != theme.fonts()
                or self.console_top.isChecked() != self.settings.value(CONSOLE_ON_TOP_SETTING, False, type=bool))

    def apply(self) -> None:
        theme.set_fonts(self.fonts())
        theme.save_fonts(self.settings, self.fonts())
        self.settings.setValue(CONSOLE_ON_TOP_SETTING, self.console_top.isChecked())
        self.applied = True

    def undo_apply(self) -> None:
        theme.set_fonts(self._fonts_at_open)
        theme.save_fonts(self.settings, self._fonts_at_open)
        self.settings.setValue(CONSOLE_ON_TOP_SETTING, self._top_at_open)
        self.applied = False

    def reset_all(self) -> None:
        self._set_fonts(theme.DEFAULT_FONTS, False)

    def _search_labels(self) -> list[QLabel]:
        return [self.text_title, self.layout_title, self.ui_size_label, self.mono_label]

    def search_texts(self) -> list[str]:
        return [lbl.text() for lbl in self._search_labels()] + [self.console_top.text(), "font"]

    def highlight(self, needle: str) -> int:
        needle = needle.strip().lower()
        for label in self._search_labels():
            set_match(label, bool(needle) and needle in label.text().lower())
        return sum(needle in t.lower() for t in self.search_texts()) if needle else 0


def _bold(label: QLabel, on: bool) -> None:
    font = label.font()
    font.setBold(on)
    label.setFont(font)
