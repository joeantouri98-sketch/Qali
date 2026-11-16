"""Preferences > Advanced: moving the preferences to another computer, and starting over."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .extrapolator_settings import set_match

# Kept on this computer only: paths and window sizes mean nothing on another one.
LOCAL_KEYS = ("root_path", "orca/executable", "startup/last_species", "preferences/size", "preferences/page")


def portable_keys(settings) -> list[str]:
    """The keys an export carries: everything except this computer's paths and window state."""
    return [k for k in settings.allKeys()
            if k not in LOCAL_KEYS and not k.startswith(("geometry", "window", "layout/splitter"))]


def export_settings(settings, path: str) -> int:
    """Write the portable preferences to the .ini file *path*; returns how many were written."""
    out = QSettings(path, QSettings.IniFormat)
    out.clear()
    keys = portable_keys(settings)
    for key in keys:
        out.setValue(key, settings.value(key))
    out.sync()
    return len(keys)


def import_settings(settings, path: str) -> int:
    """Copy the preferences in the .ini file *path* into *settings*; returns how many were read."""
    src = QSettings(path, QSettings.IniFormat)
    keys = [k for k in src.allKeys() if k not in LOCAL_KEYS]
    for key in keys:
        settings.setValue(key, src.value(key))
    settings.sync()
    return len(keys)


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


class AdvancedPage(QWidget):
    """*reset_requested*: put every page back to its defaults (kept until OK). *imported*: preferences
    were read from a file and are in the store; the dialog closes so the app can use them."""

    reset_requested = Signal()
    imported = Signal()
    changed = Signal()
    applied = False

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        page = QVBoxLayout(self)
        page.setContentsMargins(4, 0, 12, 0)
        page.setSpacing(14)

        card, lay, self.move_title = _card("Move to another computer")
        note = QLabel("Theme, colours, units and start-up options. Folders and the ORCA path stay on this computer.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        row = QHBoxLayout()
        export = QPushButton("Export...")
        export.clicked.connect(self._export)
        load = QPushButton("Import...")
        load.clicked.connect(self._import)
        row.addWidget(export)
        row.addWidget(load)
        row.addStretch(1)
        lay.addLayout(row)
        page.addWidget(card)

        card, lay, self.reset_title = _card("Start over")
        note = QLabel("Every page back to its defaults, including the extrapolator settings and edited tables. "
                      "Nothing changes until OK; Cancel keeps everything.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        row = QHBoxLayout()
        reset = QPushButton("Reset all preferences")
        reset.setObjectName("danger")
        reset.clicked.connect(self._reset)
        row.addWidget(reset)
        row.addStretch(1)
        lay.addLayout(row)
        page.addWidget(card)
        page.addStretch(1)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export preferences", "qali-preferences.ini",
                                              "Preferences (*.ini)")
        if path:
            n = export_settings(self.settings, path)
            QMessageBox.information(self, "Export preferences", f"{n} preferences written to\n{path}")

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import preferences", "", "Preferences (*.ini)")
        if not path:
            return
        if QMessageBox.question(self, "Import preferences",
                                "Replace your preferences with the ones in this file?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        import_settings(self.settings, path)
        self.imported.emit()

    def _reset(self) -> None:
        if QMessageBox.question(self, "Reset all preferences", "Put every page back to its defaults?",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.reset_requested.emit()

    # ---- Preferences hooks (nothing of its own to count or apply) ---------------------------------------
    def changed_count(self) -> int:
        return 0

    def _labels(self) -> list[QLabel]:
        return [self.move_title, self.reset_title]

    def search_texts(self) -> list[str]:
        return [lbl.text() for lbl in self._labels()] + ["Export", "Import", "Reset all preferences"]

    def highlight(self, needle: str) -> int:
        needle = needle.strip().lower()
        for label in self._labels():
            set_match(label, bool(needle) and needle in label.text().lower())
        return sum(needle in t.lower() for t in self.search_texts()) if needle else 0
