"""Preferences > General: the output folder, the ORCA program, and what Qali does on start.

Edits stay on the page until Apply or OK; ``undo_apply()`` puts back what was in force when the page opened.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qali.core.paths import project_root_from_env

from .extrapolator_settings import set_match

ROOT_SETTING = "root_path"                       # also written by File > Open output folder...
ORCA_SETTING = "orca/executable"                 # passed to the ORCA stages as ORCA_EXE
REOPEN_SETTING = "startup/reopen_last_species"
LAST_SPECIES_SETTING = "startup/last_species"


def orca_from_settings(settings) -> str:
    """The ORCA program chosen in Preferences, or '' for automatic."""
    value = settings.value(ORCA_SETTING, "")
    return value if isinstance(value, str) else ""


def reopen_enabled(settings) -> bool:
    return settings.value(REOPEN_SETTING, False, type=bool)


def detect_orca() -> tuple[str, str]:
    """(path, '') of the ORCA program found from ORCA_EXE or PATH, or ('', why it was not found)."""
    from qali.integrations.orca_runner import resolve_orca_executable

    try:
        return resolve_orca_executable(), ""
    except RuntimeError as exc:
        reason = re.split(r"(?<=[.;]) | \(in the app", str(exc))[0]     # the cause, not the advice
        return "", reason.rstrip(";.") + "."


def check_orca(path: str) -> str:
    """'' if *path* looks like the ORCA program, else what is wrong with it."""
    from qali.integrations.orca_runner import _has_orca_modules

    p = Path(path).expanduser()
    if not p.is_file():
        return "No file at this path."
    if not _has_orca_modules(p.resolve().parent):
        return "No ORCA modules next to this file, so it is not the ORCA program."
    return ""


def orca_version(path: str) -> str:
    """ORCA's version as it reports it (e.g. '6.1.0'), or '' if it cannot be read quickly."""
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"(?:Version|version)\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)", done.stdout + done.stderr)
    return m.group(1) if m else ""


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


class GeneralPage(QWidget):
    """Output folder, ORCA program and start-up options. *set_root(path | None)* switches the main window's
    folder (None: the default one)."""

    changed = Signal()

    def __init__(self, settings, root: Path, default_root: Path, set_root=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.default_root = Path(default_root)
        self.set_root = set_root
        saved = settings.value(ROOT_SETTING, "")
        self._root_at_open = saved if isinstance(saved, str) and saved else ""
        self._orca_at_open = orca_from_settings(settings)
        self._reopen_at_open = reopen_enabled(settings)
        self.applied = False

        page = QVBoxLayout(self)
        page.setContentsMargins(4, 0, 12, 0)
        page.setSpacing(14)

        card, lay, self.folder_title = _card("Output folder")
        row = QHBoxLayout()
        self.folder = QLineEdit(self._root_at_open)
        self.folder.setPlaceholderText(str(self.default_root))
        self.folder.setToolTip("Where the stages write their results")
        self.folder.textChanged.connect(self._edited)
        row.addWidget(self.folder, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_folder)
        row.addWidget(browse)
        default = QPushButton("Use default")
        default.clicked.connect(lambda: self.folder.clear())
        row.addWidget(default)
        lay.addLayout(row)
        self.folder_note = QLabel()
        self.folder_note.setObjectName("muted")
        self.folder_note.setWordWrap(True)
        lay.addWidget(self.folder_note)
        page.addWidget(card)

        card, lay, self.orca_title = _card("ORCA program")
        row = QHBoxLayout()
        self.orca = QLineEdit(self._orca_at_open)
        self.orca.setPlaceholderText("Automatic (ORCA_EXE or PATH)")
        self.orca.setToolTip("The ORCA executable the ORCA stages run")
        self.orca.textChanged.connect(self._edited)
        row.addWidget(self.orca, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_orca)
        row.addWidget(browse)
        detect = QPushButton("Detect")
        detect.setToolTip("Find ORCA from ORCA_EXE or PATH")
        detect.clicked.connect(self._detect)
        row.addWidget(detect)
        auto = QPushButton("Automatic")
        auto.clicked.connect(lambda: self.orca.clear())
        row.addWidget(auto)
        lay.addLayout(row)
        self.orca_status = QLabel()
        self.orca_status.setObjectName("muted")
        self.orca_status.setWordWrap(True)
        lay.addWidget(self.orca_status)
        page.addWidget(card)

        card, lay, self.start_title = _card("Start-up")
        grid = QGridLayout()
        self.reopen = QCheckBox("Reopen the last species")
        self.reopen.setChecked(self._reopen_at_open)
        self.reopen.toggled.connect(self._edited)
        grid.addWidget(self.reopen, 0, 0)
        lay.addLayout(grid)
        page.addWidget(card)
        page.addStretch(1)
        self._check_orca(version=False)
        self._note_folder()

    # ---- editing ----------------------------------------------------------------------------------------
    def _edited(self, *_):
        self._note_folder()
        self.changed.emit()

    def _note_folder(self) -> None:
        text = self.folder.text().strip()
        if project_root_from_env():
            note = f"QALI_PROJECT_ROOT is set, so this session uses {project_root_from_env()}."
        elif not text:
            note = "Default folder."
        elif not Path(text).expanduser().is_dir():
            note = "This folder does not exist yet; it is created on Apply."
        else:
            note = ""
        self.folder_note.setText(note)
        self.folder_note.setVisible(bool(note))

    def _browse_folder(self) -> None:
        start = self.folder.text().strip() or str(self.default_root)
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if chosen:
            self.folder.setText(chosen)

    def _browse_orca(self) -> None:
        start = str(Path(self.orca.text()).parent) if self.orca.text().strip() else ""
        flt = "ORCA (orca.exe);;All files (*)" if os.name == "nt" else "All files (*)"
        chosen, _ = QFileDialog.getOpenFileName(self, "ORCA program", start, flt)
        if chosen:
            self.orca.setText(chosen)
            self._check_orca()

    def _detect(self) -> None:
        path, why = detect_orca()
        if path:
            self.orca.setText(path)
            self._check_orca()
        else:
            self.orca_status.setText(f"Not found. {why}")

    def _check_orca(self, version: bool = True) -> None:
        """Say which ORCA the stages will run, and (*version*: asking ORCA, up to a few seconds) its version."""
        path = self.orca.text().strip()
        if not path:
            found, why = detect_orca()
            if not found:
                self.orca_status.setText(f"Automatic: not found. {why}")
                return
            v = orca_version(found) if version else ""
            self.orca_status.setText(f"Automatic: {found}" + (f"  (ORCA {v})" if v else ""))
            return
        problem = check_orca(path)
        if problem:
            self.orca_status.setText(problem)
            return
        v = orca_version(path) if version else ""
        self.orca_status.setText(f"ORCA {v}" if v else "ORCA found.")

    # ---- Preferences hooks ------------------------------------------------------------------------------
    def changed_count(self) -> int:
        """Settings that differ from the defaults: a chosen folder, a chosen ORCA, reopening on start."""
        return (bool(self.folder.text().strip()) + bool(self.orca.text().strip())
                + self.reopen.isChecked())

    def apply(self) -> list[str]:
        """Save the choices and switch the folder if it changed; returns what could not be used."""
        problems = []
        orca = self.orca.text().strip()
        if orca and check_orca(orca):
            problems.append(f"ORCA program: {check_orca(orca)}")
        elif orca:
            self.settings.setValue(ORCA_SETTING, str(Path(orca).expanduser()))
        else:
            self.settings.remove(ORCA_SETTING)
        self.settings.setValue(REOPEN_SETTING, self.reopen.isChecked())
        folder = self.folder.text().strip()
        problems += self._use_folder(folder)
        self.applied = True
        return problems

    def _use_folder(self, folder: str) -> list[str]:
        saved = self.settings.value(ROOT_SETTING, "")
        saved = saved if isinstance(saved, str) else ""
        if folder == saved:
            return []
        if folder:
            path = Path(folder).expanduser()
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return [f"Output folder: {exc}"]
            self.settings.setValue(ROOT_SETTING, str(path))
            if self.set_root is not None:
                self.set_root(path)
        else:
            self.settings.remove(ROOT_SETTING)
            if self.set_root is not None:
                self.set_root(None)
        return []

    def undo_apply(self) -> None:
        """Put back the folder, ORCA and start-up choices in force when the page opened."""
        if self._orca_at_open:
            self.settings.setValue(ORCA_SETTING, self._orca_at_open)
        else:
            self.settings.remove(ORCA_SETTING)
        self.settings.setValue(REOPEN_SETTING, self._reopen_at_open)
        self._use_folder(self._root_at_open)
        self.applied = False

    def reset_all(self) -> None:
        self.folder.clear()
        self.orca.clear()
        self.reopen.setChecked(False)

    def _labels(self) -> list[QLabel]:
        return [self.folder_title, self.orca_title, self.start_title]

    def search_texts(self) -> list[str]:
        return [lbl.text() for lbl in self._labels()] + ["Reopen the last species", "ORCA_EXE", "Detect"]

    def highlight(self, needle: str) -> int:
        needle = needle.strip().lower()
        for label in self._labels():
            set_match(label, bool(needle) and needle in label.text().lower())
        return sum(needle in t.lower() for t in self.search_texts()) if needle else 0
