"""Generate a stage's page again from a saved run (right click on a stage > Generate plot from...).

Lists the stage's saved JSON outputs in the data folder (the usual one and the named runs), with
what each run is, and returns the ones picked.  The page is then written by the stage itself with
``--from-json``: no calculation.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from qali.figures.catalog import RUN_SEP



@dataclass(frozen=True)
class SavedRun:
    path: Path
    species: str
    name: str            # "" for the usual (latest) output
    summary: str
    mtime: float


def list_runs(stem: str, data_dir: str | Path, species: str | None = None) -> list[SavedRun]:
    """Saved outputs <species>_<stem>.json and <species>_<stem>__<name>.json (also a copy renamed with a
    single underscore, <species>_<stem>_<name>.json), newest first."""
    pat = re.compile(rf"^(?P<sp>.+?)_{re.escape(stem)}(?:(?:{RUN_SEP}|_)(?P<run>.+))?\.json$")
    out: list[SavedRun] = []
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    for fn in names:
        m = pat.match(fn)
        if not m or (species and m.group("sp") != species):
            continue
        path = Path(data_dir) / fn
        out.append(SavedRun(path, m.group("sp"), m.group("run") or "", _summary(stem, path), path.stat().st_mtime))
    return sorted(out, key=lambda r: -r.mtime)


def _summary(stem: str, path: Path) -> str:
    try:
        if stem not in ("stark", "pair"):            # stages that keep their page in the JSON
            from qali.pipelines.page_store import has_page
            return "page kept" if has_page(str(path)) else "no page kept: run the stage again once"
        from qali.pipelines.rydberg_runs import load_run, summary
        return summary(stem, load_run(str(path)))
    except Exception:              # an unreadable file is listed, with no description
        return ""


class ReplotDialog(QDialog):
    """Pick saved runs of *stage* to write their pages from."""

    def __init__(self, stage: dict, data_dir: str | Path, species: str | None = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.stem = stage["replot"]["stem"]
        self.data_dir = str(data_dir)
        self.species = species or ""
        self.setWindowTitle(f"Generate plot from saved runs: {stage['label']}")
        self.resize(820, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Saved {stage['label']} outputs in {self.data_dir}. The page is written from "
                             "the run's data, with no calculation."))
        self.all_species = QCheckBox("All species")
        self.all_species.setChecked(not self.species)
        self.all_species.setEnabled(bool(self.species))
        self.all_species.toggled.connect(self._fill)
        top = QHBoxLayout()
        top.addWidget(self.all_species)
        top.addStretch(1)
        other = QPushButton("Other file...")
        other.setToolTip("A saved run anywhere on disk")
        other.clicked.connect(self._pick_file)
        top.addWidget(other)
        lay.addLayout(top)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Species", "Run", "What", "Saved"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        for c in (0, 1, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._sync)
        self.table.itemDoubleClicked.connect(lambda _it: self.accept())
        lay.addWidget(self.table, stretch=1)
        note = stage["replot"].get("note") or ""
        self.options = QLabel(note)
        self.options.setWordWrap(True)
        self.options.setVisible(bool(note))
        lay.addWidget(self.options)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip("File name of the page, in plots/<species>/")
        form.addRow("Page name", self.name)
        lay.addLayout(form)
        self.empty = QLabel("")
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Generate")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.runs: list[SavedRun] = []
        self.extra: list[SavedRun] = []
        self._fill()

    def _fill(self, *_):
        sp = None if self.all_species.isChecked() else self.species
        self.runs = list_runs(self.stem, self.data_dir, sp) + self.extra
        self.table.setRowCount(len(self.runs))
        for i, r in enumerate(self.runs):
            cells = [r.species, r.name or "latest", r.summary or r.path.name,
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(r.mtime))]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setToolTip(str(r.path))
                it.setData(Qt.UserRole, str(r.path))
                self.table.setItem(i, c, it)
        if self.runs:
            self.table.selectRow(0)
        self.empty.setText("" if self.runs else
                           f"No saved {self.stage['label']} outputs here"
                           + (f" for {sp}." if sp else ".") + " Run the stage first.")
        self.empty.setVisible(not self.runs)
        self._sync()

    def _pick_file(self):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Saved {self.stage['label']} runs", self.data_dir,
                                                "JSON (*.json)")
        for p in paths:
            path = Path(p)
            if any(r.path == path for r in self.runs):
                continue
            sp = path.stem.split("_" + self.stem)[0] if ("_" + self.stem) in path.stem else ""
            self.extra.append(SavedRun(path, sp, path.stem, _summary(self.stem, path), path.stat().st_mtime))
        if paths:
            self._fill()
            self.table.selectRow(self.table.rowCount() - 1)

    def _sync(self):
        paths = self.selected_paths()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(paths))
        one = len(paths) == 1
        self.name.setEnabled(one)
        if one:
            self.name.setText(Path(paths[0]).stem)
            self.name.setPlaceholderText("")
        else:
            self.name.clear()
            self.name.setPlaceholderText("each page is named after its JSON")

    def page_name(self) -> str:
        """The page's file name (without .html) for a single run; "" for several (named after the JSON)."""
        if len(self.selected_paths()) != 1:
            return ""
        return re.sub(r"[^A-Za-z0-9_.+-]", "_", self.name.text().strip()).removesuffix(".html")

    def selected_paths(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [str(self.runs[r].path) for r in rows if r < len(self.runs)]
