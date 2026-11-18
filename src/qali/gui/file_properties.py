"""Right-click > Properties... on a Plots / Data / Shared entry: dates, size, version stamp and status."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout

from qali.core import stamps

_STATUS_TEXT = {"current": "Current", "outdated": "Outdated", "unknown": "Unknown"}
_SUMMARY_KEYS = ("species_id", "element", "label", "kind", "charge", "method", "n_start", "n_max", "n_states",
                 "n_transitions", "gs_label", "isotope", "alpha0_au", "C6_au", "T_K", "units")


def _size(size: int) -> str:
    n = float(size)
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n):,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return str(size)


def _local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def created_time(st: os.stat_result) -> float | None:
    """The file's creation time where the system records one (Windows, macOS, some Linux file systems)."""
    birth = getattr(st, "st_birthtime", None)
    if birth:
        return float(birth)
    return float(st.st_ctime) if sys.platform == "win32" else None


def file_summary(path: Path) -> dict:
    """Everything the dialog shows, as plain values (also used by the tests)."""
    st = path.stat()
    out: dict[str, Any] = {"name": path.name, "folder": str(path.parent), "size": _size(st.st_size),
           "created": (_local(c) if (c := created_time(st)) else "not recorded by this system"),
           "modified": _local(st.st_mtime), "stamp": stamps.read_stamp(path)}
    out["status"], out["reasons"] = stamps.status(path)
    fields = {}
    if path.suffix.lower() == ".json" and st.st_size < 64 * 1024 * 1024:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            fields = {k: data[k] for k in _SUMMARY_KEYS
                      if data.get(k) is not None and not isinstance(data[k], (dict, list))}
            inputs = data.get("inputs")
            if isinstance(inputs, dict) and inputs:
                fields["inputs"] = ", ".join(f"{k}: {v}" for k, v in inputs.items() if not isinstance(v, (dict, list)))
    out["fields"] = fields
    return out


class FilePropertiesDialog(QDialog):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Properties: {path.name}")
        self.setMinimumWidth(600)
        info = file_summary(path)
        v = QVBoxLayout(self)

        def form(title: str) -> QFormLayout:
            box = QGroupBox(title)
            f = QFormLayout(box)
            f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            v.addWidget(box)
            return f

        def value(text: str) -> QLabel:
            lab = QLabel(text)
            lab.setWordWrap(len(text) > 48)       # short values on one line; paths, commands and reasons wrap
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            return lab

        f = form("File")
        for key, label in (("name", "Name"), ("folder", "Folder"), ("size", "Size"), ("created", "Created"),
                           ("modified", "Modified")):
            f.addRow(label, value(info[key]))

        f = form("Made by")
        st = info["stamp"]
        if st:
            gen = st.get("generated_utc", "")
            try:
                gen = datetime.fromisoformat(gen).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
            now = stamps.STAGE_REVISIONS.get(st.get("stage", ""), 1)
            f.addRow("Qali version", value(str(st.get("version", "?"))))
            f.addRow("Stage", value(str(st.get("stage", "?"))))
            f.addRow("Stage revision", value(f"{st.get('revision', '?')} (current: {now})"))
            f.addRow("Generated", value(gen))
            f.addRow("Command", value(str(st.get("command", ""))))
        elif not info["reasons"]:
            f.addRow(value("No version stamp."))
        status = QLabel(_STATUS_TEXT.get(info["status"], info["status"]))
        status.setObjectName("fileStatus")
        f.addRow("Status", status)
        for reason in info["reasons"]:
            f.addRow("", value(reason))

        if info["fields"]:
            f = form("Contents")
            for k, val in info["fields"].items():
                f.addRow(k, value(str(val)))

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        v.addWidget(buttons)


def status_mark(path: Path) -> tuple[str, str, bool]:
    """(prefix for the list entry, tooltip, italic) for an output file in the Plots / Data / Shared lists."""
    state, reasons = stamps.status(path)
    st = stamps.read_stamp(path)
    if state == "outdated":
        return "⚠ ", "Outdated. " + " ".join(reasons), False
    if st is None:
        return "", " ".join(reasons), bool(reasons)
    when = st.get("generated_utc", "")[:10]
    return "", f"Made by Qali {st.get('version')} ({st.get('stage')}), {when}.", False


__all__ = ["FilePropertiesDialog", "created_time", "file_summary", "status_mark"]
