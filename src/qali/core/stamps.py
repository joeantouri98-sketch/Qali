"""Version stamps on stage outputs, and what they say about a file being current.

Every JSON a stage writes (through :func:`qali.core.fileio.write_json`) starts with a ``"_qali"`` block,
and every HTML page (through ``report_templates.write_offline_report``) carries the same block in a
``<meta name="qali-stamp">`` tag:

    {"version": "3.0.0", "stage": "lifetimes", "revision": 2, "generated_utc": "...", "command": "..."}

``revision`` is the stage's output revision (:data:`STAGE_REVISIONS`). Bump it whenever a stage's
output changes in a way that makes older files wrong (new levels, changed physics), so the GUI can
tell an outdated file from a current one even when the package version stays the same.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from contextvars import ContextVar
from datetime import datetime, UTC
from pathlib import Path

STAMP_KEY = "_qali"

# Bump a stage's number when its outputs change meaning or physics (and say why in the CHANGELOG).
STAGE_REVISIONS: dict[str, int] = {
    "orca_to_json": 1, "dhf": 1, "brueckner": 1,
    "rydberg": 1, "transitions": 1,
    "lifetimes": 2,          # 2: ground ns->np from the calibrated model (2026-11-17)
    "polarizability": 3,     # 2: idem; core-polarized continuum dipole. 3: every magic wavelength, practicality
    "blackbody": 3,          # 3: shift-vs-T sweep by the exact principal value (no zig-zag above ~600 K)
    "tweezer": 2, "feshbach": 2, "compare_elements": 2,
    "hyperfine": 1, "plotinteractive": 1, "spectra": 1, "orbital3d": 1,
    "polarizability_page": 2,   # 2: practical-only filter and photons/s for the magic wavelengths
    "scattering_rate": 1, "figure": 1, "runorca": 1,
}

# Which stage writes <El>_<suffix>.json, and the stage outputs each one reads (same species).
_JSON_STAGE = {
    "rydberg": "rydberg", "transitions": "transitions", "lifetimes": "lifetimes",
    "polarizability": "polarizability", "blackbody": "blackbody", "tweezer": "tweezer",
    "hyperfine": "hyperfine", "feshbach_out": "feshbach", "dhf": "dhf", "brueckner": "brueckner",
}
UPSTREAM: dict[str, tuple[str, ...]] = {
    "transitions": ("rydberg",),
    "lifetimes": ("rydberg", "transitions"),
    "polarizability": ("rydberg", "transitions", "lifetimes"),
    "blackbody": ("polarizability", "lifetimes"),
    "tweezer": ("polarizability", "lifetimes"),
    "feshbach_out": ("polarizability",),
}

_ACTIVE: ContextVar[dict | None] = ContextVar("qali_stamp", default=None)
_META = re.compile(r'<meta name="qali-stamp" content=\'([^\']*)\'>')
_JSON_STAMP = re.compile(rb'^\{\s*"_qali": \{[^{}]*\},\s*')
_HTML_STAMP = re.compile(rb'<meta name="qali-stamp" content=\'[^\']*\'>')


def strip_stamp(data: bytes) -> bytes:
    """The bytes of an output without its stamp (to compare two runs of the same code)."""
    return _HTML_STAMP.sub(b"", _JSON_STAMP.sub(b"{\n  ", data, count=1), count=1)


@contextlib.contextmanager
def stamping(stage: str, argv: list[str]):
    """Stamp everything written inside the block as an output of *stage* run with *argv*."""
    from qali._version import __version__

    token = _ACTIVE.set({"version": __version__, "stage": stage, "revision": STAGE_REVISIONS.get(stage, 1),
                         "command": " ".join(["qali", "--pipeline", stage, *argv])})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def current() -> dict | None:
    """The stamp for a file written now, or None outside a stage run."""
    base = _ACTIVE.get()
    if base is None:
        return None
    epoch = os.environ.get("SOURCE_DATE_EPOCH", "")       # reproducible builds: a fixed time if asked
    when = datetime.fromtimestamp(int(epoch), UTC) if epoch.isdigit() else datetime.now(UTC)
    return {**base, "generated_utc": when.isoformat(timespec="seconds")}


def stamp_payload(payload):
    """*payload* with the stamp as its first key (a dict, inside a stage run); unchanged otherwise."""
    st = current()
    if st is None or not isinstance(payload, dict):
        return payload
    return {STAMP_KEY: st, **{k: v for k, v in payload.items() if k != STAMP_KEY}}


def stamp_html(html: str) -> str:
    st = current()
    if st is None:
        return html
    tag = f"<meta name=\"qali-stamp\" content='{json.dumps(st)}'>"
    k = html.find("<head", 0, 4000)
    if k < 0:                                  # pages that start with their content: the tag goes first,
        d = html.find(">") + 1 if html.lstrip()[:9].lower() == "<!doctype" else 0   # after any doctype
        return html[:d] + tag + html[d:]
    i = html.find(">", k) + 1
    return html[:i] + tag + html[i:]


def read_stamp(path: str | Path) -> dict | None:
    """The stamp of a JSON or HTML output (read from the start of the file only), or None."""
    p = Path(path)
    try:
        with p.open(encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
    except OSError:
        return None
    if p.suffix.lower() == ".json":
        i = head.find(f'"{STAMP_KEY}"')
        if i < 0 or i > 200:
            return None
        j = head.find("{", i)
        try:
            obj, _end = json.JSONDecoder().raw_decode(head[j:])
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None
    m = _META.search(head)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            return None
    return None


def _species_and_kind(path: Path) -> tuple[str, str]:
    stem = path.stem
    for suffix in sorted(_JSON_STAGE, key=len, reverse=True):
        if stem.endswith("_" + suffix):
            return stem[: -len(suffix) - 1], suffix
    return stem, ""


def _seeded_copy(path: Path) -> bool:
    """True if *path* is still the copy of bundled data the workspace seeding made."""
    from qali.core.workspace import SEEDED_FILE

    try:
        record = json.loads((path.parent / SEEDED_FILE).read_text(encoding="utf-8"))
        st = path.stat()
    except (OSError, ValueError):
        return False
    entry = record.get(path.name)
    return bool(entry) and list(entry.get("copy") or []) == [st.st_size, st.st_mtime_ns]


def status(path: str | Path) -> tuple[str, list[str]]:
    """('current' | 'outdated' | 'unknown', reasons) for a stage output.

    Outdated: its stage's revision has moved on, or an output it was built from is newer.
    Unknown: no stamp (made before stamps, by hand, or not a stage output)."""
    p = Path(path)
    st = read_stamp(p)
    species, kind = _species_and_kind(p)
    if st is None:
        if _seeded_copy(p):
            return "current", ["Bundled with this Qali (seeded, unchanged)."]
        if kind or p.suffix.lower() == ".html":
            return "unknown", ["No version stamp: made by a Qali older than 2026-11-17, or edited by hand. "
                               "Rerun its stage to be sure it is current."]
        return "unknown", []
    reasons = []
    now = STAGE_REVISIONS.get(st.get("stage", ""), 1)
    if int(st.get("revision") or 0) < now:
        reasons.append(f"Made by an older {st.get('stage')} stage (revision {st.get('revision')}, now {now}).")
    for up in UPSTREAM.get(kind, ()):
        q = p.with_name(f"{species}_{up}.json")
        if not q.is_file():
            continue
        ust = read_stamp(q)
        newer = (ust.get("generated_utc", "") > st.get("generated_utc", "")) if ust else False
        if newer:
            reasons.append(f"{q.name} is newer: rerun {st.get('stage')}.")
    return ("outdated" if reasons else "current"), reasons
