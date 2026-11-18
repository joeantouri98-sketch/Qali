"""Version stamps on stage outputs, the outdated check, and Right-click > Properties."""

from __future__ import annotations

import json
import os

import pytest

from qali.core import stamps
from qali.core.fileio import write_json


def test_outside_a_stage_run_nothing_is_stamped(tmp_path):
    write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == {"a": 1}
    assert stamps.read_stamp(tmp_path / "x.json") is None


def test_a_stage_run_stamps_json_first_and_html(tmp_path):
    with stamps.stamping("lifetimes", ["Na"]):
        write_json(tmp_path / "Na_lifetimes.json", {"species_id": "Na", "_qali": "old"})
        html = stamps.stamp_html("<!DOCTYPE html><title>x</title>")
    d = json.loads((tmp_path / "Na_lifetimes.json").read_text(encoding="utf-8"))
    assert list(d)[0] == "_qali" and d["_qali"]["stage"] == "lifetimes"
    assert d["_qali"]["revision"] == stamps.STAGE_REVISIONS["lifetimes"]
    assert d["_qali"]["command"] == "qali --pipeline lifetimes Na"
    assert html.startswith("<!DOCTYPE html><meta name=\"qali-stamp\"")
    (tmp_path / "p.html").write_text(html, encoding="utf-8")
    assert stamps.read_stamp(tmp_path / "p.html")["stage"] == "lifetimes"


def test_status_current_outdated_unknown(tmp_path, monkeypatch):
    with stamps.stamping("rydberg", ["Cs"]):
        write_json(tmp_path / "Cs_rydberg.json", {})
    with stamps.stamping("lifetimes", ["Cs"]):
        write_json(tmp_path / "Cs_lifetimes.json", {})
    assert stamps.status(tmp_path / "Cs_lifetimes.json")[0] == "current"
    # an older stage revision
    monkeypatch.setitem(stamps.STAGE_REVISIONS, "lifetimes", 99)
    state, why = stamps.status(tmp_path / "Cs_lifetimes.json")
    assert state == "outdated" and "revision" in why[0]
    monkeypatch.setitem(stamps.STAGE_REVISIONS, "lifetimes", 2)
    # an input rebuilt afterwards
    d = json.loads((tmp_path / "Cs_rydberg.json").read_text(encoding="utf-8"))
    d["_qali"]["generated_utc"] = "2999-01-01T00:00:00+00:00"
    (tmp_path / "Cs_rydberg.json").write_text(json.dumps(d), encoding="utf-8")
    state, why = stamps.status(tmp_path / "Cs_lifetimes.json")
    assert state == "outdated" and "Cs_rydberg.json is newer" in why[0]
    # no stamp
    (tmp_path / "Cs_transitions.json").write_text("{}", encoding="utf-8")
    assert stamps.status(tmp_path / "Cs_transitions.json")[0] == "unknown"


def test_pipeline_outputs_carry_the_stamp(workspace):
    from helpers import run_pipeline

    result = run_pipeline(["hyperfine", "Na"], workspace)
    assert result.returncode == 0, result.stderr
    out = next(workspace.rglob("Na_hyperfine.json"))
    st = stamps.read_stamp(out)
    assert st["stage"] == "hyperfine" and st["version"] and st["generated_utc"]


def test_properties_summary_and_mark(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qali.gui.file_properties import file_summary, status_mark

    with stamps.stamping("lifetimes", ["Na"]):
        write_json(tmp_path / "Na_lifetimes.json", {"species_id": "Na", "n_states": 3, "levels": [1]})
    info = file_summary(tmp_path / "Na_lifetimes.json")
    assert info["stamp"]["stage"] == "lifetimes" and info["status"] == "current"
    assert info["fields"] == {"species_id": "Na", "n_states": 3}
    (tmp_path / "Na_rydberg.json").write_text("{}", encoding="utf-8")
    prefix, tip, italic = status_mark(tmp_path / "Na_rydberg.json")
    assert prefix == "" and italic and "No version stamp" in tip


def test_stamp_is_reproducible_and_strippable(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    with stamps.stamping("lifetimes", ["Na"]):
        write_json(tmp_path / "a.json", {"x": 1})
        html = stamps.stamp_html("<html><head><title>t</title></head></html>")
    assert stamps.read_stamp(tmp_path / "a.json")["generated_utc"] == "2023-11-14T22:13:20+00:00"
    write_json(tmp_path / "b.json", {"x": 1})
    a, b = (tmp_path / "a.json").read_bytes(), (tmp_path / "b.json").read_bytes()
    assert stamps.strip_stamp(a) == b
    assert stamps.strip_stamp(html.encode()) == b"<html><head><title>t</title></head></html>"
    spec = importlib.util.spec_from_file_location("co", Path(__file__).parents[1] / "tools" / "compare_outputs.py")
    co = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(co)
    assert co.strip_stamp(a) == b
