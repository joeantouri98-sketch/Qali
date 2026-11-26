"""
Pages written again from a saved run, for stages whose page is not a plain view of their JSON.
-----------------------------------------------------------------------------
The Blackbody, Hyperfine and Feshbach pages are built from objects of the calculation (networks,
level structures), not only from the saved data.  So when such a stage writes its page it also
keeps, in its data JSON under ``"page"``, what the page was made from:

    {"template": "blackbody_page.html", "title": "...", "data_js": "<the page's data, as JS>",
     "includes": {"PHYS_JS": "feshbach_physics.js"}}

``--from-json`` then renders the page again from that, with the current template (no calculation).
Runs saved before this was kept have no ``"page"`` and need one new run.
"""

from __future__ import annotations

import json
import os
import re

from qali.report_templates import load, render, write_offline_report


def page_record(template: str, title: str, data_js: str, includes: dict[str, str] | None = None,
                file: str | None = None) -> dict:
    """What a page is made from: template name, title, the data literal, included script files and
    the page's own file name (when it differs from the JSON's)."""
    return dict(template=template, title=title, data_js=data_js, includes=dict(includes or {}), file=file)


def page_stem(path: str, name: str | None = None) -> str:
    """The file name (no .html) the page of a saved run gets: *name*, else the page's own, else the JSON's."""
    if name:
        return re.sub(r"[^A-Za-z0-9_.+-]", "_", name.strip()).removesuffix(".html")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        own = (d.get("page") or {}).get("file") if isinstance(d, dict) else None
    except (OSError, ValueError):
        own = None
    return own or os.path.splitext(os.path.basename(path))[0]


def has_page(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and isinstance(d.get("page"), dict)


def write_stored_page(stage: str, path: str, output_dir: str, name: str | None = None, flat: bool = False) -> str:
    """Write the page kept in a saved run: <output_dir>/<species>/<name or the JSON's stem>.html
    (flat: straight into output_dir)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rec = d.get("page") if isinstance(d, dict) else None
    if not isinstance(rec, dict) or "template" not in rec or "data_js" not in rec:
        raise ValueError(f"{os.path.basename(path)} holds no page: run the {stage} stage again to keep one")
    values = {"TITLE": rec.get("title", ""), "DATA_JS": rec["data_js"]}
    for token, fname in (rec.get("includes") or {}).items():
        values[token] = load(fname)
    html = render(rec["template"], **values)
    stem = page_stem(path, name)
    element = d.get("element") or os.path.basename(path).split("_")[0]
    out = os.path.join(output_dir, stem + ".html") if flat else os.path.join(output_dir, str(element), stem + ".html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_offline_report(out, html)
    return out


def pages_only(stage: str, paths: list[str], output_dir: str, name: str | None = None, flat: bool = False) -> int:
    """Write the pages of saved runs; no calculation.  Returns the stage's exit code."""
    failed = 0
    for path in paths:
        try:
            page = write_stored_page(stage, path, output_dir, name, flat)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"  ✗ {path}: {exc}")
            failed += 1
            continue
        print(f"  ✓ {page}")
    return 1 if failed else 0


def add_arguments(p) -> None:
    """The --from-json and --name options of a stage that keeps its page."""
    p.add_argument("--from-json", nargs="+", default=None, metavar="JSON",
                   help="Only write the page of saved run(s) (no calculation); the page is named after the JSON")
    p.add_argument("--name", default=None, help="With one --from-json: the page's file name (default: the JSON's)")
