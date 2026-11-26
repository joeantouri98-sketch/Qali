"""
Run names, titles and pages of the Stark and pair stages.
-----------------------------------------------------------------------------
Each run is filed under a name built from its settings, next to the usual (latest) output:

    data/Rb_stark__30s1-2_mj1-2_ion.json   plots/Rb/Rb_stark__30s1-2_mj1-2_ion.html
    data/Rb_stark.json                      plots/Rb/Rb_stark.html          (the latest run)

so runs with other states, m_j or modes no longer replace each other, and the Figures tab lists
them as named runs.  A page is its template with the run's JSON inside, so it can be written again
from a saved JSON without recomputing (``--from-json``).
"""

from __future__ import annotations

import json
import os
import re

from qali.core.fileio import write_json
from qali.figures.catalog import RUN_SEP
from qali.report_templates import render, write_offline_report

TEMPLATES = {"stark": "stark_page.html", "pair": "pair_page.html"}


def _slug(text: str) -> str:
    """'30s1/2' -> '30s1-2', '-3/2' -> 'm3-2', '2.5' -> '2p5'; letters, digits and '-' only."""
    t = str(text).strip().replace("/", "-").replace(".", "p")
    if t.startswith("-"):
        t = "m" + t[1:]
    return re.sub(r"[^A-Za-z0-9-]", "", t)


def _half(x: float) -> str:
    """m_j as a fraction: 0.5 -> '1/2', -1.5 -> '-3/2'."""
    m2 = round(2 * float(x))
    return f"{m2}/2"


def stark_kind(d: dict) -> str:
    mg = d.get("magnetic")
    if mg:
        base = stark_kind({k: v for k, v in d.items() if k != "magnetic"}).replace("Stark map", "Stark-Zeeman map")
        return base + f" ({mg['B_gauss']:g} G" + (f" at {mg['angle_deg']:g}°)" if mg.get("angle_deg") else ")")
    if d.get("mode") == "ac" and d.get("ac"):
        return f"AC Stark map at {d['ac']['frequency_ghz']:g} GHz"
    if d.get("ionisation") is not None:
        return "Stark map with ionisation"
    if d.get("ramp"):
        return f"Stark map with a {d['ramp']['ramp_ns']:g} ns ramp"
    return "Stark map"


def tag(stage: str, d: dict) -> str:
    """The run's name from its settings."""
    if stage == "stark":
        t = f"{_slug(d['state'])}_mj{_slug(_half(d['mj']))}"
        if d.get("mode") == "ac" and d.get("ac"):
            t += f"_ac{_slug(format(d['ac']['frequency_ghz'], 'g'))}GHz"
            if d["ac"].get("dc_V_cm"):
                t += f"_dc{_slug(format(d['ac']['dc_V_cm'], 'g'))}"
        elif d.get("ionisation") is not None:
            t += "_ion"
        elif d.get("ramp"):
            t += f"_ramp{_slug(format(d['ramp']['ramp_ns'], 'g'))}ns"
        mg = d.get("magnetic")
        if mg:
            t += f"_B{_slug(format(mg['B_gauss'], 'g'))}G"
            if mg.get("angle_deg"):
                t += f"_a{_slug(format(mg['angle_deg'], 'g'))}"
        return t
    if stage == "pair":
        a, b = _slug(d["state1"]), _slug(d["state2"])
        t = f"{a}_{b}_mj{_slug(_half(d['mj1']))}_{_slug(_half(d['mj2']))}"
        th = float(d.get("theta_deg") or 0.0)
        if th:
            t += f"_th{_slug(format(th, 'g'))}"
        fv = float(d.get("field_V_cm") or 0.0)
        if fv:
            t += f"_F{_slug(format(fv, 'g'))}Vcm"
        bg = float(d.get("b_gauss") or 0.0)
        if bg:
            t += f"_B{_slug(format(bg, 'g'))}G"
        if d.get("forster"):
            t += f"_forster{_slug(format(d['forster']['fields_V_cm'][-1], 'g'))}Vcm"
        return t
    raise ValueError(f"no run names for stage {stage!r}")


def title(stage: str, d: dict) -> str:
    label = d.get("label") or d.get("element", "")
    if stage == "stark":
        return f"{label} {d['state']} {stark_kind(d)}"
    fv, bg = float(d.get("field_V_cm") or 0.0), float(d.get("b_gauss") or 0.0)
    at = ", ".join(x for x in (f"{fv:g} V/cm" if fv else "", f"{bg:g} G" if bg else "") if x)
    return f"{label} {d['state1']} + {d['state2']} pair potential" + (f" at {at}" if at else "")


def summary(stage: str, d: dict) -> str:
    """One line on what a run is, for lists of runs."""
    try:
        if stage == "stark":
            F = d.get("fields_V_cm") or [0.0]
            unit = "V/cm (AC amplitude)" if d.get("mode") == "ac" else "V/cm"
            if d.get("mode") == "ac" and d.get("ac"):
                mode = f"AC {d['ac']['frequency_ghz']:g} GHz"
            else:
                mode = "static, ionisation" if d.get("ionisation") is not None else "static"
                if d.get("ramp"):
                    mode = f"static, {d['ramp']['ramp_ns']:g} ns ramp"
            mg = d.get("magnetic")
            if mg:
                mode += f", B {mg['B_gauss']:g} G" + (f" at {mg['angle_deg']:g}°" if mg.get("angle_deg") else "")
            return f"{d['state']}, m_j {_half(d['mj'])}, {mode}, {F[0]:g}–{F[-1]:.4g} {unit}, {len(F)} points"
        if stage == "pair":
            R = d.get("R_um") or [0.0]
            th = float(d.get("theta_deg") or 0.0)
            fv = float(d.get("field_V_cm") or 0.0)
            fo = d.get("forster")
            return (f"{d['state1']} + {d['state2']}, m_j {_half(d['mj1'])}, {_half(d['mj2'])}, θ {th:g}°, "
                    + (f"F {fv:g} V/cm, " if fv else "")
                    + (f"B {float(d.get('b_gauss') or 0):g} G, " if d.get("b_gauss") else "")
                    + f"R {R[0]:.3g}–{R[-1]:.3g} µm, C6 {d['c6']['chosen']:.4g} GHz µm⁶"
                    + (f", Förster map to {fo['fields_V_cm'][-1]:g} V/cm" if fo else ""))
    except (KeyError, TypeError, ValueError, IndexError):
        return ""
    return ""


def _js(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_page(stage: str, d: dict, path: str) -> str:
    html = render(TEMPLATES[stage], TITLE=title(stage, d), DATA_JS=_js(d))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_offline_report(path, html)
    return path


def write_outputs(stage: str, element: str, payload: dict, data_dir: str, output_dir: str,
                  progress=print) -> tuple[str, str]:
    """The run's JSON and page under its own name, and again as the usual (latest) output.
    Returns (named JSON, named page)."""
    name = f"{element}_{stage}{RUN_SEP}{tag(stage, payload)}"
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, element)
    paths = []
    for stem in (name, f"{element}_{stage}"):
        jp = os.path.join(data_dir, stem + ".json")
        write_json(jp, payload, indent=None)
        pp = write_page(stage, payload, os.path.join(out_dir, stem + ".html"))
        paths.append((jp, pp))
    progress(f"  ✓ {paths[0][0]}\n  ✓ {paths[0][1]}\n  (also the latest: {os.path.basename(paths[1][0])}, "
             f"{os.path.basename(paths[1][1])})")
    return paths[0]


def load_run(path: str) -> dict:
    """A saved run without its version stamp."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise TypeError(f"{path} is not a stage output")
    d.pop("_qali", None)
    return d


def page_from_json(stage: str, path: str, output_dir: str, name: str | None = None) -> str:
    """Write the page of a saved run, named *name* or after its JSON
    (Rb_stark__x.json -> plots/Rb/Rb_stark__x.html)."""
    d = load_run(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if name:
        stem = re.sub(r"[^A-Za-z0-9_.+-]", "_", name.strip()).removesuffix(".html")
    element = d.get("element") or os.path.basename(path).split("_")[0]
    needed = {"stark": ("state", "fields_V_cm", "energies_ghz"), "pair": ("state1", "R_um", "energies_ghz")}[stage]
    missing = [k for k in needed if k not in d]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is not a {stage} output (no {', '.join(missing)})")
    return write_page(stage, d, os.path.join(output_dir, element, stem + ".html"))
