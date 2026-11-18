"""A figure recipe: what to plot and how it looks, saved as a small JSON file (``.qfig``).

A recipe names a template and its parameters, the species, the size (journal and columns) and the style.
The ``axes`` block overrides what the template chose for the first panel (labels, scales, limits, legend).
:func:`resolve` turns a recipe into a :class:`~qali.figures.spec.FigureSpec`.
"""

from __future__ import annotations

import copy
import json
import os

from . import catalog, templates
from .spec import FigureSpec
from .style import default_style, figure_size, latex_problem

VERSION = 1
LEGEND_LOCS = ["best", "upper right", "upper left", "lower left", "lower right", "center left",
               "center right", "lower center", "upper center", "center"]
SCALES = ["linear", "log", "symlog"]


def new_recipe(template: str = "builder", species: str = "", journal: str = "aps") -> dict:
    return {
        "qfig": VERSION,
        "template": template,
        "species": species,
        "params": {},
        "inputs": {},                 # {stage: Save as name, or a file path}; missing or "" = the usual output
        "size": {"journal": journal, "columns": 1, "aspect": None, "width_cm": None, "height_cm": None},
        "style": default_style(journal),
        "axes": {"xlabel": "", "ylabel": "", "title": "", "xscale": "", "yscale": "", "linthresh": None,
                 "xmin": None, "xmax": None, "ymin": None, "ymax": None,
                 "legend": "auto", "legend_loc": "best", "legend_ncol": 1, "legend_frame": False,
                 "view": {}},           # limits of panels after the first, {"1": {"x": [lo, hi], "y": [lo, hi]}}
    }


def normalized(recipe: dict) -> dict:
    """*recipe* with every key present (older or hand-written files keep working)."""
    base = new_recipe(recipe.get("template", "builder"), recipe.get("species", ""),
                      (recipe.get("size") or {}).get("journal", "aps"))
    out = copy.deepcopy(base)
    for k, v in recipe.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def save_recipe(recipe: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized(recipe), f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_recipe(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "qfig" not in data:
        raise ValueError(f"{path} is not a Qali figure recipe.")
    if data.get("template") not in templates.TEMPLATES:
        raise ValueError(f"{path}: unknown template {data.get('template')!r}.")
    return normalized(data)


def resolve(recipe: dict, data_dir: str) -> FigureSpec:
    """The figure *recipe* describes, from the output files in *data_dir*.
    Raises :class:`~qali.figures.templates.TemplateError` with a message for the user when data is missing."""
    r = normalized(recipe)
    name = r["template"]
    if name not in templates.TEMPLATES:
        raise templates.TemplateError(f"Unknown template {name!r}.")
    fn = templates.TEMPLATES[name][1]
    species = r.get("species") or ""
    used = templates.inputs_of(name, r.get("params") or {})
    inputs = {st: v for st, v in (r.get("inputs") or {}).items() if v and st in used}
    for st, v in inputs.items():
        path = catalog.stage_file(species, st, data_dir, v)
        if not os.path.isfile(path):
            raise templates.TemplateError(
                f"No {st} file {path}." if os.path.isabs(v) else
                f"No {species} {st} run named '{v}' in {data_dir}.")
    with catalog.using_inputs(species, inputs):
        res = fn(r.get("params") or {}, species, data_dir, r["style"])
    w, h = figure_size(r["size"], res.aspect)
    spec = FigureSpec(w, h, dict(r["style"]), res.panels, sharex=res.sharex, notes=list(res.notes),
                      table=res.table)
    if inputs:
        spec.notes.append("Input: " + ", ".join(
            f"{st} {os.path.basename(v) if os.path.isabs(v) else repr(v)}" for st, v in inputs.items()) + ".")
    if r["style"].get("usetex") and latex_problem():
        spec.notes.append(latex_problem())
    _apply_axes(spec, r["axes"])
    return spec


def _apply_axes(spec: FigureSpec, ax: dict) -> None:
    if not spec.panels:
        return
    p = spec.panels[0]
    bottom = spec.panels[-1]
    if ax.get("xlabel"):
        bottom.xlabel = ax["xlabel"]
    if ax.get("ylabel"):
        p.ylabel = ax["ylabel"]
    if ax.get("title"):
        p.title = ax["title"]
    for axis in ("x", "y"):
        scale = ax.get(f"{axis}scale")
        if scale in SCALES:
            for q in spec.panels if (axis == "x" and spec.sharex) else [p]:
                setattr(q, f"{axis}scale", scale)
    if ax.get("linthresh"):
        p.linthresh = float(ax["linthresh"])
    for axis in ("x", "y"):
        lo, hi = ax.get(f"{axis}min"), ax.get(f"{axis}max")
        if lo is not None or hi is not None:
            cur = list(getattr(p, f"{axis}lim") or [None, None])
            setattr(p, f"{axis}lim", [lo if lo is not None else cur[0], hi if hi is not None else cur[1]])
            for q in spec.panels if (axis == "x" and spec.sharex) else [p]:
                q.clip = True
    for k, v in (ax.get("view") or {}).items():
        try:
            q = spec.panels[int(k)]
        except (ValueError, IndexError):
            continue
        if isinstance(v, dict):
            if v.get("x"):
                q.xlim = list(v["x"])
            if v.get("y"):
                q.ylim = list(v["y"])
            q.clip = q.clip or bool(v.get("x") or v.get("y"))
    mode = ax.get("legend", "auto")
    if mode == "off":
        for q in spec.panels:
            q.legend = None
    elif mode in ("auto", "on"):
        for q in spec.panels:
            if q.legend is not None or mode == "on":
                own = int((q.legend or {}).get("ncol") or 1)
                user = int(ax.get("legend_ncol") or 1)
                q.legend = {"loc": ax.get("legend_loc", "best"), "ncol": user if user > 1 else own,   # 1: the figure's
                            "frame": bool(ax.get("legend_frame")),
                            "title": (q.legend or {}).get("title")}
