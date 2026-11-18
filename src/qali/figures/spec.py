"""A resolved figure: panels holding plain drawing operations with their data.

Templates and the plot builder turn a recipe into a :class:`FigureSpec`. The same spec is drawn on the
screen (:func:`draw`), saved (:func:`save`) and written out as a standalone Matplotlib script
(:func:`to_script`), so the script always reproduces the figure the user saw.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import TypeGuard

import numpy as np

from .style import rc_params, tex_safe

# Axes methods an operation may call, and the keyword arguments a value may name a transform for.
OPS = {"plot", "errorbar", "scatter", "step", "bar", "barh", "fill_between", "axhline", "axvline", "hlines",
       "vlines", "text", "annotate", "axvspan", "axhspan", "pcolormesh", "contour", "contourf", "tricontour",
       "tricontourf"}
TARGETS = ("y2", "inset")                    # an operation drawn on the right-hand axis or the inset
SAVE_FORMATS = {"pdf": "PDF (vector)", "svg": "SVG (vector)", "eps": "EPS (vector)",
                "png": "PNG (600 dpi)", "tiff": "TIFF (600 dpi)"}


class RenderError(RuntimeError):
    """Drawing failed for a reason the user can fix (LaTeX setup, a bad value); the message says how."""


def explain(exc: Exception) -> str:
    """A short message for a drawing failure."""
    text = str(exc)
    if "cm-super" in text:
        return ("LaTeX is missing the cm-super font package that Matplotlib needs. Install it (TeX Live: "
                "tlmgr install cm-super; MiKTeX installs it on first use) or turn off LaTeX.")
    if "latex was not able" in text or "LaTeX" in text and "Error" in text:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip().startswith("!")), "")
        return "LaTeX could not typeset a label" + (f" ({first.lstrip('! ')})" if first else "") + \
            ". Check the label text, or turn off LaTeX."
    return text.splitlines()[0] if text else type(exc).__name__


@dataclass
class Panel:
    ops: list[dict] = field(default_factory=list)        # {"op": "plot", "args": [...], "kw": {...}}
    xlabel: str = ""
    ylabel: str = ""
    xscale: str = "linear"
    yscale: str = "linear"
    linthresh: float | None = None                      # symlog
    xlim: list | None = None
    ylim: list | None = None
    legend: dict | None = None                          # {"loc": "best", "ncol": 1, "frame": False}
    title: str = ""
    xticks: list | None = None
    xticklabels: list | None = None
    xminor: bool = True
    yticks: list | None = None
    yticklabels: list | None = None
    yminor: bool = True
    spines: list | None = None                          # spines to hide
    height_ratio: float = 1.0
    clip: bool = False                                  # clip text and arrows to the axes (a zoomed view)
    y2: dict | None = None                              # right-hand axis: {"label", "scale", "lim"}
    inset: dict | None = None                           # {"bounds": [x0, y0, w, h], "xlim", "ylim", "xscale", "yscale"}
    axis_break: dict | None = None                      # {"axis": "x" | "y", "at": [a, b]}: the range a..b left out
    colorbar: str | None = None                         # label of a colour bar for the last heat map or contour

    def add(self, op: str, *args, on: str | None = None, **kw) -> None:
        if op not in OPS:
            raise ValueError(f"unknown drawing operation {op}")
        if on is not None and on not in TARGETS:
            raise ValueError(f"unknown axes {on}")
        d = {"op": op, "args": list(args), "kw": kw}
        if on:
            d["on"] = on
        self.ops.append(d)


@dataclass
class FigureSpec:
    width_in: float
    height_in: float
    style: dict
    panels: list[Panel] = field(default_factory=list)
    sharex: bool = False
    notes: list[str] = field(default_factory=list)       # shown under the canvas (data used, warnings)
    table: tuple[list[str], list[list]] | None = None    # a template's own CSV (Grotrian: levels and lines)


@contextlib.contextmanager
def style_context(style: dict, raster: bool = True):
    import matplotlib as mpl

    with mpl.rc_context(rc_params(style, raster)):
        yield


def _usetex(style: dict, raster: bool = True) -> bool:
    from .style import latex_available

    return bool(style.get("usetex")) and latex_available(raster)


def draw(spec: FigureSpec, fig=None, raster: bool = True):
    """Draw *spec* into *fig* (a new ``matplotlib.figure.Figure`` when None) and return it. *raster*: drawn
    for the screen or PNG/TIFF (decides whether LaTeX can be used, see :func:`~.style.latex_available`)."""
    from matplotlib.figure import Figure

    tex = _usetex(spec.style, raster)
    with style_context(spec.style, raster):
        if fig is None:
            fig = Figure(figsize=(spec.width_in, spec.height_in))
        else:
            fig.clear()
            fig.set_size_inches(spec.width_in, spec.height_in, forward=False)
        fig.set_layout_engine("constrained", h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
        n = max(len(spec.panels), 1)
        gs = fig.add_gridspec(n, 1, height_ratios=[p.height_ratio for p in spec.panels] or [1])
        share = sharex_ok(spec)
        first = None
        heads = []
        for i, panel in enumerate(spec.panels):
            brk = panel.axis_break
            axs = make_panel_axes(fig, gs[i], brk, first if share else None)
            first = first or axs[0]
            mappable = None
            for ax in axs:
                mappable = _draw_panel(ax, panel, tex, labels=not brk) or mappable
            if brk:
                break_axes(axs, brk, tex_safe(panel.xlabel, tex), tex_safe(panel.ylabel, tex),
                           panel.ylim if brk["axis"] == "y" else panel.xlim)
            head = head_axes(axs, brk)
            heads.append(head)
            twin = _draw_extras(axs, panel, tex, fig, mappable)
            if panel.title:
                head.set_title(tex_safe(panel.title, tex))
            _legend(head, panel, tex, twin)
            if n > 1 and spec.style.get("panel_labels", True):
                if not brk and panel.yscale == "linear" and not has_range(panel.ylim):
                    lo, hi = head.get_ylim()
                    head.set_ylim(lo, hi + 0.14 * (hi - lo))       # headroom for the panel label
                head.text(0.015, 0.97, f"({chr(97 + i)})", transform=head.transAxes, va="top", ha="left")
            if share and i < n - 1:
                for ax in axs:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)
        fig._qali_panels = heads if not any(p.axis_break for p in spec.panels) else \
            [None if p.axis_break else h for p, h in zip(spec.panels, heads)]
    return fig


def sharex_ok(spec: FigureSpec) -> bool:
    """Panels share x unless one has a broken x axis."""
    return spec.sharex and not any(p.axis_break and p.axis_break.get("axis") == "x" for p in spec.panels)


def tidy_log_axis(ax, axis: str) -> None:
    """On a log axis spanning under 2.5 decades, label 1, 2, 5 x 10^k as plain numbers and drop the minor
    labels (Matplotlib would label every minor tick, and they collide)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    if min(lim) <= 0 or max(lim) / min(lim) > 10 ** 2.5:
        return
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


_TIDY_SRC = (
    "def tidy_log_axis(ax, axis):\n"
    "    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter\n"
    "    lim = ax.get_xlim() if axis == 'x' else ax.get_ylim()\n"
    "    if min(lim) <= 0 or max(lim) / min(lim) > 10 ** 2.5:\n"
    "        return\n"
    "    a = ax.xaxis if axis == 'x' else ax.yaxis\n"
    "    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))\n"
    "    a.set_major_formatter(FuncFormatter(lambda v, _pos: f'{v:g}'))\n"
    "    a.set_minor_formatter(NullFormatter())\n"
)


def clip_to_axes(ax) -> None:
    """Keep text and arrows inside the axes (a zoomed view would otherwise draw them over the margins)."""
    for t in ax.texts:
        t.set_clip_box(ax.bbox)
        t.set_clip_on(True)
        arrow = getattr(t, "arrow_patch", None)
        if arrow is not None:
            arrow.set_clip_box(ax.bbox)
            arrow.set_clip_on(True)


_CLIP_SRC = (
    "def clip_to_axes(ax):\n"
    "    for t in ax.texts:\n"
    "        t.set_clip_box(ax.bbox)\n"
    "        t.set_clip_on(True)\n"
    "        arrow = getattr(t, 'arrow_patch', None)\n"
    "        if arrow is not None:\n"
    "            arrow.set_clip_box(ax.bbox)\n"
    "            arrow.set_clip_on(True)\n"
)


# Axes layout shared by the screen and the exported script (the script carries this source verbatim).
_AXES_SRC = (
    "def make_panel_axes(fig, cell, brk=None, share=None):\n"
    "    \"\"\"The axes of one panel: one, or two for a broken axis (low side first).\"\"\"\n"
    "    if not brk:\n"
    "        return [fig.add_subplot(cell, sharex=share)]\n"
    "    if brk['axis'] == 'y':\n"
    "        g = cell.subgridspec(2, 1, hspace=0.05)\n"
    "        hi = fig.add_subplot(g[0], sharex=share)\n"
    "        return [fig.add_subplot(g[1], sharex=hi), hi]\n"
    "    g = cell.subgridspec(1, 2, wspace=0.05)\n"
    "    lo = fig.add_subplot(g[0])\n"
    "    return [lo, fig.add_subplot(g[1], sharey=lo)]\n"
    "\n"
    "\n"
    "def data_along(ax, axis):\n"
    "    \"\"\"The finite data values of the lines and collections on *ax* along *axis* ('x' or 'y').\"\"\"\n"
    "    k = 0 if axis == 'x' else 1\n"
    "    vals = [np.asarray(ln.get_xydata(), float)[:, k] for ln in ax.get_lines() if len(ln.get_xydata())]\n"
    "    for c in ax.collections:\n"
    "        for path in c.get_paths():\n"
    "            vals.append(np.asarray(path.vertices, float)[:, k])\n"
    "        off = c.get_offsets()\n"
    "        if len(off):\n"
    "            vals.append(np.asarray(off, float)[:, k])\n"
    "    v = np.concatenate(vals) if vals else np.array([])\n"
    "    return v[np.isfinite(v)]\n"
    "\n"
    "\n"
    "def break_axes(axs, brk, xlabel, ylabel, lim=None):\n"
    "    \"\"\"Leave out brk['at'] between the two axes of a broken panel, with the usual slanted marks.\n"
    "    Each side spans its own data (5 % margin) unless *lim* sets the outer ends.\"\"\"\n"
    "    lo, hi = axs\n"
    "    a, b = sorted(brk['at'])\n"
    "    axis = brk['axis']\n"
    "    v = data_along(lo, axis)\n"
    "    low, high = v[v <= a], v[v >= b]\n"
    "    get = lo.get_ylim if axis == 'y' else lo.get_xlim\n"
    "    start = low.min() - 0.05 * (a - low.min()) if low.size and low.min() < a else get()[0]\n"
    "    end = high.max() + 0.05 * (high.max() - b) if high.size and high.max() > b else b + 0.1 * (b - a)\n"
    "    if lim and lim[0] is not None:\n"
    "        start = lim[0]\n"
    "    if lim and lim[1] is not None:\n"
    "        end = lim[1]\n"
    "    mk = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle='none', color='k', mec='k', mew=0.8,\n"
    "              clip_on=False)\n"
    "    if axis == 'y':\n"
    "        lo.set_ylim(min(start, a - 1e-9 * abs(a) - 1e-300), a)\n"
    "        hi.set_ylim(b, max(end, b + 1e-9 * abs(b) + 1e-300))\n"
    "        lo.spines['top'].set_visible(False)\n"
    "        hi.spines['bottom'].set_visible(False)\n"
    "        lo.tick_params(axis='x', which='both', top=False)\n"
    "        hi.tick_params(axis='x', which='both', bottom=False, labelbottom=False)\n"
    "        hi.plot([0, 1], [0, 0], transform=hi.transAxes, **mk)\n"
    "        lo.plot([0, 1], [1, 1], transform=lo.transAxes, **mk)\n"
    "        lo.set_xlabel(xlabel)\n"
    "        hi.set_xlabel('')\n"
    "        lo.set_ylabel(ylabel)\n"
    "        lo.yaxis.label.set_y(1.0)\n"
    "        hi.set_ylabel('')\n"
    "        return\n"
    "    lo.set_xlim(min(start, a - 1e-9 * abs(a) - 1e-300), a)\n"
    "    hi.set_xlim(b, max(end, b + 1e-9 * abs(b) + 1e-300))\n"
    "    lo.spines['right'].set_visible(False)\n"
    "    hi.spines['left'].set_visible(False)\n"
    "    lo.tick_params(axis='y', which='both', right=False)\n"
    "    hi.tick_params(axis='y', which='both', left=False, labelleft=False)\n"
    "    lo.plot([1, 1], [0, 1], transform=lo.transAxes, **mk)\n"
    "    hi.plot([0, 0], [0, 1], transform=hi.transAxes, **mk)\n"
    "    lo.set_xlabel(xlabel)\n"
    "    lo.xaxis.label.set_x(1.0)\n"
    "    hi.set_xlabel('')\n"
    "    lo.set_ylabel(ylabel)\n"
    "    hi.set_ylabel('')\n"
    "\n"
    "\n"
    "def head_axes(axs, brk=None):\n"
    "    \"\"\"The axes that carry the panel label and the legend: the upper one of a broken y axis.\"\"\"\n"
    "    return axs[-1] if brk and brk['axis'] == 'y' else axs[0]\n"
)
_AXES_NS: dict = {}
_AXES_NS["np"] = np
exec(_AXES_SRC, _AXES_NS)                       # noqa: S102 - our own constant source, shared with the script
make_panel_axes = _AXES_NS["make_panel_axes"]
break_axes = _AXES_NS["break_axes"]
head_axes = _AXES_NS["head_axes"]


def _draw_ops(ax, ops: list[dict], tex: bool):
    """Draw *ops* on *ax*; returns the last colour-mapped artist (for a colour bar), or None."""
    mappable = None
    for op in ops:
        kw = dict(op.get("kw", {}))
        if kw.pop("transform", None) == "axes":
            kw["transform"] = ax.transAxes
        if "label" in kw and isinstance(kw["label"], str):
            kw["label"] = tex_safe(kw["label"], tex)
        args = list(op.get("args", []))
        if op["op"] in ("text", "annotate"):
            idx = 2 if op["op"] == "text" else 0
            if len(args) > idx and isinstance(args[idx], str):
                args[idx] = tex_safe(args[idx], tex)
        out = getattr(ax, op["op"])(*args, **kw)
        if op["op"] in FILLED_OPS:
            mappable = out
    return mappable


FILLED_OPS = ("pcolormesh", "contourf", "tricontourf")          # what a colour bar shows (not contour lines)


def has_range(lim) -> TypeGuard[list]:
    """A range to set: at least one end given, and not two equal ends (matplotlib would widen those and warn)."""
    return bool(lim) and any(v is not None for v in lim) and not (lim[0] is not None and lim[0] == lim[1])


def _set_scale(ax, axis: str, scale: str, linthresh) -> None:
    if scale == "symlog":
        getattr(ax, f"set_{axis}scale")("symlog", linthresh=linthresh or 1.0)
    else:
        getattr(ax, f"set_{axis}scale")(scale or "linear")


def _draw_extras(axs, p: Panel, tex: bool, fig, mappable):
    """The right-hand axis, the inset and the colour bar of panel *p* (drawn on its first axes)."""
    ax = axs[0]
    twin = None
    if p.y2 is not None:
        twin = ax.twinx()
        m = _draw_ops(twin, [o for o in p.ops if o.get("on") == "y2"], tex)
        mappable = m or mappable
        _set_scale(twin, "y", p.y2.get("scale", "linear"), p.y2.get("linthresh"))
        lim = p.y2.get("lim")
        if has_range(lim):
            twin.set_ylim(*lim)
        if p.y2.get("scale") == "log":
            tidy_log_axis(twin, "y")
        twin.set_ylabel(tex_safe(p.y2.get("label", ""), tex))
    if p.inset is not None:
        ins = ax.inset_axes(p.inset.get("bounds") or [0.58, 0.55, 0.38, 0.4])
        m = _draw_ops(ins, [o for o in p.ops if o.get("on") == "inset"], tex)
        mappable = m or mappable
        for axis in ("x", "y"):
            _set_scale(ins, axis, p.inset.get(f"{axis}scale", "linear"), None)
            lim = p.inset.get(f"{axis}lim")
            if has_range(lim):
                getattr(ins, f"set_{axis}lim")(*lim)
        ins.tick_params(labelsize="small")
        if p.inset.get("mark", True) and p.inset.get("xlim") and p.inset.get("ylim"):
            ax.indicate_inset_zoom(ins, edgecolor="0.4", linewidth=0.6)
    if p.colorbar is not None and mappable is not None:
        cb = fig.colorbar(mappable, ax=axs, pad=0.02)
        cb.set_label(tex_safe(p.colorbar, tex))
    return twin


def _draw_panel(ax, p: Panel, tex: bool, labels: bool = True):
    """Draw the main operations and the axis settings of *p* on *ax*; returns the colour-mapped artist."""
    mappable = _draw_ops(ax, [o for o in p.ops if not o.get("on")], tex)
    _set_scale(ax, "x", p.xscale, p.linthresh)
    _set_scale(ax, "y", p.yscale, p.linthresh)
    if p.xticks is not None:                    # before the limits: setting ticks widens the view to them
        ax.set_xticks(p.xticks)
        if p.xticklabels is not None:
            ax.set_xticklabels([tex_safe(t, tex) for t in p.xticklabels])
    if p.yticks is not None:
        ax.set_yticks(p.yticks)
        if p.yticklabels is not None:
            ax.set_yticklabels([tex_safe(t, tex) for t in p.yticklabels])
    if has_range(p.xlim):
        ax.set_xlim(*p.xlim)
    if has_range(p.ylim):
        ax.set_ylim(*p.ylim)
    for axis, scale in (("x", p.xscale), ("y", p.yscale)):
        if scale == "log":
            tidy_log_axis(ax, axis)
    if p.clip:
        clip_to_axes(ax)
    if not p.xminor:
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    if not p.yminor:
        ax.tick_params(axis="y", which="minor", left=False, right=False)
    for s in p.spines or []:
        ax.spines[s].set_visible(False)
        if s in ("top", "bottom"):
            ax.tick_params(axis="x", which="both", **{s: False})
        else:
            ax.tick_params(axis="y", which="both", **{s: False})
    if labels:
        ax.set_xlabel(tex_safe(p.xlabel, tex))
        ax.set_ylabel(tex_safe(p.ylabel, tex))
    return mappable


def _legend(ax, p: Panel, tex: bool, twin=None) -> None:
    handles, names = ax.get_legend_handles_labels()
    if twin is not None:
        h2, n2 = twin.get_legend_handles_labels()
        handles, names = handles + h2, names + n2
    if p.legend is not None and handles:
        (twin or ax).legend(handles, names, loc=p.legend.get("loc", "best"), ncol=int(p.legend.get("ncol", 1)),
                            frameon=bool(p.legend.get("frame", False)),
                            title=tex_safe(p.legend.get("title") or "", tex) or None)


def save(spec: FigureSpec, path: str, dpi: int = 600) -> str:
    """Draw *spec* and write it to *path*; the format follows the extension."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext not in SAVE_FORMATS:
        raise ValueError(f"Save as one of: {', '.join('.' + e for e in SAVE_FORMATS)}.")
    import logging

    raster = ext in ("png", "tiff")
    fig = draw(spec, raster=raster)
    quiet = logging.getLogger("fontTools")
    level = quiet.level
    quiet.setLevel(logging.ERROR)      # fontTools warns about the old timestamps inside the cmr10 font
    try:
        with style_context(spec.style, raster):
            extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
            fig.savefig(path, dpi=dpi if ext in ("png", "tiff") else None, format=ext, **extra)
    except (RuntimeError, ValueError) as exc:
        raise RenderError(explain(exc)) from exc
    finally:
        quiet.setLevel(level)
    return path


def data_rows(spec: FigureSpec):
    """(panel, series, x, y, y error) for every plotted data series of *spec* (for a CSV of the figure)."""
    for i, p in enumerate(spec.panels):
        for k, op in enumerate(p.ops):
            if op["op"] not in ("plot", "errorbar", "step", "bar", "scatter"):
                continue
            args = op.get("args", [])
            if len(args) < 2 or not hasattr(args[0], "__len__") or isinstance(args[0], str):
                continue
            label = op.get("kw", {}).get("label") or f"series {k + 1}"
            yerr = op.get("kw", {}).get("yerr")
            for j, (x, y) in enumerate(zip(args[0], args[1])):
                e = yerr[j] if yerr is not None and hasattr(yerr, "__len__") else ""
                yield [chr(97 + i), label, x, y, e]


def data_table(spec: FigureSpec) -> tuple[list[str], list[list]]:
    """Header and rows of the figure's numbers: the template's own table, else every plotted series."""
    if spec.table is not None:
        return spec.table
    return ["panel", "series", "x", "y", "y_error"], list(data_rows(spec))


# ----------------------------------------------------------------------------------------- script export
def _lit(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(e, (int, float, np.floating, np.integer)) and not isinstance(e, bool) for e in v):
            return "np.array([" + ", ".join(_num(e) for e in v) + "])"
        return "[" + ", ".join(_lit(e) for e in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool):
        return _num(v)
    return repr(v)


def _num(e) -> str:
    f = float(e)
    if math.isnan(f):
        return "np.nan"
    if math.isinf(f):
        return "np.inf" if f > 0 else "-np.inf"
    return repr(int(e)) if isinstance(e, (int, np.integer)) else f"{f:.7g}"


def _op_lines(ops: list[dict], tex: bool, ax: str = "ax", keep: str = "") -> list[str]:
    """Script lines drawing *ops* on the axes named *ax*; with *keep*, the colour-mapped result is kept."""
    out = []
    for op in ops:
        kw = dict(op.get("kw", {}))
        transform = kw.pop("transform", None)
        args = list(op.get("args", []))
        if op["op"] in ("text", "annotate"):
            idx = 2 if op["op"] == "text" else 0
            if len(args) > idx and isinstance(args[idx], str):
                args[idx] = tex_safe(args[idx], tex)
        if isinstance(kw.get("label"), str):
            kw["label"] = tex_safe(kw["label"], tex)
        parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kw.items()]
        if transform == "axes":
            parts.append(f"transform={ax}.transAxes")
        call = f"{ax}.{op['op']}(" + ", ".join(parts) + ")"
        out.append(f"{keep} = {call}" if keep and op["op"] in FILLED_OPS else call)
    return out


def _scale_line(ax: str, axis: str, scale: str, linthresh) -> list[str]:
    if scale == "symlog":
        return [f"{ax}.set_{axis}scale('symlog', linthresh={linthresh or 1.0!r})"]
    return [f"{ax}.set_{axis}scale({scale!r})"] if scale and scale != "linear" else []


def to_script(spec: FigureSpec, out_name: str = "figure.pdf") -> str:
    """A standalone Python script (numpy + matplotlib only) that draws *spec* and saves *out_name*."""
    rc = rc_params(spec.style, assume_latex=True)    # the script runs on the user's machine: keep their choice
    tex = bool(rc.get("text.usetex"))
    cycle = rc.pop("axes.prop_cycle").by_key()["color"]
    lines = [
        '"""Figure exported from Qali. Needs numpy and matplotlib' + (" and a LaTeX installation" if tex else "")
        + '."""',
        "import numpy as np",
        "import matplotlib as mpl",
        "import matplotlib.pyplot as plt",
        "",
        "mpl.rcParams.update({",
    ]
    lines += [f"    {k!r}: {v!r}," for k, v in rc.items()]
    lines += ["})", f"mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color={cycle!r})", "", _TIDY_SRC, _CLIP_SRC,
              _AXES_SRC]
    n = max(len(spec.panels), 1)
    ratios = [p.height_ratio for p in spec.panels] or [1]
    share = sharex_ok(spec)
    lines += [
        f"fig = plt.figure(figsize=({spec.width_in}, {spec.height_in}), layout='constrained')",
        "fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)",
        f"gs = fig.add_gridspec({n}, 1, height_ratios={ratios!r})",
        "first = None",
    ]
    for i, p in enumerate(spec.panels):
        brk = p.axis_break
        lines += ["", f"axs = make_panel_axes(fig, gs[{i}], {brk!r}, {'first' if share else 'None'})",
                  "first = first or axs[0]", "mappable = None", "for ax in axs:"]
        body = _op_lines([o for o in p.ops if not o.get("on")], tex, keep="mappable")
        body += _scale_line("ax", "x", p.xscale, p.linthresh) + _scale_line("ax", "y", p.yscale, p.linthresh)
        if p.xticks is not None:
            body.append(f"ax.set_xticks({p.xticks!r})")
            if p.xticklabels is not None:
                body.append(f"ax.set_xticklabels({[tex_safe(t, tex) for t in p.xticklabels]!r})")
        if p.yticks is not None:
            body.append(f"ax.set_yticks({p.yticks!r})")
            if p.yticklabels is not None:
                body.append(f"ax.set_yticklabels({[tex_safe(t, tex) for t in p.yticklabels]!r})")
        if has_range(p.xlim):
            body.append(f"ax.set_xlim({p.xlim[0]!r}, {p.xlim[1]!r})")
        if has_range(p.ylim):
            body.append(f"ax.set_ylim({p.ylim[0]!r}, {p.ylim[1]!r})")
        for axis, scale in (("x", p.xscale), ("y", p.yscale)):
            if scale == "log":
                body.append(f"tidy_log_axis(ax, {axis!r})")
        if p.clip:
            body.append("clip_to_axes(ax)")
        if not p.xminor:
            body.append("ax.tick_params(axis='x', which='minor', bottom=False, top=False)")
        if not p.yminor:
            body.append("ax.tick_params(axis='y', which='minor', left=False, right=False)")
        for sp in p.spines or []:
            body.append(f"ax.spines[{sp!r}].set_visible(False)")
            ax_name = "x" if sp in ("top", "bottom") else "y"
            body.append(f"ax.tick_params(axis={ax_name!r}, which='both', {sp}=False)")
        if not brk:
            body.append(f"ax.set_xlabel({tex_safe(p.xlabel, tex)!r})")
            body.append(f"ax.set_ylabel({tex_safe(p.ylabel, tex)!r})")
        lines += ["    " + b for b in (body or ["pass"])]
        if brk:
            lim = p.ylim if brk["axis"] == "y" else p.xlim
            lines.append(f"break_axes(axs, {brk!r}, {tex_safe(p.xlabel, tex)!r}, {tex_safe(p.ylabel, tex)!r}, "
                         f"{lim!r})")
        lines.append(f"ax = head_axes(axs, {brk!r})")
        twin = False
        if p.y2 is not None:
            twin = True
            lines.append("twin = axs[0].twinx()")
            lines += _op_lines([o for o in p.ops if o.get("on") == "y2"], tex, "twin", keep="mappable")
            lines += _scale_line("twin", "y", p.y2.get("scale", "linear"), p.y2.get("linthresh"))
            lim = p.y2.get("lim")
            if has_range(lim):
                lines.append(f"twin.set_ylim({lim[0]!r}, {lim[1]!r})")
            if p.y2.get("scale") == "log":
                lines.append("tidy_log_axis(twin, 'y')")
            lines.append(f"twin.set_ylabel({tex_safe(p.y2.get('label', ''), tex)!r})")
        if p.inset is not None:
            lines.append(f"ins = axs[0].inset_axes({p.inset.get('bounds') or [0.58, 0.55, 0.38, 0.4]!r})")
            lines += _op_lines([o for o in p.ops if o.get("on") == "inset"], tex, "ins", keep="mappable")
            for axis in ("x", "y"):
                lines += _scale_line("ins", axis, p.inset.get(f"{axis}scale", "linear"), None)
                lim = p.inset.get(f"{axis}lim")
                if has_range(lim):
                    lines.append(f"ins.set_{axis}lim({lim[0]!r}, {lim[1]!r})")
            lines.append("ins.tick_params(labelsize='small')")
            if p.inset.get("mark", True) and p.inset.get("xlim") and p.inset.get("ylim"):
                lines.append("axs[0].indicate_inset_zoom(ins, edgecolor='0.4', linewidth=0.6)")
        if p.colorbar is not None:
            lines.append("if mappable is not None:")
            lines.append(f"    fig.colorbar(mappable, ax=axs, pad=0.02).set_label({tex_safe(p.colorbar, tex)!r})")
        if p.title:
            lines.append(f"ax.set_title({tex_safe(p.title, tex)!r})")
        if p.legend is not None and any("label" in op.get("kw", {}) for op in p.ops):
            title = tex_safe(p.legend.get("title") or "", tex) or None
            src = "ax.get_legend_handles_labels()"
            if twin:
                lines.append("h1, l1 = ax.get_legend_handles_labels()")
                lines.append("h2, l2 = twin.get_legend_handles_labels()")
                src, target = "(h1 + h2, l1 + l2)", "twin"
            else:
                target = "ax"
            lines.append(f"handles, names = {src}")
            lines.append(f"{target}.legend(handles, names, loc={p.legend.get('loc', 'best')!r}, "
                         f"ncol={int(p.legend.get('ncol', 1))}, frameon={bool(p.legend.get('frame', False))}, "
                         f"title={title!r})")
        if n > 1 and spec.style.get("panel_labels", True):
            if not brk and p.yscale == "linear" and not has_range(p.ylim):
                lines.append("lo, hi = ax.get_ylim()")
                lines.append("ax.set_ylim(lo, hi + 0.14 * (hi - lo))")
            lines.append(f"ax.text(0.015, 0.97, '({chr(97 + i)})', transform=ax.transAxes, va='top', ha='left')")
        if share and i < n - 1:
            lines.append("for a_ in axs:")
            lines.append("    a_.set_xlabel('')")
            lines.append("    a_.tick_params(axis='x', labelbottom=False)")
    lines += ["", f"fig.savefig({out_name!r}" + (", dpi=600" if out_name.rsplit('.', 1)[-1] in ('png', 'tiff') else "")
              + ")", ""]
    return "\n".join(lines)
