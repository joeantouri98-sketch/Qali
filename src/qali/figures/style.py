"""Figure size, fonts and colours for publication figures.

Sizes are the printed column widths of the journals. The font size is the size at print scale, so a figure
saved at the preset width needs no scaling in the manuscript.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

CM = 1 / 2.54          # inch per cm


@dataclass(frozen=True)
class Journal:
    label: str
    single_in: float           # one-column width
    double_in: float           # full page width
    font_pt: float
    family: str                # "serif" | "sans"


# APS: 8.6 cm (3 3/8 in) one column, 17.8 cm (7 in) two; lettering at least 2 mm high when printed.
# Nature: 89 mm one column, 183 mm two; sans-serif lettering, 5-7 pt.
JOURNALS = {
    "aps": Journal("APS (PRA, PRL, PRX)", 3.375, 7.0, 8.0, "serif"),
    "nature": Journal("Nature family", 89 * CM / 10, 183 * CM / 10, 7.0, "sans"),
    "thesis": Journal("Thesis or report", 6.0, 6.0, 10.0, "serif"),
    "slides": Journal("Slides (16:9)", 9.0, 9.0, 14.0, "sans"),
    "custom": Journal("Custom size", 3.375, 7.0, 8.0, "serif"),
}

# Colour-blind safe palettes (Okabe & Ito 2008; Paul Tol's bright, vibrant and muted sets).
PALETTES = {
    "okabe-ito": ("Okabe–Ito", ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                                "#F0E442", "#000000"]),
    "tol-bright": ("Tol bright", ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
                                  "#BBBBBB"]),
    "tol-vibrant": ("Tol vibrant", ["#0077BB", "#EE7733", "#009988", "#EE3377", "#33BBEE", "#CC3311",
                                    "#BBBBBB"]),
    "tol-muted": ("Tol muted", ["#332288", "#CC6677", "#117733", "#DDCC77", "#88CCEE", "#882255",
                                "#44AA99", "#999933", "#AA4499"]),
    "grey": ("Greyscale", ["#000000", "#555555", "#888888", "#AAAAAA", "#333333", "#777777"]),
    "matplotlib": ("Matplotlib default", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]),
}

FONTS = {
    "cm": "Computer Modern (LaTeX look)",
    "stix": "Times-like (STIX)",
    "sans": "Sans-serif (DejaVu)",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":"]


def latex_available(raster: bool = True) -> bool:
    """True when Matplotlib can render text with LaTeX: ``latex`` on the PATH, and for on-screen and PNG/TIFF
    output also ``dvipng`` (PDF and EPS need only LaTeX)."""
    return shutil.which("latex") is not None and (not raster or shutil.which("dvipng") is not None)


def latex_problem(raster: bool = True) -> str:
    """Why LaTeX cannot be used, or an empty string."""
    if shutil.which("latex") is None:
        return "LaTeX is not installed (no 'latex' on the PATH): text is drawn with mathtext instead."
    if raster and shutil.which("dvipng") is None:
        return ("'dvipng' is missing, which LaTeX needs for the preview and PNG/TIFF: they use mathtext; "
                "PDF and EPS use LaTeX.")
    return ""


def figure_size(size: dict, preferred_aspect: float = 0.75) -> tuple[float, float]:
    """(width, height) in inches from a recipe's ``size`` block; *preferred_aspect* (height/width, from the
    template) applies when the recipe leaves the aspect unset."""
    j = JOURNALS.get(size.get("journal", "aps"), JOURNALS["aps"])
    if size.get("journal") == "custom" or size.get("width_cm"):
        width = float(size.get("width_cm") or j.single_in / CM) * CM
    else:
        width = j.double_in if int(size.get("columns", 1)) == 2 else j.single_in
    if size.get("height_cm"):
        height = float(size["height_cm"]) * CM
    else:
        aspect = size.get("aspect")
        height = width * float(aspect if aspect else preferred_aspect)
    return round(width, 4), round(height, 4)


def default_style(journal: str = "aps") -> dict:
    j = JOURNALS.get(journal, JOURNALS["aps"])
    return {
        "palette": "okabe-ito",
        "font": "sans" if j.family == "sans" else "cm",
        "font_size": j.font_pt,
        "usetex": False,
        "ticks_in": True,
        "minor_ticks": True,
        "mirror_ticks": True,
        "grid": False,
        "line_width": 1.0,
        "marker_size": 3.5,
        "panel_labels": True,
    }


def palette(name: str) -> list[str]:
    return list(PALETTES.get(name, PALETTES["okabe-ito"])[1])


def rc_params(style: dict, raster: bool = True, assume_latex: bool = False) -> dict:
    """Matplotlib rcParams for *style* (a recipe's ``style`` block); *raster* is True for the on-screen
    preview and PNG/TIFF output; *assume_latex* skips the LaTeX check (for a script run elsewhere)."""
    from cycler import cycler

    fs = float(style.get("font_size", 8.0))
    lw = float(style.get("line_width", 1.0))
    font = style.get("font", "cm")
    usetex = bool(style.get("usetex")) and (assume_latex or latex_available(raster))
    rc: dict = {
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "legend.fontsize": fs * 0.9,
        "xtick.labelsize": fs * 0.9,
        "ytick.labelsize": fs * 0.9,
        "axes.linewidth": 0.6 * lw,
        "lines.linewidth": lw,
        "lines.markersize": float(style.get("marker_size", 3.5)),
        "errorbar.capsize": 1.5,
        "xtick.direction": "in" if style.get("ticks_in", True) else "out",
        "ytick.direction": "in" if style.get("ticks_in", True) else "out",
        "xtick.top": bool(style.get("mirror_ticks", True)),
        "ytick.right": bool(style.get("mirror_ticks", True)),
        "xtick.minor.visible": bool(style.get("minor_ticks", True)),
        "ytick.minor.visible": bool(style.get("minor_ticks", True)),
        "xtick.major.width": 0.6 * lw, "ytick.major.width": 0.6 * lw,
        "xtick.minor.width": 0.45 * lw, "ytick.minor.width": 0.45 * lw,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "axes.grid": bool(style.get("grid", False)),
        "grid.linewidth": 0.4, "grid.alpha": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.prop_cycle": cycler(color=palette(style.get("palette", "okabe-ito"))),
        "savefig.bbox": "standard",      # keep the exact journal width; the layout engine fits the content
        "pdf.fonttype": 42,              # embed TrueType: editable text, accepted by journals
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "text.usetex": usetex,
    }
    if usetex:
        rc["font.family"] = "sans-serif" if font == "sans" else "serif"
        rc["text.latex.preamble"] = (r"\usepackage{amsmath}\usepackage{textcomp}"
                                     + (r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
                                        if font == "sans" else ""))
    elif font == "cm":
        rc.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
                   "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
    elif font == "stix":
        rc.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                   "mathtext.fontset": "stix"})
    else:
        rc.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                   "mathtext.fontset": "dejavusans"})
    return rc


def tex_safe(text: str, usetex: bool) -> str:
    """Escape LaTeX's special characters outside $...$ when LaTeX renders the text."""
    if not usetex or not text:
        return text
    out, math = [], False
    for ch in text:
        if ch == "$":
            math = not math
            out.append(ch)
        elif not math and ch in "_%&#":
            out.append("\\" + ch)
        elif not math and ch == "µ":
            out.append(r"\textmu{}")
        elif not math and ch == "−":
            out.append("-")
        else:
            out.append(ch)
    return "".join(out)
