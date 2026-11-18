"""Figure templates: each turns its parameters and a species' output files into panels to draw.

A template returns a :class:`Result`: the panels, notes for the user (what data went in, what was left
out) and the aspect ratio it suits. Every template reads only the JSON files the stages wrote.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import catalog
from .spec import Panel

L_CHARS = "spdfghik"
EV_TO_CM = 8065.543937
NM_TO_AU_OMEGA = 45.56335252767      # ω (a.u.) = 45.563 / λ (nm)


@dataclass
class Result:
    panels: list[Panel]
    notes: list[str] = field(default_factory=list)
    aspect: float = 0.75
    sharex: bool = False
    table: tuple[list[str], list[list]] | None = None    # the CSV of the figure when x/y series do not fit it


class TemplateError(ValueError):
    """The data a template needs is missing; the message says which stage to run."""


def _need(doc, species: str, stage: str):
    if doc is None:
        raise TemplateError(f"No {species}_{stage}.json in the output folder: run the {stage.capitalize()} stage "
                            f"for {species} first.")
    return doc


def _jnum(J) -> float | None:
    try:
        a, b = str(J).split("/")
        return int(a) / int(b)
    except (ValueError, AttributeError):
        try:
            return float(J)
        except (TypeError, ValueError):
            return None


def _pretty(label: str) -> str:
    """'5p3/2' -> '$5p_{3/2}$', '7s' -> '$7s$'."""
    m = re.match(r"^(\d+)([a-z])(\d+/2)?$", label)
    if not m:
        return label
    n, l, j = m.groups()
    return f"${n}{l}_{{{j}}}$" if j else f"${n}{l}$"


# ------------------------------------------------------------------------------------------- levels
@dataclass
class Level:
    label: str
    n: int
    l: int
    J: float | None
    binding_eV: float          # negative: energy relative to the ionization limit
    source: str

    @property
    def pretty(self) -> str:
        return _pretty(self.label)


def levels(species: str, data_dir: str) -> tuple[list[Level], float, float]:
    """(levels including the ground state, ionization energy eV, Rydberg constant eV) from the Rydberg stage."""
    doc = _need(catalog.load_json(species, "rydberg", data_dir), species, "rydberg")
    IE = float(doc["ionization_energy_eV"])
    Ry = float(doc.get("rydberg_constant_eV") or 13.605693)
    out: list[Level] = []
    gs = doc.get("ground_state") or {}
    if gs.get("label"):
        out.append(Level(gs["label"], int(gs.get("n", 0)), L_CHARS.index(gs.get("l", "s")),
                         _jnum(gs.get("J")), -IE, "NIST"))
    for e in doc.get("excitations", []):
        try:
            out.append(Level(e["label"], int(e["n"]), L_CHARS.index(e["l"]), _jnum(e.get("J")),
                             float(e["energy_eV"]), e.get("source", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return out, IE, Ry


def n_star(level: Level, Ry: float) -> float:
    return math.sqrt(Ry / -level.binding_eV) if level.binding_eV < 0 else math.nan


# ------------------------------------------------------------------------------------------- Grotrian
def grotrian_choices(species: str, data_dir: str) -> tuple[list[Level], list[dict], float]:
    """Levels and lines (lifetimes stage) a Grotrian diagram can show, and the ionization energy."""
    lv, IE, _ = levels(species, data_dir)
    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    lines = [t for t in lt.get("transitions", []) if (t.get("fosc") or 0) > 0 and t.get("A_s")]
    return lv, lines, IE


def default_grotrian_levels(lv: list[Level], IE: float, count: int = 14) -> list[str]:
    gs = lv[0]
    pick = [x for x in lv if x.l <= 3 and x.binding_eV + IE < 0.8 * IE and x.n <= gs.n + 3]
    pick.sort(key=lambda x: x.binding_eV)
    return [x.label for x in pick[:count]]


def lines_between(lines: list[dict], chosen: set[str]) -> list[dict]:
    return [t for t in lines if t["upper_label"] in chosen and t["lower_label"] in chosen]


def grotrian(params: dict, species: str, data_dir: str, style: dict) -> Result:
    lv, all_lines, IE = grotrian_choices(species, data_dir)
    by_label = {x.label: x for x in lv}
    chosen = [s for s in (params.get("levels") or default_grotrian_levels(lv, IE)) if s in by_label]
    if not chosen:
        raise TemplateError("Choose the levels to draw (Levels...).")
    cset = set(chosen)
    avail = lines_between(all_lines, cset)
    wanted = params.get("lines")
    if wanted:
        keys = {tuple(w.split(">")) for w in wanted}
        shown = [t for t in avail if (t["upper_label"], t["lower_label"]) in keys]
    else:
        shown = sorted(avail, key=lambda t: -t["A_s"])[: int(params.get("max_lines", 12))]
    unit = params.get("energy_unit", "eV")
    scale = EV_TO_CM if unit == "cm-1" else 1.0
    split = bool(params.get("j_split", True))

    def x_of(x: Level) -> tuple[float, float]:
        if split and x.l > 0 and x.J is not None:
            return x.l + (-0.21 if x.J < x.l else 0.21), 0.14
        return float(x.l), 0.28

    def y_of(x: Level) -> float:
        return (x.binding_eV + IE) * scale

    p = Panel()
    nodes = {}
    label_boxes: list[tuple[float, float, float, float]] = []      # x0, x1, y0, y1 of each level label
    for lab in chosen:
        x = by_label[lab]
        xc, w = x_of(x)
        nodes[lab] = (xc, w, y_of(x))
        p.add("hlines", y_of(x), xc - w, xc + w, colors="k", linewidth=1.1)
    if params.get("label_levels", True):
        # lower-J sub-column labels go left, the rest right; labels sharing a gap between columns are
        # spread apart vertically so near-degenerate levels (Li 3p and 3d) stay readable
        centres = sorted({nodes[lab][0] for lab in chosen})
        top_e = max(max(nodes[lab][2] for lab in chosen), IE * scale if params.get("limit", True) else 0.0)
        gap_d = 0.034 * (top_e * 1.05 + 0.03 * IE * scale)
        gaps: dict[float, list[tuple[float, str, bool]]] = {}
        for lab in chosen:
            x = by_label[lab]
            left = split and x.l > 0 and x.J is not None and x.J < x.l
            i = centres.index(nodes[lab][0])
            gaps.setdefault(i - 0.5 if left else i + 0.5, []).append((nodes[lab][2], lab, left))
        for items in gaps.values():
            items.sort()
            for (y, lab, left), yy in zip(items, _spread([it[0] for it in items], gap_d)):
                xc, w, _ = nodes[lab]
                xa = xc - w - 0.03 if left else xc + w + 0.03
                tw = 0.62 * len(lab) * style.get("font_size", 8) * 0.8 / (0.82 * 3.375 * 72) \
                    * (max(centres) - min(centres) + 1.3)
                label_boxes.append((xa - tw if left else xa, xa if left else xa + tw, yy - gap_d / 2, yy + gap_d / 2))
                p.add("text", xc - w - 0.03 if left else xc + w + 0.03, yy, by_label[lab].pretty, va="center",
                      ha="right" if left else "left", fontsize=style.get("font_size", 8) * 0.8)
    # spread the line ends along each level so lines do not start from one point
    ends: dict[str, list[tuple[float, int, int]]] = {}
    for i, t in enumerate(shown):
        for lab, other in ((t["upper_label"], t["lower_label"]), (t["lower_label"], t["upper_label"])):
            ends.setdefault(lab, []).append((nodes[other][0], i, 0 if lab == t["lower_label"] else 1))
    pos: dict[tuple[int, int], float] = {}
    for lab, at_level in ends.items():
        xc, w, _ = nodes[lab]
        at_level.sort()
        k = len(at_level)
        for j, (_ox, li, end) in enumerate(at_level):
            pos[(li, end)] = xc - 0.75 * w + (1.5 * w) * ((j + 0.5) / k)
    if shown:
        logA = np.log10([t["A_s"] for t in shown])
        lo, hi = float(logA.min()), float(logA.max())
    stagger: dict[tuple[float, float], int] = {}
    cols_l = sorted({by_label[lab].l for lab in chosen})
    x_span = max(cols_l) - min(cols_l) + 1.3
    y_span = max(max(nodes[lab][2] for lab in chosen), IE * scale) * 1.08
    axes_pt = 0.82 * 3.375 * 72              # single-column width: the conservative case
    short = 0
    arrow = params.get("arrows", "emission")
    color_by = params.get("color_by", "lower")
    lower_ls = sorted({by_label[t["lower_label"]].l for t in shown})
    for i, t in enumerate(shown):
        x0, y0 = pos[(i, 0)], nodes[t["lower_label"]][2]
        x1, y1 = pos[(i, 1)], nodes[t["upper_label"]][2]
        frac = 0.5 if hi == lo else (math.log10(t["A_s"]) - lo) / (hi - lo)
        lw = 0.5 + 1.8 * frac if params.get("width_by_A", True) else 1.0
        if color_by == "lower":
            col = f"C{lower_ls.index(by_label[t['lower_label']].l)}"
        elif color_by == "wavelength":
            col = _visible_colour(t["wavelength_nm"])
        else:
            col = "C0"
        style_ = {"arrowstyle": "-" if arrow == "none" else "-|>", "lw": lw, "color": col,
                  "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6}
        if arrow == "absorption":
            p.add("annotate", "", [x1, y1], xytext=[x0, y0], arrowprops=style_)
        else:
            p.add("annotate", "", [x0, y0], xytext=[x1, y1], arrowprops=style_)
        if params.get("wavelengths", True):
            wl = t["wavelength_nm"]
            txt = f"{wl:.1f} nm" if wl < 1000 else f"{wl / 1000:#.3g} $\\mu$m"
            text_pt = (len(txt) - (7 if "mu" in txt else 0)) * 0.55 * style.get("font_size", 8) * 0.62
            if math.hypot((x1 - x0) / x_span, (y1 - y0) / y_span) * axes_pt < 1.3 * text_pt:
                short += 1                   # the line is too short to carry its label
                continue
            # along the line, staggered when several lines join the same two columns, and off the level labels
            pair = (round(nodes[t["lower_label"]][0], 2), round(nodes[t["upper_label"]][0], 2))
            k = stagger.get(pair, 0)
            stagger[pair] = k + 1
            first = (0.5, 0.38, 0.62, 0.44, 0.56)[k % 5]
            half_w = 0.5 * text_pt / axes_pt * x_span * abs(math.cos(math.atan2((y1 - y0) / y_span, (x1 - x0) / x_span)))
            along: float | None = None
            for f in (first, 0.3, 0.7, 0.22, 0.78):
                cx, cy = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                if not any(bx0 - half_w < cx < bx1 + half_w and by0 < cy < by1 for bx0, bx1, by0, by1 in label_boxes):
                    along = f
                    break
            if along is None:
                short += 1
                continue
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
            p.add("text", x0 + along * (x1 - x0), y0 + along * (y1 - y0), txt, ha="center", va="center",
                  rotation=ang, rotation_mode="anchor", transform_rotates_text=True,
                  fontsize=style.get("font_size", 8) * 0.62, color=col,
                  bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none"})
    xs = sorted({nodes[lab][0] for lab in chosen})
    cols = sorted({by_label[lab].l for lab in chosen})
    if params.get("limit", True):
        p.add("axhline", IE * scale, color="0.4", linestyle="--", linewidth=0.6)
        p.add("text", max(cols) + 0.72, IE * scale, "limit", ha="right", va="bottom", color="0.4",
              fontsize=style.get("font_size", 8) * 0.75)
    p.xticks, p.xticklabels = [], []
    for xv in xs:
        lev = next(by_label[lab] for lab in chosen if nodes[lab][0] == xv)
        L = "SPDFGHIK"[lev.l]
        if split and lev.l > 0 and lev.J is not None:
            p.xticklabels.append(f"$^2{L}_{{{int(2 * lev.J)}/2}}$")
        else:
            p.xticklabels.append(f"$^2{L}$" if lev.l else "$^2S_{1/2}$")
        p.xticks.append(xv)
    p.xlim = [min(cols) - 0.55, max(cols) + 0.75]
    top = max(nodes[lab][2] for lab in chosen)
    p.ylim = [-0.03 * IE * scale, max(top, IE * scale if params.get("limit", True) else top) * 1.05]
    p.xminor = False
    p.spines = ["top", "right"]
    p.ylabel = "Energy (cm$^{-1}$)" if unit == "cm-1" else "Energy (eV)"
    notes = [f"{len(chosen)} levels, {len(shown)} of the {len(avail)} lines between them"
             + (" (strongest by A)" if not wanted else "") + "."]
    if short:
        notes.append(f"{short} wavelength label{'s' if short > 1 else ''} left off: no room on the line "
                     "(Data... lists every line).")
    eu = "cm-1" if unit == "cm-1" else "eV"
    rows: list[list] = [["level", lab, "", y_of(by_label[lab]), "", "", ""] for lab in chosen]
    rows += [["line", t["upper_label"], t["lower_label"], y_of(by_label[t["upper_label"]]), t["wavelength_nm"],
              t["A_s"], t.get("fosc", "")] for t in shown]
    head = ["item", "upper", "lower", f"energy_{eu}", "wavelength_nm", "A_s-1", "f"]
    return Result([p], notes, aspect=1.0, table=(head, rows))


def _spread(ys: list[float], d: float) -> list[float]:
    """Sorted positions *ys* moved apart to at least *d*, each crowded cluster centred where it was."""
    clusters: list[list[float]] = []          # [mean of originals, count]
    for y in ys:
        clusters.append([y, 1])
        while len(clusters) > 1:
            (m0, n0), (m1, n1) = clusters[-2], clusters[-1]
            if m1 - (n1 - 1) * d / 2 >= m0 + (n0 - 1) * d / 2 + d:
                break
            clusters[-2:] = [[(m0 * n0 + m1 * n1) / (n0 + n1), n0 + n1]]
    out: list[float] = []
    for m, n in clusters:
        out += [m + (k - (n - 1) / 2) * d for k in range(int(n))]
    return out


def _visible_colour(nm: float) -> str:
    """Approximate hue of a vacuum wavelength; grey outside the visible."""
    if nm < 380 or nm > 750:
        return "0.45"
    stops = [(380, "#7600ed"), (440, "#0028ff"), (490, "#00d5ff"), (510, "#1fff00"), (580, "#ffef00"),
             (645, "#ff0000"), (750, "#a00000")]
    for (a, ca), (b, _cb) in itertools.pairwise(stops):
        if a <= nm <= b:
            return ca
    return "#a00000"


# ------------------------------------------------------------------------------------ quantum defects
def _ritz_fit(n: np.ndarray, d: np.ndarray) -> tuple[float, ...] | None:
    """(δ0, δ2[, δ4]) of δ(n) = δ0 + δ2/n*² + δ4/n*⁴ with n* = n − δ0, least squares weighted by 1/n*³
    (an energy error δE moves δ by ≈ n*³ δE / 2Ry, so the high-n points are the least certain).
    Series of 7 or more leave out their lowest member. Against the published δ0 of Li to Cs this gives a
    median difference of 1e-4 (largest 1.5e-3, the K d series)."""
    ok = np.isfinite(d)
    n, d = n[ok], d[ok]
    o = np.argsort(n)
    n, d = n[o], d[o]
    if len(n) >= 7:                          # the lowest member penetrates the core beyond what δ4 absorbs
        n, d = n[1:], d[1:]
    if len(n) < 3:
        return None
    terms = 3 if len(n) >= 6 else 2
    d0 = float(np.median(d))
    coef = None
    for _ in range(30):                      # n* depends on δ0: iterate the linear fit
        ns = n - d0
        A = np.stack([ns ** (-2 * k) for k in range(terms)], axis=1)
        w = ns ** -3.0
        coef, *_ = np.linalg.lstsq(A * w[:, None], d * w, rcond=None)
        if abs(coef[0] - d0) < 1e-12:
            break
        d0 = float(coef[0])
    return tuple(float(c) for c in coef) if coef is not None else None


def _ritz_eval(coef: tuple[float, ...], n: np.ndarray) -> np.ndarray:
    ns = n - coef[0]
    return sum(c * ns ** (-2 * k) for k, c in enumerate(coef))


def quantum_defects(params: dict, species: str, data_dir: str, style: dict) -> Result:
    lv, IE, Ry = levels(species, data_dir)
    want = [c for c in params.get("l", ["s", "p", "d", "f"]) if c in L_CHARS]
    only_measured = bool(params.get("measured_only", True))
    panels = []
    notes = []
    for i, lc in enumerate(want):
        l = L_CHARS.index(lc)
        pts = [x for x in lv if x.l == l and (x.source == "NIST" or not only_measured)]
        if not pts:
            continue
        p = Panel()
        Js: list[float | None] = [*sorted({x.J for x in pts if x.J is not None})] or [None]
        for k, J in enumerate(Js):
            sel = [x for x in pts if x.J == J] if J is not None else pts
            n = np.array([x.n for x in sel], dtype=float)
            d = np.array([x.n - n_star(x, Ry) for x in sel])
            order = np.argsort(n)
            lab = f"${lc}_{{{int(2 * J)}/2}}$" if J is not None else f"${lc}$"
            p.add("plot", n[order], d[order], linestyle="none", marker="os"[k % 2], color=f"C{i}",
                  markerfacecolor="none" if k else f"C{i}", label=lab)
        if params.get("fit", True):
            fitted: list = []
            for k, J in enumerate(Js):
                sel = [x for x in pts if x.J == J] if J is not None else pts
                n = np.array([x.n for x in sel], dtype=float)
                d = np.array([x.n - n_star(x, Ry) for x in sel])
                coef = _ritz_fit(n, d)
                if coef is None:
                    continue
                lo = np.sort(n)[1] if len(n) >= 7 else n.min()          # the fitted range
                nn = np.linspace(lo, max(n.max(), lo + 1), 200)
                p.add("plot", nn, _ritz_eval(coef, nn), color="0.3", linestyle="--", linewidth=0.8,
                      **({"label": "Ritz fit"} if not fitted else {}))
                fitted.append(f"{lc}{'' if J is None else f'{int(2 * J)}/2'} {coef[0]:.5f}")
            if fitted:
                notes.append("Ritz δ0: " + ", ".join(fitted) + ".")
        p.ylabel = f"$\\delta_{lc}$"
        p.legend = {"loc": "best", "ncol": len(Js) + 1, "frame": False}
        panels.append(p)
    if not panels:
        raise TemplateError("No measured levels for the chosen series.")
    panels[-1].xlabel = "$n$"
    notes.insert(0, ("δ = n − n*, n* = √(Ry/E_bind) from the measured (NIST) levels" if only_measured else
                     "δ = n − n*, including the model (QDT) levels")
                 + "; dashed: a Ritz fit δ0 + δ2/n*² + δ4/n*⁴ to each J series, weighted by 1/n*³; series of 7 or more leave out their lowest member.")
    return Result(panels, notes, aspect=0.45 * len(panels) + 0.15, sharex=True)


# ----------------------------------------------------------------------------------------- scaling laws
SCALING = {
    "lifetime": ("Radiative lifetime", "lifetimes", r"$\tau$ (ns)", 3),
    "bbr_lifetime": ("Lifetime with blackbody (300 K)", "blackbody", r"$\tau_\mathrm{eff}$ (ns)", 2),
    "rydberg_alpha": ("Rydberg static polarizability", "blackbody", r"$|\alpha_0|$ (a.u.)", 7),
}


def scaling_laws(params: dict, species: str, data_dir: str, style: dict) -> Result:
    qty = params.get("quantity", "lifetime")
    if qty not in SCALING:
        raise TemplateError(f"Unknown quantity {qty}.")
    title, stage, ylabel, expected = SCALING[qty]
    lv, IE, Ry = levels(species, data_dir)
    nstar = {x.label: n_star(x, Ry) for x in lv}
    by_nl: dict[tuple[int, int], list[float]] = {}
    for x in lv:
        by_nl.setdefault((x.n, x.l), []).append(n_star(x, Ry))
    doc = _need(catalog.load_json(species, stage, data_dir), species, stage)
    if qty == "lifetime":
        rows = [(r["state"], r["n"], r["l"], r["lifetime_ns"]) for r in doc.get("lifetimes", [])]
    elif qty == "bbr_lifetime":
        rows = [(r["state"], r["n"], r["l"], (r.get("tau_eff_s") or math.nan) * 1e9)
                for r in (doc.get("rates") or {}).get("states", [])]
    else:
        rows = [(r["state"], r["n"], r["l"], abs(r.get("alpha0_au") or math.nan))
                for r in (doc.get("rydberg_shifts") or {}).get("states", [])]
    want = [c for c in params.get("l", ["s", "p", "d"]) if c in L_CHARS]
    nmin = float(params.get("fit_from", 10))
    p = Panel(xscale="log", yscale="log", xlabel="$n^*$", ylabel=ylabel)
    notes = [f"{title} of {species}."]
    for i, lc in enumerate(want):
        sel = []
        for state, n, l, val in rows:
            if (l if isinstance(l, str) else L_CHARS[int(l)]) != lc or not (val and val > 0):
                continue
            ns = nstar.get(state)
            if ns is None or not math.isfinite(ns):
                cands = by_nl.get((int(n), L_CHARS.index(lc)))
                ns = float(np.mean(cands)) if cands else math.nan
            if math.isfinite(ns):
                sel.append((ns, float(val)))
        if not sel:
            continue
        sel.sort()
        x = np.array([s[0] for s in sel])
        y = np.array([s[1] for s in sel])
        fit = x >= nmin
        label = f"${lc}$"
        if params.get("fit", True) and fit.sum() >= 3:
            g, c = np.polyfit(np.log(x[fit]), np.log(y[fit]), 1)
            xx = np.geomspace(x[fit].min(), x.max(), 50)
            p.add("plot", xx, np.exp(c) * xx ** g, color=f"C{i}", linestyle="--", linewidth=0.8)
            label += f": $\\propto n^{{*\\,{g:.2f}}}$"
            notes.append(f"{lc}: fitted exponent {g:.3f} for n* ≥ {nmin:g} (scaling law n*^{expected}).")
        p.add("plot", x, y, linestyle="none", marker="os^D"[i % 4], color=f"C{i}", label=label)
    if not p.ops:
        raise TemplateError(f"No {title.lower()} data for the chosen series.")
    p.legend = {"loc": "best", "ncol": 1, "frame": False}
    return Result([p], notes, aspect=0.8)


# ----------------------------------------------------------------------------------- theory vs experiment
QUANTITIES = {
    "alpha0": r"$\alpha_0$", "C6": r"$C_6$", "tau_D1": r"$\tau(np_{1/2})$", "tau_D2": r"$\tau(np_{3/2})$",
}


def theory_vs_experiment(params: dict, species: str, data_dir: str, style: dict) -> Result:
    from qali.core.constants import ALPHA0_EXPT, C6_EXPT
    from qali.core.spectroscopy import LITERATURE

    want = [q for q in params.get("quantities", list(QUANTITIES)) if q in QUANTITIES]
    elements = [e for e in ("Li", "Na", "K", "Rb", "Cs", "Fr") if e in catalog.available_species(data_dir)]
    if params.get("species"):
        elements = [e for e in elements if e in params["species"]]
    p = Panel(ylabel="theory / experiment $-$ 1 (%)")
    p.add("axhline", 0.0, color="0.5", linewidth=0.6)
    width = 0.6
    offs = np.linspace(-width / 2, width / 2, len(want)) if len(want) > 1 else np.zeros(1)
    notes = []
    for k, q in enumerate(want):
        xs, ys, es = [], [], []
        for i, el in enumerate(elements):
            th = sig = ex = None
            if q in ("alpha0", "C6"):
                d = catalog.load_json(el, "polarizability", data_dir) or {}
                th = d.get("alpha0_au" if q == "alpha0" else "C6_au")
                sig = d.get("alpha0_sigma_au" if q == "alpha0" else "C6_sigma_au")
                ex = (ALPHA0_EXPT if q == "alpha0" else C6_EXPT).get(el)
            else:
                lit = LITERATURE.get(el, {})
                d = catalog.load_json(el, "lifetimes", data_dir) or {}
                lab = f"{lit.get('np', '')}{'1/2' if q == 'tau_D1' else '3/2'}"
                row = next((r for r in d.get("lifetimes", []) if r.get("state") == lab), None)
                if row:
                    th, sig = row.get("lifetime_ns"), row.get("sigma_tau_ns")
                v = lit.get("tau_D1_ns" if q == "tau_D1" else "tau_D2_ns")
                ex = float(v) if isinstance(v, (int, float)) else None
            if th and ex:
                xs.append(i + offs[k])
                ys.append(100 * (th / ex - 1))
                es.append(100 * (sig or 0) / ex)
        if xs:
            p.add("errorbar", xs, ys, yerr=es, linestyle="none", marker="os^D"[k % 4], color=f"C{k}",
                  label=QUANTITIES[q])
    if len(p.ops) == 1:
        raise TemplateError("No polarizability or lifetime results to compare: run those stages first.")
    p.xticks = list(range(len(elements)))
    p.xticklabels = elements
    p.xlim = [-0.6, len(elements) - 0.4]
    p.xminor = False
    p.legend = {"loc": "best", "ncol": min(len(want), 4), "frame": False}
    notes.append("Reference values: α0 and C6 from constants.ALPHA0_EXPT / C6_EXPT, D-line lifetimes from "
                 "spectroscopy.LITERATURE (Tables and variables). Error bars: the theory 1σ.")
    return Result([p], notes, aspect=0.62)


# ----------------------------------------------------------------------------------------------- α(λ)
def _alpha_curve(trans: list[dict], wl: np.ndarray) -> np.ndarray:
    dE = np.array([abs(t["delta_E_au"]) for t in trans])
    f = np.array([t["fosc"] for t in trans])
    s = np.array([t.get("sign", 1.0) for t in trans])
    om = NM_TO_AU_OMEGA / wl
    denom = dE[None, :] ** 2 - om[:, None] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (s * f / denom).sum(axis=1)
    # Break the curve only at the poles themselves (the grid holds every resonance wavelength), so it
    # still reaches the magic crossings a fraction of a nm away.
    on_pole = (np.abs(denom) <= 1e-9 * dE[None, :] ** 2).any(axis=1)
    a[on_pole | ~np.isfinite(a)] = np.nan
    cont = [abs(t["delta_E_au"]) for t in trans if t.get("source") == "continuum"]
    if cont:
        a[om >= min(cont)] = np.nan                    # no Re α above the photoionization threshold
    return a


def _magic_n_filter(params: dict, d: dict, magic: list, wl: np.ndarray, alpha_e: np.ndarray,
                    notes: list[str]) -> np.ndarray:
    """Keep the magic wavelengths next to excited-state lines with n in the chosen range; outside it, hide the
    excited curve around each line (out past its crossing), so no unmarked crossing is left on the plot."""
    from qali.physics.polarizability_core import default_magic_n_range

    rng = d.get("magic_n_range") or default_magic_n_range(d.get("gs_label", ""))
    n_lo = int(params["magic_n_min"]) if params.get("magic_n_min") is not None else int(rng[0])
    n_hi = int(params["magic_n_max"]) if params.get("magic_n_max") is not None else int(rng[1])
    n_lo, n_hi = min(n_lo, n_hi), max(n_lo, n_hi)
    lines = [(NM_TO_AU_OMEGA / abs(t["delta_E_au"]), t.get("other_label", "")) for t in d["transitions_ex"]
             if t.get("source") not in ("core", "continuum") and t["delta_E_au"]]
    if not lines:
        return np.ones(len(magic), dtype=bool)
    lam = np.array([x[0] for x in lines])
    n_of = np.array([int(m.group(1)) if (m := re.match(r"^(\d+)", lab)) else 0 for _w, lab in lines])
    out = (n_of < n_lo) | (n_of > n_hi)
    reach = np.full(len(lines), MAGIC_CUTOFF_NM)            # how far around each line to hide
    keep = np.ones(len(magic), dtype=bool)
    for i, m in enumerate(magic):
        j = int(np.argmin(np.abs(lam - m["wl_nm"])))
        if out[j]:
            keep[i] = False
            reach[j] = max(reach[j], 2.0 * abs(m["wl_nm"] - lam[j]) + 1e-3)
    kept = np.array([m["wl_nm"] for m, k in zip(magic, keep) if k])
    for j in np.flatnonzero(out):
        if len(kept):                                        # never hide a crossing that is kept
            reach[j] = min(reach[j], 0.9 * float(np.min(np.abs(kept - lam[j]))))
        alpha_e[np.abs(wl - lam[j]) <= reach[j]] = np.nan
    notes.append(f"Magic wavelengths next to {d['excited_label']} lines with n = {n_lo}–{n_hi}: "
                 f"{int(keep.sum())} of the {len(magic)} in range.")
    return keep


ALPHA_POINTS = 4000          # default samples of α(λ) across the range (plus the points near each resonance)
MAGIC_SHOW = {"range": "Cut near lines, full in a window", "all": "All crossings", "practical": "Practical only"}
MAGIC_CUTOFF_NM = 0.5        # default half-width of the cut around each line ("range")


def polarizability_spectrum(params: dict, species: str, data_dir: str, style: dict) -> Result:
    d = _need(catalog.load_json(species, "polarizability", data_dir), species, "polarizability")
    r = d.get("wl_range_nm") or [400, 1600]
    lo = float(params.get("wl_min") or max(float(r[0]), 400.0))
    hi = float(params.get("wl_max") or min(float(r[1]), 1600.0))
    if lo > hi:
        lo, hi = hi, lo
    res = sorted({NM_TO_AU_OMEGA / abs(t["delta_E_au"]) for t in d["transitions_gs"] + d["transitions_ex"]
                  if t.get("source") not in ("core", "continuum") and t["delta_E_au"]})
    n_pts = int(params.get("points") or ALPHA_POINTS)
    n_pts = min(max(n_pts, 200), 200_000)
    magic = [m for m in d.get("magic_wavelengths", []) if lo <= m["wl_nm"] <= hi and m.get("alpha_au") is not None]
    # Near each resonance α runs off to ±∞ within ~0.01 nm, so sample it on a log scale down to 1e-5 nm;
    # the resonances themselves (where the curve breaks) and the magic wavelengths are grid points.
    side = np.geomspace(1e-5, 2.0, 80)
    grid = [np.linspace(lo, hi, n_pts), np.array(res), np.array([m["wl_nm"] for m in magic])]
    grid += [np.concatenate([r - side, r + side]) for r in res if lo - 2 < r < hi + 2]
    wl = np.unique(np.concatenate(grid))
    wl = wl[(wl >= lo) & (wl <= hi)]
    which = params.get("states", "both")
    show = params.get("magic_show") or "range"
    if show not in MAGIC_SHOW:
        raise TemplateError(f"Unknown magic display '{show}'.")
    p = Panel(xlabel=r"Wavelength (nm)", ylabel=r"$\alpha$ (a.u.)")
    curves = []
    if which in ("both", "ground"):
        curves.append(("g", d["gs_label"], d["transitions_gs"], "C0"))
    if which in ("both", "excited") and d.get("excited_label"):
        curves.append(("e", d["excited_label"], d["transitions_ex"], "C1"))
    om = NM_TO_AU_OMEGA / wl
    alphas = {key: _alpha_curve(trans, wl) for key, _lab, trans, _c in curves}
    search = d.get("magic_search_nm")
    if "e" in alphas and search:          # below the excited state's ionization threshold Re α is not a line sum
        alphas["e"][wl < float(search[0])] = np.nan
    keep_magic = np.ones(len(magic), dtype=bool)
    mw = np.array([m["wl_nm"] for m in magic], dtype=float)
    notes = [f"α(λ) from the {species}_polarizability.json line lists (bound lines, ionic core, continuum)."]
    if show == "range":
        # Outside the window, cut each curve within ±cutoff of its own lines (and the crossings there);
        # inside it, draw everything: sweep the window to look at any part of the spectrum in full.
        cut = max(float(params.get("cutoff_nm") or MAGIC_CUTOFF_NM), 0.0)
        w_lo, w_hi = params.get("range_min"), params.get("range_max")
        if w_lo is not None or w_hi is not None:            # one end set: the other is the plot's edge
            w_lo, w_hi = sorted((lo if w_lo is None else float(w_lo), hi if w_hi is None else float(w_hi)))
        inside = np.zeros(len(wl), dtype=bool) if w_lo is None or w_hi is None else (wl >= w_lo) & (wl <= w_hi)
        lines_all = []
        for key, _lab, trans, _c in curves:
            lines = np.array([NM_TO_AU_OMEGA / abs(t["delta_E_au"]) for t in trans
                              if t.get("source") not in ("core", "continuum") and t["delta_E_au"]])
            lines_all.append(lines)
            if len(lines) and cut > 0:
                near = np.abs(wl[:, None] - lines[None, :]).min(axis=1) < cut
                alphas[key][near & ~inside] = np.nan
        if len(mw) and cut > 0 and lines_all:
            every = np.concatenate(lines_all)
            m_in = np.zeros(len(mw), dtype=bool) if w_lo is None or w_hi is None else (mw >= w_lo) & (mw <= w_hi)
            keep_magic = m_in | (np.abs(mw[:, None] - every[None, :]).min(axis=1) >= cut)
        window = f"; drawn in full from {w_lo:g} to {w_hi:g} nm" if w_lo is not None and w_hi is not None else ""
        notes.append(f"Curves cut within ±{cut:g} nm of each line{window}.")
    elif show == "practical":
        from qali.core.constants import ATOMIC_MASS_AMU
        from qali.physics.polarizability_core import MAGIC_HEATING_MIN_S, TrapScattering, state_decay_rates

        if which != "both" or not d.get("excited_label"):
            raise TemplateError("Practical magic wavelengths need both states: pick Ground and excited.")
        rates = state_decay_rates(catalog.load_json(species, "lifetimes", data_dir))
        mass = d.get("mass_amu") or ATOMIC_MASS_AMU.get(d.get("element") or species)
        sc = TrapScattering(d["transitions_gs"], d["gs_label"], d["transitions_ex"], d["excited_label"], rates,
                            mass_amu=mass)
        t_min = float(params.get("heating_min") or d.get("magic_heating_min_s") or MAGIC_HEATING_MIN_S)
        p_max = params.get("excite_max")
        if p_max is None:
            p_max = d.get("magic_excite_max")
        with np.errstate(invalid="ignore"):
            alphas["g"][sc.heating_time(om, alphas["g"]) < t_min] = np.nan
            if p_max is not None:
                alphas["e"][sc.excited_per_decay(om, alphas["e"]) > p_max] = np.nan
            if len(mw):
                om_m = NM_TO_AU_OMEGA / mw
                a_m = np.array([m["alpha_au"] for m in magic], dtype=float)
                keep_magic = ~(sc.heating_time(om_m, a_m) < t_min)
                if p_max is not None:
                    keep_magic &= ~(sc.excited_per_decay(om_m, a_m) > p_max)
        w_min = params.get("window_min")
        if w_min is None:
            w_min = d.get("magic_window_min_ghz")
        if w_min is not None and len(mw):
            from qali.physics.polarizability_core import magic_window_ghz

            for i, m in enumerate(magic):
                slope = m.get("delta_alpha_slope_au_per_nm")
                if slope is None or magic_window_ghz(m["wl_nm"], m["alpha_au"], slope) >= w_min:
                    continue
                keep_magic[i] = False
                # hide the excited curve's branch through this crossing: out to its line and as far beyond
                r = m.get("nearby_resonance_nm")
                half = 2.0 * abs(m["wl_nm"] - r) if r is not None else 0.0
                alphas["e"][np.abs(wl - (r if r is not None else m["wl_nm"])) <= max(half, 1e-4)] = np.nan
        extra = (f"; at most {p_max:g} trap photons absorbed by {d['excited_label']} per decay (1 mK deep)"
                 if p_max is not None else "")
        extra += f"; magic window (|Δα| ≤ 1 % of α) at least {w_min:g} GHz wide" if w_min is not None else ""
        notes.append(f"Practical: photon recoil takes at least {t_min:g} s to heat {d['gs_label']} atoms out of "
                     f"the trap (any depth){extra}; curves hidden where that fails.")
        if not rates:
            notes.append(f"No {species}_lifetimes.json: line widths from each line's own A only.")
    else:
        notes.append("Curves break only at the resonances; every crossing is a magic wavelength.")
    if "e" in alphas:
        keep_magic &= _magic_n_filter(params, d, magic, wl, alphas["e"], notes)
    scale0 = []
    for key, lab, trans, col in curves:
        p.add("plot", wl, alphas[key], color=col, label=f"$\\alpha_{{{_pretty(lab).strip('$')}}}$")
        scale0.append(abs(_alpha_curve(trans, np.array([1e7]))[0]))
    shown = [m for m, k in zip(magic, keep_magic) if k]
    if params.get("magic", True) and which == "both" and shown:
        look = {"marker": params.get("magic_marker") or "o", "color": params.get("magic_colour") or "k"}
        if params.get("magic_fill", "open") == "open":
            look["markerfacecolor"] = "none"
        if params.get("magic_size"):
            look["markersize"] = float(params["magic_size"])
        p.add("plot", [m["wl_nm"] for m in shown], [m["alpha_au"] for m in shown], linestyle="none",
              label="magic", zorder=3, **look)
    if which == "both" and magic:
        notes.append(f"{len(shown)} of the {len(magic)} magic wavelengths in range shown.")
    ymax = float(params.get("ymax") or 0) or 3 * max(scale0 + [100.0])
    p.ylim = [-ymax, ymax]
    p.add("axhline", 0.0, color="0.6", linewidth=0.5)
    p.legend = {"loc": "best", "ncol": 1, "frame": False}
    return Result([p], notes, aspect=0.7)


# ---------------------------------------------------------------------------------------- free builder
KINDS = {"markers": "Markers", "line": "Line", "line+markers": "Line and markers", "errorbar": "Error bars",
         "band": "Line with ±error band", "step": "Steps", "bar": "Bars", "heatmap": "Heat map (z)",
         "contour": "Contours (z)"}
MAP_KINDS = ("heatmap", "contour")
SERIES_AXES = {"": "Left", "y2": "Right", "inset": "Inset"}
BREAKS = {"": "None", "x": "Break x axis", "y": "Break y axis"}
RESIDUALS = {"": "None", "difference": "Difference from series 1", "ratio": "Ratio to series 1",
             "percent": "Deviation from series 1 (%)"}


def builder(params: dict, species: str, data_dir: str, style: dict) -> Result:
    series = params.get("series") or []
    if not series:
        raise TemplateError("Add a series: pick a table and the columns for x and y.")
    p = Panel()
    tables: dict[tuple[str, str], catalog.Table] = {}
    xs_used: set[str] = set()
    ys_used: set[str] = set()
    y2_used: set[str] = set()
    inset_used: set[str] = set()
    notes = []
    drawn: list[tuple[np.ndarray, np.ndarray, dict, str]] = []          # for the residual panel
    for i, s in enumerate(series):
        src_file = s.get("file") or ""
        sp = s.get("species") or species
        key = (src_file or sp, s.get("table", ""))
        if key not in tables:
            t = catalog.find_file_table(src_file, key[1]) if src_file else catalog.find_table(sp, data_dir, key[1])
            if t is None:
                where = os.path.basename(src_file) if src_file else sp
                raise TemplateError(f"Series {i + 1}: no table '{key[1]}' in {where}.")
            tables[key] = t
        t = tables[key]
        try:
            x = catalog.evaluate_column(s.get("x") or "", t).astype(float)
            y = catalog.evaluate_column(s.get("y") or "", t).astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if s.get("where"):
                mask &= np.asarray(catalog.evaluate_column(s["where"], t), dtype=bool)
            yerr = catalog.evaluate_column(s["yerr"], t).astype(float) if s.get("yerr") else None
            xerr = catalog.evaluate_column(s["xerr"], t).astype(float) if s.get("xerr") else None
        except catalog.ExpressionError as exc:
            raise TemplateError(f"Series {i + 1}: {exc}") from None
        except (TypeError, ValueError):
            raise TemplateError(f"Series {i + 1}: x and y must be numbers.") from None
        x, y = x[mask], y[mask]
        yerr = yerr[mask] if yerr is not None else None
        xerr = xerr[mask] if xerr is not None else None
        kind = s.get("kind", "markers")
        on = s.get("on") or None
        if on not in (None, "y2", "inset"):
            raise TemplateError(f"Series {i + 1}: unknown axes '{on}'.")
        if kind in MAP_KINDS:
            try:
                z = catalog.evaluate_column(s.get("z") or "", t).astype(float)
            except catalog.ExpressionError as exc:
                raise TemplateError(f"Series {i + 1}: z: {exc}") from None
            except (TypeError, ValueError):
                raise TemplateError(f"Series {i + 1}: z must be numbers.") from None
            z = z[mask]
            ok = np.isfinite(z)
            x, y, z = x[ok], y[ok], z[ok]
            if len(x) < 3:
                raise TemplateError(f"Series {i + 1}: a map needs at least 3 points with x, y and z.")
            _add_map(p, kind, x, y, z, s, on)
            p.colorbar = s.get("zlabel") or (catalog.label_for(s["z"]) if (s.get("z") or "").isidentifier()
                                             else s.get("z") or "")
            if on != "inset":
                xs_used.add(s.get("x"))
            (inset_used if on == "inset" else y2_used if on == "y2" else ys_used).add(s.get("y"))
            origin = os.path.basename(src_file) if src_file else f"{sp} {t.stage}" + (f" ({t.run})" if t.run else "")
            notes.append(f"Series {i + 1}: {len(x)} points from {origin} ({t.path}).")
            continue
        if kind in ("line", "line+markers", "step", "band") and s.get("sort", True):
            o = np.argsort(x)
            x, y = x[o], y[o]
            yerr = yerr[o] if yerr is not None else None
            xerr = xerr[o] if xerr is not None else None
        col = s.get("color") or "auto"
        col = f"C{i}" if col == "auto" else col
        mk = s.get("marker") or "os^Dv"[i % 5]
        mlook = {"marker": mk}                       # marker type, and its own size and fill when set
        if float(s.get("marker_size") or 0) > 0:
            mlook["markersize"] = float(s["marker_size"])
        if s.get("marker_fill") == "open":
            mlook["markerfacecolor"] = "none"
        ls = s.get("linestyle") or "-"
        kw = {"color": col}
        if s.get("label"):
            kw["label"] = s["label"]
        if kind == "markers":
            look = {"linestyle": "none", **mlook}
            p.add("plot", x, y, on=on, **look, **kw)
        elif kind == "line":
            look = {"linestyle": ls}
            p.add("plot", x, y, on=on, **look, **kw)
        elif kind == "line+markers":
            look = {"linestyle": ls, **mlook}
            p.add("plot", x, y, on=on, **look, **kw)
        elif kind == "errorbar":
            look = {"linestyle": "none", **mlook}
            p.add("errorbar", x, y, yerr=yerr, xerr=xerr, on=on, **look, **kw)
        elif kind == "band":
            look = {"linestyle": ls}
            if yerr is not None:
                p.add("fill_between", x, y - np.abs(yerr), y + np.abs(yerr), on=on, color=col, alpha=0.25,
                      linewidth=0)
            p.add("plot", x, y, on=on, **look, **kw)
        elif kind == "step":
            look = {"linestyle": ls}
            p.add("step", x, y, where="mid", on=on, **look, **kw)
        elif kind == "bar":
            look = {}
            width = float(np.min(np.diff(np.unique(x)))) * 0.8 if len(np.unique(x)) > 1 else 0.8
            p.add("bar", x, y, width=width, yerr=yerr, on=on, **kw)
        else:
            raise TemplateError(f"Series {i + 1}: unknown kind '{kind}'.")
        if on is None:
            drawn.append((x, y, {**look, **kw}, kind))
            xs_used.add(s.get("x"))
            ys_used.add(s.get("y"))
        elif on == "y2":
            xs_used.add(s.get("x"))
            y2_used.add(s.get("y"))
        else:
            inset_used.add(s.get("y"))
        origin = os.path.basename(src_file) if src_file else f"{sp} {t.stage}" + (f" ({t.run})" if t.run else "")
        notes.append(f"Series {i + 1}: {len(x)} points from {origin} ({t.path}).")
    if len(xs_used) == 1:
        xc = next(iter(xs_used)) or ""
        p.xlabel = catalog.label_for(xc) if xc.isidentifier() else xc
    if len(ys_used) == 1:
        yc = next(iter(ys_used)) or ""
        p.ylabel = catalog.label_for(yc) if yc.isidentifier() else yc
    if any(s.get("label") for s in series):
        p.legend = {"loc": "best", "ncol": 1, "frame": False}
    _builder_axes(p, params, y2_used, inset_used, notes)
    mode = params.get("residual") or ""
    if mode and mode in RESIDUALS and len(drawn) > 1:
        rx, ry = drawn[0][0], drawn[0][1]
        o = np.argsort(rx)
        rx, ry = rx[o], ry[o]
        q = Panel(height_ratio=0.4)
        base = 1.0 if mode == "ratio" else 0.0
        q.add("axhline", base, color=drawn[0][2].get("color", "C0"), linewidth=0.7)
        for x, y, look, kind in drawn[1:]:
            ref = np.interp(x, rx, ry, left=np.nan, right=np.nan)       # series 1 at these x
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (y - ref if mode == "difference" else y / ref if mode == "ratio" else 100 * (y / ref - 1))
            ok = np.isfinite(r)
            if ok.any():
                kw2 = {k: v for k, v in look.items() if k != "label"}
                if kind in ("markers", "errorbar"):
                    kw2 = {"linestyle": "none", "color": look.get("color"),
                           **{k: v for k, v in look.items() if k.startswith("marker")}}
                    kw2.setdefault("marker", "o")
                q.add("plot", x[ok], r[ok], **kw2)
        q.ylabel = {"difference": "difference", "ratio": "ratio", "percent": "deviation (%)"}[mode]
        q.xlabel, p.xlabel = p.xlabel, ""
        q.xscale = p.xscale
        notes.append("Lower panel: series 2 onwards against series 1, interpolated at their x "
                     "(outside its range: not shown).")
        return Result([p, q], notes, aspect=0.95, sharex=True)
    return Result([p], notes, aspect=0.75)

def _label_of(cols: set) -> str:
    if len(cols) != 1:
        return ""
    c = next(iter(cols)) or ""
    return catalog.label_for(c) if c.isidentifier() else c


def _num_pair(v, both: bool = False) -> list | None:
    """A range from two fields: an empty end stays automatic (None), unless *both* ends are required."""
    if not v or len(v) != 2:
        return None
    ends: list = []
    for e in v:
        try:
            ends.append(None if e is None or e == "" else float(e))
        except (TypeError, ValueError):
            ends.append(None)
    if ends == [None, None] or (both and None in ends):
        return None
    return None if ends[0] is not None and ends[0] == ends[1] else ends   # equal ends: no range


def _builder_axes(p: Panel, params: dict, y2_used: set, inset_used: set, notes: list[str]) -> None:
    """The plot builder's right-hand axis, inset and broken axis."""
    if any(o.get("on") == "y2" for o in p.ops):
        p.y2 = {"label": params.get("y2label") or _label_of(y2_used), "scale": params.get("y2scale") or "linear",
                "lim": _num_pair(params.get("y2lim"))}
    if any(o.get("on") == "inset" for o in p.ops):
        p.inset = {"bounds": params.get("inset_bounds") or [0.58, 0.55, 0.38, 0.4],
                   "xlim": _num_pair(params.get("inset_xlim")), "ylim": _num_pair(params.get("inset_ylim")),
                   "xscale": params.get("inset_xscale") or "linear", "yscale": params.get("inset_yscale") or "linear"}
    for key, what in (("y2lim", "right-hand axis"), ("inset_xlim", "inset x"), ("inset_ylim", "inset y")):
        raw = params.get(key)
        if raw and len(raw) == 2 and None not in raw and "" not in raw and _num_pair(raw) is None:
            notes.append(f"The {what} range was left out: its two ends are equal.")
    axis = params.get("break_axis") or ""
    if axis:
        at = _num_pair(params.get("break_at"), both=True)
        if axis not in ("x", "y"):
            raise TemplateError(f"Unknown axis break '{axis}'.")
        if at is None or at[0] == at[1]:
            raise TemplateError("Give the range to leave out of the broken axis (from, to).")
        if p.y2 is not None or p.inset is not None:
            raise TemplateError("A broken axis can't be combined with a right-hand axis or an inset.")
        p.axis_break = {"axis": axis, "at": sorted(at)}
        notes.append(f"The {axis} axis leaves out {min(at):g} to {max(at):g}.")


def _on_grid(x: np.ndarray, y: np.ndarray):
    """(unique x, unique y, index into the grid) when the points fill a rectangular grid once, else None."""
    ux, ix = np.unique(x, return_inverse=True)
    uy, iy = np.unique(y, return_inverse=True)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(x):
        return None
    flat = iy * len(ux) + ix
    if len(np.unique(flat)) != len(x):
        return None
    return ux, uy, flat


def _add_map(p: Panel, kind: str, x, y, z, s: dict, on) -> None:
    """A heat map or contours of z(x, y): on a grid when the points form one, else triangulated."""
    cmap = s.get("cmap") or "viridis"
    levels = int(s.get("levels") or 12)
    g = _on_grid(x, y)
    if g is not None:
        ux, uy, flat = g
        Z = np.full(len(ux) * len(uy), np.nan)
        Z[flat] = z
        Z = Z.reshape(len(uy), len(ux))
        if kind == "heatmap":
            p.add("pcolormesh", ux, uy, Z, on=on, shading="nearest", cmap=cmap, rasterized=True)
        else:
            p.add("contourf", ux, uy, Z, levels, on=on, cmap=cmap)
            p.add("contour", ux, uy, Z, levels, on=on, colors="k", linewidths=0.4, linestyles="solid")
        return
    if kind == "heatmap":
        p.add("tricontourf", x, y, z, max(levels, 64), on=on, cmap=cmap)
    else:
        p.add("tricontourf", x, y, z, levels, on=on, cmap=cmap)
        p.add("tricontour", x, y, z, levels, on=on, colors="k", linewidths=0.4, linestyles="solid")


# --------------------------------------------------------------------------- contributions, convergence, budget
CONTRIB_QTY = {"alpha0": r"$\alpha_0$", "C6": r"$C_6$"}
CONTRIB_VIEWS = {"channels": "Contributions by line", "convergence": "Convergence with n",
                 "budget": "Uncertainty budget"}
_NL = re.compile(r"^(\d+)([a-z])")


def _line_name(gs: str, other: str) -> str:
    return f"{_pretty(gs).strip('$')}$-${_pretty(other).strip('$')}" if other not in ("core", "continuum") else other


def _polarizability_terms(species: str, data_dir: str, qty: str):
    """Per line of the ground state's sum: (label, value, sigma, n), adding up to α0 or C6 of the
    Polarizability stage. α0 = Σ f/ΔE²; C6 = (3/2) ΣΣ f_i f_j / (ΔE_i ΔE_j (ΔE_i + ΔE_j)), split by rows;
    σ from σ_f by first-order propagation (as the stage does). Core and continuum are pseudo-lines."""
    d = _need(catalog.load_json(species, "polarizability", data_dir), species, "polarizability")
    tg = [t for t in d.get("transitions_gs", []) if t.get("delta_E_au")]
    if not tg:
        raise TemplateError(f"No transitions in {species}'s polarizability output.")
    f = np.array([t["fosc"] * t.get("sign", 1) for t in tg])
    E = np.array([abs(t["delta_E_au"]) for t in tg])
    sf = np.array([t.get("sigma_f") or 0.0 for t in tg])
    if qty == "C6":
        K = 1.0 / (np.outer(E, E) * (E[:, None] + E[None, :]))
        val = 1.5 * f * (K @ f)
        sig = 3.0 * (K @ f) * sf
        total, total_sig = d.get("C6_au"), d.get("C6_sigma_au")
    else:
        val = f / E ** 2
        sig = sf / E ** 2
        total, total_sig = d.get("alpha0_au"), d.get("alpha0_sigma_au")
    gs = d.get("gs_label", "")
    rows = []
    for t, v, sg in zip(tg, val, sig):
        other = t.get("other_label", "")
        if t.get("source") in ("core", "continuum"):
            other = t["source"]
        m = _NL.match(other)
        rows.append((other, float(v), float(sg), int(m.group(1)) if m else None, t.get("source", "")))
    return gs, rows, float(total if total is not None else val.sum()), float(total_sig or 0.0)


def contributions(params: dict, species: str, data_dir: str, style: dict) -> Result:
    qty = params.get("quantity", "alpha0")
    if qty not in CONTRIB_QTY:
        raise TemplateError(f"Unknown quantity {qty}.")
    view = params.get("view", "channels")
    top = max(1, int(params.get("top", 6)))
    gs, rows, total, total_sig = _polarizability_terms(species, data_dir, qty)
    sym = CONTRIB_QTY[qty]
    lines = [r for r in rows if r[0] not in ("core", "continuum")]
    core = sum(r[1] for r in rows if r[0] == "core")
    cont = sum(r[1] for r in rows if r[0] == "continuum")
    # the lines, J components merged (5p1/2 + 5p3/2 -> 5p) unless asked otherwise
    merge = bool(params.get("merge_j", True))
    groups: dict[str, list] = {}
    for r in lines:
        m = _NL.match(r[0])
        key = m.group(0) if (merge and m) else r[0]
        g = groups.setdefault(key, [key, 0.0, 0.0, r[3]])
        g[1] += r[1]
        g[2] = math.hypot(g[2], r[2])
    p = Panel()
    plain = {"alpha0": "α0", "C6": "C6"}[qty]
    notes = [f"{species} {gs} ground state: {plain} = {total:.6g} ± {total_sig:.3g} a.u. (Polarizability stage)."]
    head = ["item", "value_au", "sigma_au", "share_pct"]
    table_rows: list[list] = []
    if view == "channels":
        ordered = sorted(groups.values(), key=lambda g: -abs(g[1]))
        shown, rest = ordered[:top], ordered[top:]
        items = [(f"${_line_name(gs, g[0])}$", g[1]) for g in shown]
        if rest:
            items.append((f"other lines ({len(rest)})", sum(g[1] for g in rest)))
        items += [("continuum", cont), ("core", core)]
        items = [it for it in items if it[1] != 0.0]
        y = np.arange(len(items))[::-1]
        vals = np.array([it[1] for it in items])
        p.add("barh", y, vals, color=[("C0" if i < len(shown) else "0.6") for i in range(len(items))], height=0.7)
        log = bool(params.get("log", True))
        for yy, v in zip(y, vals):
            pct = 100 * v / total
            txt = f"{pct:.3g} %" if abs(pct) >= 0.01 else f"{pct:.1e} %"
            p.add("text", v * (1.15 if log else 1.0) if v > 0 else 0, yy, " " + txt, va="center", ha="left",
                  fontsize=style.get("font_size", 8) * 0.75)
        p.yticks = list(y)
        p.yticklabels = [it[0] for it in items]
        p.xscale = "log" if log else "linear"
        p.xlabel = f"{sym} contribution (a.u.)"
        hi = vals.max()
        p.xlim = [max(vals[vals > 0].min() / 3, hi * 1e-7), hi * 20] if log else [0, hi * 1.25]
        p.yminor = False
        table_rows = [[it[0].strip("$").replace("$-$", "-"), it[1], "", 100 * it[1] / total] for it in items]
        aspect = 0.2 + 0.075 * len(items)
    elif view == "convergence":
        with_n = sorted([g for g in groups.values() if g[3] is not None], key=lambda g: g[3])
        ns = sorted({g[3] for g in with_n})
        if not ns:
            raise TemplateError("No lines with a principal quantum number.")
        run = []
        acc = 0.0
        for n in ns:
            acc += sum(g[1] for g in with_n if g[3] == n)
            run.append(acc)
        run_arr = np.array(run) + cont + core          # lines up to n_max, with the continuum and core
        p.add("axhspan", total - total_sig, total + total_sig, color="0.85", linewidth=0, label=r"total $\pm\sigma$")
        p.add("axhline", total, color="0.3", linestyle="--", linewidth=0.8)
        p.add("plot", ns, run_arr, marker="o", linestyle="-", color="C0", label="lines up to $n_\\mathrm{max}$")
        p.xlabel = r"$n_\mathrm{max}$"
        p.xscale = "log"
        p.ylabel = f"{sym} (a.u.)"
        p.legend = {"loc": "lower right", "ncol": 1, "frame": False}
        notes.append(f"Lines summed up to n_max, plus the continuum ({cont:.3g}) and the core ({core:.3g}).")
        if params.get("relative", False):
            for op in p.ops:
                if op["op"] == "plot":
                    op["args"][1] = 100 * (np.asarray(op["args"][1]) / total - 1)
                elif op["op"] == "axhspan":
                    op["args"] = [-100 * total_sig / total, 100 * total_sig / total]
                elif op["op"] == "axhline":
                    op["args"] = [0.0]
            p.ylabel = f"{sym} / total $-$ 1 (%)"
            p.yscale = "symlog"
            p.linthresh = max(100 * total_sig / total, 1e-3)
        else:
            lo = min(run_arr.min(), total - 3 * total_sig)
            p.ylim = [lo - 0.05 * (total - lo), total + max(3 * total_sig, 0.04 * (total - lo))]
        table_rows = [[f"n_max={n}", r, "", 100 * r / total] for n, r in zip(ns, run_arr)]
        head = ["item", "partial_sum_au", "", "share_pct"]
        aspect = 0.75
    elif view == "budget":
        budget = [(g[0], g[2] ** 2) for g in groups.values() if g[2] > 0]
        budget += [(lab, sum(r[2] ** 2 for r in rows if r[0] == lab)) for lab in ("continuum",)]
        budget = [v for v in budget if v[1] > 0]
        tot_var = sum(v[1] for v in budget)
        if tot_var <= 0:
            raise TemplateError("No line of this sum has an uncertainty.")
        budget.sort(key=lambda v: -v[1])
        top_b, rest_b = budget[:top], budget[top:]
        items = [(f"${_line_name(gs, v[0])}$" if v[0] != "continuum" else "continuum", v[1]) for v in top_b]
        if rest_b:
            items.append((f"other lines ({len(rest_b)})", sum(v[1] for v in rest_b)))
        y = np.arange(len(items))[::-1]
        share = np.array([100 * it[1] / tot_var for it in items])
        p.add("barh", y, share, color="C3", height=0.7)
        for yy, v, it in zip(y, share, items):
            p.add("text", v * 1.15, yy, f" $\\pm${math.sqrt(it[1]):.2g}", va="center", ha="left",
                  fontsize=style.get("font_size", 8) * 0.75)
        p.yticks = list(y)
        p.yticklabels = [it[0] for it in items]
        p.xlabel = f"share of $\\sigma^2$ of {sym} (%)"
        p.xscale = "log"
        p.xlim = [max(share.min() / 3, share.max() * 1e-8), share.max() * 30]
        p.yminor = False
        notes.append("Core: no uncertainty assigned by the stage.")
        table_rows = [[it[0].strip("$").replace("$-$", "-"), "", math.sqrt(it[1]), s_] for it, s_ in zip(items, share)]
        head = ["item", "", "sigma_au", "share_of_variance_pct"]
        aspect = 0.2 + 0.075 * len(items)
    else:
        raise TemplateError(f"Unknown view {view}.")
    return Result([p], notes, aspect=aspect, table=(head, table_rows))


# --------------------------------------------------------------------------------------- decay branching
def lifetime_states(species: str, data_dir: str) -> list[str]:
    lt = catalog.load_json(species, "lifetimes", data_dir) or {}
    return [r["state"] for r in lt.get("lifetimes", []) if r.get("channels")]


def _default_branching_state(rows: dict, energy: dict[str, float]) -> str:
    """The lowest-lying state with three or more decay channels (else the one with the most)."""
    ordered = sorted(rows, key=lambda st: energy.get(st, math.inf))
    for st in ordered:
        if len(rows[st].get("channels") or []) >= 3:
            return st
    return max(ordered, key=lambda st: len(rows[st].get("channels") or [])) if ordered else ""


def branching(params: dict, species: str, data_dir: str, style: dict) -> Result:
    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    rows = {r["state"]: r for r in lt.get("lifetimes", []) if r.get("channels")}
    if not rows:
        raise TemplateError(f"No decay channels in {species}'s lifetimes output.")
    state = params.get("state")
    if not state:
        try:
            energy = {x.label: x.binding_eV for x in levels(species, data_dir)[0]}
        except TemplateError:
            energy = {}
        state = _default_branching_state(rows, energy)
    if state not in rows:
        raise TemplateError(f"No state {state} in {species}'s lifetimes output.")
    r = rows[state]
    ch = sorted(r["channels"], key=lambda c: -c.get("A_s", 0))
    top = max(1, int(params.get("top", 8)))
    shown, rest = ch[:top], ch[top:]
    items = [(f"{_pretty(c['lower_label'])}", 100 * c.get("branching_ratio", 0), c.get("wavelength_nm"),
              c.get("A_s"), c.get("fosc_source", "")) for c in shown]
    if rest:
        items.append((f"others ({len(rest)})", 100 * sum(c.get("branching_ratio", 0) for c in rest), None,
                      sum(c.get("A_s", 0) for c in rest), ""))
    p = Panel()
    y = np.arange(len(items))[::-1]
    share = np.array([it[1] for it in items])
    p.add("barh", y, share, color=["C0" if it[2] else "0.6" for it in items], height=0.7)
    for yy, it in zip(y, items):
        wl = it[2]
        lam = "" if wl is None else (f"{wl:.1f} nm" if wl < 1000 else f"{wl / 1000:#.3g} $\\mu$m")
        txt = f" {it[1]:.3g} %" + (f", {lam}" if lam else "")
        p.add("text", it[1], yy, txt, va="center", ha="left", fontsize=style.get("font_size", 8) * 0.75)
    p.yticks = list(y)
    p.yticklabels = [it[0] for it in items]
    p.yminor = False
    log = bool(params.get("log", False))
    if log:
        p.xscale = "log"
        p.xlim = [max(share[share > 0].min() / 3, 1e-6), 1000]
    else:
        p.xlim = [0, 135]
    p.xlabel = "branching ratio (%)"
    p.ylabel = f"decay of {_pretty(state)}"
    tau = r.get("lifetime_ns")
    sig = r.get("sigma_tau_ns")
    notes = [f"{species} {state}: τ = {tau:.4g} ns" + (f" ± {sig:.2g} ns" if sig else "")
             + f", {len(ch)} decay channels (Lifetimes stage)."]
    table = (["lower", "branching_pct", "wavelength_nm", "A_s-1", "f_source"],
             [[it[0].strip("$"), it[1], it[2] if it[2] is not None else "", it[3], it[4]] for it in items])
    return Result([p], notes, aspect=0.22 + 0.075 * len(items), table=table)


# ------------------------------------------------------------------------------------------- decay cascade
_LABEL = re.compile(r"^(\d+)([a-z])(\d+/2)?$")


def _level_energies(species: str, data_dir: str) -> tuple[dict[str, float], float]:
    """Excitation energy (eV above the ground state) by level label, and the ionization energy."""
    lv, IE, _Ry = levels(species, data_dir)
    return {x.label: x.binding_eV + IE for x in lv}, IE


def _energy_of(label: str, E: dict[str, float]) -> float | None:
    """A label's energy; a label without J (high n) takes the mean of its fine-structure levels."""
    if label in E:
        return E[label]
    m = _LABEL.match(label)
    if not m or m.group(3):
        return None
    vals = [e for k, e in E.items() if (mm := _LABEL.match(k)) and mm.group(1, 2) == m.group(1, 2)]
    return sum(vals) / len(vals) if vals else None


def cascade_flux(rows: dict, start: str, E: dict[str, float], floor: float = 1e-7):
    """Radiative cascade from *start*: the fraction of atoms passing through each level and the photons per
    atom on each line. Levels are taken in order of falling energy, so every level has collected all its
    feeding before it decays."""
    P = {start: 1.0}
    F: dict[tuple[str, str], float] = {}
    pending = {start}
    while pending:
        s = max(pending, key=lambda k: _energy_of(k, E) or -math.inf)
        pending.discard(s)
        for c in (rows.get(s) or {}).get("channels") or []:
            f = P[s] * float(c.get("branching_ratio") or 0.0)
            if f < floor:
                continue
            lo = c["lower_label"]
            P[lo] = P.get(lo, 0.0) + f
            F[(s, lo)] = F.get((s, lo), 0.0) + f
            pending.add(lo)
    return P, F


def cascade_time(rows: dict, start: str, E: dict[str, float]) -> float:
    """Mean time (ns) from *start* to the ground state: T(s) = τ(s) + Σ BR T(lower), radiative decay only."""
    memo: dict[str, float] = {}

    def T(s: str, depth: int = 0) -> float:
        if s in memo:
            return memo[s]
        r = rows.get(s)
        if not r or not r.get("channels") or depth > 200:
            return 0.0
        t = float(r.get("lifetime_ns") or 0.0) + sum(float(c.get("branching_ratio") or 0.0) *
                                                       T(c["lower_label"], depth + 1) for c in r["channels"])
        memo[s] = t
        return t

    return T(start)


def _default_cascade_state(rows: dict, E: dict[str, float]) -> str:
    """The lowest-lying level whose cascade passes through four or more levels above 1 %."""
    ordered = sorted((s for s in rows if _energy_of(s, E) is not None), key=lambda s: _energy_of(s, E) or 0.0)
    for s in ordered:
        P, _F = cascade_flux(rows, s, E)
        if sum(1 for v in P.values() if v >= 0.01) >= 5:
            return s
    return ordered[-1] if ordered else ""


def cascade_states(species: str, data_dir: str) -> list[str]:
    """Levels with decay channels, lowest first (the starts a cascade can have)."""
    lt = catalog.load_json(species, "lifetimes", data_dir) or {}
    names = [r["state"] for r in lt.get("lifetimes", []) if r.get("channels")]
    try:
        E, _IE = _level_energies(species, data_dir)
    except TemplateError:
        return names
    return sorted(names, key=lambda k: (_energy_of(k, E) is None, _energy_of(k, E) or 0.0))


def cascade(params: dict, species: str, data_dir: str, style: dict) -> Result:
    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    rows = {r["state"]: r for r in lt.get("lifetimes", []) if r.get("channels")}
    if not rows:
        raise TemplateError(f"No decay channels in {species}'s lifetimes output.")
    E, _IE = _level_energies(species, data_dir)
    state = params.get("state") or _default_cascade_state(rows, E)
    if state not in rows:
        raise TemplateError(f"No state {state} with decay channels in {species}'s lifetimes output.")
    cut = float(params.get("min_pct", 1.0)) / 100.0
    P, F = cascade_flux(rows, state, E)
    shown = {k for k, v in P.items() if v >= cut and _energy_of(k, E) is not None}
    lines = {k: v for k, v in F.items() if v >= cut and k[0] in shown and k[1] in shown}
    gs = min(shown, key=lambda k: _energy_of(k, E) or 0.0)
    p = Panel()
    fs = style.get("font_size", 8)

    def xpos(label: str) -> float:
        m = _LABEL.match(label)
        l_ = L_CHARS.index(m.group(2)) if m else 0
        if m and m.group(3) and l_ > 0:
            return l_ + (-0.2 if (_jnum(m.group(3)) or 0.0) < l_ else 0.2)
        return float(l_)

    def right_side(label: str) -> bool:
        """The upper-J level of a pair is labelled on its right, everything else on its left."""
        m = _LABEL.match(label)
        return bool(m and m.group(3) and (_jnum(m.group(3)) or 0.0) > L_CHARS.index(m.group(2)))

    unit = params.get("energy_unit", "eV")
    scale = EV_TO_CM if unit == "cm-1" else 1.0

    def en(label: str) -> float:
        return scale * (_energy_of(label, E) or 0.0)

    half = 0.12
    for k in sorted(shown, key=lambda k: _energy_of(k, E) or 0.0):
        x, y = xpos(k), en(k)
        p.add("hlines", y, x - half, x + half, color="k", linewidth=1.2)
        text = _pretty(k) + ("" if k == state else f" ({100 * P[k]:.3g} %)")
        box = {"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none", "alpha": 0.85}
        if right_side(k):
            p.add("text", x + half + 0.03, y, text, ha="left", va="center", fontsize=fs * 0.7, bbox=box, zorder=4)
        else:
            p.add("text", x - half - 0.03, y, text, ha="right", va="center", fontsize=fs * 0.7, bbox=box, zorder=4)
    wmax = max(lines.values()) if lines else 1.0
    label_lines = bool(params.get("line_labels", True))
    label_cut = float(params.get("label_pct", 0.0)) / 100.0         # every drawn line by default
    def colour(f: float) -> str:
        return "C0" if f >= 0.1 else "C1" if f >= 0.01 else "0.55"

    for (u, lo), f in sorted(lines.items(), key=lambda t: t[1]):          # strongest drawn last, on top
        lw = 0.4 + 2.4 * f / wmax
        p.add("annotate", "", [xpos(lo), en(lo)], xytext=[xpos(u), en(u)],
              arrowprops={"arrowstyle": "-|>", "color": colour(f), "linewidth": lw, "shrinkA": 0, "shrinkB": 0,
                          "mutation_scale": 6 + 3 * lw})
    ys = [en(k) for k in shown]
    yspan = (max(ys) - min(ys)) or 1.0
    placed: list[tuple[float, float, str]] = []

    def clash(x: float, y: float) -> bool:
        return any(abs(x - a) < 0.3 and abs(y - b) < 0.05 * yspan for a, b, _t in placed)

    for (u, lo), f in sorted(lines.items(), key=lambda t: -t[1]):         # strongest labelled first
        if not label_lines or f < label_cut:
            continue
        wl = next((c.get("wavelength_nm") for c in rows[u]["channels"] if c["lower_label"] == lo), None)
        lam = "" if wl is None else (f"{wl:.0f} nm" if wl < 1000 else f"{wl / 1000:.3g} $\\mu$m")
        spots = [(xpos(u) + a * (xpos(lo) - xpos(u)), en(u) + a * (en(lo) - en(u))) for a in (0.4, 0.25, 0.6, 0.75, 0.5)]
        x0, y0 = next((xy for xy in spots if not clash(*xy)), spots[0])   # else the first free spot along the line
        placed.append((x0, y0, lam))
        p.add("text", x0, y0, lam, ha="center", va="center", fontsize=fs * 0.55, color=colour(f),
              bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})
    ls = sorted({int(round(xpos(k))) for k in shown})
    p.xticks = ls
    p.xticklabels = [L_CHARS[i] for i in ls]
    p.xminor = False
    p.xlim = [min(ls) - 0.9, max(ls) + (1.3 if any(right_side(k) and xpos(k) > max(ls) for k in shown) else 0.9)]
    p.ylabel = r"energy above the ground state (cm$^{-1}$)" if unit == "cm-1" else \
        "energy above the ground state (eV)"
    p.spines = ["top", "right"]
    p.clip = True
    tau = rows[state].get("lifetime_ns")
    t_all = cascade_time(rows, state, E)
    left = 1.0 - sum(v for (u, lo), v in F.items() if lo == gs)
    notes = [f"{species} {state} (τ = {tau:.4g} ns): radiative cascade to {gs}, no blackbody transfer (0 K); "
             f"mean time to reach {gs}: {t_all:.4g} ns.",
             f"Beside each level: the share of atoms passing through it. Lines carrying under {100 * cut:g} % "
             "of the atoms are left out; arrow width follows the photons per atom."]
    if abs(left) > 1e-3:
        notes.append(f"{100 * left:.2g} % of the decay is in branches too weak to follow.")
    table = (["upper", "lower", "wavelength_nm", "photons_per_atom"],
             [[u, lo, next((c.get("wavelength_nm") for c in rows[u]["channels"] if c["lower_label"] == lo), ""),
               f] for (u, lo), f in sorted(F.items(), key=lambda t: -t[1])])
    return Result([p], notes, aspect=0.8, table=table)


# ------------------------------------------------------------------------------- oscillator strength density
def oscillator_density(params: dict, species: str, data_dir: str, style: dict) -> Result:
    """df/dε of the ground state across the ionization threshold: f n*³ for the bound lines (ε < 0) and the
    photoionization df/dε of the Polarizability stage's continuum pseudo-lines (ε > 0), in Hartree⁻¹."""
    from qali.physics.valence_continuum import energy_grid

    d = _need(catalog.load_json(species, "polarizability", data_dir), species, "polarizability")
    tg = d.get("transitions_gs", [])
    cont = sorted((t for t in tg if t.get("source") == "continuum"), key=lambda t: t["delta_E_au"])
    bound = [t for t in tg if t.get("source") not in ("core", "continuum") and t.get("delta_E_au", 0) > 0]
    if not bound:
        raise TemplateError(f"No bound lines in {species}'s polarizability output.")
    eps_grid = energy_grid()
    cavity = str(d.get("continuum_model") or "").startswith(("DHF", "Brueckner"))
    if cont and not cavity:
        E_bind = cont[0]["delta_E_au"] - eps_grid[0]
    else:
        lv, IE, _ = levels(species, data_dir)
        E_bind = IE / 27.211386245988
    unit = params.get("energy_unit", "eV")
    k = 27.211386245988 if unit == "eV" else 1.0
    merge = bool(params.get("merge_j", False))
    groups: dict[str, list] = {}
    for t in bound:
        eps = t["delta_E_au"] - E_bind
        if eps >= 0:
            continue
        lab = t.get("other_label", "")
        m = _NL.match(lab)
        if merge and m:
            key, series = m.group(0), m.group(2)
        else:
            j = lab[len(m.group(0)):] if m else ""
            key, series = lab, (f"{m.group(2)}_{{{j}}}" if m and j else (m.group(2) if m else lab))
        g = groups.setdefault(key, [0.0, 0.0, series, set()])       # Σf, Σ f ε, series, sources
        g[0] += t["fosc"]
        g[1] += t["fosc"] * eps
        g[3].add(t.get("source", ""))
    p = Panel()
    pts: dict[tuple[str, str], list[tuple[float, float]]] = {}
    rows = []
    for key, (fs, fe, series, srcs) in groups.items():
        eps = fe / fs                                   # f-weighted mean energy of merged J components
        nstar = 1.0 / math.sqrt(-2.0 * eps)
        dens = fs * nstar ** 3
        src = ("measured" if srcs & {"NIST", "literature", "precision"}
               else "extrapolated" if srcs & {"NIST-extrap", "extrapolated"} else "model")
        pts.setdefault((series, src), []).append((eps * k, dens / k))
        rows.append([key, eps * k, dens / k, "+".join(sorted(srcs))])
    all_series = sorted({sr for sr, _ in pts})
    for (series, src), xy in sorted(pts.items()):
        xy.sort()
        i = all_series.index(series)
        p.add("plot", [q[0] for q in xy], [q[1] for q in xy], linestyle="none", marker="os^D"[i % 4],
              color=f"C{i}", markerfacecolor="none" if src != "measured" else f"C{i}",
              markersize=(style.get("marker_size", 3.5) * (0.7 if src == "extrapolated" else 1.0)),
              label=f"${series}$, {src}")
    if cont and cavity:
        # a model's basis-cavity continuum: pseudo-lines at the cavity's own energies; df/dε is the
        # slope of their running sum, over ±0.04 eV, where the cavity resolves it (up to ~1 eV)
        ce_all = np.array([t["delta_E_au"] - E_bind for t in cont])
        F = np.cumsum([t["fosc"] for t in cont])
        half = 0.04 / 27.211386245988
        top = min(float(ce_all.max()), 1.0 / 27.211386245988)
        ce = np.linspace(0.0, top, 60)
        lo_, hi_ = np.maximum(ce - half, 0.0), ce + half
        cf = (np.interp(hi_, ce_all, F) - np.interp(lo_, ce_all, F)) / (hi_ - lo_)
        name = str(d["continuum_model"]).split(" ")[0]
        p.add("plot", ce * k, cf / k, color="0.25", linestyle="-", label=rf"continuum ({name}), $df/d\varepsilon$")
        rows += [["continuum", e * k, f / k, "model"] for e, f in zip(ce, cf)]
    elif cont:
        n = min(len(cont), len(eps_grid))
        w = np.zeros(len(eps_grid))
        dd = np.diff(eps_grid)
        w[:-1] += dd / 2
        w[1:] += dd / 2
        w[0] += eps_grid[0]
        ce = np.array([t["delta_E_au"] - E_bind for t in cont[:n]])
        cf = np.array([t["fosc"] for t in cont[:n]]) / w[:n]
        p.add("plot", ce * k, cf / k, color="0.25", linestyle="-", label=r"continuum, $df/d\varepsilon$")
        rows += [["continuum", e * k, f / k, "model"] for e, f in zip(ce, cf)]
    p.add("axvline", 0.0, color="0.5", linestyle="--", linewidth=0.6)
    p.xlabel = r"$\varepsilon = \hbar\omega - E_\mathrm{ion}$" + (" (eV)" if unit == "eV" else " (a.u.)")
    p.ylabel = r"$df/d\varepsilon$" + (" (eV$^{-1}$)" if unit == "eV" else " (a.u.)")
    p.yscale = "log"
    lo = params.get("eps_min")
    hi = params.get("eps_max")
    bound_n = sorted((int(m.group(1)), r[1]) for r in rows if r[0] != "continuum" and (m := _NL.match(r[0])))
    ns_ = sorted({n for n, _ in bound_n})
    # from just below the second member of the series (the resonance line is far left and much stronger)
    default_lo = 1.15 * min(e for n, e in bound_n if n == ns_[1]) if len(ns_) > 1 else -1.0
    p.xlim = [lo if lo is not None else default_lo, hi if hi is not None else -default_lo]
    p.legend = {"loc": "best", "ncol": 1, "frame": False}
    gs = d.get("gs_label", "")
    notes = [f"{species} {gs}: bound lines below the limit (ε < 0) as f·n*³, the photoionization continuum "
             "above it (Polarizability stage; filled: measured, small open: extrapolated from the measured series, "
             "open: model).",
             "df/dε is continuous across the limit, so a step at ε = 0 or between measured and model points "
             "shows f-values that do not agree.",
             "The resonance line lies left of the default range."]
    return Result([p], notes, aspect=0.75, table=(["line", f"eps_{unit}", f"df_deps_per_{unit}", "source"], rows))


# ----------------------------------------------------------------------------------------- Feshbach a(B)
def feshbach_channels(species: str, data_dir: str) -> list[str]:
    d = catalog.load_json(species, "feshbach_out", data_dir) or {}
    return [c["channel"] for c in d.get("channels", []) if c.get("a_a0")]


def feshbach(params: dict, species: str, data_dir: str, style: dict) -> Result:
    d = _need(catalog.load_json(species, "feshbach_out", data_dir), species, "feshbach")
    chans = {c["channel"]: c for c in d.get("channels", []) if c.get("a_a0")}
    if not chans:
        raise TemplateError(f"No scattering lengths in {species}'s Feshbach output.")
    want = params.get("channels") or [d.get("default_channel") or next(iter(chans))]
    want = [c for c in want if c in chans] or [next(iter(chans))]
    B = np.asarray(d.get("B_G", []), dtype=float)
    p = Panel()
    rows = []
    for i, name in enumerate(want):
        c = chans[name]
        a = np.asarray([math.nan if v is None else v for v in c["a_a0"]], dtype=float)
        # break the curve across each pole so no vertical line joins +∞ and −∞
        poles = sorted(c.get("poles_G") or [])
        a_plot = a.copy()
        for B0 in poles:
            j = np.searchsorted(B, B0)
            if 0 < j < len(B):
                a_plot[j - 1:j + 1] = np.where(np.sign(a[j - 1]) != np.sign(a[j]), np.nan, a_plot[j - 1:j + 1])
        lab = name.replace("|", r"$|$").replace(">", r"$\rangle$") if len(want) > 1 else None
        kw = {"label": lab} if lab else {}
        p.add("plot", B, a_plot, color=f"C{i}", **kw)
        if params.get("poles", True):
            for B0 in poles:
                p.add("axvline", B0, color=f"C{i}", linestyle=":", linewidth=0.6)
        if c.get("a_bg_a0") is not None:
            p.add("axhline", float(c["a_bg_a0"]), color="0.6", linestyle="--", linewidth=0.6)
        rows += [[name, b, v] for b, v in zip(B, a)]
    p.add("axhline", 0.0, color="0.4", linewidth=0.5)
    p.xlabel = "$B$ (G)"
    p.ylabel = "$a$ ($a_0$)"
    p.yscale = "symlog"
    p.linthresh = float(params.get("linthresh") or max(100.0, abs(float(d.get("a_bar_a0") or 100.0))))
    ymax = float(params.get("ymax") or 2e4)
    p.ylim = [-ymax, ymax]
    if params.get("B_min") is not None or params.get("B_max") is not None:
        p.xlim = [params.get("B_min") if params.get("B_min") is not None else float(B.min()),
                  params.get("B_max") if params.get("B_max") is not None else float(B.max())]
    else:
        p.xlim = [float(B.min()), float(B.max())]
    if len(want) > 1:
        p.legend = {"loc": "best", "ncol": 1, "frame": False}
    iso = d.get("isotope_A")
    notes = [f"{iso or ''}{species}, channel{'s' if len(want) > 1 else ''} " + ", ".join(want)
             + " (Feshbach stage); dotted: resonance poles, dashed: background a_bg."]
    return Result([p], notes, aspect=0.6, table=(["channel", "B_G", "a_a0"], rows))


# ------------------------------------------------------------------------------------------- blackbody
BBR_VIEWS = {"shifts": "Rydberg BBR shifts vs n", "rates": "Decay rates vs n", "temperature": "Temperature sweep",
             "map": "Map over n and temperature"}
BBR_MAP_QTY = {"tau": "Lifetime τ_eff", "gamma": "Blackbody rate Γ_BBR", "ratio": "Γ_BBR / Γ_rad"}


def blackbody_map(params: dict, species: str, data_dir: str, style: dict) -> Result:
    """τ_eff, Γ_BBR or Γ_BBR/Γ_rad of one series over (T, n), with the Blackbody stage's bound-bound physics
    (blackbody_core.bbr_depopulation: stimulated emission A n̄, absorption (g_u/g_l) A n̄) on the Lifetimes
    network. Blackbody ionization is not included."""
    from qali.physics.blackbody_core import build_lifetime_graph, nbar_photon, omega_from_nm

    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    graph = build_lifetime_graph(lt)
    lc = (params.get("l") or ["s"])[0]
    qty = params.get("quantity", "tau")
    if qty not in BBR_MAP_QTY:
        raise TemplateError(f"Unknown quantity {qty}.")
    l_int = L_CHARS.index(lc) if lc in L_CHARS else 0
    jw = params.get("J") or (f"{2 * l_int - 1}/2" if l_int else "")      # default: the lower J of the series
    T = np.linspace(float(params.get("T_min") or 10.0), float(params.get("T_max") or 600.0),
                    int(params.get("T_points") or 60))
    picked = []
    for label, info in graph.states.items():
        m = _LABEL.match(label)
        if not m or m.group(2) != lc or not info.get("n") or not info.get("A_total_s"):
            continue
        if jw and m.group(3) and m.group(3) != jw:
            continue
        picked.append((int(info["n"]), label))
    by_n: dict[int, str] = {}
    for n_, label in sorted(picked):
        by_n.setdefault(n_, label)                   # one level per n (labels without J at high n included)
    if len(by_n) < 2:
        raise TemplateError(f"Fewer than two {lc} levels with decay rates in {species}'s lifetimes output.")
    ns = sorted(by_n)
    Z = np.zeros((len(ns), len(T)))
    for i, n_ in enumerate(ns):
        label = by_n[n_]
        rate = np.zeros_like(T)
        for _partner, A, g_u, g_l, wl, kind in graph.edges_from.get(label, []):
            w = omega_from_nm(wl)
            nb = np.array([nbar_photon(w, t) for t in T])
            rate += A * nb * (1.0 if kind == "stim" else g_u / max(g_l, 1e-30))
        g_rad = float(graph.states[label].get("A_total_s") or 0.0)
        Z[i] = {"tau": 1e9 / (g_rad + rate), "gamma": rate, "ratio": rate / g_rad}[qty]
    with np.errstate(divide="ignore"):
        L = np.log10(np.where(Z > 0, Z, np.nan))
    p = Panel()
    hi = float(np.nanmax(L))
    lo = max(float(np.nanmin(L)), hi - float(params.get("decades") or 6.0))   # rates: ~zero at low n and T
    p.add("pcolormesh", T, np.asarray(ns, dtype=float), L, shading="nearest", vmin=lo, vmax=hi,
          cmap=params.get("cmap") or ("viridis" if qty == "tau" else "magma"), rasterized=True)
    if qty == "ratio":
        p.add("contour", T, np.asarray(ns, dtype=float), L, [0.0], colors="w", linewidths=0.8)
    p.colorbar = {"tau": r"$\log_{10}\,\tau_\mathrm{eff}$ (ns)", "gamma": r"$\log_{10}\,\Gamma_\mathrm{BBR}$ (s$^{-1}$)",
                  "ratio": r"$\log_{10}\,\Gamma_\mathrm{BBR}/\Gamma_\mathrm{rad}$"}[qty]
    p.xlabel, p.ylabel = "$T$ (K)", f"$n$ (${lc}$" + (f"$_{{{jw}}}$" if jw else "") + " levels)"
    p.ylabel = p.ylabel.replace("$$", "")
    notes = [f"{species} {lc}{jw} levels n = {ns[0]}-{ns[-1]}: bound-bound blackbody transfer on the Lifetimes "
             "network (Blackbody stage physics), no blackbody ionization."]
    if qty == "ratio":
        notes.append("White line: Γ_BBR = Γ_rad; above it the blackbody empties the level faster than it decays.")
    rows = [[by_n[n_], t_, z_] for n_, zr in zip(ns, Z) for t_, z_ in zip(T, zr)]
    head = ["state", "T_K", {"tau": "tau_eff_ns", "gamma": "Gamma_BBR_s", "ratio": "Gamma_BBR_over_rad"}[qty]]
    return Result([p], notes, aspect=0.75, table=(head, rows))


def blackbody(params: dict, species: str, data_dir: str, style: dict) -> Result:
    if params.get("view") == "map":
        return blackbody_map(params, species, data_dir, style)
    d = _need(catalog.load_json(species, "blackbody", data_dir), species, "blackbody")
    view = params.get("view", "shifts")
    want_l = [c for c in params.get("l", ["s", "p", "d"]) if c in L_CHARS]
    p = Panel()
    rows: list[list] = []
    T = (d.get("rates") or {}).get("T_K") or d.get("T_K")
    notes = []
    if view == "shifts":
        rs = d.get("rydberg_shifts") or {}
        st = rs.get("states") or []
        if not st:
            raise TemplateError("No Rydberg shifts in the blackbody output: run Blackbody with Rydberg shifts on.")
        T = rs.get("T_K", T)
        for i, lc in enumerate(want_l):
            for k, j2 in enumerate((None, 1, 3, 5)):
                sel = [r for r in st if r.get("l") == lc and (j2 is None and "/" not in r["state"]
                                                              or j2 is not None and r["state"].endswith(f"{j2}/2"))]
                if not sel:
                    continue
                sel.sort(key=lambda r: r["n"])
                n = [r["n"] for r in sel]
                y = [r.get("shift_dyn_hz") if r.get("dyn_reliable", True) else math.nan for r in sel]
                lab = f"${lc}$" if j2 is None else f"${lc}_{{{j2}/2}}$"
                p.add("plot", n, y, linestyle="-" if k < 2 else "--", marker="os^D"[k % 4], color=f"C{i}",
                      markerfacecolor=f"C{i}" if k < 2 else "none", label=lab)
                rows += [[r["state"], r["n"], r.get("shift_dyn_hz"), r.get("shift_stat_hz")] for r in sel]
        p.xlabel = "$n$"
        p.ylabel = f"BBR shift at {T:g} K (Hz)"
        p.yscale = "symlog"
        p.linthresh = float(params.get("linthresh") or 100.0)
        p.add("axhline", 0.0, color="0.5", linewidth=0.5)
        head = ["state", "n", "shift_dyn_hz", "shift_stat_hz"]
        notes.append("Dynamic (frequency-dependent) BBR shift of each Rydberg level; points the stage marks "
                     "unreliable are left out.")
    elif view == "rates":
        st = (d.get("rates") or {}).get("states") or []
        if not st:
            raise TemplateError("No BBR rates in the blackbody output.")
        lc = want_l[0] if want_l else "s"
        sel = sorted([r for r in st if r.get("l") == lc and (lc == "s" or r["state"].endswith(f"{2 * L_CHARS.index(lc) + 1}/2")
                                                             or "/" not in r["state"])], key=lambda r: r["n"])
        if not sel:
            raise TemplateError(f"No {lc} states in the blackbody rates.")
        n = np.array([r["n"] for r in sel], dtype=float)
        for i, (key, lab) in enumerate((("Gamma_rad_s", r"$\Gamma_\mathrm{rad}$"),
                                        ("Gamma_BBR_s", r"$\Gamma_\mathrm{BBR}$ (bound)"),
                                        ("Gamma_ion_s", r"$\Gamma_\mathrm{ion}$ (photo- and field ionization)"))):
            y = np.array([r.get(key) if r.get(key) else math.nan for r in sel], dtype=float)
            if np.isfinite(y).any():
                p.add("plot", n, y, marker="os^"[i], linestyle="-", color=f"C{i}", label=lab)
        p.xscale = "log"
        p.yscale = "log"
        p.xlabel = "$n$"
        p.ylabel = r"rate (s$^{-1}$)"
        rows = [[r["state"], r["n"], r.get("Gamma_rad_s"), r.get("Gamma_BBR_s"), r.get("Gamma_ion_s")] for r in sel]
        head = ["state", "n", "Gamma_rad_s", "Gamma_BBR_s", "Gamma_ion_s"]
        notes.append(f"{lc} states at {T:g} K (Blackbody stage).")
    elif view == "temperature":
        ts = (d.get("rates") or {}).get("T_sweep") or {}
        TK = ts.get("T_K") or []
        states = ts.get("states") or {}
        if not TK or not states:
            raise TemplateError("No temperature sweep in the blackbody output.")
        pick = [s_ for s_ in (params.get("states") or list(states)) if s_ in states]

        pick.sort(key=sweep_order)
        gam_rad = {r["state"]: r.get("Gamma_rad_s") for r in (d.get("rates") or {}).get("states", [])}
        qty = params.get("quantity", "tau")
        as_tau = qty == "tau"
        TK_arr = np.asarray(TK, dtype=float)
        for i, sname in enumerate(pick):
            g = np.asarray(states[sname], dtype=float)
            if qty in ("tau", "ratio"):
                gr = gam_rad.get(sname)
                if not gr:
                    continue
                y = 1e9 / (gr + g) if as_tau else g / gr
            else:
                y = g
            n_col = len(style_palette(style))
            p.add("plot", TK_arr, y, color=f"C{i % n_col}", linestyle="-" if i < n_col else "--",
                  label=_pretty(sname))
            rows += [[sname, t_, v] for t_, v in zip(TK_arr, y)]
        if not p.ops:
            raise TemplateError("No radiative rates for the swept states.")
        p.xlabel = "$T$ (K)"
        p.ylabel = {"tau": r"$\tau_\mathrm{eff}$ (ns)", "ratio": r"$\Gamma_\mathrm{BBR}/\Gamma_\mathrm{rad}$"}.get(
            qty, r"$\Gamma_\mathrm{BBR}$ (s$^{-1}$)")
        p.yscale = "log"
        head = ["state", "T_K", {"tau": "tau_eff_ns", "ratio": "Gamma_BBR_over_rad"}.get(qty, "Gamma_BBR_s")]
        p.legend = {"loc": "best", "ncol": 2 if len(pick) > 5 else 1, "frame": False}
        notes.append("τ_eff = 1/(Γ_rad + Γ_BBR(T)), Γ_rad and the Γ_BBR temperature sweep from the Blackbody stage."
                     if as_tau else "Blackbody transfer rate Γ_BBR(T) (Blackbody stage temperature sweep).")
    else:
        raise TemplateError(f"Unknown view {view}.")
    if p.legend is None:
        p.legend = {"loc": "best", "ncol": 1, "frame": False}
    return Result([p], notes, aspect=0.75, table=(head, rows))


def style_palette(style: dict) -> list[str]:
    from .style import palette

    return palette(style.get("palette", "okabe-ito"))


def sweep_order(name: str) -> tuple[int, int, str]:
    """Sort key of a level label: n, then l."""
    m = _NL.match(name)
    return (int(m.group(1)), L_CHARS.index(m.group(2)), name) if m else (999, 9, name)


# ------------------------------------------------------------------------------------------- Breit–Rabi
def hyperfine_states(species: str, data_dir: str) -> list[str]:
    d = catalog.load_json(species, "hyperfine", data_dir) or {}
    return [s_["label"] for s_ in d.get("states", []) if s_.get("A_MHz") is not None]


def breit_rabi(params: dict, species: str, data_dir: str, style: dict) -> Result:
    from qali.physics.hyperfine_core import breit_rabi_energies

    d = _need(catalog.load_json(species, "hyperfine", data_dir), species, "hyperfine")
    states = {s_["label"]: s_ for s_ in d.get("states", []) if s_.get("A_MHz") is not None}
    if not states:
        raise TemplateError(f"No hyperfine constants in {species}'s hyperfine output.")
    label = params.get("state") or next(iter(states))
    if label not in states:
        raise TemplateError(f"No state {label} in {species}'s hyperfine output.")
    st = states[label]
    I = float(d["I"])
    J = float(st["J"])
    Bmax = float(params.get("B_max") or 0.0)
    A, Bq, C = float(st["A_MHz"]), float(st.get("B_MHz") or 0.0), float(st.get("C_kHz") or 0.0)
    gJ, gI = float(st["gJ"]), float(d.get("gI") or 0.0)
    if Bmax <= 0:             # to where the electron spin decouples: μB B ≈ the hyperfine splitting
        split = abs(A) * (I + 0.5) * (2 * J + 1) / 2 + abs(Bq)
        Bmax = float(f"{2.5 * split / (1.39962449 * abs(gJ)):.1g}")
    Bs = np.linspace(0.0, Bmax, 400)
    E, labels = breit_rabi_energies(I, J, A, Bq, gJ, gI, Bs, C_kHz=C)
    unit = params.get("unit") or ("GHz" if abs(A) * (I + 0.5) > 500 else "MHz")
    k = 1e-3 if unit == "GHz" else 1.0
    Fs = sorted({lab[0] for lab in labels})
    p = Panel()
    rows = []
    for col, (F, mF) in enumerate(labels):
        i = Fs.index(F)
        kw = {"label": f"$F={F:g}$"} if mF == -F else {}
        p.add("plot", Bs, E[:, col] * k, color=f"C{i}", linewidth=0.9, **kw)
        rows += [[f"F={F:g},mF={mF:g}", b, e] for b, e in zip(Bs[::8], E[::8, col])]
    p.xlabel = "$B$ (G)"
    p.ylabel = f"$E/h$ ({unit})"
    p.legend = {"loc": "best", "ncol": 1, "frame": False}
    notes = [f"{d.get('isotope_A', '')}{species} {label}: I = {I:g}, A = {A:.6g} MHz"
             + (f", B = {Bq:.4g} MHz" if Bq else "") + f", gJ = {gJ:.6g} (Hyperfine stage); exact diagonalization."]
    return Result([p], notes, aspect=0.75, table=(["level", "B_G", "E_MHz"], rows))


# ----------------------------------------------------------------------------------------- radial functions
def default_radial_states(lv: list[Level]) -> list[str]:
    """The ground state and the lowest p and d levels (upper J)."""
    out = [lv[0].label]
    for l in (1, 2):
        cand = sorted((x for x in lv if x.l == l), key=lambda x: (x.binding_eV, -(x.J or 0)))
        if cand:
            low = [x for x in cand if x.n == cand[0].n]
            out.append(max(low, key=lambda x: x.J or 0).label)
    return out


def radial(params: dict, species: str, data_dir: str, style: dict) -> Result:
    from qali.core.species import resolve_species
    from qali.physics.orbital_radial import resample, state_radial

    lv, IE, Ry = levels(species, data_dir)
    by = {x.label: x for x in lv}
    want = [s_.strip() for s_ in (params.get("states") or default_radial_states(lv)) if s_.strip()]
    missing = [s_ for s_ in want if s_ not in by]
    if missing:
        raise TemplateError("No level " + ", ".join(missing) + f" for {species} (Rydberg stage).")
    sp = resolve_species(species)
    z_core = float(sp.get("charge", 0)) + 1.0
    qty = params.get("quantity", "u")
    p = Panel()
    rows = []
    notes = []
    for i, lab in enumerate(want):
        x = by[lab]
        j = x.J if x.J is not None else x.l + 0.5
        d = state_radial(sp["symbol"], int(sp["Z"]), int(sp.get("charge", 0)), z_core, x.n, x.l, j,
                         n_star(x, Ry), "model")
        xs, u = resample(d["r"], d["u"], 1500)      # on the log grid x = ln r
        r = np.exp(xs)
        y = u ** 2 if qty == "density" else u
        p.add("plot", r, y, color=f"C{i}", label=x.pretty)
        if params.get("core", True) and d.get("rc") and i == 0:
            p.add("axvspan", float(r[0]) if params.get("xscale", "log") == "log" else 0.0, float(d["rc"]),
                  color="0.9", linewidth=0)
        notes.append(f"{lab}: {d['method']} function, ⟨r⟩ = {d['r1']:.4g} a0, {len(d['nodes'])} node{'' if len(d['nodes']) == 1 else 's'}"
                     + (f", {100 * d['p_core']:.3g} % inside the core" if d.get("p_core") else "") + ".")
        rows += [[lab, rr, uu] for rr, uu in zip(r[::5], u[::5])]
    p.add("axhline", 0.0, color="0.5", linewidth=0.5)
    p.xscale = params.get("xscale", "log")
    p.xlabel = "$r$ ($a_0$)"
    p.ylabel = r"$u(r) = rR(r)$ ($a_0^{-1/2}$)" if qty == "u" else r"$r^2|R(r)|^2$ ($a_0^{-1}$)"
    if p.xscale == "log":
        p.xlim = [float(params.get("r_min") or 1e-2), None]
    p.legend = {"loc": "best", "ncol": 1, "frame": False}
    if params.get("core", True):
        notes.append("Shaded: the ionic core (99 % of the core electrons).")
    return Result([p], notes, aspect=0.75, table=(["state", "r_a0", "u"], rows))


# ---------------------------------------------------------------------------------------------- spectra
SPECTRUM_VIEWS = {"absorption": "Absorption", "emission": "Emission", "dlines": "D lines (hyperfine)"}
SPECTRUM_QTY = {"sigma": "Cross-section per atom", "od": "Optical depth", "transmission": "Transmission"}


def _spectrum_payload(species: str, data_dir: str, T: float | None, L: float):
    """Levels, lines and D-line components exactly as the Spectrum page builds them."""
    from qali.core.species import resolve_species
    from qali.pipelines import spectra as SP

    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    sp = resolve_species(species)
    args = SP.build_parser().parse_args([species])
    args.T, args.cell_length = T, L
    return sp, SP.build_payload(sp["species_id"], sp, lt, data_dir, args)


def _dense_nu_grid(nu_lo: float, nu_hi: float, centres: list[float], widths: list[float], n: int = 3000):
    """A frequency grid with extra points across every line (lines are pm wide over a nm-wide view)."""
    parts = [np.linspace(nu_lo, nu_hi, n)]
    for c, w in zip(centres, widths):
        if nu_lo <= c <= nu_hi:
            parts.append(c + w * np.sinh(np.linspace(-4.0, 4.0, 81)))      # dense at the centre, out to ~27 widths
    nu = np.unique(np.concatenate(parts))
    return nu[(nu >= nu_lo) & (nu <= nu_hi)]


def spectrum(params: dict, species: str, data_dir: str, style: dict) -> Result:
    from qali.core.constants import C_SI
    from qali.physics import spectrum_core as sc

    view = params.get("view", "absorption")
    L = float(params.get("cell_cm") or 5.0)
    sp, pay = _spectrum_payload(species, data_dir, params.get("T_K"), L)
    T = float(pay["opts"]["T"])
    mass = float(pay["mass"])
    inst_nm = float(params.get("instrument_pm") or 0.0) * 1e-3
    qty = params.get("quantity", "sigma" if view != "dlines" else "transmission")
    N_cm3 = sc.vapour_density_cm3(sp["symbol"], T) if not pay["is_ion"] else None
    if qty != "sigma" and not N_cm3:
        if params.get("quantity"):
            raise TemplateError("No vapour density for this species: plot the cross-section instead.")
        qty = "sigma"                       # no vapour data (Fr, ions): the cross-section per atom
    air = params.get("air", False)
    notes = []
    p = Panel()
    rows: list[list] = []
    if view == "dlines":
        which = params.get("line", "D2")
        dl = next((d for d in pay["dlines"] if d["name"] == which), None)
        if dl is None:
            raise TemplateError(f"No {which} line with hyperfine constants for {species} ({pay['dl_note'] or 'run Hyperfine'}).")
        nu0 = C_SI / (dl["lam"] * 1e-9)
        gam = sc.natural_fwhm_hz(dl["tau"]) / 2.0
        span = float(params.get("span_GHz") or 0.0) * 1e9
        comps = [(iso, c) for iso in dl["iso"] for c in iso["comps"]]
        dets = [c[2] * 1e6 for _iso, c in comps]
        sig_d = {iso["A"]: math.hypot(sc.doppler_sigma_hz(nu0, iso["m"], T), sc.instrument_sigma_hz(nu0, inst_nm)
                                      if inst_nm else 0.0) for iso in dl["iso"]}
        smax = max(sig_d.values())
        if not span:
            span = (max(dets) - min(dets)) / 2 + 6 * smax
        mid = (max(dets) + min(dets)) / 2
        d = _dense_nu_grid(mid - span, mid + span, dets, [max(sig_d[iso["A"]], gam) for iso, _c in comps])
        sig = np.zeros_like(d)
        for iso, c in comps:
            sig += sc.LINE_K * dl["f"] * c[3] * sc.voigt(nu0 + d, nu0 + c[2] * 1e6, sig_d[iso["A"]], gam)
        y = _spectrum_quantity(sig, qty, N_cm3, L)
        p.add("plot", d / 1e9, y, color="C0", linewidth=style.get("line_width", 1.0))
        if params.get("sticks", True):
            top = (np.nanmax(y) if qty != "transmission" else 1.0)
            wmax = max(c[3] for _iso, c in comps)
            for i, iso in enumerate(dl["iso"]):
                xs = [c[2] / 1e3 for c in iso["comps"]]
                hs = [0.12 * top * c[3] / wmax for c in iso["comps"]]
                base = 0.0 if qty != "transmission" else 1.0
                p.add("vlines", xs, [base] * len(xs), [base - h if qty == "transmission" else base + h for h in hs],
                      colors=f"C{i + 1}", linewidth=0.8,
                      label=f"$^{{{iso['A']}}}${sp['symbol']} ({100 * iso['ab']:.3g} %)")
            p.legend = {"loc": "best", "ncol": 1, "frame": False}
        p.xlabel = f"detuning from the {which} line (GHz)"
        rows = [[float(a / 1e9), float(b)] for a, b in zip(d[::4], y[::4])]
        notes.append(f"{species} {which} {dl['lower']} → {dl['upper']}, λ = {dl['lam']:.4f} nm: hyperfine components "
                     "of every isotope with the natural abundances, Doppler-broadened at "
                     f"{T:g} K (Spectrum stage physics).")
        head = ["detuning_GHz", qty]
    else:
        levels, lines = pay["levels"], pay["lines"]
        E = [lv[2] for lv in levels]
        g = [lv[3] for lv in levels]
        lo_wl, hi_wl = pay["opts"]["wl"]
        lo_wl = float(params.get("wl_min") or lo_wl)
        hi_wl = float(params.get("wl_max") or hi_wl)
        nu_lo, nu_hi = C_SI / (hi_wl * 1e-9), C_SI / (lo_wl * 1e-9)
        if view == "absorption":
            pop = sc.boltzmann_weights(E, g, T)
            amp = [pop[ln[0]] * ln[3] for ln in lines]
        else:
            texc = params.get("T_exc") or 5000.0
            pop = sc.boltzmann_weights(E, g, float(texc) if texc != "inf" else math.inf)
            amp = [pop[ln[1]] * ln[4] * (C_SI / (ln[2] * 1e-9)) for ln in lines]
        top = max(a for a, ln in zip(amp, lines) if lo_wl <= ln[2] <= hi_wl) if any(lo_wl <= ln[2] <= hi_wl for ln in lines) else 0
        if not top:
            raise TemplateError(f"No lines between {lo_wl:g} and {hi_wl:g} nm.")
        keep = [(a, ln) for a, ln in zip(amp, lines) if lo_wl * 0.98 <= ln[2] <= hi_wl * 1.02 and a > 1e-7 * top]
        cen, wid, pars = [], [], []
        for a, ln in keep:
            nu0 = C_SI / (ln[2] * 1e-9)
            sd = math.hypot(sc.doppler_sigma_hz(nu0, mass, T), sc.instrument_sigma_hz(nu0, inst_nm) if inst_nm else 0.0)
            ga = sc.natural_fwhm_hz(levels[ln[1]][4], levels[ln[0]][4]) / 2.0
            cen.append(nu0)
            wid.append(max(sd, ga))
            pars.append((a, nu0, sd, ga, ln))
        nu = _dense_nu_grid(nu_lo, nu_hi, cen, wid, n=4000)
        y = np.zeros_like(nu)
        for a, nu0, sd, ga, ln in pars:
            m = np.abs(nu - nu0) < 2000 * max(sd, ga)          # far wings add nothing visible
            if view == "absorption":
                y[m] += sc.LINE_K * a * sc.stimulated_factor(nu0, T) * sc.voigt(nu[m], nu0, sd, ga)
            else:
                y[m] += a * sc.voigt(nu[m], nu0, sd, ga)
        lam = C_SI / nu * 1e9
        if air:
            lam = np.array([sc.vac_to_air_nm(v) for v in lam])
        if view == "absorption":
            y = _spectrum_quantity(y, qty, N_cm3, L)
            notes.append(f"{species} at {T:g} K: Boltzmann populations, Voigt lines (Doppler, natural"
                         + (f", {inst_nm * 1e3:g} pm instrument" if inst_nm else "") + "), Spectrum stage physics.")
            head = ["wavelength_nm", qty]
        else:
            y = y / np.nanmax(y)
            p.ylabel = "relative intensity"
            notes.append(f"{species}: optically thin emission, upper levels at T_exc = {params.get('T_exc') or 5000} K"
                         + (f", {inst_nm * 1e3:g} pm instrument" if inst_nm else "") + ".")
            head = ["wavelength_nm", "relative_intensity"]
        o = np.argsort(lam)
        p.add("plot", lam[o], y[o], color="C0", linewidth=style.get("line_width", 1.0))
        rows = [[float(a), float(b)] for a, b in zip(lam[o][::3], y[o][::3])]
        nlab = int(params.get("labels", 6))
        if nlab:
            placed: list[float] = []
            for a, ln in sorted(keep, key=lambda t: -t[0]):
                x0 = sc.vac_to_air_nm(ln[2]) if air else ln[2]
                if len(placed) >= nlab or not lo_wl <= ln[2] <= hi_wl:
                    continue
                if any(abs(x0 - q) < 0.025 * (hi_wl - lo_wl) for q in placed):
                    continue                     # a close, stronger line is labelled already
                placed.append(x0)
                p.add("annotate", f"{_pretty(levels[ln[0]][0]).strip('$')}–{_pretty(levels[ln[1]][0]).strip('$')}"
                      .join(["$", "$"]).replace("–", "$-$"),
                      [x0, 1.0], xycoords=("data", "axes fraction"), xytext=[0, -2], textcoords="offset points",
                      ha="center", va="top", rotation=90, fontsize=style.get("font_size", 8) * 0.6, color="0.3")
        p.xlabel = "wavelength in air (nm)" if air else "wavelength (nm)"
        p.xlim = [lo_wl, hi_wl] if not air else [sc.vac_to_air_nm(lo_wl), sc.vac_to_air_nm(hi_wl)]
    if view != "emission":
        p.ylabel = {"sigma": r"$\sigma$ (cm$^2$)", "od": "optical depth",
                    "transmission": "transmission"}[qty]
        if qty == "sigma" or (qty == "od" and params.get("log", True)):
            p.yscale = "log"
            mx = float(np.nanmax([r[1] for r in rows])) if rows else 1.0
            p.ylim = [mx * 1e-6, mx * 3]
        if qty != "sigma":
            notes.append(f"Cell {L:g} cm, saturated vapour {N_cm3:.3g} atoms/cm³ at {T:g} K.")
    return Result([p], notes, aspect=0.55, table=(head, rows))


def _spectrum_quantity(sig_m2, qty: str, N_cm3, L_cm: float):
    sig_cm2 = np.asarray(sig_m2) * 1e4
    if qty == "sigma":
        return sig_cm2
    od = sig_cm2 * N_cm3 * L_cm
    return od if qty == "od" else np.exp(-od)


# ---------------------------------------------------------------------------------------- trap depth
def trap(params: dict, species: str, data_dir: str, style: dict) -> Result:
    from qali.core.constants import ATOMIC_MASS_AMU
    from qali.physics.tweezer_core import attach_gamma, compute_spectra, strongest_resonances

    pol = _need(catalog.load_json(species, "polarizability", data_dir), species, "polarizability")
    lt = _need(catalog.load_json(species, "lifetimes", data_dir), species, "lifetimes")
    rates = {r["state"]: float(r["A_total_s"]) for r in lt.get("lifetimes", []) if r.get("A_total_s")}
    ch, _n, _c = attach_gamma(pol.get("transitions_gs") or [], rates)
    if not ch:
        raise TemplateError(f"No ground-state transitions in {species}'s polarizability output.")
    I = float(params.get("I_kWcm2") or 50.0)
    lo, hi = float(params.get("wl_min") or 400.0), float(params.get("wl_max") or 1600.0)
    sym = species.split("_")[0]
    mass = float(ATOMIC_MASS_AMU.get(sym, 1.0))
    sp = compute_spectra(ch, lo, hi, 1200, I, mass)
    wl = np.asarray(sp.wl_nm)
    U = np.asarray(sp.U_uK, dtype=float)
    R = np.asarray(sp.R_sc_Hz, dtype=float)
    a = Panel(ylabel=r"$U$ ($\mu$K)", yscale="symlog", linthresh=float(params.get("linthresh") or 10.0))
    a.add("plot", wl, U, color="C0")
    a.add("axhline", 0.0, color="0.5", linewidth=0.5)
    b = Panel(ylabel=r"$R_\mathrm{sc}$ (s$^{-1}$)", yscale="log", height_ratio=0.7)
    b.add("plot", wl, R, color="C1")
    if params.get("resonances", True):
        for r_ in strongest_resonances(ch, 8):
            if lo <= r_.wl_nm <= hi:
                for q in (a, b):
                    q.add("axvline", r_.wl_nm, color="0.75", linestyle=":", linewidth=0.6)
    b.xlabel = "wavelength (nm)"
    a.xlim = [lo, hi]
    finite = R[np.isfinite(R) & (R > 0)]
    if finite.size:
        b.ylim = [float(np.percentile(finite, 1)) / 2, float(np.percentile(finite, 99)) * 5]
    ymax = float(params.get("U_max") or 3 * np.nanpercentile(np.abs(U), 90))
    a.ylim = [-ymax, ymax]
    notes = [f"{species} ground state {pol.get('gs_label', '')}, I = {I:g} kW/cm²: U = −Re α I / (2 ε0 c), "
             "R_sc from Im α with each line's natural width (Tweezer stage physics); dotted: strongest lines."]
    rows = [[float(x), float(u), float(r)] for x, u, r in zip(wl[::3], U[::3], R[::3])]
    return Result([a, b], notes, aspect=0.85, sharex=True, table=(["wavelength_nm", "U_uK", "R_sc_s-1"], rows))


def inputs_of(template: str, params: dict) -> list[str]:
    """The stage outputs of the figure's species that *template* reads, in the order shown."""
    if template == "grotrian":
        return ["rydberg", "lifetimes"]
    if template in ("quantum_defects", "radial"):
        return ["rydberg"]
    if template == "scaling":
        return ["rydberg", SCALING.get(params.get("quantity", "lifetime"), SCALING["lifetime"])[1]]
    if template == "theory_vs_expt":
        return ["polarizability", "lifetimes"]
    if template == "alpha":
        return ["polarizability", "lifetimes"] if params.get("magic_show") == "practical" else ["polarizability"]
    if template in ("contributions", "oscillator_density"):
        return ["polarizability"]
    if template == "branching":
        return ["lifetimes"]
    if template == "cascade":
        return ["rydberg", "lifetimes"]
    if template == "spectrum":
        return ["lifetimes"]
    if template == "trap":
        return ["polarizability", "lifetimes"]
    if template == "feshbach":
        return ["feshbach_out"]
    if template == "blackbody":
        return ["lifetimes"] if params.get("view") == "map" else ["blackbody"]
    if template == "breit_rabi":
        return ["hyperfine"]
    return []                                   # the builder picks a table, named runs included, per series


TEMPLATES = {
    "builder": ("Plot builder", builder),
    "grotrian": ("Grotrian diagram", grotrian),
    "quantum_defects": ("Quantum defects", quantum_defects),
    "scaling": ("Scaling laws (vs n*)", scaling_laws),
    "theory_vs_expt": ("Theory vs experiment", theory_vs_experiment),
    "alpha": ("Polarizability α(λ)", polarizability_spectrum),
    "contributions": ("α₀ and C₆: lines, convergence, uncertainty", contributions),
    "branching": ("Decay branching ratios", branching),
    "cascade": ("Decay cascade", cascade),
    "oscillator_density": ("df/dε across the ionization limit", oscillator_density),
    "blackbody": ("Blackbody: shifts, rates, temperature", blackbody),
    "breit_rabi": ("Breit–Rabi (Zeeman) diagram", breit_rabi),
    "feshbach": ("Feshbach scattering length a(B)", feshbach),
    "radial": ("Radial wavefunctions", radial),
    "spectrum": ("Spectrum: absorption, emission, D lines", spectrum),
    "trap": ("Trap depth and scattering vs λ", trap),
}
