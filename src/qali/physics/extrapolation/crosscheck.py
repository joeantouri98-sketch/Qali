"""The Extrapolator against the DHF and Brueckner stages: ground ns -> np_J oscillator strengths.

The Extrapolator predicts f(ns -> np_J) beyond NIST from a calibrated model potential with a fitted correction;
DHF + core polarisation and Brueckner (Σ⁽²⁾ + RPA) compute the same lines from first principles. Where they
agree within their uncertainties the extrapolation stands on an independent footing; where they do not, one of
them is off, and the measured lines say which one is closer in that series.

Uncertainties:
* Extrapolator: its 68 % band (half its width in ln f);
* a structure model: its scatter against the measured lines, per J, σ_J = median |ln(f_model / f_measured)| over
  the measured lines above the resonance line whose integral cancels less than 95 % (at least 5 %), widened line by
  line for cancellation: where the
  radial integral is a near-total cancellation of its parts (c = |net| / Σ|parts|; Cs 6s -> 12p1/2: c = 0.008),
  a 1 % error in the parts becomes 1 %/c in the line, so σ = sqrt(σ_J² + (0.01/c)²).
The pull z = ln(f_extrap / f_model) / sqrt(σ_ext² + σ_model²) flags a disagreement when |z| > 2.
"""
from __future__ import annotations

import math
import re

from .params import species_params
from .stage import J_VALUES, ExtrapolationConfig, ground_model, run_for_species

MODEL_NAMES = {"dhf": "DHF", "brueckner": "Brueckner"}
SIGMA_FLOOR = 0.05
_P = re.compile(r"^(\d+)p(1/2|3/2)$")


PART_ERROR = 0.01


def model_f(doc: dict, gs: str) -> dict[tuple[int, str], float]:
    """{(n, '1/2'|'3/2'): f} of gs -> np_J from a DHF or Brueckner output (its transitions list)."""
    return {k: v[0] for k, v in _model_lines(doc, gs).items()}


def _model_lines(doc: dict, gs: str) -> dict[tuple[int, str], tuple[float, float | None]]:
    """{(n, J): (f, cancellation)}."""
    out: dict[tuple[int, str], tuple[float, float | None]] = {}
    for t in doc.get("transitions") or []:
        lo, up = str(t.get("lower", "")), str(t.get("upper", ""))
        m = _P.match(up)
        if not m or not lo.startswith(gs) or t.get("f_absorption") is None:
            continue
        c = t.get("cancellation")
        out[(int(m.group(1)), m.group(2))] = (float(t["f_absorption"]), float(c) if c else None)
    return out


CLEAN = 0.05          # lines with less cancellation than this set a model's own scatter σ_J


def _model_sigma(lines: dict, measured: dict[tuple[int, str], float], nl: int) -> dict[str, float]:
    out = {}
    for jstr in J_VALUES:
        pairs = [(lines[k], f) for k, f in measured.items() if k[1] == jstr and k[0] > nl and k in lines
                 and lines[k][0] > 0 and f > 0]
        clean = [(fc, f) for fc, f in pairs if fc[1] is None or fc[1] >= CLEAN] or pairs
        r = sorted(abs(math.log(fc[0] / f)) for fc, f in clean)
        out[jstr] = max(SIGMA_FLOOR, r[len(r) // 2]) if r else 0.25
    return out


def cross_check(element: str, nist: dict, rydberg: dict, models: dict[str, dict],
                config: ExtrapolationConfig | None = None) -> dict:
    """Rows per (J, n): measured (the Extrapolator's anchors), Extrapolator (fit and band), its uncorrected model ('Off'), and each structure model.

    *models*: {'dhf': doc, 'brueckner': doc} (either may be missing). Returns
    {element, rows, sigma_models, summary, warnings}."""
    nl = species_params(element)["nl"]
    gs = f"{nl}s"
    cfg = config or ExtrapolationConfig(mode="use", n_max=40)
    ext = run_for_species(element, cfg, nist, rydberg)
    gm = ground_model(element, rydberg, nist)
    # the Extrapolator's own anchors: precision values where there are some, and no source it knows is wrong
    # (Fr T6898 splits each multiplet 1:2 and is off by 2-20x)
    # NIST rows from spliced or extrapolated sources (K 10p-13p) are not measurements: shown apart, not used
    anchors = [a for a in ext.get("anchors", []) if not a.get("unreliable") and a.get("f")]
    measured = {(int(a["n"]), a["jstr"]): float(a["f"]) for a in anchors if not a.get("low_weight")}
    nist_extrap = {(int(a["n"]), a["jstr"]): float(a["f"]) for a in anchors if a.get("low_weight")}
    ml = {name: _model_lines(doc, gs) for name, doc in models.items() if doc}
    mf = {name: {k: v[0] for k, v in lines.items()} for name, lines in ml.items()}
    sig = {name: _model_sigma(ml[name], measured, nl) for name in mf}
    n_top = max((k[0] for fm in mf.values() for k in fm), default=nl)
    rows = []
    for jstr, ch in (ext.get("channels") or {}).items():
        by_n = {r["n"]: r for r in ch["rows"]}
        for n in range(nl, n_top + 1):
            r = by_n.get(n)
            f_ext = r["f"] if r else None
            s_ext = (0.5 * math.log(r["hi68"] / r["lo68"]) if r and r["lo68"] > 0 and r["hi68"] > 0 else None)
            row = {"J": jstr, "n": n, "label": f"{n}p{jstr}", "f_measured": measured.get((n, jstr)),
                   "f_nist_extrapolated": nist_extrap.get((n, jstr)),
                   "f_extrap": f_ext, "lo68": r["lo68"] if r else None, "hi68": r["hi68"] if r else None,
                   "f_model_off": gm.f(f"{n}p{jstr}") if gm else None}
            for name, fm in mf.items():
                fv = fm.get((n, jstr))
                c = ml[name].get((n, jstr), (None, None))[1]
                row[f"f_{name}"] = fv
                row[f"cancel_{name}"] = c
                s_mod = math.hypot(sig[name][jstr], PART_ERROR / c) if c else sig[name][jstr]
                row[f"sigma_{name}"] = s_mod
                if fv and f_ext and s_ext is not None and fv > 0 and f_ext > 0:
                    z = math.log(f_ext / fv) / math.hypot(s_ext, s_mod)
                    row[f"ratio_{name}"] = f_ext / fv
                    row[f"z_{name}"] = z
            rows.append(row)
    summary = {}
    for name in mf:
        zs = [(r["label"], r[f"z_{name}"]) for r in rows if r.get(f"z_{name}") is not None and r["f_measured"] is None]
        summary[name] = {"compared": len(zs), "beyond_2sigma": [lab for lab, z in zs if abs(z) > 2],
                         "median_ratio": _median([r[f"ratio_{name}"] for r in rows
                                                  if r.get(f"ratio_{name}") and r["f_measured"] is None]),
                         "sigma_vs_nist": sig[name]}
    return {"element": element, "rows": rows, "summary": summary, "warnings": ext.get("warnings", [])}


def _median(v: list[float]) -> float | None:
    v = sorted(v)
    return v[len(v) // 2] if v else None
