"""Level energies from the DHF and Brueckner stages, for the series NIST has no level of.

The Rydberg stage builds each series from its measured levels: the levels themselves where NIST lists them,
and a quantum-defect fit to them above. A series with no measured level at all (Fr nf) gets no fit, so its
quantum defect falls back to 0 (hydrogenic) and its lowest members can be missing entirely (Fr 5f, below the
first n the stage generates). There a calculation is the better value:

* the model's levels of that series replace the hydrogenic ones, J by J, up to the model's highest n;
* above that, the series continues with a Ritz fit δ(n) = δ0 + δ2/(n − δ0)² to the model's levels
  (J-averaged, weights 2J+1), the same law the stage fits to measured levels.

Series with at least one measured level keep the quantum-defect fit to those: one measured level of a series
and the Ritz law are more accurate than a calculation (Li 4f: the Brueckner level is off by 2.4 cm⁻¹, the fit
to 4f and 5f by far less). Model levels are tagged with the model's name as their source and marked
theoretical; each carries an uncertainty from the model's own misfit to measured levels the model did not fit
to (same l when there is one, else any l), scaled with the level density as (n*_ref/n*)³.
"""
from __future__ import annotations

import json
import math
import os
import re

MODELS = ("dhf", "brueckner")
MODEL_NAMES = {"dhf": "DHF", "brueckner": "Brueckner"}
L_CHARS = "spdfghik"
CM_TO_EV = 1.0 / 8065.543937


class ModelLevelsError(ValueError):
    """The model output is missing or does not fit the species."""


def load_model(element: str, model: str, data_dir: str, path: str | None = None) -> dict:
    """The DHF or Brueckner stage output of *element* (or the file at *path*)."""
    if model not in MODELS:
        raise ModelLevelsError(f"unknown model '{model}' (choose {', '.join(MODELS)})")
    path = path or os.path.join(data_dir, f"{element}_{model}.json")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except OSError:
        raise ModelLevelsError(
            f"{os.path.basename(path)} not found: run the {MODEL_NAMES[model]} stage for {element} first") from None
    if not doc.get("levels"):
        raise ModelLevelsError(f"{os.path.basename(path)} has no levels")
    return doc


def _j_str(j: float) -> str:
    return f"{int(round(2 * j))}/2"


def model_levels(doc: dict) -> list[dict]:
    """[{n, l (char), J ('5/2'), binding_eV, label}] of a model output, ground state included."""
    out = []
    for x in doc["levels"]:
        try:
            l_char = L_CHARS[int(x["l"])]
            out.append(dict(n=int(x["n"]), l=l_char, J=_j_str(float(x["j"])), binding_eV=float(x["binding_eV"]),
                            label=x.get("label")))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return out


def _n_star(binding_eV: float, ry_eff: float) -> float:
    return math.sqrt(ry_eff / binding_eV)


def misfit_scale(doc: dict, ry_eff: float, ionization_cm: float | None = None) -> dict[str, float]:
    """Per l (and '*' for any l): max |E_model − E_NIST| (cm⁻¹) · n*³ over the levels the model was not fitted
    to; σ(n*) = this / n*³. Empty when the model was compared with nothing it did not fit."""
    ion = ionization_cm or doc.get("ionization_cm")
    out: dict[str, float] = {}
    if not ion:
        return out
    for c in doc.get("nist_comparison") or []:
        if c.get("fitted"):
            continue
        m = re.match(r"^(\d+)([a-z])", c.get("state", ""))
        diff = c.get("diff_cm")
        nist = c.get("nist_cm")
        if not m or diff is None or nist is None or nist >= ion:
            continue
        ns = _n_star((ion - nist) * CM_TO_EV, ry_eff)
        v = abs(float(diff)) * ns ** 3
        for key in (m.group(2), "*"):
            out[key] = max(out.get(key, 0.0), v)
    return out


def ritz_fit(levels: list[dict], ry_eff: float) -> dict | None:
    """δ(n) = δ0 + δ2/(n − δ0)² fitted to one series' model levels (J-averaged by 2J+1), lowest n left out when
    there are 4 or more; None with fewer than 2 points."""
    by_n: dict[int, list[tuple[float, float]]] = {}
    for x in levels:
        g = int(x["J"].split("/")[0]) + 1
        by_n.setdefault(x["n"], []).append((x["n"] - _n_star(x["binding_eV"], ry_eff), g))
    pts = sorted((n, sum(d * g for d, g in v) / sum(g for _, g in v)) for n, v in by_n.items())
    if len(pts) >= 4:
        pts = pts[1:]
    if len(pts) < 2:
        return None
    d0 = pts[-1][1]
    for _ in range(20):                               # δ0, δ2 by linear least squares in 1/(n − δ0)²
        xs = [1.0 / (n - d0) ** 2 for n, _ in pts]
        ys = [d for _, d in pts]
        k = len(xs)
        mx, my = sum(xs) / k, sum(ys) / k
        sxx = sum((x - mx) ** 2 for x in xs)
        d2 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        new_d0 = my - d2 * mx
        if abs(new_d0 - d0) < 1e-12:
            break
        d0 = new_d0
    rms = math.sqrt(sum((d0 + d2 / (n - d0) ** 2 - d) ** 2 for n, d in pts) / len(pts))
    return dict(delta0=d0, delta2=d2, n_min=pts[0][0], n_max=pts[-1][0], n_points=len(pts), rms=rms)


def unmeasured_series(doc: dict, measured_l: set[str]) -> list[str]:
    """The orbitals the model covers and NIST lists no level of."""
    return sorted({x["l"] for x in model_levels(doc)} - set(measured_l), key=L_CHARS.index)


def apply_model_levels(states: list[dict], doc: dict, model: str, series: list[str], ionization_energy: float,
                       ry_eff: float, n_max: int) -> dict:
    """Put the model's levels of *series* into *states* (the Rydberg stage's excitations), in place.

    Replaces the quantum-defect levels of those series up to the model's highest n with the model's J levels,
    adds the members the stage did not generate (below its first n), and leaves the higher members to the
    caller's quantum-defect series (built with the Ritz fits returned here). Returns a summary for the JSON."""
    name = MODEL_NAMES[model]
    lev = [x for x in model_levels(doc) if x["l"] in series and x["n"] <= n_max]
    scale = misfit_scale(doc, ry_eff, (doc.get("ionization_cm")))
    top = {l_: max((x["n"] for x in lev if x["l"] == l_), default=0) for l_ in series}
    keep = [s for s in states if not (s["l"] in series and s["n"] <= top[s["l"]] and s.get("source") == "QDT model")]
    E_ground = -ionization_energy
    added = []
    for x in lev:
        E = -x["binding_eV"]
        if E - E_ground <= 0:
            continue
        ns = _n_star(x["binding_eV"], ry_eff)
        ref = scale.get(x["l"], scale.get("*"))
        sigma = round(ref / ns ** 3 * CM_TO_EV, 7) if ref else None
        label = f"{x['n']}{x['l']}{x['J']}"
        added.append({
            "n": x["n"], "l": x["l"], "J": x["J"], "term": None,
            "energy_eV": round(E, 6), "wavelength_nm": round(1239.84 / (E - E_ground), 4),
            "label": label, "label_pretty": label, "source": name, "method": f"{name} (model level)",
            "theoretical": True, "uncertainty_eV": sigma,
        })
    merged = keep + added
    merged.sort(key=lambda s: (s["n"], L_CHARS.index(s["l"]) if s["l"] in L_CHARS else 99,
                               int(s["J"].split("/")[0]) if s.get("J") and "/" in str(s["J"]) else 0))
    states[:] = merged
    for i, s in enumerate(states, 1):
        s["state_number"] = i
    return dict(model=model, name=name, series=series, n_max={l_: top[l_] for l_ in series},
                levels=len(added), labels=[s["label"] for s in added],
                uncertainty_note=("max |model − NIST| · n*³ over the levels the model did not fit, same l "
                                  "where there are some, else any l; σ = that / n*³"),
                misfit_cm_nstar3={k: round(v, 3) for k, v in scale.items()})
