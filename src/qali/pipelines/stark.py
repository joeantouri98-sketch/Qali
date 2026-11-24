"""
The ``stark`` pipeline: Stark map of an alkali Rydberg state.
-----------------------------------------------------------------------------
H0 + F z diagonalised at each field in a |n l j m_j> basis (one m_j), with fine-structure-resolved
energies from precision quantum defects and model-potential radial integrals.  The physics is in
``qali.physics.rydberg_stark``.

Each eigenvalue near the chosen state is kept with its character, |<chosen state|eigenstate>|², so
the map shows where the state's weight goes through the avoided crossings.  Also written: the
state's polarisability for every |m_j| (second order, ΔE = -½ α F²), its scalar and tensor parts,
and the avoided crossings the followed state meets.

AC mode (--ac-frequency): F(t) = F_dc + F_ac cos(2π f t), solved with Shirley's time-independent
Floquet Hamiltonian (Phys. Rev. 138, B979 (1965); the method of ARC's ShirleyMethod).  The map is
then quasienergy against the peak amplitude F_ac, with photon sidebands q = -Q..Q; the
polarisability is α(ω), with AC shift -¼ α(ω) F_ac².  Cost grows as (2Q + 1)³.

Defaults: n - 4 to n + 4, every l (full manifolds), fields from 0 to the Inglis-Teller field
1/(3n*⁵) a.u., energies within ±0.6 manifold spacings.  Rb 50S: ~900 states, about 30 s.

Outputs:
    data/<species>_stark.json
    plots/<species>/<species>_stark.html

Usage:
    qali --pipeline stark Rb --state 50s1/2
    qali --pipeline stark Cs --state 60d5/2 --mj 5/2 --field-max 2
    qali --pipeline stark Rb --state 56d5/2 --ac-frequency 2 --field-max 3 --n-range 6 --l-max 10
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from qali.core.errors import MissingDataError
from qali.core.fileio import write_json
from qali.core.paths import reference_path
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.physics.extrapolation.exact import MARINESCU
from qali.physics.rydberg_stark.defects import L_LETTERS, RydbergEnergies, j_label
from qali.physics.rydberg_stark.stark import (avoided_crossings, build_basis, dynamic_polarizability, floquet_map,
                                              follow, inglis_teller_V_cm, manifold_spacing_ghz, polarizability,
                                              scalar_tensor, StarkMap, stark_map, with_m)
from qali.report_templates import render, write_offline_report

TEMPLATE = "stark_page.html"
SPECIES = ("Li", "Na", "K", "Rb", "Cs", "Fr")
MHZ_KVCM_PER_AU = 2.48832e-4                     # 1 a.u. of polarisability in MHz/(kV/cm)²
_STATE = re.compile(r"^\s*(\d+)\s*([a-z])\s*(\d+/2)?\s*$")


def parse_state(text: str) -> tuple[int, int, int]:
    """'50s1/2', '50s', '60d5/2' -> (n, l, 2j); without j the lower one."""
    m = _STATE.match(text.lower())
    if not m or m.group(2) not in L_LETTERS:
        raise ValueError(f"state {text!r}: give n, l and j, for example 50s1/2 or 60d5/2")
    n, l = int(m.group(1)), L_LETTERS.index(m.group(2))
    if m.group(3):
        j2 = int(m.group(3).split("/")[0])
        if j2 not in (2 * l - 1, 2 * l + 1) or j2 < 1:
            raise ValueError(f"state {text!r}: j must be l ± 1/2")
    else:
        j2 = max(1, 2 * l - 1)
    if l >= n:
        raise ValueError(f"state {text!r}: l must be below n")
    return n, l, j2


def parse_half(text: str) -> int:
    """'1/2' or '-3/2' or '0.5' -> doubled value."""
    t = text.strip()
    v = float(t.split("/")[0]) / float(t.split("/")[1]) if "/" in t else float(t)
    m2 = round(2 * v)
    if abs(2 * v - m2) > 1e-9 or m2 % 2 == 0:
        raise ValueError(f"m_j {text!r} must be a half-integer")
    return m2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qali --pipeline stark",
                                description="Stark map of an alkali Rydberg state (exact diagonalisation)")
    p.add_argument("element", nargs="?", default=None, help="Li, Na, K, Rb, Cs or Fr")
    p.add_argument("--state", default="50s1/2", help="Chosen state, for example 50s1/2, 40p3/2, 60d5/2")
    p.add_argument("--mj", default="1/2", help="m_j of the map, for example 1/2 or -3/2")
    p.add_argument("--field-min", type=float, default=0.0, help="Lowest field (AC: amplitude), V/cm")
    p.add_argument("--field-max", type=float, default=None,
                   help="Highest field (AC: amplitude), V/cm (default: the Inglis-Teller field 1/(3n*^5) a.u.)")
    p.add_argument("--points", type=int, default=300, help="Number of field values")
    p.add_argument("--n-range", type=int, default=4, help="Basis n from n - this to n + this")
    p.add_argument("--l-max", type=int, default=None, help="Highest l in the basis (default: every l)")
    p.add_argument("--window", type=float, default=None,
                   help="Keep energies within ± this many GHz of the state (default: 0.6 manifold spacings; AC: 1.5 f)")
    p.add_argument("--ac-frequency", type=float, default=None,
                   help="AC field frequency, GHz; the map is then against the AC amplitude")
    p.add_argument("--sidebands", type=int, default=2, help="AC only: photon sidebands each side")
    p.add_argument("--dc-field", type=float, default=0.0, help="AC only: static field added, V/cm")
    p.add_argument("--nist-dir", default=None, help="Folder of NIST level exports; default: bundled")
    p.add_argument("--data-dir", default=str(Workspace.current().data_json_dir))
    p.add_argument("--output-dir", default=str(Workspace.current().plots_dir),
                   help="Plots folder; the page goes in <output-dir>/<species>/")
    return p


def energies_for(element: str, nist_dir: str) -> RydbergEnergies:
    from qali.pipelines.dhf import nist_inputs

    ip, nist = nist_inputs(element, nist_dir)
    if not nist:
        raise MissingDataError(f"No NIST levels for {element} in {nist_dir}")
    return RydbergEnergies(element, nist, ip, MARINESCU[element]["alphaC"])


def compute(element: str, n: int, l: int, j2: int, m2: int, *, energies: RydbergEnergies,
            field_min: float = 0.0, field_max: float | None = None, points: int = 300,
            n_range: int = 4, l_max: int | None = None, window: float | None = None,
            ac_frequency: float | None = None, sidebands: int = 2, dc_field: float = 0.0,
            progress=print) -> dict:
    """The stage's whole calculation, as the JSON payload (no files written)."""
    n_lo, n_hi = max(1, n - n_range), n + n_range
    lmax = n_hi - 1 if l_max is None else min(l_max, n_hi - 1)
    if l > lmax:
        raise ValueError(f"--l-max {lmax} is below the state's l = {l}")
    progress(f"  Basis n {n_lo}-{n_hi}, l <= {lmax}, m_j = {m2}/2 ...")
    base = build_basis(energies, element, n, l, j2, 1 if m2 > 0 else -1, n_lo, n_hi, lmax)
    basis = base if abs(m2) == 1 else with_m(base, m2)
    target = basis.levels[basis.target]
    ns = target.n_star
    fmax = inglis_teller_V_cm(ns) if field_max is None else field_max
    if fmax <= field_min:
        raise ValueError("--field-max must be above --field-min")
    W = 0.6 * manifold_spacing_ghz(ns)
    if ac_frequency is not None and ac_frequency > 0:
        W = min(W, 1.5 * ac_frequency)             # AC: the state's zone and the first copies either side
    W = W if window is None else window
    F = np.linspace(field_min, fmax, max(2, points))
    ac = ac_frequency is not None
    if ac and (ac_frequency is None or ac_frequency <= 0):
        raise ValueError("--ac-frequency must be positive")
    if ac and sidebands < 1:
        raise ValueError("--sidebands must be at least 1")
    alphas = {mm: polarizability(base if mm == 1 else with_m(base, mm)) for mm in range(1, j2 + 1, 2)}
    a0, a2 = scalar_tensor(alphas, j2)
    alpha_m = alphas[abs(m2)]
    acinfo = None
    smap: StarkMap
    if ac:
        assert ac_frequency is not None
        nbig = (2 * sidebands + 1) * len(basis.levels)
        progress(f"  {len(basis.levels)} states x {2 * sidebands + 1} photon blocks ({nbig}); {len(F)} amplitudes "
                 f"up to {fmax:.4g} V/cm at {ac_frequency:.6g} GHz, window ±{W:.4g} GHz ...")
        smap = floquet_map(basis, ac_frequency, F, sidebands, W, dc_field)
        dyn = {mm: dynamic_polarizability(base if mm == 1 else with_m(base, mm), ac_frequency)
               for mm in range(1, j2 + 1, 2)}
        d0, d2 = scalar_tensor(dyn, j2)
        # second order: -½ α(0) F_dc² - ¼ α(ω) F_ac²
        quad = -0.5 * alpha_m * dc_field ** 2 - 0.25 * dyn[abs(m2)] * F ** 2
        # truncation check: one more sideband, at the largest amplitude where the state is still clean
        # (character >= 0.9; past that the largest overlap hops between near-degenerate dressed states)
        cm = smap.character.max(axis=1)
        clean = [k for k in range(len(F)) if cm[k] >= 0.9 and F[k] > 0]
        check = None
        if clean:
            kc = clean[-1]
            e_q = float(smap.energies_ghz[kc, int(np.argmax(smap.character[kc]))])
            top = floquet_map(basis, ac_frequency, F[kc:kc + 1], sidebands + 1, W, dc_field)
            e_q1 = float(top.energies_ghz[0, int(np.argmax(top.character[0]))])
            check = dict(amplitude_V_cm=float(F[kc]), change_MHz=abs(e_q1 - e_q) * 1e3,
                         shift_MHz=e_q * 1e3)
            progress(f"  one more sideband moves the state by {check['change_MHz']:.3g} MHz "
                     f"(shift {e_q * 1e3:.4g} MHz) at {F[kc]:.4g} V/cm")
        acinfo = dict(frequency_ghz=float(ac_frequency), sidebands=int(sidebands), dc_V_cm=float(dc_field),
                      alpha_dynamic=dict(by_mj={f"{mm}/2": a for mm, a in dyn.items()}, alpha0=d0, alpha2=d2,
                                         unit="MHz/(V/cm)^2", alpha0_au=d0 * 1e3 / MHZ_KVCM_PER_AU,
                                         alpha2_au=d2 * 1e3 / MHZ_KVCM_PER_AU),
                      sideband_check=check)
    else:
        progress(f"  {len(basis.levels)} states; {len(F)} fields up to {fmax:.4g} V/cm, window ±{W:.4g} GHz ...")
        smap = stark_map(basis, F, W)
        quad = -0.5 * alpha_m * F ** 2
    cross = avoided_crossings(smap) if len(F) > 2 else []
    # past the field where no level holds half of the state any more (3 fields running), "the state's"
    # crossings mean nothing: keep the ones before
    cmax = smap.character.max(axis=1)
    low = cmax < 0.5
    end = next((k for k in range(len(F) - 2) if low[k] and low[k + 1] and low[k + 2]), len(F))
    f_end = float(F[end]) if end < len(F) else float("inf")
    cross = [c for c in cross if c[0] < f_end]
    ef = follow(smap)
    rows_e, rows_c = [], []
    for k in range(len(F)):
        ok = ~np.isnan(smap.energies_ghz[k])
        rows_e.append([round(float(v), 6) for v in smap.energies_ghz[k][ok]])
        rows_c.append([round(float(v), 4) for v in smap.character[k][ok]])
    in_win = [dict(label=lv.label, E_ghz=float(e), source=lv.source)
              for lv, e in zip(basis.levels, basis.E0_ghz) if abs(e) <= W]
    return dict(
        element=element, mode="ac" if ac else "dc", ac=acinfo, state=target.label, n=n, l=l, j=j2 / 2, mj=m2 / 2, n_star=ns,
        binding_cm=target.binding_cm, energy_source=target.source,
        ionization_cm=energies.ip_cm, rydberg_cm=energies.rm_cm,
        basis=dict(n_lo=n_lo, n_hi=n_hi, l_max=lmax, size=len(basis.levels)),
        fields_V_cm=[float(f) for f in F], window_ghz=W, inglis_teller_V_cm=inglis_teller_V_cm(ns),
        energies_ghz=rows_e, character=rows_c, followed_ghz=[float(v) for v in ef],
        quadratic_ghz=[float(v) * 1e-3 for v in quad],
        polarizability=dict(by_mj={f"{mm}/2": a for mm, a in alphas.items()}, alpha0=a0, alpha2=a2,
                            unit="MHz/(V/cm)^2",
                            alpha0_au=a0 * 1e3 / MHZ_KVCM_PER_AU, alpha2_au=a2 * 1e3 / MHZ_KVCM_PER_AU),
        crossings=[dict(field_V_cm=f, gap_MHz=g * 1e3) for f, g in cross[:40]],
        dissolved_V_cm=None if end >= len(F) else f_end,
        levels_in_window=in_win,
    )


def _js(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.element:
        parser.error("give an element, for example Rb")
    try:
        sp = resolve_species(args.element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = sp["species_id"]
    if element not in SPECIES:
        parser.error(f"Stark maps are for the alkali atoms {', '.join(SPECIES)}")
    try:
        n, l, j2 = parse_state(args.state)
        m2 = parse_half(args.mj) if args.mj else 1
    except ValueError as exc:
        parser.error(str(exc))
    if abs(m2) > j2:
        parser.error(f"|m_j| = {abs(m2)}/2 is above j = {j2}/2")
    if args.points < 2 or args.n_range < 1:
        parser.error("--points must be at least 2 and --n-range at least 1")
    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))
    en = energies_for(element, nist_dir)
    if not en.exists(n, l):
        parser.error(f"{n}{L_LETTERS[l]} is not a level of {element}")
    kind = "Stark map" if args.ac_frequency is None else f"AC Stark map at {args.ac_frequency:g} GHz"
    print(f"\n{'=' * 65}\n  {element} {n}{j_label(l, j2)}, m_j = {m2}/2: {kind}\n{'=' * 65}")
    try:
        payload = compute(element, n, l, j2, m2, energies=en, field_min=args.field_min, field_max=args.field_max,
                          points=args.points, n_range=args.n_range, l_max=args.l_max, window=args.window,
                          ac_frequency=args.ac_frequency, sidebands=args.sidebands, dc_field=args.dc_field)
    except ValueError as exc:
        parser.error(str(exc))
    payload["label"] = sp["label"]
    pol = payload["polarizability"]
    if payload["ac"]:
        dp = payload["ac"]["alpha_dynamic"]
        print(f"  alpha0(omega) = {dp['alpha0']:.6g} MHz/(V/cm)^2" + (f", alpha2(omega) = {dp['alpha2']:.6g}" if j2 > 1 else ""))
    print(f"  n* = {payload['n_star']:.5f} ({payload['energy_source']}); "
          f"alpha0 = {pol['alpha0']:.6g} MHz/(V/cm)^2" + (f", alpha2 = {pol['alpha2']:.6g}" if j2 > 1 else ""))
    if payload["crossings"]:
        c = payload["crossings"][0]
        print(f"  first avoided crossing: {c['field_V_cm']:.5g} V/cm (gap {c['gap_MHz']:.4g} MHz)")
    os.makedirs(args.data_dir, exist_ok=True)
    path = os.path.join(args.data_dir, f"{element}_stark.json")
    write_json(path, payload, indent=None)
    print(f"  ✓ {path}")
    html = render(TEMPLATE, TITLE=f"{sp['label']} {payload['state']} {kind}", DATA_JS=_js(payload))
    out_dir = os.path.join(args.output_dir, element)
    os.makedirs(out_dir, exist_ok=True)
    page = os.path.join(out_dir, f"{element}_stark.html")
    write_offline_report(page, html)
    print(f"  ✓ {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
