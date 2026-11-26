"""
The ``pair`` pipeline: interaction of two alkali Rydberg atoms.
-----------------------------------------------------------------------------
Pair potentials from exact diagonalisation of E1 + E2 + V_dd(R) over pair states near the chosen
pair, with each eigenvalue coloured by its weight on the chosen pair; C6 (and C3 when the two
levels are dipole coupled) from perturbation theory in the pair's degenerate manifold; the
channels that make up C6 with their Förster defects; and the blockade radius for a given Rabi
frequency, from C6 and from the diagonalised potential.  The physics is in
``qali.physics.rydberg_stark.pair``; energies and radial integrals as in the Stark stage.

Sign convention: ΔE = C6/R⁶ (repulsive: C6 > 0).  ARC quotes -C6.

In a static field along the quantisation axis (--field, V/cm; ``qali.physics.rydberg_stark.forster``)
each atom's states are its Stark states (every l mixed, labelled by the level they continue): the
pair potentials, C3 and C6 then include the field.  The states have permanent dipoles (a first-order
C3 for the pair itself), the field mixes parity (about twice the pair states), and any pair within
hΩ of the chosen one counts as resonant (first order).  --forster-max adds a Förster map: the pair
energies from 0 to that field with their dipole coupling to the chosen pair, and the fields where a
coupled pair crosses it (as ARC's StarkMapResonances; Rb 37P3/2 + 37P3/2 -> 37S + 38S at 1.79 V/cm).
A magnetic field along the same axis (--b-field, gauss) adds each atom's Zeeman and diamagnetic terms
(``qali.physics.rydberg_stark.zeeman``) to the dressing, with or without the electric field; it
lifts the ±m_j degeneracy and shifts the Förster defects (the 1.79 V/cm resonance moves to 1.71 at
10 G).

Defaults: n ± 4, l up to the pair's highest l + 3, pair states within 1.7 manifold spacings,
R from 0.3 to 2 blockade radii at Ω/2π = 1 MHz.  Rb 60S + 60S: ~1500 pair states, about 20 s;
Rb 62D: ~5500, about 10 min (cost grows as the cube of the pair basis).

Outputs:
    data/<species>_pair.json
    plots/<species>/<species>_pair.html

Usage:
    qali --pipeline pair Rb --state 60s1/2
    qali --pipeline pair Rb --state 60d5/2 --mj 5/2 --theta 90
    qali --pipeline pair Cs --state 60s1/2 --state2 60p3/2 --mj2 3/2
    qali --pipeline pair Rb --state 37p3/2 --forster-max 3 --field 1.79
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qali.core.paths import reference_path
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.physics.rydberg_stark.defects import L_LETTERS, RydbergEnergies, j_label
from qali.physics.rydberg_stark.forster import Atom, dressed_states, forster_map
from qali.physics.rydberg_stark.pair import (A0_UM, SingleStates, blockade_radius_um, dispersion, leroy_radius_um,
                                             pair_basis, pair_map, single_states)
from qali.physics.rydberg_stark.stark import HARTREE_GHZ, build_basis, manifold_spacing_ghz
from qali.pipelines.stark import SPECIES, energies_for, parse_half, parse_state
from qali.pipelines.rydberg_runs import page_from_json, write_outputs

TEMPLATE = "pair_page.html"
C6_AU_PER_GHZ_UM6 = 1.0 / (HARTREE_GHZ * A0_UM ** 6)
C3_AU_PER_GHZ_UM3 = 1.0 / (HARTREE_GHZ * A0_UM ** 3)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qali --pipeline pair",
                                description="Pair potentials, C6 and blockade radius of two alkali Rydberg atoms")
    p.add_argument("element", nargs="?", default=None, help="Li, Na, K, Rb, Cs or Fr")
    p.add_argument("--state", default="60s1/2", help="State of atom 1, for example 60s1/2 or 60d5/2")
    p.add_argument("--mj", default="1/2", help="m_j of atom 1")
    p.add_argument("--state2", default="", help="State of atom 2 (default: the same)")
    p.add_argument("--mj2", default="", help="m_j of atom 2 (default: the same)")
    p.add_argument("--theta", type=float, default=0.0, help="Angle of the axis to the quantisation axis, degrees")
    p.add_argument("--r-min", type=float, default=None, help="Smallest R, µm (default: 0.3 blockade radii)")
    p.add_argument("--r-max", type=float, default=None, help="Largest R, µm (default: 2 blockade radii)")
    p.add_argument("--points", type=int, default=200, help="Number of R values")
    p.add_argument("--rabi", type=float, default=1.0, help="Rabi frequency Ω/2π for the blockade radius, MHz")
    p.add_argument("--n-range", type=int, default=4, help="Single-atom basis n from n - this to n + this")
    p.add_argument("--l-max", type=int, default=None, help="Highest l in the basis (default: pair's highest l + 3)")
    p.add_argument("--ecut", type=float, default=None,
                   help="Pair states within ± this many GHz (default: 1.7 manifold spacings)")
    p.add_argument("--window", type=float, default=None,
                   help="Keep energies within ± this many GHz of the pair (default: 1.5 x the shift at r-min)")
    p.add_argument("--c6-cut", type=float, default=None,
                   help="C6: leave out pairs further than this many GHz (default: none; ARC uses 25)")
    p.add_argument("--field", type=float, default=0.0,
                   help="Static field along the quantisation axis, V/cm: pair potentials of the Stark-dressed states")
    p.add_argument("--b-field", type=float, default=0.0,
                   help="Magnetic field along the quantisation axis, gauss (Zeeman and diamagnetic terms)")
    p.add_argument("--forster-max", type=float, default=None,
                   help="Also a Förster map: pair energies from 0 to this field (V/cm) with their coupling to the pair")
    p.add_argument("--forster-points", type=int, default=121, help="Fields on the Förster map")
    p.add_argument("--forster-window", type=float, default=1.0,
                   help="Förster map: pairs within ± this many GHz of the chosen pair")
    p.add_argument("--dress-l-max", type=int, default=None,
                   help="Highest l for the Stark dressing (default: n - 1, every l)")
    p.add_argument("--from-json", nargs="+", default=None, metavar="JSON",
                   help="Only write the page of saved pair run(s) (no calculation); the page is named after the JSON")
    p.add_argument("--name", default=None, help="With one --from-json: the page's file name (default: the JSON's)")
    p.add_argument("--nist-dir", default=None, help="Folder of NIST level exports; default: bundled")
    p.add_argument("--data-dir", default=str(Workspace.current().data_json_dir))
    p.add_argument("--output-dir", default=str(Workspace.current().plots_dir),
                   help="Plots folder; the page goes in <output-dir>/<species>/")
    return p


def compute(element: str, s1: tuple[int, int, int], m2a: int, s2: tuple[int, int, int], m2b: int, *,
            energies: RydbergEnergies, theta_deg: float = 0.0, r_min: float | None = None,
            r_max: float | None = None, points: int = 200, rabi_mhz: float = 1.0, n_range: int = 4,
            l_max: int | None = None, ecut: float | None = None, window: float | None = None,
            c6_cut: float | None = None, field: float = 0.0, b_gauss: float = 0.0, forster_max: float | None = None,
            forster_points: int = 121, forster_window: float = 1.0, dress_l_max: int | None = None,
            progress=print) -> dict:
    """The stage's whole calculation, as the JSON payload (no files written)."""
    (n1, l1, j1), (n2, l2, j2) = s1, s2
    if abs(m2a) > j1 or abs(m2b) > j2:
        raise ValueError("|m_j| is above j")
    if rabi_mhz <= 0:
        raise ValueError("--rabi must be positive")
    lmax = max(l1, l2) + 3 if l_max is None else l_max
    if lmax < max(l1, l2) + 1:
        raise ValueError(f"--l-max must be at least {max(l1, l2) + 1} (the pair's l + 1)")
    n_lo, n_hi = max(1, min(n1, n2) - n_range), max(n1, n2) + n_range
    lmax = min(lmax, n_hi - 1)
    if field < 0 or (forster_max is not None and forster_max <= 0):
        raise ValueError("--field must be ≥ 0 and --forster-max > 0")
    if b_gauss < 0:
        raise ValueError("--b-field must be ≥ 0 (the sign of B is the choice of m_j)")
    dressed = bool(field) or bool(b_gauss)              # the pair states themselves change
    dressing = dressed or forster_max is not None
    l_dress = n_hi - 1 if dress_l_max is None else min(max(dress_l_max, lmax), n_hi - 1)
    progress(f"  Single-atom basis n {n_lo}-{n_hi}, l <= {lmax}"
             + (f" (Stark dressing with l <= {l_dress})" if dressing else "") + " ...")
    b = build_basis(energies, element, n1, l1, j1, 1, n_lo, n_hi, l_dress if dressing else lmax, r2=bool(b_gauss))
    assert b.R_au is not None
    idx = {(lv.n, lv.l, lv.j2): i for i, lv in enumerate(b.levels)}
    if (n2, l2, j2) not in idx:
        raise ValueError(f"{n2}{L_LETTERS[l2]}{j2}/2 is not in the basis")
    atom = Atom.from_levels(b.levels, b.R_au, b_gauss, b.R2_au, energies.rm_cm) if dressing else None
    fmap = None
    if forster_max is not None:
        assert atom is not None
        Ff = np.linspace(0.0, forster_max, max(2, forster_points))
        progress(f"  Förster map: {len(Ff)} fields to {forster_max:g} V/cm, pairs within ±{forster_window:g} GHz ...")
        fmap = forster_map(atom, idx[(n1, l1, j1)], m2a, atom, idx[(n2, l2, j2)], m2b, Ff,
                           np.radians(theta_deg), forster_window)
        progress(f"  {len(fmap.resonances)} Förster resonances"
                 + (f"; strongest {fmap.resonances[0]['pair']} at {fmap.resonances[0]['field_V_cm']:.4g} V/cm"
                    if fmap.resonances else ""))
    char_chosen = None
    ss: SingleStates
    if dressed:
        assert atom is not None
        keep = [i for i, lv in enumerate(b.levels) if lv.l <= lmax]
        ss = dressed_states(atom, field, keep)
        pos = {g: r for r, g in enumerate(keep)}
        a, c = pos[idx[(n1, l1, j1)]], pos[idx[(n2, l2, j2)]]
        assert ss.character is not None
        char_chosen = [float(ss.character[ss.states.index((a, m2a))]), float(ss.character[ss.states.index((c, m2b))])]
        b_levels = ss.levels
        at = ", ".join(x for x in (f"{field:g} V/cm" if field else "", f"{b_gauss:g} G" if b_gauss else "") if x)
        progress(f"  Dressed at {at}: the chosen states keep "
                 f"{char_chosen[0]:.3f} and {char_chosen[1]:.3f} of their levels")
    else:
        if dressing:
            keep = [i for i, lv in enumerate(b.levels) if lv.l <= lmax]
            levels_small = [b.levels[i] for i in keep]
            ss = single_states(levels_small, b.R_au[np.ix_(keep, keep)])
            pos = {g: r for r, g in enumerate(keep)}
            a, c = pos[idx[(n1, l1, j1)]], pos[idx[(n2, l2, j2)]]
            b_levels = levels_small
        else:
            a, c = idx[(n1, l1, j1)], idx[(n2, l2, j2)]
            ss = single_states(b.levels, b.R_au)
            b_levels = b.levels
    th = np.radians(theta_deg)
    progress(f"  {len(ss.states)} single-atom states; C3 and C6 ...")
    # in a field, pairs closer than hΩ to the chosen one count as resonant (C3) rather than as C6 channels
    disp = dispersion(ss, a, c, m2a, m2b, th, ecut_ghz=c6_cut,
                      deg_tol_ghz=max(1e-6, rabi_mhz * 1e-3) if dressed else 1e-6)
    c6 = disp.c6_chosen
    c3 = disp.c3_ghz_um3
    resonant = abs(disp.c3_chosen) > 1e-9
    la, lb = b_levels[a], b_levels[c]
    leroy = leroy_radius_um(la, lb)
    rb6 = blockade_radius_um(c6, rabi_mhz) if c6 else float("nan")
    c3max = abs(disp.c3_chosen)
    rb3 = (c3max * 1e3 / rabi_mhz) ** (1 / 3) if resonant else float("nan")
    rb = rb3 if resonant else rb6
    if dressed and resonant:
        # permanent dipoles and the van der Waals term together: the largest R where |C3/R³ + C6/R⁶| = Ω
        Rg = np.geomspace(0.05, 1e4, 20000)
        over = np.abs(disp.c3_chosen / Rg ** 3 + c6 / Rg ** 6) >= rabi_mhz * 1e-3
        rb = float(Rg[np.flatnonzero(over).max()]) if over.any() else float("nan")
    if not np.isfinite(rb):
        raise ValueError("no interaction at second order in this basis")
    R1 = 2.0 * rb if r_max is None else r_max
    R0 = max(0.3 * rb, 1.2 * leroy) if r_min is None else r_min
    if not 0 < R0 < R1:
        raise ValueError("need 0 < --r-min < --r-max")
    ns = min(la.n_star, lb.n_star)
    E_cut = 1.7 * manifold_spacing_ghz(ns) if ecut is None else ecut
    shift0 = (c3max / R0 ** 3 if resonant else 0.0) + (abs(c6) / R0 ** 6 if (dressed or not resonant) else 0.0)
    W = min(max(1.5 * shift0, 2e-3 * rabi_mhz), E_cut) if window is None else window
    pb = pair_basis(ss, a, c, m2a, m2b, th, E_cut, lmax)
    sizes = [B.shape[1] for B in pb.blocks]
    R = np.linspace(R0, R1, max(2, points))
    progress(f"  {len(pb.pairs)} pair states ({sizes[0]} symmetric, {sizes[1]} antisymmetric) within "
             f"±{E_cut:.4g} GHz; {len(R)} R from {R0:.4g} to {R1:.4g} µm, window ±{W:.4g} GHz ...")
    pm = pair_map(pb, R, W)
    k = np.argmax(pm.character, axis=1)
    fol = pm.energies_ghz[np.arange(len(R)), k]
    # blockade from the diagonalised potential: the largest R where the pair has moved by Ω
    om = rabi_mhz * 1e-3
    hit = np.flatnonzero(np.abs(fol) >= om)
    rb_diag = None
    if hit.size and hit.max() < len(R) - 1:
        i = int(hit.max())
        y0, y1 = abs(fol[i]), abs(fol[i + 1])
        rb_diag = float(R[i] + (R[i + 1] - R[i]) * (y0 - om) / (y0 - y1)) if y0 != y1 else float(R[i])
    cmax = pm.character.max(axis=1)
    mixed = np.flatnonzero(cmax < 0.5)
    rows_e, rows_c = [], []
    for i in range(len(R)):
        ok = ~np.isnan(pm.energies_ghz[i])
        rows_e.append([round(float(v), 7) for v in pm.energies_ghz[i][ok]])
        rows_c.append([round(float(v), 4) for v in pm.character[i][ok]])
    vdw = [float(v) for v in (c6 / R ** 6)]
    label1, label2 = la.label, lb.label
    return dict(
        element=element, state1=label1, state2=label2, mj1=m2a / 2, mj2=m2b / 2, theta_deg=theta_deg,
        n_star1=la.n_star, n_star2=lb.n_star, energy_source1=la.source, energy_source2=lb.source,
        basis=dict(n_lo=n_lo, n_hi=n_hi, l_max=lmax, single=len(ss.states), ecut_ghz=E_cut, size=len(pb.pairs),
                   symmetric=sizes[0], antisymmetric=sizes[1]),
        R_um=[float(r) for r in R], window_ghz=W, energies_ghz=rows_e, character=rows_c,
        followed_ghz=[float(v) for v in fol], vdw_ghz=vdw,
        c6=dict(chosen=c6, chosen_au=c6 * C6_AU_PER_GHZ_UM6, eigen=[float(v) for v in disp.c6_ghz_um6],
                manifold=list(disp.manifold), cut_ghz=c6_cut, unit="GHz µm^6", convention="dE = C6/R^6"),
        c3=dict(resonant=resonant, chosen=disp.c3_chosen, eigen=[float(v) for v in c3], unit="GHz µm^3",
                eigen_au=[float(v) * C3_AU_PER_GHZ_UM3 for v in c3]),
        channels=disp.channels,
        leroy_um=leroy,
        blockade=dict(rabi_MHz=rabi_mhz, R_perturbative_um=rb, R_diagonalised_um=rb_diag,
                      from_c3=resonant),
        mixed_below_um=float(R[mixed.max()]) if mixed.size else None,
        field_V_cm=float(field), b_gauss=float(b_gauss),
        dressing=dict(l_max=l_dress, character=char_chosen) if dressing else None,
        forster=None if fmap is None else dict(
            fields_V_cm=[float(v) for v in fmap.fields_V_cm], window_ghz=forster_window,
            target_ghz=[float(v) for v in fmap.target_ghz], target_character=[float(v) for v in fmap.target_character],
            energies_ghz=[[round(float(v), 7) for v in e] for e in fmap.energies_ghz],
            c3_ghz_um3=[[float(f"{v:.5g}") for v in cc] for cc in fmap.c3_ghz_um3],
            labels=fmap.labels, resonances=fmap.resonances[:60]),
    )


def _js(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _clean(o):
    """JSON-safe: NaN -> None, numpy scalars -> float."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float | np.floating):
        return float(o) if np.isfinite(o) else None
    return o


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.from_json:
        if args.name and len(args.from_json) > 1:
            parser.error("--name needs a single --from-json file")
        return pages_only(args.from_json, args.output_dir, args.name)
    if not args.element:
        parser.error("give an element, for example Rb")
    try:
        sp = resolve_species(args.element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = sp["species_id"]
    if element not in SPECIES:
        parser.error(f"Pair potentials are for the alkali atoms {', '.join(SPECIES)}")
    try:
        s1 = parse_state(args.state)
        s2 = parse_state(args.state2) if args.state2.strip() else s1
        m2a = parse_half(args.mj) if args.mj else 1
        m2b = parse_half(args.mj2) if args.mj2.strip() else m2a
    except ValueError as exc:
        parser.error(str(exc))
    for (n, l, j2), m2 in ((s1, m2a), (s2, m2b)):
        if abs(m2) > j2:
            parser.error(f"|m_j| = {abs(m2)}/2 is above j = {j2}/2")
    if args.points < 2 or args.n_range < 1:
        parser.error("--points must be at least 2 and --n-range at least 1")
    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))
    en = energies_for(element, nist_dir)
    for n, l, _ in (s1, s2):
        if not en.exists(n, l):
            parser.error(f"{n}{L_LETTERS[l]} is not a level of {element}")
    name = f"{s1[0]}{j_label(s1[1], s1[2])} + {s2[0]}{j_label(s2[1], s2[2])}"
    at = (f" at {args.field:g} V/cm" if args.field else "") + (f", {args.b_field:g} G" if args.b_field else "")
    print(f"\n{'=' * 65}\n  {element} {name}: pair potential{at}\n{'=' * 65}")
    try:
        payload = compute(element, s1, m2a, s2, m2b, energies=en, theta_deg=args.theta, r_min=args.r_min,
                          r_max=args.r_max, points=args.points, rabi_mhz=args.rabi, n_range=args.n_range,
                          l_max=args.l_max, ecut=args.ecut, window=args.window, c6_cut=args.c6_cut,
                          field=args.field, b_gauss=args.b_field, forster_max=args.forster_max, forster_points=args.forster_points,
                          forster_window=args.forster_window, dress_l_max=args.dress_l_max)
    except ValueError as exc:
        parser.error(str(exc))
    payload["label"] = sp["label"]
    payload = _clean(payload)
    print(f"  C6 = {payload['c6']['chosen']:.6g} GHz µm^6 (dE = C6/R^6; {payload['c6']['chosen_au']:.4g} a.u.)")
    if payload["c3"]["resonant"]:
        print(f"  resonant: C3 = {abs(payload['c3']['chosen']):.5g} GHz µm^3 for the chosen pair "
              f"(manifold: {min(payload['c3']['eigen']):.5g} to {max(payload['c3']['eigen']):.5g})")
    bl = payload["blockade"]
    print(f"  blockade radius at {bl['rabi_MHz']:g} MHz: {bl['R_perturbative_um']:.4g} µm"
          + (f" ({bl['R_diagonalised_um']:.4g} µm diagonalised)" if bl["R_diagonalised_um"] else ""))
    write_outputs("pair", element, payload, args.data_dir, args.output_dir)
    return 0



def pages_only(paths: list[str], output_dir: str, name: str | None = None) -> int:
    """Write the pages of saved runs; no calculation."""
    failed = 0
    for path in paths:
        try:
            page = page_from_json("pair", path, output_dir, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"  ✗ {path}: {exc}")
            failed += 1
            continue
        print(f"  ✓ {page}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
