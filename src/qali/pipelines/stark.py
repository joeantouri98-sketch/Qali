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

Ionisation (--ionisation): the static map is solved instead by complex scaling on a radial mesh
(``qali.physics.rydberg_stark.ionisation``), which includes the continuum: every level gets its
ionisation width Γ as well as its position.  The default field range runs to 1.2 x the classical
ionisation field of the state, where the saddle point -2√F (a.u.) reaches its energy.  Rb 30S:
about 12 s per field per core (120 by default), the fields solved in parallel (--workers).

Magnetic field (--b-field, gauss; --b-angle to the electric field, degrees): the paramagnetic
Zeeman term μ_B B·(g_L L + g_S S) is added (``qali.physics.rydberg_stark.zeeman``), with the
mixing of j = l ± 1/2 of each (n, l) that ARC's diagonal-only term leaves out, and the diamagnetic
term (B²/8) r² sin²θ_B (also left out by ARC; ~n⁴ B²).  Parallel fields keep m_j; crossed fields
mix it and the basis holds every |m_j| ≤ --m-max (default |m_j| + 2), about (2 m_max + 1) times
the states.

Defaults: n - 4 to n + 4, every l (full manifolds), fields from 0 to the Inglis-Teller field
1/(3n*⁵) a.u., energies within ±0.6 manifold spacings.  Rb 50S: ~900 states, about 30 s.

Outputs:
    data/<species>_stark.json
    plots/<species>/<species>_stark.html

Usage:
    qali --pipeline stark Rb --state 50s1/2
    qali --pipeline stark Cs --state 60d5/2 --mj 5/2 --field-max 2
    qali --pipeline stark Rb --state 56d5/2 --ac-frequency 2 --field-max 3 --n-range 6 --l-max 10
    qali --pipeline stark Rb --state 30s1/2 --field-max 40 --b-field 500 --b-angle 90
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from qali.core.errors import MissingDataError
from qali.core.paths import reference_path
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.physics.extrapolation.exact import MARINESCU
from qali.physics.rydberg_stark.defects import L_LETTERS, RydbergEnergies, j_label
from qali.physics.rydberg_stark.ramp import (
    adiabatic_step,
    crossing_slopes,
    landau_zener,
    max_step_for,
    ramp,
    ramp_adiabatic,
)
from qali.physics.rydberg_stark.ionisation import build_system, classical_limit_ghz, default_workers, resonance_map
from qali.physics.rydberg_stark.stark import (AU_FIELD_V_CM, CM1_GHZ, HARTREE_GHZ, avoided_crossings, build_basis,
                                              dynamic_polarizability, floquet_map, follow, inglis_teller_V_cm, manifold_spacing_ghz, polarizability,
                                              scalar_tensor, StarkMap, stark_map, with_m)
from qali.physics.rydberg_stark.zeeman import diamagnetic_estimate_ghz, zeeman_basis
from qali.pipelines.rydberg_runs import page_from_json, write_outputs

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
    p.add_argument("--points", type=int, default=None, help="Number of field values (default 300; 120 with --ionisation)")
    p.add_argument("--n-range", type=int, default=4, help="Basis n from n - this to n + this")
    p.add_argument("--l-max", type=int, default=None, help="Highest l in the basis (default: every l)")
    p.add_argument("--window", type=float, default=None,
                   help="Keep energies within ± this many GHz of the state (default: 0.6 manifold spacings; AC: 1.5 f; ionisation: 0.3)")
    p.add_argument("--ionisation", action="store_true",
                   help="Static field: also the ionisation widths (complex scaling; slower). "
                        "Default field range up to 1.2 x the classical ionisation field")
    p.add_argument("--ion-rate", type=float, default=1.0,
                   help="Ionisation: the state's rate Γ/h (MHz) taken as the onset of ionisation")
    p.add_argument("--ramp-time", type=float, default=None,
                   help="Static map: also carry the state through a linear ramp from --field-min to --field-max "
                        "in this many ns (time-dependent Schrödinger equation in the instantaneous eigenbasis; Landau-Zener at "
                        "each crossing)")
    p.add_argument("--b-field", type=float, default=0.0, help="Magnetic field, gauss (paramagnetic Zeeman term)")
    p.add_argument("--b-angle", type=float, default=0.0,
                   help="Angle of B to the electric field (the quantisation axis), degrees; non-zero mixes m_j")
    p.add_argument("--m-max", default="",
                   help="Crossed fields: highest |m_j| in the basis, for example 5/2 (default: |m_j| + 2)")
    p.add_argument("--workers", type=int, default=0,
                   help="Ionisation: processes solving fields in parallel (default 0: every core but one)")
    p.add_argument("--ac-frequency", type=float, default=None,
                   help="AC field frequency, GHz; the map is then against the AC amplitude")
    p.add_argument("--sidebands", type=int, default=2, help="AC only: photon sidebands each side")
    p.add_argument("--dc-field", type=float, default=0.0, help="AC only: static field added, V/cm")
    p.add_argument("--from-json", nargs="+", default=None, metavar="JSON",
                   help="Only write the page of saved stark run(s) (no calculation); the page is named after the JSON")
    p.add_argument("--name", default=None, help="With one --from-json: the page's file name (default: the JSON's)")
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
            field_min: float = 0.0, field_max: float | None = None, points: int | None = None,
            n_range: int = 4, l_max: int | None = None, window: float | None = None,
            ac_frequency: float | None = None, sidebands: int = 2, dc_field: float = 0.0,
            ionisation: bool = False, ion_rate_mhz: float = 1.0, workers: int = 0,
            ramp_ns: float | None = None, b_gauss: float = 0.0, b_angle_deg: float = 0.0,
            m_max2: int | None = None, progress=print) -> dict:
    """The stage's whole calculation, as the JSON payload (no files written)."""
    n_lo, n_hi = max(1, n - n_range), n + n_range
    lmax = n_hi - 1 if l_max is None else min(l_max, n_hi - 1)
    if l > lmax:
        raise ValueError(f"--l-max {lmax} is below the state's l = {l}")
    progress(f"  Basis n {n_lo}-{n_hi}, l <= {lmax}, m_j = {m2}/2 ...")
    base = build_basis(energies, element, n, l, j2, 1 if m2 > 0 else -1, n_lo, n_hi, lmax, r2=bool(b_gauss))
    basis = base if abs(m2) == 1 else with_m(base, m2)
    target = basis.levels[basis.target]
    maginfo = None
    if b_gauss:
        if ionisation:
            raise ValueError("--b-field is not available with --ionisation")
        crossed = abs(np.sin(np.radians(b_angle_deg))) > 1e-12
        basis = zeeman_basis(base, m2, b_gauss, b_angle_deg, m_max2 if crossed else None, energies.rm_cm)
        w0, v0 = np.linalg.eigh(basis.h0())
        k0 = int(np.argmax((basis.tvec() @ v0) ** 2))
        ns_t = target.n_star
        maginfo = dict(B_gauss=float(b_gauss), angle_deg=float(b_angle_deg), crossed=bool(crossed),
                       m_max=(basis.meta.get("m_max2", abs(m2)) / 2), zeeman_shift_ghz=float(w0[k0]),
                       diamagnetic=True,
                       diamagnetic_state_ghz=diamagnetic_estimate_ghz(ns_t, l, b_gauss),
                       states=len(basis.levels))
        progress(f"  B = {b_gauss:g} G at {b_angle_deg:g}° to E"
                 + (f" (m_j mixed, |m_j| <= {maginfo['m_max']:g})" if crossed else "")
                 + f": {len(basis.levels)} states; the state moves {w0[k0] * 1e3:.5g} MHz at F = 0 "
                 f"(of which diamagnetic ~{maginfo['diamagnetic_state_ghz'] * 1e3:.3g} MHz)")
    ns = target.n_star
    if ramp_ns is not None and (ionisation or ac_frequency is not None):
        raise ValueError("--ramp-time is for static maps (leave out --ionisation and --ac-frequency)")
    if ramp_ns is not None and ramp_ns <= 0:
        raise ValueError("--ramp-time must be positive")
    if ionisation and ac_frequency is not None:
        raise ValueError("--ionisation is for static fields (leave out --ac-frequency)")
    # classical field ionisation of the state: the saddle point -2√F reaches its energy
    f_class = (target.binding_cm * CM1_GHZ / HARTREE_GHZ) ** 2 / 4.0 * AU_FIELD_V_CM
    if field_max is not None:
        fmax = field_max
    else:
        fmax = 1.2 * f_class if ionisation else inglis_teller_V_cm(ns)
    points = (120 if ionisation else 300) if points is None else points
    if fmax <= field_min:
        raise ValueError("--field-max must be above --field-min")
    W = (0.3 if ionisation else 0.6) * manifold_spacing_ghz(ns)       # ionisation: every level costs time
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
    ioninfo = None
    rampinfo = None
    pops = None
    widths = None
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
    elif ionisation:
        progress(f"  Ionisation: complex-scaled mesh, n {n_lo}-{n_hi}, l <= {lmax} ...")
        rsys = build_system(energies, element, n, l, j2, m2, n_lo, n_hi, lmax)
        mt = rsys.meta
        progress(f"  {len(rsys.channels)} channels x {mt['N']} mesh points (box {mt['r_max_a0']:.4g} a0); "
                 f"{len(F)} fields up to {fmax:.4g} V/cm, window ±{W:.4g} GHz ...")
        nw = workers or default_workers(len(F))
        if nw > 1:
            progress(f"  {nw} worker processes")
        rmap = resonance_map(rsys, F, W, progress=progress, workers=nw)
        smap = StarkMap(F, rmap.energies_ghz, rmap.character, tuple(lv.label for lv in basis.levels), target.label)
        quad = -0.5 * alpha_m * F ** 2
        widths = rmap.widths_ghz
        # the state's ionisation rate: the widths of the levels it is spread over, weighted by its
        # share of each (within the window); a single level's width is not the state's once it has
        # dissolved into the manifold, and the weighting leaves out spurious levels with no share
        def state_rate(ch, gw):
            c = np.nan_to_num(np.asarray(ch, float))
            g = np.nan_to_num(np.asarray(gw, float))
            tot = c.sum()
            return float((c * g).sum() / tot) if tot > 0 else 0.0
        g_state = np.array([state_rate(rmap.character[k], widths[k]) for k in range(len(F))])
        rate = ion_rate_mhz * 1e-3
        onset = next((float(F[k]) for k in range(len(F)) if g_state[k] >= rate), None)
        # convergence check at the onset field (else the last one): again with a smaller rotation
        # angle and a finer mesh; positions and the state's rate should not move
        kc = next((k for k in range(len(F)) if F[k] == onset), len(F) - 1)
        N2 = int(round(mt["N"] * 1.25))
        alt = build_system(energies, element, n, l, j2, m2, n_lo, n_hi, lmax, theta=0.15,
                           N=N2, scale=mt["scale_a0"] * mt["N"] / N2)
        am = resonance_map(alt, F[kc:kc + 1], W)
        e1, e2 = rmap.energies_ghz[kc], am.energies_ghz[0]
        ok1, ok2 = ~np.isnan(e1), ~np.isnan(e2)
        de = [abs(e2[ok2][int(np.argmin(np.abs(e2[ok2] - e)))] - e) * 1e3 for e in e1[ok1]]
        mde = float(np.median(de)) if de else 0.0
        r1, r2 = g_state[kc], state_rate(am.character[0], am.widths_ghz[0])
        ion_check = dict(field_V_cm=float(F[kc]), levels=int(ok1.sum()), median_shift_change_MHz=mde,
                         state_rate_MHz=r1 * 1e3, state_rate_alt_MHz=r2 * 1e3)
        progress(f"  check at {F[kc]:.4g} V/cm (θ 0.15, mesh x1.25): levels move by {mde:.3g} MHz (median); "
                 f"the state's rate {r1 * 1e3:.4g} -> {r2 * 1e3:.4g} MHz")
        ioninfo = dict(method="complex scaling, Lagrange-Laguerre mesh", theta=mt["theta"], mesh_points=mt["N"],
                       box_a0=mt["r_max_a0"], channels=len(rsys.channels), classical_field_V_cm=f_class,
                       saddle_ghz=[float(v) for v in classical_limit_ghz(F, rsys.E_target_au)],
                       state_rate_MHz=[float(v) * 1e3 for v in g_state],
                       rate_MHz=ion_rate_mhz, onset_V_cm=onset, check=ion_check)
        if onset is not None:
            progress(f"  the state's ionisation rate reaches Γ/h = {ion_rate_mhz:g} MHz at {onset:.4g} V/cm "
                     f"(classical {f_class:.4g} V/cm)")
    else:
        progress(f"  {len(basis.levels)} states; {len(F)} fields up to {fmax:.4g} V/cm, window ±{W:.4g} GHz ...")
        smap = stark_map(basis, F, W)
        quad = -0.5 * alpha_m * F ** 2
    if maginfo is not None:
        quad = quad + maginfo["zeeman_shift_ghz"] * 1e3                # MHz; the Zeeman-shifted start
    cross = avoided_crossings(smap) if len(F) > 2 else []
    if ramp_ns is not None:
        smap, pops, rampinfo = _ramp(basis, smap, cross, F, W, ramp_ns, progress)
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
    rows_p = None if pops is None else [[round(float(v), 5) for v in pops[k][~np.isnan(smap.energies_ghz[k])]]
                                        for k in range(len(F))]
    rows_g = None if widths is None else [[round(float(v), 7) for v in widths[k][~np.isnan(smap.energies_ghz[k])]]
                                          for k in range(len(F))]
    in_win = [dict(label=lab, E_ghz=float(e), source=lv.source)
              for lv, lab, e in zip(basis.levels, basis.labels(), basis.E0_ghz) if abs(e) <= W]
    return dict(
        element=element, mode="ac" if ac else "dc", ac=acinfo, ionisation=ioninfo, ramp=rampinfo, magnetic=maginfo,
        state=target.label, n=n, l=l, j=j2 / 2, mj=m2 / 2, n_star=ns,
        binding_cm=target.binding_cm, energy_source=target.source,
        ionization_cm=energies.ip_cm, rydberg_cm=energies.rm_cm,
        basis=dict(n_lo=n_lo, n_hi=n_hi, l_max=lmax, size=len(basis.levels)),
        fields_V_cm=[float(f) for f in F], window_ghz=W, inglis_teller_V_cm=inglis_teller_V_cm(ns),
        energies_ghz=rows_e, character=rows_c, widths_ghz=rows_g, population=rows_p,
        followed_ghz=[float(v) for v in ef],
        quadratic_ghz=[float(v) * 1e-3 for v in quad],
        polarizability=dict(by_mj={f"{mm}/2": a for mm, a in alphas.items()}, alpha0=a0, alpha2=a2,
                            unit="MHz/(V/cm)^2",
                            alpha0_au=a0 * 1e3 / MHZ_KVCM_PER_AU, alpha2_au=a2 * 1e3 / MHZ_KVCM_PER_AU),
        crossings=[dict(field_V_cm=f, gap_MHz=g * 1e3) for f, g in cross[:40]],
        dissolved_V_cm=None if end >= len(F) else f_end,
        levels_in_window=in_win,
    )


DIRECT_RAMP_NS = 100.0     # ramps up to this long are propagated directly in the full basis


def _ramp(basis, smap, cross, F, W, ramp_ns, progress):
    """The state through a linear ramp over the map's fields, and the Landau-Zener picture of its
    avoided crossings.  Returns (map, populations, payload block)."""
    ks = [int(np.argmin(np.abs(F - f))) for f, _g in cross]
    slopes = [crossing_slopes(smap.energies_ghz, smap.character, F, k) for k in ks]
    grid = (F[-1] - F[0]) / max(1, len(F) - 1)
    # direct propagation costs ~ ramp time, the eigenbasis one ~ √(ramp time): the latter wins above ~100 ns
    method = "direct" if ramp_ns <= DIRECT_RAMP_NS else "eigenbasis"
    if method == "direct":
        step = max_step_for(cross, slopes, grid / 4)
    else:
        step = adiabatic_step(basis, (F[-1] - F[0]) / ramp_ns, grid)
    progress(f"  Ramp {F[0]:g} -> {F[-1]:.4g} V/cm in {ramp_ns:g} ns ({method}): steps of {step:.3g} V/cm ...")
    r = ramp(basis, F, ramp_ns, W, step) if method == "direct" else ramp_adiabatic(basis, F, ramp_ns, W, step)
    rmap = StarkMap(F, r.energies_ghz, r.character, smap.labels, smap.target)
    # what the propagation keeps on the state's own character across each crossing (diabatic share)
    fol = [float(r.population[k, int(np.argmax(r.character[k]))]) for k in range(len(F))]
    lz = []
    for (f, g), k, ds in zip(cross, ks, slopes):
        a, b = max(0, k - 4), min(len(F) - 1, k + 4)
        p_tdse = fol[b] / fol[a] if fol[a] > 1e-6 else None
        lz.append(dict(field_V_cm=float(f), gap_MHz=float(g) * 1e3, slope_diff_MHz_per_V_cm=None if ds is None else ds * 1e3,
                       p_lz=None if ds is None else landau_zener(g, ds, r.slew_V_cm_ns), p_tdse=p_tdse))
    info = dict(ramp_ns=float(ramp_ns), method=method, slew_V_cm_ns=r.slew_V_cm_ns, steps=r.steps, field_step_V_cm=float(step),
                in_window=[float(v) for v in r.in_window], on_state=fol, crossings=lz[:40])
    progress(f"  {r.steps} steps; at the end {fol[-1]:.3f} of the population is on the level with most of the state, "
             f"{r.in_window[-1]:.3f} in the window")
    return rmap, r.population, info


def _js(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


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
        parser.error(f"Stark maps are for the alkali atoms {', '.join(SPECIES)}")
    try:
        n, l, j2 = parse_state(args.state)
        m2 = parse_half(args.mj) if args.mj else 1
    except ValueError as exc:
        parser.error(str(exc))
    if abs(m2) > j2:
        parser.error(f"|m_j| = {abs(m2)}/2 is above j = {j2}/2")
    if (args.points is not None and args.points < 2) or args.n_range < 1:
        parser.error("--points must be at least 2 and --n-range at least 1")
    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))
    en = energies_for(element, nist_dir)
    if not en.exists(n, l):
        parser.error(f"{n}{L_LETTERS[l]} is not a level of {element}")
    try:
        mm2 = parse_half(args.m_max) if args.m_max.strip() else None
    except ValueError as exc:
        parser.error(str(exc))
    if mm2 is not None and mm2 < abs(m2):
        parser.error("--m-max must be at least |m_j|")
    kind = (("Stark-Zeeman map" if args.b_field else "Stark map") + (" with ionisation" if args.ionisation else "")
            + (f" with a {args.ramp_time:g} ns ramp" if args.ramp_time else "")) if args.ac_frequency is None \
        else f"AC Stark map at {args.ac_frequency:g} GHz"
    print(f"\n{'=' * 65}\n  {element} {n}{j_label(l, j2)}, m_j = {m2}/2: {kind}\n{'=' * 65}")
    try:
        payload = compute(element, n, l, j2, m2, energies=en, field_min=args.field_min, field_max=args.field_max,
                          points=args.points, n_range=args.n_range, l_max=args.l_max, window=args.window,
                          ac_frequency=args.ac_frequency, sidebands=args.sidebands, dc_field=args.dc_field,
                          ionisation=args.ionisation, ion_rate_mhz=args.ion_rate, workers=args.workers,
                          ramp_ns=args.ramp_time, b_gauss=args.b_field, b_angle_deg=args.b_angle, m_max2=mm2)
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
    write_outputs("stark", element, payload, args.data_dir, args.output_dir)
    return 0



def pages_only(paths: list[str], output_dir: str, name: str | None = None) -> int:
    """Write the pages of saved runs; no calculation."""
    failed = 0
    for path in paths:
        try:
            page = page_from_json("stark", path, output_dir, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"  ✗ {path}: {exc}")
            failed += 1
            continue
        print(f"  ✓ {page}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
