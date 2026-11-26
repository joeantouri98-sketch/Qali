"""Stark maps with a magnetic field as well: parallel or crossed E and B.

H = H0 + F z + μ_B B·(g_L L + g_S S),  E along the quantisation axis z, B at angle β to it in the
xz plane.  The paramagnetic term acts within each (n, l): in |l j m_j> it is diagonal in m_j for
the z part (g_j m_j on the diagonal, coupling j = l ± 1/2) and changes m_j by ±1 for the x part,
from the uncoupled |m_l m_s> with Clebsch-Gordan coefficients (the coupling order l then s of the
spin-angle functions behind ``angular_z``).  The radial overlap of j = l ± 1/2 of the same n is
taken as 1 (as ARC does).

* β = 0 (parallel): m_j stays good; the basis is the Stark basis of one m_j plus the Zeeman term.
  This is ARC's StarkMap with Bz (Šibalić et al., CPC 220, 319 (2017)).
* β ≠ 0 (crossed): m_j is mixed; the basis holds every m_j with |m_j| ≤ m_max of each level, and the
  chosen m_j is taken along B (the good quantum number at F = 0): the chosen state is |j m_j>
  turned onto b̂ with Wigner's d^j(β).

The diamagnetic term (B²/8) r² (1 - (r̂·b̂)²) = (B²/8) r² (2/3)(1 - P2(r̂·b̂)), with
P2(r̂·b̂) = Σ_q (-1)^q C²_q(b̂) C²_{-q}(r̂), is included when the basis has the <r²> integrals
(``build_basis(..., r2=True)``): <r²> within each l and between l and l ± 2, the rank-2 angular
factors as for the dipole.  ARC leaves it out; it grows as n⁴ B² (Rb 30S at 500 G: ~30 MHz).

g_L = 1 - m_e/M (from the species' reduced Rydberg constant), g_S = 2.00231930436.
"""

from __future__ import annotations

import math

import numpy as np

from qali.physics.atomic_structure.angular import reduced_c, threej2
from qali.physics.rydberg_stark.stark import HARTREE_GHZ, StarkBasis, _kappa, _z_matrix

MU_B_GHZ_PER_G = 1.39962449361e-3          # μ_B/h, GHz per gauss
G_S = 2.00231930436
R_INF_CM = 109737.31568160
B_AU_GAUSS = 2.35051757077e9               # atomic unit of magnetic field, gauss


def g_l(rydberg_cm: float | None) -> float:
    """Orbital g factor 1 - m_e/M, with m_e/M from R_M = R_inf/(1 + m_e/M)."""
    if not rydberg_cm:
        return 1.0
    return 1.0 - (R_INF_CM / rydberg_cm - 1.0)


def _cg(l: int, ml2: int, ms2: int, j2: int, m2: int) -> float:
    """<l m_l; 1/2 m_s | j m_j> (doubled m_l, m_s, j, m_j)."""
    if ml2 + ms2 != m2:
        return 0.0
    ph = -1.0 if ((2 * l - 1 + m2) // 2) % 2 else 1.0
    return ph * np.sqrt(j2 + 1.0) * threej2(2 * l, 1, j2, ml2, ms2, -m2)


def _uncoupled(l: int, j2: int, m2: int) -> dict[tuple[int, int], float]:
    out = {}
    for ms2 in (-1, 1):
        ml2 = m2 - ms2
        if abs(ml2) <= 2 * l:
            c = _cg(l, ml2, ms2, j2, m2)
            if c:
                out[(ml2, ms2)] = c
    return out


def _op(l: int, a: dict, b: dict, kind: str, gl: float, gs: float) -> float:
    """<a|O|b> with a, b given by their uncoupled components; kind 'z': gl Lz + gs Sz,
    '+' : gl L+ + gs S+ (raises m by 1)."""
    tot = 0.0
    for (ml2, ms2), cb in b.items():
        if kind == "z":
            ca = a.get((ml2, ms2))
            if ca:
                tot += ca * cb * (gl * ml2 / 2 + gs * ms2 / 2)
        else:
            ca = a.get((ml2 + 2, ms2))                       # L+ |m_l>
            if ca:
                ml = ml2 / 2
                tot += ca * cb * gl * np.sqrt(l * (l + 1) - ml * (ml + 1))
            ca = a.get((ml2, ms2 + 2))                       # S+ |m_s = -1/2>
            if ca and ms2 == -1:
                tot += ca * cb * gs
    return tot


def magnetic_matrix(levels, m2s, cos_b: float, sin_b: float, gl: float, gs: float = G_S) -> np.ndarray:
    """<i|cos β (g_L L_z + g_S S_z) + sin β (g_L L_x + g_S S_x)|k> over states (levels[i], m2s[i])
    (dimensionless; times μ_B B)."""
    N = len(levels)
    M = np.zeros((N, N))
    by_nl: dict[tuple[int, int], list[int]] = {}
    for i, lv in enumerate(levels):
        by_nl.setdefault((lv.n, lv.l), []).append(i)
    for (_n, l), idx in by_nl.items():
        comp = {i: _uncoupled(l, levels[i].j2, int(m2s[i])) for i in idx}
        for i in idx:
            for k in idx:
                dm = int(m2s[i]) - int(m2s[k])
                v = 0.0
                if dm == 0 and cos_b:
                    v += cos_b * _op(l, comp[i], comp[k], "z", gl, gs)
                elif dm == 2 and sin_b:                       # <i|O_x|k> = ½ <i|O_+|k>
                    v += 0.5 * sin_b * _op(l, comp[i], comp[k], "+", gl, gs)
                elif dm == -2 and sin_b:
                    v += 0.5 * sin_b * _op(l, comp[k], comp[i], "+", gl, gs)
                M[i, k] = v
    return M


def wigner_d(j2: int, mp2: int, m2: int, beta: float) -> float:
    """d^j_{m'm}(β) (doubled j, m', m): |j m> turned by β about y = Σ_m' d^j_{m'm}(β) |j m'>."""
    f = math.factorial
    j, m, mp = j2 / 2, m2 / 2, mp2 / 2
    c, s = math.cos(beta / 2), math.sin(beta / 2)
    pre = math.sqrt(f(round(j + m)) * f(round(j - m)) * f(round(j + mp)) * f(round(j - mp)))
    tot = 0.0
    for k in range(round(2 * j) + 1):
        a, b, cc, d = round(j + m - k), k, round(j - k - mp), round(k - m + mp)
        if min(a, b, cc, d) < 0:
            continue
        tot += (-1) ** (k - round(m - mp)) / (f(a) * f(b) * f(cc) * f(d)) \
            * c ** round(2 * j - 2 * k + m - mp) * s ** round(2 * k - m + mp)
    return pre * tot


def _c2_of_axis(beta: float) -> dict[int, float]:
    """C²_q(b̂) for b̂ = (sin β, 0, cos β)."""
    s, c = np.sin(beta), np.cos(beta)
    return {0: 0.5 * (3 * c * c - 1), 1: -np.sqrt(1.5) * s * c, -1: np.sqrt(1.5) * s * c,
            2: np.sqrt(3 / 8) * s * s, -2: np.sqrt(3 / 8) * s * s}


def diamagnetic_matrix(levels, m2s, R2: np.ndarray, beta: float) -> np.ndarray:
    """<i|r² (1 - (r̂·b̂)²)|k> (a.u.) over states (levels[i], m2s[i]); R2 over the same rows."""
    N = len(levels)
    M = np.zeros((N, N))
    cb = _c2_of_axis(beta)
    cache: dict[tuple, float] = {}
    for i in range(N):
        a, ma = levels[i], int(m2s[i])
        for k in range(N):
            b, mb = levels[k], int(m2s[k])
            r2 = R2[i, k]
            if r2 == 0.0 or abs(a.l - b.l) not in (0, 2):
                continue
            key = (a.l, a.j2, ma, b.l, b.j2, mb)
            ang = cache.get(key)
            if ang is None:
                ang = 0.0
                if a.l == b.l and a.j2 == b.j2 and ma == mb:
                    ang += 2.0 / 3.0
                q2 = ma - mb                                  # <a|C²_{-q}|b> needs m_a - m_b = -q
                q = -q2 // 2
                if abs(q2) <= 4 and q in cb and cb[q]:
                    c = reduced_c(_kappa(a.l, a.j2), _kappa(b.l, b.j2), 2)
                    if c:
                        ph = -1.0 if ((a.j2 - ma) // 2) % 2 else 1.0
                        el = ph * threej2(a.j2, 4, b.j2, -ma, q2, mb) * c       # <a|C²_{-q}|b>
                        sgn = -1.0 if q % 2 else 1.0
                        ang -= 2.0 / 3.0 * sgn * cb[q] * el
                cache[key] = ang
            M[i, k] = ang * r2
    return M


def zeeman_basis(base: StarkBasis, m2: int, B_gauss: float, angle_deg: float = 0.0,
                 m_max2: int | None = None, rydberg_cm: float | None = None) -> StarkBasis:
    """The Stark basis of ``base`` (any m_j; its levels and radial integrals) with the Zeeman term
    of B at angle_deg to the field.  Parallel: the states of m_j = m2/2.  Crossed: every m_j with
    |m_j| ≤ m_max2/2 (default |m_j| + 2) of each level; the chosen state is (target, m2)."""
    assert base.R_au is not None
    beta = np.radians(angle_deg)
    cb, sb = float(np.cos(beta)), float(np.sin(beta))
    if abs(sb) < 1e-12:
        sb, cb = 0.0, (1.0 if cb > 0 else -1.0)
    gl = g_l(rydberg_cm)
    E0b = base.E0_ghz
    if sb == 0.0:
        keep = [i for i, lv in enumerate(base.levels) if lv.j2 >= abs(m2)]
        if base.target not in keep:
            raise ValueError("|m_j| > j of the chosen state")
        levels = [base.levels[i] for i in keep]
        R = base.R_au[np.ix_(keep, keep)]
        m2s = np.full(len(levels), m2)
        Z = _z_matrix(levels, R, m2)
        Hb = MU_B_GHZ_PER_G * B_gauss * magnetic_matrix(levels, m2s, cb, 0.0, gl)
        if base.R2_au is not None:
            Hb = Hb + _dia_ghz(B_gauss) * diamagnetic_matrix(levels, m2s, base.R2_au[np.ix_(keep, keep)], 0.0)
        return StarkBasis(levels, m2, keep.index(base.target), E0b[keep], Z,
                          dict(base.meta, B_gauss=B_gauss, angle_deg=angle_deg, diamagnetic=base.R2_au is not None),
                          R, H0_extra=Hb)
    mm = abs(m2) + 4 if m_max2 is None else m_max2
    if mm < abs(m2):
        raise ValueError("--m-max is below |m_j| of the chosen state")
    states = [(i, mq) for mq in range(-mm, mm + 1, 2) for i, lv in enumerate(base.levels) if lv.j2 >= abs(mq)]
    levels = [base.levels[i] for i, _ in states]
    m2s = np.array([mq for _, mq in states])
    E0 = E0b[[i for i, _ in states]]
    N = len(states)
    Z = np.zeros((N, N))
    for mq in range(-mm, mm + 1, 2):
        rows = [r for r, (_, q) in enumerate(states) if q == mq]
        if not rows:
            continue
        ids = [states[r][0] for r in rows]
        Z[np.ix_(rows, rows)] = _z_matrix([base.levels[i] for i in ids], base.R_au[np.ix_(ids, ids)], mq)
    Hb = MU_B_GHZ_PER_G * B_gauss * magnetic_matrix(levels, m2s, cb, sb, gl)
    if base.R2_au is not None:
        ids = [i for i, _ in states]
        Hb = Hb + _dia_ghz(B_gauss) * diamagnetic_matrix(levels, m2s, base.R2_au[np.ix_(ids, ids)], beta)
    t = states.index((base.target, m2))
    # m_j is taken along B: the chosen state is |j m_j> turned from z onto b̂ (the Zeeman sublevel
    # at F = 0 for a weak field); components with |m_j'| above m_max are missing (renormalised)
    jt = base.levels[base.target].j2
    tv = np.zeros(N)
    for r, (i, q) in enumerate(states):
        if i == base.target and abs(q) <= jt:
            tv[r] = wigner_d(jt, q, m2, beta)
    kept = float(np.linalg.norm(tv))
    tv /= kept
    return StarkBasis(levels, m2, t, E0, Z, dict(base.meta, B_gauss=B_gauss, angle_deg=angle_deg, m_max2=mm,
                                                 diamagnetic=base.R2_au is not None, target_norm=kept),
                      None, H0_extra=Hb, m2s=m2s, target_vec=tv)


def _dia_ghz(B_gauss: float) -> float:
    """B²/8 in GHz per a.u. of r²(1 - cos²)."""
    return (B_gauss / B_AU_GAUSS) ** 2 / 8.0 * HARTREE_GHZ


def r2_au(n_star: float, l: int) -> float:
    return n_star ** 2 / 2.0 * (5 * n_star ** 2 + 1 - 3 * l * (l + 1))


def diamagnetic_estimate_ghz(n_star: float, l: int, B_gauss: float, sin2: float = 2.0 / 3.0) -> float:
    """(B²/8) <r² sin²θ> in GHz, with <sin²θ> = 2/3 (exact for S states): the diamagnetic shift's size."""
    b = B_gauss / B_AU_GAUSS
    return b ** 2 / 8.0 * sin2 * r2_au(n_star, l) * HARTREE_GHZ
