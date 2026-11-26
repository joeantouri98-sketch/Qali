"""Interactions of two alkali Rydberg atoms: dipole-dipole pair potentials, C3 and C6, blockade radius.

Two atoms at separation R along a unit vector n at angle θ to the quantisation axis interact through

    V = [d1·d2 - 3 (d1·n)(d2·n)] / R³ = Σ_{q1 q2} W_{q1 q2}(θ) d1_{q1} d2_{q2} / R³,

d_q the spherical components of each dipole, <n l j m|d_q|n' l' j' m'> =
(-1)^(j-m) (j 1 j'; -m q m') <l j||C1||l' j'> <n l j|r|n' l' j'>, with the energies and radial
integrals of ``defects`` and ``radial``.  W is real for n in the xz plane; at θ = 0
W_00 = -2, W_{+1,-1} = W_{-1,+1} = -1 and M = m1 + m2 is conserved.  (-1)^(l1 + l2) is always
conserved, and so is the symmetry under exchange of the two (identical) atoms.

Pair potentials: H = E1 + E2 + V(R) diagonalised over pair states within an energy cut of the
chosen pair, in the exchange-symmetric and antisymmetric blocks separately (the method of ARC's
PairStateInteractions, Šibalić et al., CPC 220, 319 (2017)).  The character of an eigenstate is its
weight on the chosen pair |a m1, b m2> and its swap |b m2, a m1>.

C6 from second-order perturbation theory in the degenerate manifold of the chosen pair (every
m1, m2, and both orders when a ≠ b): ΔE = C6 / R⁶, with

    C6_{αβ} = Σ_p <α|V R³|p><p|V R³|β> / (E_pair - E_p),

so a repulsive potential has C6 > 0.  (ARC quotes -C6, as V(R) = -C6/R⁶.)  If the two levels of
the pair are dipole coupled (nS + nP), the first-order resonant C3 matrix is given as well.

Units: energies GHz (relative to the chosen pair), R in µm, C3 in GHz µm³, C6 in GHz µm⁶.

Engine contract: inputs as arguments, results as dataclasses; no files, prints or plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from qali.physics.atomic_structure.angular import reduced_c, threej2
from qali.physics.rydberg_stark.defects import Level
from qali.physics.rydberg_stark.stark import CM1_GHZ, HARTREE_GHZ, _kappa

A0_UM = 5.29177210903e-5                       # Bohr radius, µm
C3_UNIT = HARTREE_GHZ * A0_UM ** 3             # (e a0)² / R³ in GHz µm³


# ---------------------------------------------------------------- single-atom states and dipoles

@dataclass
class SingleStates:
    """|level, m_j> states of one atom with the spherical dipole matrices d_q (a.u.)."""
    levels: list[Level]
    states: list[tuple[int, int]]            # (level index, 2 m_j)
    E_ghz: np.ndarray                        # energy of each state, GHz (absolute, below the limit < 0)
    D: dict[int, np.ndarray]                 # q -> <s|d_q|s'>


def single_states(levels: list[Level], R_au: np.ndarray) -> SingleStates:
    """Every m_j of every level, and <s|d_q|s'> from the radial matrix R_au (l -> l ± 1 blocks)."""
    states = [(i, m2) for i, lv in enumerate(levels) for m2 in range(-lv.j2, lv.j2 + 1, 2)]
    S = len(states)
    E = np.array([-levels[i].binding_cm * CM1_GHZ for i, _ in states])
    D = {q: np.zeros((S, S)) for q in (-1, 0, 1)}
    lv_of = np.array([i for i, _ in states])
    # angular factors depend on (l, j, m) only: cache them
    cache: dict[tuple, float] = {}
    nz_i, nz_k = np.nonzero(R_au)
    pairs = set(zip(nz_i.tolist(), nz_k.tolist()))
    by_level: dict[int, list[int]] = {}
    for s, i in enumerate(lv_of.tolist()):
        by_level.setdefault(i, []).append(s)
    for i, k in pairs:
        a, b = levels[i], levels[k]
        c = reduced_c(_kappa(a.l, a.j2), _kappa(b.l, b.j2), 1)
        if c == 0.0:
            continue
        for s in by_level[i]:
            m = states[s][1]
            for t in by_level[k]:
                mp = states[t][1]
                q2 = m - mp
                if abs(q2) > 2:
                    continue
                key = (a.l, a.j2, m, b.l, b.j2, mp)
                ang = cache.get(key)
                if ang is None:
                    ph = -1.0 if ((a.j2 - m) // 2) % 2 else 1.0
                    ang = ph * threej2(a.j2, 2, b.j2, -m, q2, mp) * c
                    cache[key] = ang
                if ang:
                    D[q2 // 2][s, t] = ang * R_au[i, k]
    return SingleStates(levels, states, E, D)


def angular_weights(theta: float) -> dict[tuple[int, int], float]:
    """W_{q1 q2}(θ) with V R³ = Σ W d1_q1 d2_q2, the axis n = (sin θ, 0, cos θ)."""
    nv = np.array([np.sin(theta), 0.0, np.cos(theta)])
    T = np.eye(3) - 3.0 * np.outer(nv, nv)
    s = 1 / np.sqrt(2)
    # d_x = (d_-1 - d_+1)/√2, d_y = i (d_-1 + d_+1)/√2, d_z = d_0:  d_a = Σ_q U[a, q] d_q
    U = np.array([[s, 0, -s], [1j * s, 0, 1j * s], [0, 1, 0]])      # columns q = -1, 0, +1
    W = U.T @ T @ U
    out = {}
    for i1, q1 in enumerate((-1, 0, 1)):
        for i2, q2 in enumerate((-1, 0, 1)):
            w = W[i1, i2]
            if abs(w) > 1e-12:
                if abs(w.imag) > 1e-12:
                    raise AssertionError("W must be real for an axis in the xz plane")
                out[(q1, q2)] = float(w.real)
    return out


def _v_columns(ss: SingleStates, W: dict, cols: list[tuple[int, int]]) -> np.ndarray:
    """<p|V R³|α> (a.u.) for every product state p = (s1, s2) (flattened s1 * S + s2) and each α."""
    S = len(ss.states)
    out = np.zeros((S * S, len(cols)))
    for c, (a1, a2) in enumerate(cols):
        acc = np.zeros((S, S))
        for (q1, q2), w in W.items():
            acc += w * np.outer(ss.D[q1][:, a1], ss.D[q2][:, a2])
        out[:, c] = acc.ravel()
    return out


# ---------------------------------------------------------------- C3 and C6 by perturbation theory

@dataclass(frozen=True)
class Dispersion:
    manifold: tuple[str, ...]                # labels of the degenerate pair states, "m1,m2" (and swaps)
    c3_ghz_um3: np.ndarray                   # eigenvalues of the first-order matrix (zeros unless resonant)
    c3_chosen: float                         # largest |C3| among the eigenstates the chosen pair is part of
    c6_ghz_um6: np.ndarray                   # eigenvalues of the second-order matrix
    c6_vectors: np.ndarray                   # columns in the manifold basis
    c6_chosen: float                         # <chosen|C6|chosen>
    channels: list[dict] = field(default_factory=list)   # largest contributions to c6_chosen


def _state_index(ss: SingleStates, level: int, m2: int) -> int:
    return ss.states.index((level, m2))


def dispersion(ss: SingleStates, a: int, b: int, m2a: int, m2b: int, theta: float = 0.0,
               ecut_ghz: float | None = None, deg_tol_ghz: float = 1e-6) -> Dispersion:
    """C3 and C6 of the pair (levels a, b) in its degenerate manifold, and C6 of |a m2a, b m2b>.

    ecut_ghz: leave out intermediate pairs further than this from the chosen pair (None: all of
    those the single-atom basis gives)."""
    W = angular_weights(theta)
    la, lb = ss.levels[a], ss.levels[b]
    man = [(_state_index(ss, a, ma), _state_index(ss, b, mb))
           for ma in range(-la.j2, la.j2 + 1, 2) for mb in range(-lb.j2, lb.j2 + 1, 2)]
    if a != b:
        man += [(t, s) for s, t in man]
    # the degenerate manifold: every m in zero field; in a field only those still degenerate (±m in E,
    # none other in B), and any other pair within deg_tol_ghz of the chosen one (a Förster resonance)
    E_c = ss.E_ghz[_state_index(ss, a, m2a)] + ss.E_ghz[_state_index(ss, b, m2b)]
    man = [(s, t) for s, t in man if abs(ss.E_ghz[s] + ss.E_ghz[t] - E_c) <= deg_tol_ghz]
    if getattr(ss, "field_V_cm", 0.0) or getattr(ss, "B_gauss", 0.0):
        near = np.argwhere(np.abs(ss.E_ghz[:, None] + ss.E_ghz[None, :] - E_c) <= deg_tol_ghz)
        man += [(int(s), int(t)) for s, t in near if (int(s), int(t)) not in man]
    labels = tuple(f"{ss.levels[ss.states[s][0]].label} {ss.states[s][1]}/2, "
                   f"{ss.levels[ss.states[t][0]].label} {ss.states[t][1]}/2" for s, t in man)
    S = len(ss.states)
    cols = _v_columns(ss, W, man)                                  # [S*S, K] a.u.
    E_t = E_c
    Ep = (ss.E_ghz[:, None] + ss.E_ghz[None, :]).ravel()
    dE = E_t - Ep
    in_man = np.zeros(S * S, bool)
    for s, t in man:
        in_man[s * S + t] = True
    # first order inside the manifold
    C3m = cols[[s * S + t for s, t in man]] * C3_UNIT
    C3m = 0.5 * (C3m + C3m.T)
    keep = (~in_man) & (np.abs(dE) > 1e-9) & np.any(cols != 0, axis=1)
    if ecut_ghz is not None:
        keep &= np.abs(dE) <= ecut_ghz
    V = cols[keep] * C3_UNIT                                       # GHz µm³
    C6m = (V / dE[keep][:, None]).T @ V                            # GHz µm⁶
    C6m = 0.5 * (C6m + C6m.T)
    w6, v6 = np.linalg.eigh(C6m)
    i0 = man.index((_state_index(ss, a, m2a), _state_index(ss, b, m2b)))
    if np.any(C3m):
        w3, v3 = np.linalg.eigh(C3m)
        part = v3[i0] ** 2 > 1e-8
        c3c = float(w3[part][np.argmax(np.abs(w3[part]))])
    else:
        w3, c3c = np.zeros(len(man)), 0.0
    contrib = V[:, i0] ** 2 / dE[keep]
    idx = np.flatnonzero(keep)
    groups: dict[tuple[int, int], list[float]] = {}
    for p, cv in zip(idx.tolist(), contrib.tolist()):
        s1, s2 = divmod(p, S)
        k = tuple(sorted((ss.states[s1][0], ss.states[s2][0])))
        g = groups.setdefault(k, [0.0, 0.0])
        g[0] += cv
        g[1] = E_t - Ep[p]
    total = float(C6m[i0, i0])
    chans = sorted(groups.items(), key=lambda kv: -abs(kv[1][0]))[:12]
    channels = [dict(pair=f"{ss.levels[i].label} + {ss.levels[k].label}", forster_ghz=-g[1],
                     c6_ghz_um6=g[0], share=g[0] / total if total else 0.0) for (i, k), g in chans]
    return Dispersion(labels, w3, c3c, w6, v6, total, channels)


# ---------------------------------------------------------------- pair potentials

@dataclass
class PairBasis:
    ss: SingleStates
    pairs: np.ndarray                          # [P, 2] single-state indices
    E0_ghz: np.ndarray                         # [P] pair energies relative to the chosen pair
    V_ghz_um3: np.ndarray                      # [P, P] V R³
    target: list[int]                          # rows of the chosen pair and its swap
    blocks: list[np.ndarray]                   # orthonormal columns: exchange-symmetric, antisymmetric
    meta: dict = field(default_factory=dict)


def pair_basis(ss: SingleStates, a: int, b: int, m2a: int, m2b: int, theta: float = 0.0,
               ecut_ghz: float = 25.0, l_max: int | None = None) -> PairBasis:
    """Product states (s1, s2) within ecut_ghz of the chosen pair, with its parity (-1)^(l1+l2)
    and, at θ = 0, its M = m1 + m2."""
    S = len(ss.states)
    sa, sb = _state_index(ss, a, m2a), _state_index(ss, b, m2b)
    E_t = ss.E_ghz[sa] + ss.E_ghz[sb]
    lpar = (ss.levels[a].l + ss.levels[b].l) % 2
    lv = np.array([ss.levels[i].l for i, _ in ss.states])
    m2 = np.array([m for _, m in ss.states])
    ok1 = np.ones(S, bool) if l_max is None else lv <= l_max
    dE = ss.E_ghz[:, None] + ss.E_ghz[None, :] - E_t
    mask = (np.abs(dE) <= ecut_ghz) & ok1[:, None] & ok1[None, :]
    if not getattr(ss, "field_V_cm", 0.0):           # a field mixes parity
        mask &= ((lv[:, None] + lv[None, :]) % 2) == lpar
    if abs(theta) < 1e-12:
        mask &= (m2[:, None] + m2[None, :]) == (m2a + m2b)
    mask[sa, sb] = mask[sb, sa] = True
    pairs = np.argwhere(mask)
    P = len(pairs)
    W = angular_weights(theta)
    V = np.zeros((P, P))
    s1, s2 = pairs[:, 0], pairs[:, 1]
    for (q1, q2), w in W.items():
        V += w * ss.D[q1][np.ix_(s1, s1)] * ss.D[q2][np.ix_(s2, s2)]
    V *= C3_UNIT
    E0 = ss.E_ghz[s1] + ss.E_ghz[s2] - E_t
    row = {(int(x), int(y)): r for r, (x, y) in enumerate(pairs)}
    target = sorted({row[(sa, sb)], row[(sb, sa)]})
    # exchange blocks
    sym, anti = [], []
    for r, (x, y) in enumerate(pairs):
        x, y = int(x), int(y)
        r2 = row[(y, x)]
        if r2 < r:
            continue
        if r2 == r:
            sym.append({r: 1.0})
        else:
            h = 1 / np.sqrt(2)
            sym.append({r: h, r2: h})
            anti.append({r: h, r2: -h})
    blocks = []
    for vecs in (sym, anti):
        B = np.zeros((P, len(vecs)))
        for c, d in enumerate(vecs):
            for r, v in d.items():
                B[r, c] = v
        blocks.append(B)
    return PairBasis(ss, pairs, E0, V, target, blocks, dict(ecut_ghz=ecut_ghz, theta=theta, l_max=l_max))


@dataclass(frozen=True)
class PairMap:
    R_um: np.ndarray                     # [nR]
    energies_ghz: np.ndarray             # [nR, N] eigenvalues, ascending (NaN padded)
    character: np.ndarray                # [nR, N] weight on the chosen pair and its swap


def pair_map(pb: PairBasis, R_um, window_ghz: float | None = None) -> PairMap:
    R = np.asarray(R_um, float)
    blk = [(B, B.T @ np.diag(pb.E0_ghz) @ B, B.T @ pb.V_ghz_um3 @ B, B[pb.target]) for B in pb.blocks if B.shape[1]]
    rows_e, rows_c = [], []
    for r in R:
        ws, cs = [], []
        for _B, H0, V, T in blk:
            H = H0 + V / r ** 3
            if window_ghz is None:
                w, v = eigh(H)
            else:
                w, v = eigh(H, subset_by_value=(-window_ghz, window_ghz), driver="evr")
            ws.append(w)
            cs.append(np.sum((T @ v) ** 2, axis=0))
        w = np.concatenate(ws)
        o = np.argsort(w)
        rows_e.append(w[o])
        rows_c.append(np.concatenate(cs)[o])
    width = max((len(x) for x in rows_e), default=0)
    En = np.full((len(R), width), np.nan)
    ch = np.zeros((len(R), width))
    for k, (e, c) in enumerate(zip(rows_e, rows_c)):
        En[k, :len(e)] = e
        ch[k, :len(c)] = c
    return PairMap(R, En, ch)


def r2_expect_au(n_star: float, l: int) -> float:
    """<r²> of a hydrogenic orbital at n*, a.u."""
    return n_star ** 2 / 2.0 * (5 * n_star ** 2 + 1 - 3 * l * (l + 1))


def leroy_radius_um(la: Level, lb: Level) -> float:
    """LeRoy radius 2 (sqrt<r²>_a + sqrt<r²>_b): below it the electron clouds overlap."""
    return 2.0 * (np.sqrt(r2_expect_au(la.n_star, la.l)) + np.sqrt(r2_expect_au(lb.n_star, lb.l))) * A0_UM


def blockade_radius_um(c6_ghz_um6: float, rabi_mhz: float) -> float:
    """R where |C6|/R⁶ equals h·Ω/2π."""
    return float((abs(c6_ghz_um6) * 1e3 / rabi_mhz) ** (1 / 6))
