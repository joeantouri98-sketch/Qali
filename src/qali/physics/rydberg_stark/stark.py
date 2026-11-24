"""Stark maps of alkali Rydberg states: exact diagonalisation in a |n l j m_j> basis.

H = H0 + F z at one m_j (a static field along z keeps m_j), with H0 the field-free levels of
``defects`` and

    <n l j m|z|n' l' j' m> = <n l j|r|n' l' j'> (-1)^(j-m) (j 1 j'; -m 0 m) <l j||C1||l' j'>,

radial integrals from ``radial``.  The basis takes every n in [n_lo, n_hi] and l up to l_max
(and below n), both j.  Each eigenvector's overlap with the chosen state gives its character;
following the largest overlap through the map tracks the state across avoided crossings.

Polarisabilities follow ΔE = -½ α F²: α(m_j) = -2 Σ_k |<s|z|k>|² / (E_s - E_k) over the basis,
and for j ≥ 3/2 the scalar and tensor parts from α(m_j) = α0 + α2 (3m² - j(j+1)) / (j(2j-1)).

Units: energies in GHz (relative to the chosen state's field-free level), fields in V/cm,
polarisabilities in MHz/(V/cm)².

Engine contract: inputs as arguments, results as frozen dataclasses; no files, prints or plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from qali.physics.atomic_structure.angular import reduced_c, threej2
from qali.physics.rydberg_stark import radial
from qali.physics.rydberg_stark.defects import Level, RydbergEnergies

HARTREE_GHZ = 6.579683920502e6
CM1_GHZ = 29.9792458
AU_FIELD_V_CM = 5.14220674763e9


def _kappa(l: int, j2: int) -> int:
    return -(l + 1) if j2 == 2 * l + 1 else l


def angular_z(l: int, j2: int, lp: int, jp2: int, m2: int) -> float:
    """(-1)^(j-m) (j 1 j'; -m 0 m) <l j||C1||l' j'> (doubled j, m)."""
    if abs(l - lp) != 1:
        return 0.0
    c = reduced_c(_kappa(l, j2), _kappa(lp, jp2), 1)
    if c == 0.0:
        return 0.0
    ph = -1.0 if ((j2 - m2) // 2) % 2 else 1.0
    return ph * threej2(j2, 2, jp2, -m2, 0, m2) * c


@dataclass
class StarkBasis:
    """States |n l j m_j> and the field-free Hamiltonian and z matrix (GHz, a.u.)."""
    levels: list[Level]
    m2: int                          # 2 m_j
    target: int                      # index of the chosen state
    E0_ghz: np.ndarray               # field-free energies relative to the chosen state
    Z_au: np.ndarray                 # <i|z|j>, a.u.
    meta: dict = field(default_factory=dict)
    R_au: np.ndarray | None = None   # radial <i|r|j> (l -> l±1 blocks), a.u.


def build_basis(energies: RydbergEnergies, symbol: str | None, n: int, l: int, j2: int, m2: int,
                n_lo: int, n_hi: int, l_max: int) -> StarkBasis:
    """Basis around n, l, j at m_j = m2/2.  symbol: Marinescu species for the radial functions
    (None: hydrogen)."""
    if abs(m2) > j2:
        raise ValueError("|m_j| > j")
    levels: list[Level] = []
    for nn in range(n_lo, n_hi + 1):
        for ll in range(min(l_max, nn - 1) + 1):
            if not energies.exists(nn, ll):
                continue
            for jj in (2 * ll - 1, 2 * ll + 1):
                if jj < 1 or abs(m2) > jj:
                    continue
                levels.append(energies.level(nn, ll, jj))
    idx = {(lv.n, lv.l, lv.j2): i for i, lv in enumerate(levels)}
    if (n, l, j2) not in idx:
        raise ValueError(f"{n}{'spdfghik'[l]}{j2}/2 is not in the basis")
    t = idx[(n, l, j2)]
    E = np.array([-lv.binding_cm for lv in levels]) * CM1_GHZ
    E0 = E - E[t]
    # radial functions, one grid for all
    nst_max = max(lv.n_star for lv in levels)
    x = radial.x_grid(nst_max, radial.inner_limit(energies.alpha_d if symbol else 0.0))
    chis = np.zeros((len(levels), len(x)))
    groups: dict[tuple[int, int], list[int]] = {}
    for i, lv in enumerate(levels):
        groups.setdefault((lv.l, lv.j2), []).append(i)
    for (ll, jj), ids in groups.items():
        chis[ids] = radial.solve(symbol, ll, jj / 2, [levels[i].n_star for i in ids], x)
    R = np.zeros((len(levels), len(levels)))
    for (la, ja), ia in groups.items():
        for (lb, jb), ib in groups.items():
            if lb == la + 1:
                Rab = radial.dipole(chis[ia], chis[ib], x)
                R[np.ix_(ia, ib)] = Rab
                R[np.ix_(ib, ia)] = Rab.T
    return StarkBasis(levels, m2, t, E0, _z_matrix(levels, R, m2), dict(n_lo=n_lo, n_hi=n_hi, l_max=l_max), R)


def _z_matrix(levels: list[Level], R: np.ndarray, m2: int) -> np.ndarray:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, lv in enumerate(levels):
        groups.setdefault((lv.l, lv.j2), []).append(i)
    Z = np.zeros_like(R)
    for (la, ja), ia in groups.items():
        for (lb, jb), ib in groups.items():
            if abs(la - lb) == 1:
                a = angular_z(la, ja, lb, jb, m2)
                if a:
                    Z[np.ix_(ia, ib)] = a * R[np.ix_(ia, ib)]
    return Z


def with_m(basis: StarkBasis, m2: int) -> StarkBasis:
    """The same basis at another m_j (states with j < |m_j| dropped); radial integrals reused."""
    assert basis.R_au is not None
    keep = [i for i, lv in enumerate(basis.levels) if lv.j2 >= abs(m2)]
    if basis.target not in keep:
        raise ValueError("|m_j| > j of the chosen state")
    levels = [basis.levels[i] for i in keep]
    R = basis.R_au[np.ix_(keep, keep)]
    return StarkBasis(levels, m2, keep.index(basis.target), basis.E0_ghz[keep], _z_matrix(levels, R, m2),
                      dict(basis.meta), R)


@dataclass(frozen=True)
class StarkMap:
    fields_V_cm: np.ndarray          # [nF]
    energies_ghz: np.ndarray         # [nF, N] eigenvalues, ascending
    character: np.ndarray            # [nF, N] |<chosen state|eigenvector>|²
    labels: tuple[str, ...]          # basis labels (field-free order)
    target: str


def stark_map(basis: StarkBasis, fields_V_cm, window_ghz: float | None = None) -> StarkMap:
    """Eigenvalues at each field; character = overlap with the chosen state.

    window_ghz: keep only the eigenvalues within ± this of the chosen state's field-free level
    (faster for large bases); None keeps all.  Rows are padded with NaN to a common width."""
    F = np.asarray(fields_V_cm, float)
    H0 = np.diag(basis.E0_ghz)
    Zg = basis.Z_au * HARTREE_GHZ / AU_FIELD_V_CM            # GHz per V/cm
    rows_e, rows_c = [], []
    for f in F:
        if window_ghz is None:
            w, v = eigh(H0 + f * Zg)
        else:
            w, v = eigh(H0 + f * Zg, subset_by_value=(-window_ghz, window_ghz), driver="evr")
        rows_e.append(w)
        rows_c.append(v[basis.target] ** 2)
    width = max((len(r) for r in rows_e), default=0)
    En = np.full((len(F), width), np.nan)
    ch = np.zeros((len(F), width))
    for k, (e, c) in enumerate(zip(rows_e, rows_c)):
        En[k, :len(e)] = e
        ch[k, :len(c)] = c
    return StarkMap(F, En, ch, tuple(lv.label for lv in basis.levels), basis.levels[basis.target].label)


def inglis_teller_V_cm(n_star: float) -> float:
    """Field where the adjacent hydrogenic manifolds first touch, 1/(3 n*^5) a.u."""
    return AU_FIELD_V_CM / (3.0 * n_star ** 5)


def manifold_spacing_ghz(n_star: float) -> float:
    return HARTREE_GHZ / n_star ** 3


def follow(smap: StarkMap) -> np.ndarray:
    """Energy (GHz) of the eigenstate with the largest character at each field."""
    return smap.energies_ghz[np.arange(len(smap.fields_V_cm)), np.argmax(smap.character, axis=1)]


def gap_to_nearest(smap: StarkMap) -> np.ndarray:
    """Distance (GHz) from the followed state to the nearest other eigenvalue, at each field."""
    k = np.argmax(smap.character, axis=1)
    E = smap.energies_ghz
    rows = np.arange(len(k))
    own = E[rows, k]
    d = np.abs(E - own[:, None])
    other = np.where((np.arange(E.shape[1])[None, :] == k[:, None]) | np.isnan(d), np.inf, d)
    return other.min(axis=1)


def avoided_crossings(smap: StarkMap, depth: float = 0.3) -> list[tuple[float, float]]:
    """[(field V/cm, gap GHz)] at the local minima of the gap deeper than depth x its median."""
    g = gap_to_nearest(smap)
    med = float(np.median(g))
    return [(float(smap.fields_V_cm[i]), float(g[i])) for i in range(1, len(g) - 1)
            if g[i] < g[i - 1] and g[i] <= g[i + 1] and g[i] < depth * med]


def polarizability(basis: StarkBasis) -> float:
    """α(m_j) of the chosen state from second-order perturbation theory, MHz/(V/cm)²."""
    t = basis.target
    zt = basis.Z_au[t]
    dE = basis.E0_ghz - basis.E0_ghz[t]
    mask = np.abs(dE) > 1e-12
    shift2 = np.sum(zt[mask] ** 2 / (-dE[mask]))            # E2 = F² Σ |z|²/(E_t - E_k), z in a.u.
    zfac = (HARTREE_GHZ / AU_FIELD_V_CM) ** 2               # GHz² per (V/cm)² per a.u.²
    return float(-2.0 * shift2 * zfac * 1e3)                # E2 = -½ α F², in MHz/(V/cm)²


def scalar_tensor(alphas: dict[int, float], j2: int) -> tuple[float, float]:
    """(α0, α2) from α(m_j) of two or more |m_j| (doubled keys); α2 = 0 for j = 1/2."""
    j = j2 / 2
    if j2 < 3:
        return float(np.mean(list(alphas.values()))), 0.0
    ms = sorted(alphas)
    A = np.array([[1.0, (3 * (m / 2) ** 2 - j * (j + 1)) / (j * (2 * j - 1))] for m in ms])
    sol, *_ = np.linalg.lstsq(A, np.array([alphas[m] for m in ms]), rcond=None)
    return float(sol[0]), float(sol[1])


# ---------------------------------------------------------------- AC fields (Shirley Floquet)

@dataclass(frozen=True)
class FloquetMap(StarkMap):
    """Quasienergies of the atom in F(t) = F_dc + F_ac cos(2π f t): fields_V_cm holds the peak
    amplitudes F_ac, character the weight of |chosen state, 0 photons>."""
    frequency_ghz: float
    sidebands: int
    dc_V_cm: float


def floquet_map(basis: StarkBasis, frequency_ghz: float, amplitudes_V_cm, sidebands: int = 3,
                window_ghz: float | None = None, dc_V_cm: float = 0.0) -> FloquetMap:
    """Shirley's time-independent Floquet Hamiltonian (Phys. Rev. 138, B979 (1965)).

    Blocks q = -Q..Q (photon number): H0 + F_dc z + q h f on the diagonal, (F_ac/2) z between
    q and q ± 1, so F_ac is the peak amplitude (ARC's ShirleyMethod convention).  Quasienergies
    repeat every f; window_ghz keeps those within ± this of the chosen state's level.  Only
    processes of up to Q + 1 photons are converged."""
    if frequency_ghz <= 0:
        raise ValueError("AC frequency must be positive")
    if sidebands < 1:
        raise ValueError("need at least one sideband")
    A = np.asarray(amplitudes_V_cm, float)
    N = len(basis.levels)
    Zg = basis.Z_au * HARTREE_GHZ / AU_FIELD_V_CM            # GHz per V/cm
    nb = 2 * sidebands + 1
    qs = np.arange(-sidebands, sidebands + 1)
    H0 = np.kron(np.eye(nb), np.diag(basis.E0_ghz) + dc_V_cm * Zg)
    H0 += np.diag(np.repeat(qs * frequency_ghz, N))
    B = np.kron(np.eye(nb, k=1) + np.eye(nb, k=-1), 0.5 * Zg)
    t0 = sidebands * N + basis.target                      # |chosen state, q = 0>
    # without a static field, (-1)^(l + q) is conserved: z changes l by one and the drive q by one,
    # so the matrix splits into two halves (same quasienergies, a quarter of the work)
    if dc_V_cm == 0.0:
        par = (np.tile([lv.l for lv in basis.levels], nb) + np.repeat(qs, N)) % 2
        blocks = [np.flatnonzero(par == 0), np.flatnonzero(par == 1)]
    else:
        blocks = [np.arange(nb * N)]
    rows_e, rows_c = [], []
    for a in A:
        H = H0 + a * B
        ws, cs = [], []
        for ids in blocks:
            Hb = H[np.ix_(ids, ids)]
            if window_ghz is None:
                w, v = eigh(Hb)
            else:
                w, v = eigh(Hb, subset_by_value=(-window_ghz, window_ghz), driver="evr")
            pos = np.flatnonzero(ids == t0)
            ws.append(w)
            cs.append(v[pos[0]] ** 2 if len(pos) else np.zeros(len(w)))
        w = np.concatenate(ws)
        o = np.argsort(w)
        rows_e.append(w[o])
        rows_c.append(np.concatenate(cs)[o])
    width = max((len(r) for r in rows_e), default=0)
    En = np.full((len(A), width), np.nan)
    ch = np.zeros((len(A), width))
    for k, (e, c) in enumerate(zip(rows_e, rows_c)):
        En[k, :len(e)] = e
        ch[k, :len(c)] = c
    return FloquetMap(A, En, ch, tuple(lv.label for lv in basis.levels),
                      basis.levels[basis.target].label, float(frequency_ghz), int(sidebands), float(dc_V_cm))


def dynamic_polarizability(basis: StarkBasis, frequency_ghz: float) -> float:
    """α(ω) of the chosen state, MHz/(V/cm)²; the AC shift is -¼ α(ω) F_ac² (F_ac peak).

    α(ω) = 2 Σ_k ω_k |<s|z|k>|² / (ω_k² - ω²), ω_k = E_k - E_s; α(0) is ``polarizability``."""
    t = basis.target
    zt = basis.Z_au[t]
    wk = basis.E0_ghz - basis.E0_ghz[t]
    mask = np.abs(wk) > 1e-12
    s = np.sum(wk[mask] * zt[mask] ** 2 / (wk[mask] ** 2 - frequency_ghz ** 2))   # a.u.² / GHz
    zfac = (HARTREE_GHZ / AU_FIELD_V_CM) ** 2
    return float(2.0 * s * zfac * 1e3)
