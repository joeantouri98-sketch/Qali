"""Field ionisation of alkali Rydberg states: Stark resonances by complex scaling.

The bound-state Stark map (``stark``) cannot ionise: its basis has no continuum.  Here the atom in
a static field is solved on a radial Lagrange-Laguerre mesh (Baye, Phys. Rep. 565, 1 (2015)) in
channels (l, j) at fixed m_j, with r rotated into the complex plane, r -> r e^(iθ) (complex
scaling; Reinhardt 1982, Ho 1983).  Outgoing waves then decay, the continuum rotates away from the
real axis, and each Stark state becomes an isolated eigenvalue E - iΓ/2: position and ionisation
width, independent of θ once converged.

Hamiltonian per channel, with r = h e^(iθ) x on the Laguerre zeros x_i (regularised by x):
    T/(2 h² e^(2iθ)) + l(l+1)/(2r²) + V_lj(r),   and F r <l j m|cos θ|l' j' m> between channels,
diagonal on the mesh.

Core: Rydberg states, their Stark mixing and their ionisation depend on the core only through the
quantum defects (quantum-defect theory), and a mesh that reaches 10⁴ a0 cannot resolve a Z = 37
core.  So each channel has a soft, analytic model potential
    V_lj(r) = -1/r - α_d / (2 (r² + a²)²) - c_lj e^(-r/a)/r,   a = 3 a0,
with c_lj tuned so that the channel's Rydberg levels have the precision quantum defect of
``defects`` modulo 1 (the integer part only counts deep core-like states).  The levels that
remain slightly off (the defect's energy dependence) are then moved exactly onto the precision
energies by adding Σ_k ΔE_k φ_k φ_kᵀ, φ_k the channel's complex-scaled eigenvectors
(c-normalised, φᵀφ = 1), the analytic continuation of a small short-range correction.  The
potential is analytic for any θ < π/2; the default θ = 0.2 (larger angles amplify the coarse mesh near the core).

Checks (hydrogen, against Fernández & Garcia, Appl. Math. Comput. 273, 11 (2016), and the
1s values of Benassi & Grecchi): 1s at F = 0.04-0.1 a.u. and |n1 n2 m> = |0 9 0>, |0 0 9> to
1e-4 in Γ at θ = 0.2 (1e-7 at θ = 0.45, allowed for hydrogen).

Units: GHz, V/cm; Γ is the full width (ionisation rate Γ/ħ).
Engine contract: inputs as arguments, results as dataclasses; no files, prints or plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import eig, eigvalsh_tridiagonal

from qali.physics.rydberg_stark.defects import Level, RydbergEnergies
from qali.physics.rydberg_stark.stark import AU_FIELD_V_CM, CM1_GHZ, HARTREE_GHZ, angular_z

THETA = 0.2
SOFT = 3.0                  # a0: range of the soft core terms (resolved by the mesh)
N_CORR_MIN = 6              # levels below n* = 6 are not moved (core-like, far from any window)
L_TUNE = 4                  # channels above: polarisation term only, levels set by the projector


def laguerre_mesh(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Zeros of L_N (Golub-Welsch, stable for any N) and the regularised kinetic matrix -d²/dx²."""
    k = np.arange(1, N)
    x = eigvalsh_tridiagonal(2.0 * np.arange(N) + 1.0, k.astype(float))
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    sgn = (-1.0) ** (np.arange(N)[:, None] - np.arange(N)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        T = sgn * (X1 + X2) / (np.sqrt(X1 * X2) * (X1 - X2) ** 2)
    T[np.diag_indices(N)] = -(x ** 2 - 2 * (2 * N + 1) * x - 4) / (12 * x ** 2)
    return x, T


def mesh_for(n_max: float, box: float = 8.0) -> tuple[int, float]:
    """Mesh size and scale for states up to n_max, the box reaching ~box n_max² a0."""
    N = int(np.ceil(6 * n_max))
    x_last = 4.0 * N + 2.0
    return N, box * n_max ** 2 / x_last


@dataclass
class Channel:
    l: int
    j2: int
    levels: list[Level]                   # precision levels assigned to its bound states
    H: np.ndarray                         # field-free channel Hamiltonian (a.u., complex scaled, corrected)
    target_vec: np.ndarray | None = None  # φ of the chosen state (c-normalised)


@dataclass
class ResonanceSystem:
    channels: list[Channel]
    x: np.ndarray
    h: complex                                # complex mesh scale h e^(iθ)
    m2: int
    target: Level
    E_target_au: float
    meta: dict = field(default_factory=dict)


def build_system(energies: RydbergEnergies, symbol: str | None, n: int, l: int, j2: int, m2: int,
                 n_lo: int, n_hi: int, l_max: int, theta: float = THETA, N: int | None = None,
                 scale: float | None = None) -> ResonanceSystem:
    """Channels (l', j') with l' <= l_max and j' >= |m_j|; field-free levels moved to the precision
    energies for n' in [n_lo, n_hi + 6] (and every lower level of the channel the mesh holds)."""
    if abs(m2) > j2:
        raise ValueError("|m_j| > j")
    if not 0 < theta < 1.0:
        raise ValueError("θ must be between 0 and 1 rad")
    tgt = energies.level(n, l, j2)
    n_top = n_hi + 6
    N0, h0 = mesh_for(n_top)
    N = N or N0
    h = scale or h0
    x, T = laguerre_mesh(N)
    hc = h * np.exp(1j * theta)
    r = hc * x
    chans: list[Channel] = []
    for ll in range(l_max + 1):
        for jj in (2 * ll - 1, 2 * ll + 1):
            if jj < 1 or abs(m2) > jj:
                continue
            c = tune_core(energies, ll, jj, n, x, T, h) if symbol is not None and ll <= L_TUNE else 0.0
            V = soft_potential(r, energies.alpha_d if symbol else 0.0, c)
            H = T / (2 * hc * hc) + np.diag(ll * (ll + 1) / (2 * r * r) + V)
            w, v = eig(H)
            # bound states: (nearly) real and negative, and inside the mesh
            # well-resolved bound states only: (nearly) real, negative; box states and deep core-like
            # states are left alone (their c-norm is ill-conditioned or they do not matter)
            bound = np.flatnonzero((w.real < 0) & (np.abs(w.imag) < 1e-5 * np.abs(w.real))
                                   & (w.real > -0.5 / N_CORR_MIN ** 2))
            bound = bound[np.argsort(w[bound].real)]
            lev: list[Level] = []
            corr = np.zeros_like(H)
            tvec = None
            # assign by energy: the model's k-th bound level to the nearest precision level
            for b in bound:
                e_model = w[b].real
                nstar_model = 1.0 / np.sqrt(-2.0 * e_model)
                nn = int(round(nstar_model + _defect_guess(energies, ll, jj)))
                if nn <= ll or nn > n_top or not energies.exists(nn, ll):
                    continue
                lv = energies.level(nn, ll, jj)
                e_prec = _au(lv)
                if abs(e_prec - e_model) > 0.3 / lv.n_star ** 3:      # not the same level
                    continue
                phi = v[:, b] / np.sqrt(np.sum(v[:, b] ** 2))
                corr += (e_prec - w[b]) * np.outer(phi, phi)
                lev.append(lv)
                if (nn, ll, jj) == (n, l, j2):
                    tvec = phi
            chans.append(Channel(ll, jj, lev, H + corr, tvec))
    t_ch = [c for c in chans if (c.l, c.j2) == (l, j2)]
    if not t_ch or t_ch[0].target_vec is None:
        raise ValueError(f"{tgt.label} is not resolved on the mesh: raise N")
    meta = dict(N=N, scale_a0=h, theta=theta, r_max_a0=float(h * x[-1]), n_lo=n_lo, n_hi=n_hi, l_max=l_max)
    return ResonanceSystem(chans, x, hc, m2, tgt, _au(tgt), meta)


def _au(lv: Level) -> float:
    """Level energy in hartree (below the limit), from its binding energy."""
    return -lv.binding_cm * CM1_GHZ / HARTREE_GHZ


def soft_potential(r, alpha_d: float, c: float):
    """-1/r - α_d / (2 (r² + a²)²) - c e^(-r/a)/r with a = SOFT, a.u. (r may be complex)."""
    return -1.0 / r - alpha_d / (2.0 * (r * r + SOFT * SOFT) ** 2) - c * np.exp(-r / SOFT) / r


def _frac_defect(l: int, c: float, alpha_d: float, n_ref: float, x, T, h) -> float:
    """Signed fractional quantum defect of the channel's level nearest n_ref (real mesh)."""
    from scipy.linalg import eigh
    r = h * x
    w = eigh(T / (2 * h * h) + np.diag(l * (l + 1) / (2 * r * r) + soft_potential(r, alpha_d, c)),
             eigvals_only=True)
    w = w[w < 0]
    ns = 1.0 / np.sqrt(-2.0 * w)
    k = int(np.argmin(np.abs(ns - n_ref)))
    return float((-ns[k]) % 1.0)


def tune_core(energies: RydbergEnergies, l: int, j2: int, n: int, x, T, h) -> float:
    """c_lj that gives the channel the precision quantum defect (mod 1) near n."""
    from scipy.optimize import brentq
    n_ref = max(n, l + 2)
    if not energies.exists(n_ref, l):
        return 0.0
    target = energies.defect(n_ref, l, j2)[0] % 1.0
    ad = energies.alpha_d
    n_star = n_ref - energies.defect(n_ref, l, j2)[0]

    def g(c: float) -> float:
        d = _frac_defect(l, c, ad, n_star, x, T, h) - target
        return (d + 0.5) % 1.0 - 0.5
    g0 = g(0.0)
    if abs(g0) < 1e-9:
        return 0.0
    # δ grows with c (more attraction): walk towards the root and bracket it
    step = 0.25 if g0 < 0 else -0.25
    a, ga = 0.0, g0
    for _ in range(400):
        b = a + step
        gb = g(b)
        if ga * gb <= 0 and abs(ga - gb) < 0.5:          # a real crossing, not the wrap of δ mod 1
            return float(brentq(g, min(a, b), max(a, b), xtol=1e-12))
        a, ga = b, gb
    raise ValueError(f"could not tune the core of l = {l}, j = {j2}/2")


def _defect_guess(energies: RydbergEnergies, l: int, j2: int) -> float:
    try:
        return float(energies.defect(40, l, j2)[0])
    except Exception:
        return 0.0


def _groups(sys_: ResonanceSystem) -> list[list[int]]:
    """Channel indices grouped by l (the field couples a group only to its neighbours)."""
    out: dict[int, list[int]] = {}
    for i, c in enumerate(sys_.channels):
        out.setdefault(c.l, []).append(i)
    return [out[k] for k in sorted(out)]


def _blocks(sys_: ResonanceSystem, F_au: float):
    """Diagonal blocks D_g (one per l) and couplings C_g between groups g and g+1, dense."""
    ch = sys_.channels
    N = len(sys_.x)
    r = sys_.h * sys_.x
    groups = _groups(sys_)
    D, C = [], []
    for g, ids in enumerate(groups):
        B = np.zeros((len(ids) * N, len(ids) * N), complex)
        for a, i in enumerate(ids):
            B[a * N:(a + 1) * N, a * N:(a + 1) * N] = ch[i].H
        D.append(B)
        if g + 1 < len(groups):
            nxt = groups[g + 1]
            Cg = np.zeros((len(ids) * N, len(nxt) * N), complex)
            for a, i in enumerate(ids):
                for b, k in enumerate(nxt):
                    ang = angular_z(ch[i].l, ch[i].j2, ch[k].l, ch[k].j2, sys_.m2)
                    if ang:
                        Cg[a * N:(a + 1) * N, b * N:(b + 1) * N] = np.diag(F_au * ang * r)
            C.append(Cg)
    return D, C


class _ShiftInvert(sla.LinearOperator):
    """(H - σ)⁻¹ for the block-tridiagonal H (groups of l), by block elimination; H is complex
    symmetric.  S_0 = D_0 - σ, S_g = D_g - σ - C_(g-1)ᵀ S_(g-1)⁻¹ C_(g-1); the S_g⁻¹ are kept."""

    def __init__(self, D, C, sigma: complex):
        from scipy.linalg import inv
        self.sizes = [d.shape[0] for d in D]
        self.off = np.cumsum([0] + self.sizes)
        self.Ct = [sp.csr_matrix(c.T) for c in C]           # couplings are diagonal per channel pair
        self.Si: list[np.ndarray] = []
        self.G: list[np.ndarray] = []                        # G_g = S_g⁻¹ C_g
        for g in range(len(D)):
            S = D[g] - sigma * np.eye(self.sizes[g])
            if g:
                S -= C[g - 1].T @ self.G[g - 1]
            Si = inv(S, check_finite=False)
            self.Si.append(Si)
            if g < len(C):
                self.G.append(Si @ C[g])
        super().__init__(complex, (int(self.off[-1]), int(self.off[-1])))

    def _matvec(self, b):
        b = np.asarray(b, complex).ravel()
        n = len(self.sizes)
        z = []
        for g in range(n):                                     # forward: z_g = S_g⁻¹ y_g
            yg = b[self.off[g]:self.off[g + 1]]
            if g:
                yg = yg - self.Ct[g - 1] @ z[g - 1]
            z.append(self.Si[g] @ yg)
        x = [None] * n
        x[n - 1] = z[n - 1]
        for g in range(n - 2, -1, -1):                         # back: x_g = z_g - G_g x_(g+1)
            x[g] = z[g] - self.G[g] @ x[g + 1]
        return np.concatenate(x)


def _apply(D, C, v: np.ndarray) -> np.ndarray:
    """H @ v for the block-tridiagonal H of _blocks (v: one vector per column)."""
    sizes = [d.shape[0] for d in D]
    off = np.cumsum([0] + sizes)
    out = np.empty_like(v)
    for g in range(len(D)):
        a, b = off[g], off[g + 1]
        y = D[g] @ v[a:b]
        if g:
            y += C[g - 1].T @ v[off[g - 1]:a]
        if g < len(C):
            y += C[g] @ v[b:off[g + 2]]
        out[a:b] = y
    return out


def hamiltonian(sys_: ResonanceSystem, F_V_cm: float) -> sp.csr_matrix:
    """The whole complex-scaled Hamiltonian (a.u.), sparse."""
    D, C = _blocks(sys_, F_V_cm / AU_FIELD_V_CM)
    n = len(D)
    bl: list[list] = [[None] * n for _ in range(n)]
    for g in range(n):
        bl[g][g] = sp.csr_matrix(D[g])
        if g + 1 < n:
            bl[g][g + 1] = sp.csr_matrix(C[g])
            bl[g + 1][g] = sp.csr_matrix(C[g].T)
    return sp.bmat(bl, format="csr")


@dataclass(frozen=True)
class ResonanceMap:
    fields_V_cm: np.ndarray
    energies_ghz: np.ndarray          # [nF, K] positions relative to the chosen state's level (NaN padded)
    widths_ghz: np.ndarray            # [nF, K] Γ (full width)
    character: np.ndarray             # [nF, K] |φ_targetᵀ v|² (c-product, v c-normalised), clipped to [0, 1]


def _target_vector(sys_: ResonanceSystem) -> np.ndarray:
    ch = sys_.channels
    N = len(sys_.x)
    order = [i for g in _groups(sys_) for i in g]          # the block order of the matrix
    pos = {i: k for k, i in enumerate(order)}
    it = next(i for i, c in enumerate(ch) if (c.l, c.j2) == (sys_.target.l, sys_.target.j2))
    tv = np.zeros(len(order) * N, complex)
    tv[pos[it] * N:(pos[it] + 1) * N] = ch[it].target_vec
    return tv


def _solve_field(sys_: ResonanceSystem, tv: np.ndarray, f: float, W: float, kk: int):
    """Positions (GHz from the level), widths (GHz) and shares of the resonances at one field, and
    the number of eigenvalues to ask for at the next field."""
    E0 = sys_.E_target_au
    D, C = _blocks(sys_, f / AU_FIELD_V_CM)
    # the shift sits a little off the real axis: E0 itself is an exact eigenvalue of its
    # channel's block, which the block LU would have to invert
    sigma = E0 + W * (0.013 - 0.021j)
    op = _ShiftInvert(D, C, sigma)
    kk_i = min(kk, op.shape[0] - 2)
    # ask for the k eigenvalues nearest σ until they reach beyond every corner of the region
    # kept (|E - E0| <= W, Γ/2 <= 2W): then none in it is missing
    reach = abs(complex(W, 2 * W)) + abs(sigma - E0)
    while True:
        mu, v = sla.eigs(op, k=kk_i, which="LM", tol=1e-11)  # μ = 1/(E - σ)
        w = sigma + 1.0 / mu
        if np.max(np.abs(w - sigma)) >= reach or kk_i >= op.shape[0] - 2:
            break
        kk_i = min(int(kk_i * 1.6) + 10, op.shape[0] - 2)
    keep = (np.abs(w.real - E0) <= W) & (-w.imag <= 2 * W)
    # next field: as many as this one needed inside the reach, with a margin (k can fall)
    kk_next = max(12, int(1.3 * np.sum(np.abs(w - sigma) <= reach)) + 10)
    w, v = w[keep], v[:, keep]
    # the eigenvalues of the shift-inverted operator carry its rounding (~1 MHz at n = 30);
    # the Rayleigh quotient vᵀHv / vᵀv with the Hamiltonian itself is second order in the
    # eigenvector's error (one more inverse-iteration step moves it by ~1 kHz)
    if v.shape[1]:
        Hv = _apply(D, C, v)
        w = np.sum(v * Hv, axis=0) / np.sum(v * v, axis=0)
    nrm = np.sqrt(np.sum(v * v, axis=0))
    c = np.abs((tv @ v) / nrm) ** 2
    o = np.argsort(w.real)
    return ((w.real[o] - E0) * HARTREE_GHZ, np.maximum(-2.0 * w.imag[o], 0.0) * HARTREE_GHZ,
            np.clip(c[o], 0.0, 1.0), kk_next)


# one copy of the system per worker process, sent once when the worker starts
_WORKER: dict = {}


def _worker_init(sys_: ResonanceSystem, W: float, kk: int) -> None:
    _WORKER.update(sys=sys_, tv=_target_vector(sys_), W=W, kk=kk)


def _worker_field(i: int, f: float):
    e, g, c, kk = _solve_field(_WORKER["sys"], _WORKER["tv"], f, _WORKER["W"], _WORKER["kk"])
    _WORKER["kk"] = kk                      # the worker's next field starts from this one's need
    return i, e, g, c


def default_workers(n_fields: int) -> int:
    """Worker processes for a map of n_fields: every core but one, at most 16, none for a few fields.
    The packaged app runs on one: its executable is not set up to start worker processes."""
    import os
    from qali.core.paths import is_frozen_app
    if is_frozen_app() or n_fields < 4:
        return 1
    return max(1, min((os.cpu_count() or 1) - 1, 16, n_fields // 2))


def resonance_map(sys_: ResonanceSystem, fields_V_cm, window_ghz: float, k: int | None = None,
                  progress=None, workers: int = 1) -> ResonanceMap:
    """Positions and widths of the resonances within ± window of the chosen level at each field.

    workers > 1 solves the fields in that many processes (each gets one copy of the system, and
    BLAS runs on one thread in each, so the cores are not oversubscribed)."""
    F = np.asarray(fields_V_cm, float)
    W = window_ghz / HARTREE_GHZ
    E0 = sys_.E_target_au
    tv = _target_vector(sys_)
    # number of field-free levels in the window, a guide for how many eigenvalues to ask for
    nlev = sum(1 for c in sys_.channels for lv in c.levels if abs(_au(lv) - E0) <= W)
    kk = k or max(12, int(1.6 * nlev) + 10)
    rows: list = [None] * len(F)
    workers = max(1, min(int(workers), len(F)))
    if workers > 1:
        rows = _parallel(sys_, F, W, kk, workers, progress)
    else:
        for i, f in enumerate(F):
            e, g, c, kk = _solve_field(sys_, tv, f, W, kk)
            rows[i] = (e, g, c)
            if progress and (i % 10 == 0 or i == len(F) - 1):
                progress(f"    field {i + 1}/{len(F)}")
    width = max((len(r[0]) for r in rows), default=0)
    En = np.full((len(F), width), np.nan)
    Gm = np.full((len(F), width), np.nan)
    Ch = np.zeros((len(F), width))
    for i, (e, g, c) in enumerate(rows):
        En[i, :len(e)] = e
        Gm[i, :len(g)] = g
        Ch[i, :len(c)] = c
    return ResonanceMap(F, En, Gm, Ch)


def _parallel(sys_: ResonanceSystem, F: np.ndarray, W: float, kk: int, workers: int, progress) -> list:
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows: list = [None] * len(F)
    # one BLAS thread in each worker (read when the worker imports numpy)
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    for k in saved:
        os.environ[k] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(sys_, W, kk)) as ex:
            # interleaved, so each worker walks up the field and its eigenvalue count follows
            futs = [ex.submit(_worker_field, i, float(F[i])) for i in range(len(F))]
            for done, fut in enumerate(as_completed(futs), 1):
                i, e, g, c = fut.result()
                rows[i] = (e, g, c)
                if progress and (done % 10 == 0 or done == len(F)):
                    progress(f"    {done}/{len(F)} fields ({workers} processes)")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rows


def classical_limit_ghz(F_V_cm, E_target_au: float) -> np.ndarray:
    """Saddle-point energy -2√F (a.u.) of the Coulomb + field potential, relative to the state."""
    F = np.asarray(F_V_cm, float) / AU_FIELD_V_CM
    return (-2.0 * np.sqrt(F) - E_target_au) * HARTREE_GHZ
