"""Radial wavefunctions and dipole integrals of Rydberg states (the ARC method).

Each state is integrated inward at its own energy E = -1/(2 n*²) (n* from ``defects``) in the
l-dependent Marinescu model potential plus the spin-orbit term α²/(4r³) [j(j+1) - l(l+1) - 3/4],
on a uniform grid in x = √r, which spaces the points like the local wavelength.  With
u(r) = r R(r) = x^½ χ(x),

    χ'' = [8x² (V(x²) - E) + (2l + ½)(2l + 3/2)/x²] χ.

The integration runs from r = 2n*(n* + 15) down to r_in = α_d^(1/3) (ARC's inner limit), and is
stopped where the solution starts to grow inward inside the inner classical turning point (the
energy is not an eigenvalue of the model potential, so the core region would diverge).  Dipole
integrals <a|r|b> = ∫ u_a u_b r dr = 2 ∫ x⁴ χ_a χ_b dx between normalised functions.

All states of one (l, j) are integrated together; the functions of a basis share one grid.
Hydrogen (no core) is the check: <n l|r|n l+1> = (3/2) n √(n² - (l+1)²).

Engine contract: inputs as arguments, results as arrays; no files, prints or plots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ALPHA_FS = 1.0 / 137.035999084
H_X = 0.01                 # grid step in √r (√a0)
RESCALE = 1e100


def potential(symbol: str | None, l: int, j: float, r: np.ndarray) -> np.ndarray:
    """Model potential V(r), a.u.; symbol None: hydrogen (-1/r)."""
    if symbol is None:
        V = -1.0 / r
    else:
        from qali.physics.orbital_radial import _vc
        V = _vc(symbol, l, r)
    if l > 0:
        V = V + ALPHA_FS ** 2 / (4 * r ** 3) * (j * (j + 1) - l * (l + 1) - 0.75)
    return V


def inner_limit(alpha_d: float) -> float:
    return max(alpha_d ** (1.0 / 3.0), 0.05)


def x_grid(n_star_max: float, r_in: float) -> np.ndarray:
    x_max = np.sqrt(2 * n_star_max * (n_star_max + 15.0))
    x0 = np.sqrt(r_in)
    return x0 + H_X * np.arange(int(np.ceil((x_max - x0) / H_X)) + 3)


def solve(symbol: str | None, l: int, j: float, n_stars: Sequence[float], x: np.ndarray) -> np.ndarray:
    """χ(x) of every n* of one (l, j) on grid x: [len(n_stars), len(x)], normalised ∫u² dr = 1."""
    ns = np.asarray(n_stars, float)
    N, G = len(ns), len(x)
    r = x * x
    V = potential(symbol, l, j, r)
    E = -0.5 / ns ** 2
    k2 = 8.0 * r[None, :] * (V[None, :] - E[:, None]) + (2 * l + 0.5) * (2 * l + 1.5) / r[None, :]
    f = 1.0 - H_X * H_X / 12.0 * k2                   # Numerov: f χ at three points
    chi = np.zeros((N, G))
    start = np.minimum(np.searchsorted(x, np.sqrt(2 * ns * (ns + 15.0))), G - 1)
    tp_in = ns * ns - ns * np.sqrt(np.maximum(ns * ns - l * (l + 1), 0.0))   # inner turning point
    for i in range(N):
        chi[i, start[i]] = 1e-10
        chi[i, start[i] - 1] = 1e-10 * (1.0 + 1e-6)
    top = int(start.max())
    active = np.zeros(N, bool)
    for k in range(top - 2, -1, -1):
        active |= start - 2 >= k
        if not active.any():
            continue
        nxt = ((12.0 - 10.0 * f[:, k + 1]) * chi[:, k + 1] - f[:, k + 2] * chi[:, k + 2]) / f[:, k]
        chi[:, k] = np.where(active & (start - 2 >= k), nxt, chi[:, k])
        big = np.abs(chi[:, k]) > RESCALE
        if big.any():
            chi[big, :] /= RESCALE
    # stop the inward divergence: inside the inner turning point, cut where |u| grows inward
    u = np.abs(chi) * np.sqrt(x)[None, :]
    for i in range(N):
        it = int(np.searchsorted(r, tp_in[i]))
        if it <= 1:
            continue
        seg = u[i, :it]
        grow = np.nonzero(seg[:-1] > seg[1:])[0]        # u decreasing outward = growing inward
        if grow.size:
            cut = grow[-1]
            chi[i, :cut + 1] = 0.0
    norm = 2.0 * H_X * np.sum(chi * chi * (x * x)[None, :], axis=1)
    chi /= np.sqrt(norm)[:, None]
    # one sign convention: positive just inside the outer turning point
    ref = np.minimum(np.searchsorted(r, 2.0 * ns * ns * 0.9), G - 1)
    sgn = np.sign(chi[np.arange(N), ref])
    sgn[sgn == 0] = 1.0
    return chi * sgn[:, None]


def moment(chi_a: np.ndarray, chi_b: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """<a|r^k|b> for stacks of χ on the same grid (u = χ √x, dr = 2x dx), a.u."""
    w = 2.0 * H_X * x ** (2 * k + 2)
    return (chi_a * w[None, :]) @ chi_b.T


def dipole(chi_a: np.ndarray, chi_b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """<a|r|b> for stacks of χ on the same grid: [len(a), len(b)], a.u."""
    w = 2.0 * H_X * x ** 4
    return (chi_a * w[None, :]) @ chi_b.T
