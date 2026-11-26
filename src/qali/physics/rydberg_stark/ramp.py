"""A Rydberg state carried through its Stark map by a field ramp: time-dependent Schrödinger
equation and the Landau-Zener picture.

The field rises linearly from F0 to F1 in a time T.  The state starts in the chosen level (at F0
the field-free level, or the eigenstate holding most of it when F0 > 0) and is propagated in the
basis of the static map,

    i ħ dc/dt = (H0 + F(t) Z) c,

by exponential midpoint steps exp(-i H(F_mid) dt / ħ) c (scipy expm_multiply on the sparse H;
H0 is diagonal and Z couples l to l ± 1 only).  The steps are short enough that the field moves
by less than a tenth of the narrowest crossing's width (gap / slope difference).

At each field of the map the population of every eigenstate in the window is |<μ|c>|²: a slow
ramp leaves the population on one adiabatic level (it goes round every avoided crossing), a fast
one carries it along the state's own character through the crossings (diabatic).  For each
avoided crossing of the followed state the Landau-Zener probability of the diabatic passage is

    P_LZ = exp(-π² Δ² / |dν/dt|),   Δ the gap and dν/dt = (slope difference) × dF/dt (in Hz),

against the share the propagation keeps on the state's own character there.

Units: energies GHz, fields V/cm, times ns (ν in GHz, so π² Δ² / (|ds| Ḟ) with Ḟ in V/cm/ns).
Engine contract: inputs as arguments, results as dataclasses; no files, prints or plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from qali.physics.rydberg_stark.stark import AU_FIELD_V_CM, HARTREE_GHZ, StarkBasis

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class RampResult:
    fields_V_cm: np.ndarray          # [nF] the map's fields
    energies_ghz: np.ndarray         # [nF, K] eigenvalues in the window (NaN padded)
    character: np.ndarray            # [nF, K] share of the chosen state in each
    population: np.ndarray           # [nF, K] population left by the ramp on each
    in_window: np.ndarray            # [nF] population inside the window
    ramp_ns: float
    slew_V_cm_ns: float
    steps: int


def landau_zener(gap_ghz: float, slope_diff_ghz_per_V_cm: float, slew_V_cm_ns: float) -> float:
    """Probability of the diabatic passage of an avoided crossing (full gap, frequency units)."""
    rate = abs(slope_diff_ghz_per_V_cm * slew_V_cm_ns)          # GHz/ns
    if rate == 0.0:
        return 0.0
    return float(np.exp(-np.pi ** 2 * gap_ghz ** 2 / rate))


def _hamiltonians(basis: StarkBasis):
    H0 = sp.csr_matrix(basis.h0())
    Z = sp.csr_matrix(basis.Z_au * HARTREE_GHZ / AU_FIELD_V_CM)        # GHz per V/cm
    Z.eliminate_zeros()
    return H0, Z


def ramp(basis: StarkBasis, fields_V_cm, ramp_ns: float, window_ghz: float | None = None,
         max_field_step: float | None = None, start: np.ndarray | None = None) -> RampResult:
    """Propagate the chosen state through a linear ramp over the map's fields (first to last).
    start: another initial state (basis coefficients)."""
    F = np.asarray(fields_V_cm, float)
    if len(F) < 2 or F[-1] <= F[0]:
        raise ValueError("the ramp needs at least two rising fields")
    if ramp_ns <= 0:
        raise ValueError("the ramp time must be positive")
    slew = (F[-1] - F[0]) / ramp_ns                                   # V/cm per ns
    H0, Z = _hamiltonians(basis)
    Hd = basis.h0()
    Zd = basis.Z_au * HARTREE_GHZ / AU_FIELD_V_CM
    W = window_ghz

    def eig_at(f):
        if W is None:
            return eigh(Hd + f * Zd)
        return eigh(Hd + f * Zd, subset_by_value=(-W, W), driver="evr")

    # start: the chosen level (adiabatically prepared at F0 > 0)
    tv = basis.tvec()
    if start is not None:
        c = np.asarray(start, complex) / np.linalg.norm(start)
    elif F[0] == 0.0:
        c = tv.astype(complex)
    else:
        w0, v0 = eigh(Hd + F[0] * Zd)
        c = v0[:, int(np.argmax((tv @ v0) ** 2))].astype(complex)
    dF = max_field_step or (F[1] - F[0]) / 4
    rows = []
    steps = 0
    for k in range(len(F)):
        if k:
            n_sub = max(1, int(np.ceil((F[k] - F[k - 1]) / dF)))
            h = (F[k] - F[k - 1]) / n_sub
            dt = h / slew                                             # ns
            for s in range(n_sub):
                fm = F[k - 1] + (s + 0.5) * h
                c = expm_multiply((-1j * TWO_PI * dt) * (H0 + fm * Z), c)
            steps += n_sub
        w, v = eig_at(F[k])
        rows.append((w, (tv @ v) ** 2, np.abs(v.T @ c) ** 2))
    width = max(len(r[0]) for r in rows)
    En = np.full((len(F), width), np.nan)
    Ch = np.zeros((len(F), width))
    Po = np.zeros((len(F), width))
    for k, (w, ch, po) in enumerate(rows):
        En[k, :len(w)] = w
        Ch[k, :len(w)] = ch
        Po[k, :len(w)] = po
    return RampResult(F, En, Ch, Po, Po.sum(axis=1), float(ramp_ns), float(slew), steps)


def sudden_width(slew_V_cm_ns: float, slope_diff_max: float, x_sudden: float = 0.01) -> float:
    """Width (V/cm) below which an avoided crossing is passed suddenly at this ramp rate: its
    Landau-Zener exponent π²Δ²/(Δs Ḟ) = π² w² Δs / Ḟ (w = Δ/Δs) is below x_sudden, so the
    diabatic passage is certain to 1 - x_sudden."""
    return float(np.sqrt(x_sudden * abs(slew_V_cm_ns) / abs(slope_diff_max)) / np.pi)


def ramp_adiabatic(basis: StarkBasis, fields_V_cm, ramp_ns: float, window_ghz: float,
                   field_step: float, band: float = 2.0, sub_band: float = 3.0,
                   start: np.ndarray | None = None) -> RampResult:
    """The same propagation in the instantaneous eigenbasis, for slow ramps.

    Each step from F to F + h: the amplitudes take their dynamical phase over half the step, are
    projected onto the eigenstates at F + h (the overlap matrix carries the non-adiabatic
    transitions), and take the other half (second order).  Its cost does not grow with the ramp
    time: only the field step matters.  Steps far shorter than a crossing's width follow it;
    crossings far narrower than the step are passed suddenly (diabatically), which is right when
    the step is below ``sudden_width``.

    Between two fields of the grid the eigenstates are found in the span of the eigenstates
    within ± sub_band × window at the first field (H = diag(E) + (F - F_k) Z there), and the
    amplitudes are carried back to the full basis at each grid field.  Eigenstates are followed
    within ± band × window of the chosen level; population that leaves that band is dropped
    (reported as the population left in the window)."""
    F = np.asarray(fields_V_cm, float)
    if len(F) < 2 or F[-1] <= F[0]:
        raise ValueError("the ramp needs at least two rising fields")
    if sub_band < band:
        raise ValueError("sub_band must be at least band")
    slew = (F[-1] - F[0]) / ramp_ns
    Hd = basis.h0()
    Zd = basis.Z_au * HARTREE_GHZ / AU_FIELD_V_CM
    Wb, Ws = band * window_ghz, sub_band * window_ghz

    def full(f, w):
        return eigh(Hd + f * Zd, subset_by_value=(-w, w), driver="evr")

    tv = basis.tvec()
    Es, B = full(F[0], Ws)                     # subspace at the start of the interval
    E, V = full(F[0], Wb)
    if start is not None:
        c = np.asarray(start, complex) / np.linalg.norm(start)
        a = V.T @ c
    elif F[0] == 0.0:
        a = (tv @ V).astype(complex)                   # <μ|target>
    else:
        a = np.zeros(len(E), complex)
        a[int(np.argmax((tv @ V) ** 2))] = 1.0
    rows = []
    steps = 0

    def record(E, V, a):
        inw = np.abs(E) <= window_ghz
        rows.append((E[inw], (tv @ V[:, inw]) ** 2, np.abs(a[inw]) ** 2))

    record(E, V, a)
    for k in range(1, len(F)):
        Zs = B.T @ Zd @ B                              # the field operator in the subspace
        U = B.T @ V                                    # the followed states in subspace coordinates
        n_sub = max(1, int(np.ceil((F[k] - F[k - 1]) / field_step)))
        h = (F[k] - F[k - 1]) / n_sub
        dt = h / slew
        for s_ in range(1, n_sub + 1):
            a = np.exp(-1j * np.pi * dt * E) * a       # half step of phase at the old energies
            if s_ < n_sub:
                E2, U2 = eigh(np.diag(Es) + (s_ * h) * Zs, subset_by_value=(-Wb, Wb), driver="evr")
                a = (U2.T @ U) @ a                     # projection onto the new eigenstates
                U = U2
            else:                                      # the grid field: back to the full basis
                E2, V2 = full(F[k], Wb)
                a = (V2.T @ (B @ U)) @ a
                V = V2
                Es, B = full(F[k], Ws)                 # the next interval's subspace
            a = np.exp(-1j * np.pi * dt * E2) * a      # the other half at the new energies
            E = E2
        steps += n_sub
        record(E, V, a)
    width = max(len(r[0]) for r in rows)
    En = np.full((len(F), width), np.nan)
    Ch = np.zeros((len(F), width))
    Po = np.zeros((len(F), width))
    for k, (w, ch, po) in enumerate(rows):
        En[k, :len(w)] = w
        Ch[k, :len(w)] = ch
        Po[k, :len(w)] = po
    return RampResult(F, En, Ch, Po, Po.sum(axis=1), float(ramp_ns), float(slew), steps)


def crossing_slopes(energies_ghz: np.ndarray, character: np.ndarray, fields: np.ndarray, k: int,
                    m: int = 3) -> float | None:
    """|slope difference| (GHz per V/cm) of the two levels meeting at the avoided crossing at index k:
    the level holding most of the state and its nearest neighbour, m fields before the crossing."""
    a = k - m
    if a < 1:
        return None
    E, C = energies_ghz, character
    i = int(np.nanargmax(C[a]))
    e = E[a, i]
    others = [j for j in range(E.shape[1]) if j != i and np.isfinite(E[a, j])]
    if not others:
        return None
    j = min(others, key=lambda q: abs(E[a, q] - e))
    # each level's slope from the field before, following the nearest level there
    def slope(idx):
        prev = E[a - 1]
        q = int(np.nanargmin(np.abs(prev - E[a, idx])))
        return (E[a, idx] - prev[q]) / (fields[a] - fields[a - 1])
    return float(abs(slope(i) - slope(j)))


def adiabatic_step(basis: StarkBasis, slew_V_cm_ns: float, grid_step: float, x_sudden: float = 0.02) -> float:
    """Field step for ``ramp_adiabatic``: a fifth of the narrowest crossing that is not passed
    suddenly at this slew (the widest slope difference in the basis is ~3 n_max² a.u.), and at most a
    quarter of the map's grid step.  A step near a crossing's width gets it wrong; a fifth gets
    Landau-Zener to ~1e-4."""
    n_max = max(lv.n for lv in basis.levels)
    ds_max = 3.0 * n_max ** 2 * HARTREE_GHZ / AU_FIELD_V_CM
    return min(sudden_width(slew_V_cm_ns, ds_max, x_sudden) / 5.0, grid_step / 4.0)


def max_step_for(crossings: list[tuple[float, float]], slope_diffs: list[float | None],
                 default: float) -> float:
    """Field step that resolves every crossing: a tenth of its width (gap / slope difference)."""
    widths = [g / d for (_f, g), d in zip(crossings, slope_diffs) if d]
    return min([default] + [w / 10 for w in widths if w > 0])
