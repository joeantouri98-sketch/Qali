"""Two Rydberg atoms in a static electric field: Förster maps and Stark-dressed pair states.

A field F along the quantisation axis z mixes each atom's levels of the same m_j (the Stark map of
``stark``); m_j stays good.  Two atoms in the field:

* Förster map: the energies E_i(F) + E_j(F) of every pair of Stark states, against F, beside the
  chosen pair's own, each with its dipole-dipole coupling to the chosen pair,

      C3_p = <i j| V R³ |a b> = Σ_{q1 q2} W_{q1 q2}(θ) <i|d_q1|a> <j|d_q2|b>   (GHz µm³),

  in the dressed states.  A Förster resonance is a field where a coupled pair crosses the chosen
  one: there the interaction becomes C3/R³ (resonant) instead of C6/R⁶.  This is the calculation of
  ARC's StarkMapResonances (Šibalić et al., CPC 220, 319 (2017)), with the coupling of the dressed
  states in place of ARC's rule that the main components differ by Δl = ±1.

* Dressed single states: the Stark eigenstates at one field that continue the levels with l ≤
  l_max (matched one-to-one by their overlap with those levels), with their energies and the dipole
  matrices between them.  These replace the field-free states in ``pair``: the pair potentials,
  C3 and C6 then include the field.  A field mixes parity, so pairs of either parity are coupled,
  and dressed states have permanent dipoles (a first-order C3 for the pair itself).  Each block of
  m_j is diagonalised in the full basis (l up to n - 1 by default), so the Stark shifts are those of
  the Stark map; only the matching of states to field-free labels uses l_max.

A magnetic field B along the same axis (``Atom(B_gauss=...)``) is added to every block of m_j: the
paramagnetic term with its j = l ± 1/2 mixing and the diamagnetic term (``zeeman``), so the
dressed states, pair potentials and Förster maps hold in E and B together.  m_j stays good; B
across the axis would mix it and is not done here (the Stark stage does crossed fields).

Units: GHz, V/cm, GHz µm³.  Engine contract: inputs as arguments, results as dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from qali.physics.atomic_structure.angular import reduced_c, threej2
from qali.physics.rydberg_stark.defects import Level
from qali.physics.rydberg_stark.pair import C3_UNIT, SingleStates, angular_weights
from qali.physics.rydberg_stark.stark import AU_FIELD_V_CM, CM1_GHZ, HARTREE_GHZ, _kappa, _z_matrix

F_UNIT = HARTREE_GHZ / AU_FIELD_V_CM          # GHz per (a.u. dipole × V/cm)


# ---------------------------------------------------------------- one atom, blocks of m_j

@dataclass
class Atom:
    """Levels of one atom with their radial dipole integrals (a.u.) and absolute energies (GHz)."""
    levels: list[Level]
    R_au: np.ndarray
    E_ghz: np.ndarray
    B_gauss: float = 0.0                        # magnetic field along the quantisation axis
    R2_au: np.ndarray | None = None             # <r²> integrals, for the diamagnetic term
    rydberg_cm: float | None = None             # the species' R_M, for g_L
    _mag: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_levels(cls, levels: list[Level], R_au: np.ndarray, B_gauss: float = 0.0,
                    R2_au: np.ndarray | None = None, rydberg_cm: float | None = None) -> Atom:
        if B_gauss and R2_au is None:
            raise ValueError("a magnetic field needs the <r²> integrals (build_basis(..., r2=True))")
        return cls(levels, R_au, np.array([-lv.binding_cm * CM1_GHZ for lv in levels]), float(B_gauss), R2_au,
                   rydberg_cm)

    def magnetic(self, m2: int) -> np.ndarray | None:
        """The Zeeman and diamagnetic terms in the block of m_j (GHz), or None without a field."""
        if not self.B_gauss:
            return None
        if m2 not in self._mag:
            from qali.physics.rydberg_stark import zeeman as zm
            k = self.block(m2)
            lv = [self.levels[i] for i in k]
            ms = np.full(len(k), m2)
            assert self.R2_au is not None
            self._mag[m2] = (zm.MU_B_GHZ_PER_G * self.B_gauss * zm.magnetic_matrix(lv, ms, 1.0, 0.0, zm.g_l(self.rydberg_cm))
                             + zm._dia_ghz(self.B_gauss) * zm.diamagnetic_matrix(lv, ms, self.R2_au[np.ix_(k, k)], 0.0))
        return self._mag[m2]

    def block(self, m2: int) -> np.ndarray:
        """Indices of the levels with a state of this m_j (j ≥ |m_j|)."""
        return np.array([i for i, lv in enumerate(self.levels) if lv.j2 >= abs(m2)], int)

    def z(self, m2: int) -> np.ndarray:
        k = self.block(m2)
        return _z_matrix([self.levels[i] for i in k], self.R_au[np.ix_(k, k)], m2)

    def d(self, m2: int, m2p: int) -> np.ndarray:
        """<i, m|d_q|k, m'> with q = m - m' (|q| ≤ 1), between the blocks of m and m'."""
        ki, kk = self.block(m2), self.block(m2p)
        q2 = m2 - m2p
        out = np.zeros((len(ki), len(kk)))
        if abs(q2) > 2:
            return out
        gi: dict[tuple[int, int], list[int]] = {}
        gk: dict[tuple[int, int], list[int]] = {}
        for r, i in enumerate(ki):
            gi.setdefault((self.levels[i].l, self.levels[i].j2), []).append(r)
        for r, i in enumerate(kk):
            gk.setdefault((self.levels[i].l, self.levels[i].j2), []).append(r)
        for (la, ja), ra in gi.items():
            for (lb, jb), rb in gk.items():
                if abs(la - lb) != 1:
                    continue
                c = reduced_c(_kappa(la, ja), _kappa(lb, jb), 1)
                if c == 0.0:
                    continue
                ph = -1.0 if ((ja - m2) // 2) % 2 else 1.0
                ang = ph * threej2(ja, 2, jb, -m2, q2, m2p) * c
                if ang:
                    out[np.ix_(ra, rb)] = ang * self.R_au[np.ix_(ki[ra], kk[rb])]
        return out

    def dress(self, m2: int, F_V_cm: float) -> tuple[np.ndarray, np.ndarray]:
        """Stark (and Zeeman) eigenstates of the block of m_j at F: energies (GHz, absolute) and vectors."""
        k = self.block(m2)
        H = np.diag(self.E_ghz[k]) + F_V_cm * F_UNIT * self.z(m2)
        mg = self.magnetic(m2)
        E, V = eigh(H if mg is None else H + mg)
        # eigh's signs are arbitrary (they differ between LAPACK builds): make each vector's largest
        # component positive, so couplings keep their sign from one field to the next
        big = V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])]
        return E, V * np.where(big < 0, -1.0, 1.0)


def _follow_level(V: np.ndarray, row: int) -> int:
    """The eigenstate with most of basis row `row`."""
    return int(np.argmax(V[row] ** 2))


# ---------------------------------------------------------------- Förster map

@dataclass(frozen=True)
class ForsterMap:
    fields_V_cm: np.ndarray                   # [nF]
    target_ghz: np.ndarray                    # [nF] chosen pair, relative to its field-free energy
    target_character: np.ndarray              # [nF] product of the two atoms' weights on their levels
    energies_ghz: list[np.ndarray]            # per field: coupled pairs within the window
    c3_ghz_um3: list[np.ndarray]              # their coupling to the chosen pair
    labels: list[list[str]]                   # their main components
    resonances: list[dict]                    # fields where a coupled pair crosses the chosen one


def _label(atom: Atom, k: np.ndarray, v: np.ndarray, m2: int) -> str:
    i = int(np.argmax(v ** 2))
    return f"{atom.levels[k[i]].label} {m2:+d}/2"


def forster_map(atom1: Atom, a: int, m2a: int, atom2: Atom, b: int, m2b: int, fields_V_cm,
                theta: float = 0.0, window_ghz: float = 1.0, c3_rel: float = 1e-3) -> ForsterMap:
    """Pair-state energies against field and their coupling to the chosen pair |a m2a, b m2b>.

    a, b: level indices in atom1.levels / atom2.levels.  Pairs are kept when they lie within
    ± window_ghz of the chosen pair at that field and couple to it by more than c3_rel of the
    strongest coupling on the map.  For two identical atoms in the same state, |i j> and |j i> are
    one exchange-symmetric pair, coupled by √2 <i j|V R³|a a>."""
    F = np.asarray(fields_V_cm, float)
    W = angular_weights(theta)
    qs = sorted({q for q, _ in W}), sorted({q for _, q in W})
    ka, kb = atom1.block(m2a), atom2.block(m2b)
    ra, rb = int(np.flatnonzero(ka == a)[0]), int(np.flatnonzero(kb == b)[0])
    blocks1 = {m2a + 2 * q: None for q in qs[0] if atom1.block(m2a + 2 * q).size}
    blocks2 = {m2b + 2 * q: None for q in qs[1] if atom2.block(m2b + 2 * q).size}
    d1 = {m: atom1.d(m, m2a) for m in blocks1}
    d2 = {m: atom2.d(m, m2b) for m in blocks2}
    E_pair0 = atom1.E_ghz[a] + atom2.E_ghz[b]
    same = atom1 is atom2 and a == b and m2a == m2b
    tgt, tch, rows_e, rows_c, rows_l = [], [], [], [], []
    seen: dict[tuple, tuple[float, float, float, str]] = {}     # key -> (F, defect, C3, label) at the last field
    res: list[dict] = []
    for f in F:
        Ea, Va = atom1.dress(m2a, f)
        Eb, Vb = atom2.dress(m2b, f)
        ia, ib = _follow_level(Va, ra), _follow_level(Vb, rb)
        va, vb = Va[:, ia], Vb[:, ib]
        Et = Ea[ia] + Eb[ib]
        tgt.append(Et - E_pair0)
        tch.append(float(va[ra] ** 2 * vb[rb] ** 2))
        dr1 = {m: atom1.dress(m, f) for m in blocks1}
        dr2 = {m: (dr1[m] if atom2 is atom1 and m in dr1 else atom2.dress(m, f)) for m in blocks2}
        es, cs, ls = [], [], []
        now: dict[tuple, tuple[float, float, float, str]] = {}
        for m1 in blocks1:
            E1, V1 = dr1[m1]
            x1 = V1.T @ d1[m1] @ va                    # <i|d_q1|a> for every dressed i
            for m2 in blocks2:
                w = W.get(((m1 - m2a) // 2, (m2 - m2b) // 2))
                if not w:
                    continue
                E2, V2 = dr2[m2]
                x2 = V2.T @ d2[m2] @ vb
                dE = E1[:, None] + E2[None, :] - Et
                C = w * C3_UNIT * np.outer(x1, x2)
                sel = np.argwhere((np.abs(dE) <= window_ghz) & (np.abs(C) > 1e-12))
                for i, j in sel.tolist():
                    if m1 == m2a and m2 == m2b and i == ia and j == ib:
                        continue
                    if same and (m1, i) > (m2, j):
                        continue                       # the swap of a pair already counted
                    if same and (m1, i) != (m2, j):
                        C[i, j] *= np.sqrt(2.0)
                    lab = f"{_label(atom1, atom1.block(m1), V1[:, i], m1)} + {_label(atom2, atom2.block(m2), V2[:, j], m2)}"
                    de, c3 = float(dE[i, j]), float(C[i, j])
                    es.append(de + Et - E_pair0)
                    cs.append(c3)
                    ls.append(lab)
                    key = (m1, m2, i, j)
                    now[key] = (float(f), de, c3, lab)
                    prev = seen.get(key)
                    if prev is not None and prev[1] * de < 0:
                        t = prev[1] / (prev[1] - de)
                        # |C3| interpolated, its sign from the nearer field (a sign flip of an
                        # eigenvector between the two fields must not average +C3 with -C3)
                        mag = abs(prev[2]) + t * (abs(c3) - abs(prev[2]))
                        sgn = np.sign(prev[2] if t < 0.5 else c3) or 1.0
                        res.append(dict(field_V_cm=prev[0] + t * (f - prev[0]), pair=lab,
                                        c3_ghz_um3=float(sgn * mag),
                                        slope_ghz_per_V_cm=(de - prev[1]) / (f - prev[0])))
        seen = now
        rows_e.append(np.array(es))
        rows_c.append(np.array(cs))
        rows_l.append(ls)
    cmax = max((float(np.max(np.abs(c))) for c in rows_c if len(c)), default=0.0)
    cut = c3_rel * cmax
    for k in range(len(F)):
        ok = np.abs(rows_c[k]) >= cut
        rows_e[k], rows_c[k] = rows_e[k][ok], rows_c[k][ok]
        rows_l[k] = [lab for lab, o in zip(rows_l[k], ok.tolist()) if o]
    res = [r for r in res if abs(r["c3_ghz_um3"]) >= cut]
    res.sort(key=lambda r: -abs(r["c3_ghz_um3"]))
    return ForsterMap(F, np.array(tgt), np.array(tch), rows_e, rows_c, rows_l, res)


# ---------------------------------------------------------------- dressed single states for ``pair``

@dataclass
class DressedStates(SingleStates):
    field_V_cm: float = 0.0
    B_gauss: float = 0.0
    character: np.ndarray | None = None        # weight of each dressed state on its field-free level


def dressed_states(atom: Atom, F_V_cm: float, keep_levels: list[int]) -> DressedStates:
    """Stark states at F that continue the field-free states of keep_levels (every m_j), with the
    dipole matrices between them, as a ``pair.SingleStates`` (levels = the kept levels)."""
    keep = sorted(keep_levels)
    pos = {g: r for r, g in enumerate(keep)}
    m2max = max(atom.levels[i].j2 for i in keep)
    blocks = list(range(-m2max, m2max + 1, 2))
    vecs: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}   # m -> (E, V columns, kept level per column)
    for m2 in blocks:
        k = atom.block(m2)
        rows = [r for r, g in enumerate(k) if g in pos]
        if not rows:
            continue
        E, V = atom.dress(m2, F_V_cm)
        # match the kept field-free states to eigenstates one to one, by their weight
        rr, cc = linear_sum_assignment(-(V[rows] ** 2))
        vecs[m2] = (E[cc], V[:, cc], [int(k[rows[r]]) for r in rr])
    states: list[tuple[int, int]] = []
    E_all, ch_all, where = [], [], []
    for m2, (E, V, lv) in vecs.items():
        for c, g in enumerate(lv):
            states.append((pos[g], m2))
            E_all.append(E[c])
            ch_all.append(float(V[np.flatnonzero(atom.block(m2) == g)[0], c] ** 2))
            where.append((m2, c))
    S = len(states)
    D = {q: np.zeros((S, S)) for q in (-1, 0, 1)}
    idx_of: dict[int, list[int]] = {}
    for s, (m2, _c) in enumerate(where):
        idx_of.setdefault(m2, []).append(s)
    for m2, ss in idx_of.items():
        cols_m = [where[s][1] for s in ss]
        Vm = vecs[m2][1][:, cols_m]
        for m2p, tt in idx_of.items():
            q2 = m2 - m2p
            if abs(q2) > 2:
                continue
            Vp = vecs[m2p][1][:, [where[t][1] for t in tt]]
            D[q2 // 2][np.ix_(ss, tt)] = Vm.T @ atom.d(m2, m2p) @ Vp
    levels = [atom.levels[g] for g in keep]
    return DressedStates(levels, states, np.array(E_all), D, field_V_cm=float(F_V_cm), B_gauss=atom.B_gauss,
                         character=np.array(ch_all))
