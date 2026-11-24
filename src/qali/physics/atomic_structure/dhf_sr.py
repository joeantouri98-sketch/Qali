"""Structural radiation and normalisation: the third-order corrections to Brueckner E1 dipoles.

The Brueckner dipoles of ``brueckner`` (Brueckner orbitals + RPA) leave out two third-order
pieces (Johnson, Idrees & Sapirstein, PRA 35, 3218 (1987); Roberts, Fairhall & Ginges,
PRA 107, 052812 (2023), eqs. 5-6):

  structural radiation  the photon acting inside the correlation potential: Sigma(2) of
                        ``dhf_sigma`` differentiated with respect to the field.  Every internal
                        line takes one insertion: core-core and excited-excited (the core and
                        excited orbitals turned by the field, at E_w on one side and E_v on the
                        other) and core-excited (the RPA-dressed core response of ``dhf_rpa``,
                        replacing a core orbital or rotating an excited one into the core);
  normalisation         -(d/2) (<w|dSigma/dE|w> + <v|dSigma/dE|v>) in magnitude: the weight the
                        valence orbital loses to the correlated part of the state.

The m-scheme diagrams are reduced numerically: each angular factor is the explicit sum over
magnetic substates of 3j products (cached by the j's and multipoles), so no 6j algebra is
hand-coded.  Checks: inserting the unit operator gives SR = -dSigma/dE exactly, so SR + N = 0
(particle number); the <w|Sigma|w> of these sums equals ``sigma2_matrices``; w <-> v swapped
agree.  Against Roberts et al.'s SR + N (Cs, Rb, K): with lambda = 1 the weak lines come out ~15 %
above theirs (Cs 6s-7p +8.3 / +5.1 % against +6.8 / +4.2 %); scaled by the lambda of Sigma (the
default here: SR and N are derivatives of the scaled Sigma) +7.0 / +4.3 %.  On the resonance lines
SR and N nearly cancel (Cs D1: -0.01 % scaled, theirs -0.16 %).

Cost: the sums run over core x excited x excited states for every pair of reported levels; the
photon frequency enters through the core response and is interpolated on three nodes.  The whole
Brueckner stage at --n-max 25 on two cores: Na 3, K 6, Rb 12, Cs 18, Fr 32 minutes.

Engine contract: inputs as arguments, results as arrays; no files, prints or plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from qali.physics.atomic_structure.angular import reduced_c, threej2
from qali.physics.atomic_structure.dhf_rpa import CoreRPA
from qali.physics.atomic_structure.dhf_sigma import Brueckner, Orb, _j2
from qali.physics.atomic_structure.dirac_basis import psi_k

N_OMEGA = 3                  # photon-frequency nodes for the core-response (T) diagrams
_LET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ------------------------------------------------------------------ angular factors (m-sums)
def _ph(n: int) -> float:
    return -1.0 if n % 2 else 1.0


@lru_cache(None)
def _ct(j1: int, K: int, j2: int) -> np.ndarray:
    """[m1, Q, m2] = (-1)^(j1-m1) (j1 K j2; -m1 Q m2); j doubled, K integer."""
    A = np.zeros((j1 + 1, 2 * K + 1, j2 + 1))
    for a, m1 in enumerate(range(-j1, j1 + 1, 2)):
        for c, m2 in enumerate(range(-j2, j2 + 1, 2)):
            q2 = m1 - m2
            if abs(q2) <= 2 * K:
                A[a, q2 // 2 + K, c] = _ph((j1 - m1) // 2) * threej2(j1, 2 * K, j2, -m1, q2, m2)
    return A


@lru_cache(None)
def _ctneg(j1: int, K: int, j2: int) -> np.ndarray:
    """[m1, Q, m2] = (-1)^Q _ct[m1, -Q, m2] (the first operator of C^K(1).C^K(2))."""
    A = _ct(j1, K, j2)[:, ::-1, :].copy()
    for i, Q in enumerate(range(-K, K + 1)):
        A[:, i, :] *= _ph(Q)
    return A


@lru_cache(None)
def _external(jw: int, kt: int, jv: int) -> tuple[int, int, int, float]:
    """(m_w, q, m_v, (-1)^(jw-mw) (jw kt jv; -mw q mv)): the largest Wigner-Eckart factor."""
    best = (0, 0, 0, 0.0)
    for mw in range(-jw, jw + 1, 2):
        for q in range(-kt, kt + 1):
            mv = mw - 2 * q
            if abs(mv) <= jv:
                val = _ph((jw - mw) // 2) * threej2(jw, 2 * kt, jv, -mw, 2 * q, mv)
                if abs(val) > abs(best[3]) + 1e-12:
                    best = (mw, q, mv, val)
    return best


@lru_cache(None)
def angular(spec: tuple, js: tuple, Ks: tuple, kt: int) -> float:
    """Reduced angular factor of an m-scheme diagram.

    spec  factors ('g', i, j, k, l, K-index) for the Coulomb element g_ijkl and ('t', i, j) for the
          one-body operator <i|t_q|j> (rank kt); 'w' (bra) and 'v' (ket) are the external orbitals
    js    ((name, 2j), ...) for every orbital;  Ks  the multipoles the K-indices refer to
    Returns sum_m prod(3j factors) / (Wigner-Eckart factor of <w||t||v>).  Reduced C^K factors
    and radial integrals are left to the caller."""
    jd = dict(js)
    mw, q, mv, ext = _external(jd['w'], kt, jd['v'])
    if ext == 0.0:
        return 0.0
    names = sorted({x for f in spec for x in f[1:5] if isinstance(x, str)} | {'w', 'v'})
    let = {nm: _LET[i] for i, nm in enumerate(names)}
    nq = len(names)
    ops: list[np.ndarray] = []
    subs: list[str] = []
    for f in spec:
        if f[0] == 'g':
            _, i, j, k, l, kk = f
            K = Ks[kk]
            Q = _LET[nq]
            nq += 1
            ops += [_ctneg(jd[i], K, jd[k]), _ct(jd[j], K, jd[l])]
            subs += [let[i] + Q + let[k], let[j] + Q + let[l]]
        else:
            _, i, j = f
            ops.append(_ct(jd[i], kt, jd[j])[:, q + kt, :])
            subs.append(let[i] + let[j])
    ew = np.zeros(jd['w'] + 1)
    ew[(mw + jd['w']) // 2] = 1.0
    ev = np.zeros(jd['v'] + 1)
    ev[(mv + jd['v']) // 2] = 1.0
    val = np.einsum(','.join(subs + [let['w'], let['v']]) + '->', *ops, ew, ev, optimize='greedy')
    return float(val) / ext


def _js(**kw: int) -> tuple:
    return tuple(sorted(kw.items()))


# ------------------------------------------------------------------ orbitals
@dataclass
class Stack:
    """Valence orbitals of one kappa projected on the excited MBPT states (the space Sigma acts in)."""
    kappa: int
    U: np.ndarray            # [n, p]: <MBPT p | orbital>
    P: np.ndarray            # [n, grid] on the MBPT grid
    Q: np.ndarray
    E: np.ndarray            # [n] energies (a.u.)


@dataclass
class _Pert:
    """A core orbital's first-order response, one kappa channel, on the MBPT grid."""
    kappa: int
    P: np.ndarray
    Q: np.ndarray


class ThirdOrder:
    """Structural radiation and dSigma/dE for reduced E1 elements between Brueckner orbitals.

    br     the Brueckner model (its MBPT basis and the Sigma(2) matrices at the energy nodes)
    rpa    the core RPA response on the same basis (dressed core-excited amplitudes)
    on_v   {id(MBPT orbital): (P, Q) on the valence grid} (``ValenceE1.on_v``), bv that grid
    Reduced elements follow the ``reduced_c`` phase convention of the rest of the engine."""

    def __init__(self, br: Brueckner, rpa: CoreRPA, on_v: dict, bv):
        mb = self.mb = br.mb
        self.br, self.rpa, self.on_v, self.bv = br, rpa, on_v, bv
        self.wq, r = mb.w, mb.b.r
        self.core = mb.core
        self.kaps = mb.kappas
        self.kmax = 2 * mb.lmax + 3
        self.ex = {k: mb.excited(k) for k in self.kaps}
        self.PE = {k: np.array([o.P for o in v]) for k, v in self.ex.items()}
        self.QE = {k: np.array([o.Q for o in v]) for k, v in self.ex.items()}
        self.eE = {k: np.array([o.eps for o in v]) for k, v in self.ex.items()}
        self._Y: dict = {}
        self._Tee: dict = {}
        self._Y1: dict = {}
        self.idx: dict = {}
        for i, (b, n) in enumerate(rpa.pairs):
            self.idx.setdefault((id(b), n.kappa), []).append(i)
        self.r = r

    # ---------------- orbitals and radial pieces
    def stack(self, kappa: int, orbs, energies) -> Stack:
        """Project valence-grid orbitals (P, Q) of one kappa on the excited MBPT states."""
        on = [self.on_v[id(p)] for p in self.ex[kappa]]
        U = np.array([[self.bv.integrate(Pp * P + Qp * Q) for Pp, Qp in on] for P, Q in orbs])
        U = U.reshape(len(orbs), len(on))
        return Stack(kappa, U, U @ self.PE[kappa], U @ self.QE[kappa], np.asarray(energies, float))

    def dsigma(self, s: Stack) -> np.ndarray:
        """<w|dSigma(2)/dE|w> (m-scheme, lambda = 1) for every orbital of the stack, from the
        Sigma matrices at the energy nodes (the interpolant Brueckner levels are solved with)."""
        nodes, S, _O = self.br.sig[s.kappa]
        n = len(nodes)
        out = np.empty(len(s.E))
        for i, E in enumerate(s.E):
            E = float(np.clip(E, nodes.min(), nodes.max()))
            wd = np.zeros(n)
            for a in range(n):
                for k in range(n):
                    if k != a:
                        wd[a] += np.prod([(E - nodes[j]) / (nodes[a] - nodes[j])
                                          for j in range(n) if j not in (a, k)]) / (nodes[a] - nodes[k])
            u = s.U[i]
            out[i] = float(u @ np.tensordot(wd, S, axes=1) @ u)
        return out

    def _Yc(self, b: Orb, kx: int, K: int) -> np.ndarray:
        """psi_K(rho_{b x}) for core b and every excited x of kappa kx: [nx, grid] (cached)."""
        key = (id(b), kx, K)
        if key not in self._Y:
            self._Y[key] = psi_k(self.mb.b, self.PE[kx] * b.P + self.QE[kx] * b.Q, K)
        return self._Y[key]

    def _Ycc(self, a: Orb, b: Orb, K: int) -> np.ndarray:
        key = (id(a), id(b), K)
        if key not in self._Y1:
            self._Y1[key] = psi_k(self.mb.b, (a.P * b.P + a.Q * b.Q)[None], K)[0]
        return self._Y1[key]

    def _Yp(self, d: _Pert, kx: int, K: int) -> np.ndarray:
        return psi_k(self.mb.b, self.PE[kx] * d.P + self.QE[kx] * d.Q, K)

    def _rhoS(self, s, kx: int) -> np.ndarray:
        """rho_{s x} * w for the stack's orbitals s and excited x: [ns, nx, grid]."""
        return (s.P[:, None, :] * self.PE[kx][None] + s.Q[:, None, :] * self.QE[kx][None]) * self.wq

    def _rho1S(self, s, o) -> np.ndarray:
        """rho_{s o} * w: [ns, grid] for a stack and one orbital."""
        return (s.P * o.P + s.Q * o.Q) * self.wq

    def _Y1S(self, o, s, K: int) -> np.ndarray:
        """psi_K(rho_{o s}) for every orbital of the stack: [ns, grid]."""
        return psi_k(self.mb.b, s.P * o.P + s.Q * o.Q, K)

    def _rhoE(self, o, kx: int) -> np.ndarray:
        return (self.PE[kx] * o.P + self.QE[kx] * o.Q) * self.wq

    def _ks(self, ka, kc, kb, kd) -> list[int]:
        return [K for K in range(self.kmax) if reduced_c(ka, kc, K) and reduced_c(kb, kd, K)]

    @staticmethod
    def _cg(ka, kc, kb, kd, K) -> float:
        return reduced_c(ka, kc, K) * reduced_c(kb, kd, K)

    def _t_ee(self, k1: int, k2: int):
        """bare reduced <i||d||j> between excited states (None if forbidden)."""
        key = (k1, k2)
        if key not in self._Tee:
            c = reduced_c(k1, k2)
            self._Tee[key] = None if c == 0.0 else c * (
                (self.PE[k1] * self.r * self.wq) @ self.PE[k2].T + (self.QE[k1] * self.r * self.wq) @ self.QE[k2].T)
        return self._Tee[key]

    def _t_cc(self, a: Orb, b: Orb) -> float:
        c = reduced_c(a.kappa, b.kappa)
        return 0.0 if c == 0.0 else c * float(np.sum(self.wq * self.r * (a.P * b.P + a.Q * b.Q)))

    def _amp(self, xy: np.ndarray, c: Orb, kr: int) -> np.ndarray:
        ii = self.idx.get((id(c), kr))
        return np.zeros(len(self.ex[kr])) if ii is None else xy[ii]

    # ---------------- structural radiation
    def structural_radiation(self, sw: Stack, sv: Stack) -> np.ndarray:
        """SR[w, v]: reduced <w||SR||v> (lambda = 1) for every pair, each at its own photon
        frequency omega = E_w - E_v (core response interpolated over N_OMEGA nodes)."""
        om = sw.E[:, None] - sv.E[None, :]
        lo, hi = float(om.min()), float(om.max())
        if hi - lo < 1e-6:
            nodes = np.array([lo])
        else:
            nodes = (lo + hi) / 2 + (hi - lo) / 2 * np.cos(np.pi * (np.arange(N_OMEGA) + 0.5) / N_OMEGA)
        base, tparts = self._diagrams(sw, sv, nodes)
        out = base.copy()
        for i, tn in enumerate(tparts):
            wt = np.ones_like(om)
            for j in range(len(nodes)):
                if j != i:
                    wt *= (om - nodes[j]) / (nodes[i] - nodes[j])
            out += wt * tn
        return out

    def _diagrams(self, sw: Stack, sv: Stack, nodes) -> tuple[np.ndarray, list[np.ndarray]]:
        kw, kv = sw.kappa, sv.kappa
        jw, jv = _j2(kw), _j2(kv)
        nw, nv = len(sw.E), len(sv.E)
        Ew, Ev = sw.E, sv.E
        kaps, core = self.kaps, self.core
        eE = self.eE
        base = np.zeros((nw, nv))
        tn = [np.zeros((nw, nv)) for _ in nodes]
        sols = [self.rpa._solve(float(o)) for o in nodes]          # (x, y) at each node

        def A(spec, js, Ks):
            return angular(spec, js + (('w', jw), ('v', jv)), Ks, 1)

        def mm(L, R):
            return L.reshape(nw, -1) @ R.reshape(nv, -1).T

        rhoW = {k: self._rhoS(sw, k) for k in kaps}
        rhoV = {k: self._rhoS(sv, k) for k in kaps}
        pert = []                                                   # per node: (dX, dY)
        for o, (x, y) in zip(nodes, sols):
            dX, dY = {}, {}
            for c in core:
                for kr in kaps:
                    if reduced_c(kr, c.kappa) == 0.0:
                        continue
                    cx = self._amp(x, c, kr) / (c.eps - eE[kr] + o)
                    cy = self._amp(y, c, kr) / (c.eps - eE[kr] - o)
                    dX[(id(c), kr)] = _Pert(kr, cx @ self.PE[kr], cx @ self.QE[kr])
                    dY[(id(c), kr)] = _Pert(kr, cy @ self.PE[kr], cy @ self.QE[kr])
            pert.append((dX, dY))

        # ================= term 1: sum_bmn g_wbmn g~_mnvb / (E + e_b - e_m - e_n)
        def term1_arrays(b):
            kb = b.kappa
            LD, RD = {}, {}
            for km in kaps:
                for kn in kaps:
                    den = b.eps - eE[km][:, None] - eE[kn][None, :]
                    Ks = self._ks(kw, km, kb, kn)
                    if Ks:
                        Dw = Ew[:, None, None] + den[None]
                        for K in Ks:
                            LD[(km, kn, K)] = self._cg(kw, km, kb, kn, K) * (rhoW[km] @ self._Yc(b, kn, K).T) / Dw
                    Dv = Ev[:, None, None] + den[None]
                    lst = []
                    for K in self._ks(km, kv, kn, kb):
                        lst.append((K, 1.0, ('g', 'm', 'n', 'v', 'b', 1),
                                    self._cg(km, kv, kn, kb, K) * (rhoV[km] @ self._Yc(b, kn, K).T) / Dv))
                    for K in self._ks(kn, kv, km, kb):
                        lst.append((K, -1.0, ('g', 'n', 'm', 'v', 'b', 1), self._cg(kn, kv, km, kb, K)
                                    * np.transpose(rhoV[kn] @ self._Yc(b, km, K).T, (0, 2, 1)) / Dv))
                    if lst:
                        RD[(km, kn)] = lst
            return LD, RD

        for b in core:
            kb, jb = b.kappa, _j2(b.kappa)
            LD, RD = term1_arrays(b)
            for (km, kn), rlist in RD.items():
                jm, jn = _j2(km), _j2(kn)
                # Dm: g(w b m' n) t(m' m) g~(m n v b) / (D_w D_v)
                for kmp in kaps:
                    T = self._t_ee(kmp, km)
                    if T is None:
                        continue
                    js = _js(b=jb, mp=_j2(kmp), m=jm, n=jn)
                    for K2, sg, tail, R in rlist:
                        TR = None
                        for K1 in self._ks(kw, kmp, kb, kn):
                            a = A((('g', 'w', 'b', 'mp', 'n', 0), ('t', 'mp', 'm'), tail), js, (K1, K2))
                            if a:
                                if TR is None:
                                    TR = np.einsum('pm,vmn->vpn', T, R)
                                base += sg * a * mm(LD[(kmp, kn, K1)], TR)
                # Dn: g(w b m n') t(n' n) g~(m n v b)
                for knp in kaps:
                    T = self._t_ee(knp, kn)
                    if T is None:
                        continue
                    js = _js(b=jb, np=_j2(knp), m=jm, n=jn)
                    for K2, sg, tail, R in rlist:
                        RT = None
                        for K1 in self._ks(kw, km, kb, knp):
                            a = A((('g', 'w', 'b', 'm', 'np', 0), ('t', 'np', 'n'), tail), js, (K1, K2))
                            if a:
                                if RT is None:
                                    RT = R @ T.T
                                base += sg * a * mm(LD[(km, knp, K1)], RT)
                for i, (dX, dY) in enumerate(pert):
                    x, y = sols[i]
                    o = nodes[i]
                    # T1bL: g(w r m n) t(b r) g~(m n v b)       (core bra on the left, Y response)
                    for kr in kaps:
                        d = dY.get((id(b), kr))
                        if d is None:
                            continue
                        js = _js(b=jb, r=_j2(kr), m=jm, n=jn)
                        for K1 in self._ks(kw, km, kr, kn):
                            Lp = None
                            for K2, sg, tail, R in rlist:
                                a = A((('g', 'w', 'r', 'm', 'n', 0), ('t', 'b', 'r'), tail), js, (K1, K2))
                                if a:
                                    if Lp is None:
                                        Lp = self._cg(kw, km, kr, kn, K1) * (rhoW[km] @ self._Yp(d, kn, K1).T)
                                    tn[i] += sg * a * mm(Lp, R)
                    for a_ in core:
                        ka, ja = a_.kappa, _j2(a_.kappa)
                        js = _js(b=jb, a=ja, m=jm, n=jn)
                        # T1mL: g(w b a n) t(a m) g~(m n v b), y_am / (e_m - e_a + w)
                        if reduced_c(ka, km):
                            cm = self._amp(y, a_, km) / (eE[km] - a_.eps + o)
                            for K2, sg, tail, R in rlist:
                                Rc = None
                                for K1 in self._ks(kw, ka, kb, kn):
                                    a = A((('g', 'w', 'b', 'a', 'n', 0), ('t', 'a', 'm'), tail), js, (K1, K2))
                                    if a:
                                        if Rc is None:
                                            Rc = np.einsum('m,vmn->vn', cm, R)
                                        Ln = self._cg(kw, ka, kb, kn, K1) * (self._rho1S(sw, a_) @ self._Yc(b, kn, K1).T)
                                        tn[i] += sg * a * (Ln @ Rc.T)
                        # T1nL: g(w b m a) t(a n) g~(m n v b), y_an / (e_n - e_a + w)
                        if reduced_c(ka, kn):
                            cn = self._amp(y, a_, kn) / (eE[kn] - a_.eps + o)
                            for K2, sg, tail, R in rlist:
                                Rc = None
                                for K1 in self._ks(kw, km, kb, ka):
                                    a = A((('g', 'w', 'b', 'm', 'a', 0), ('t', 'a', 'n'), tail), js, (K1, K2))
                                    if a:
                                        if Rc is None:
                                            Rc = R @ cn
                                        Lm = self._cg(kw, km, kb, ka, K1) * (rhoW[km] @ self._Ycc(b, a_, K1))
                                        tn[i] += sg * a * (Lm @ Rc.T)
            # right-side core-response diagrams use L/D_w with a perturbed right vertex
            for (km, kn, K1), L in LD.items():
                jm, jn = _j2(km), _j2(kn)
                for i, (dX, dY) in enumerate(pert):
                    x, y = sols[i]
                    o = nodes[i]
                    # T1bR: g(w b m n) g~(m n v r) t(r b)     (core ket on the right, X response)
                    for kr in kaps:
                        d = dX.get((id(b), kr))
                        if d is None:
                            continue
                        js = _js(b=jb, r=_j2(kr), m=jm, n=jn)
                        for K in self._ks(km, kv, kn, kr):
                            a = A((('g', 'w', 'b', 'm', 'n', 0), ('g', 'm', 'n', 'v', 'r', 1), ('t', 'r', 'b')), js, (K1, K))
                            if a:
                                R = self._cg(km, kv, kn, kr, K) * (rhoV[km] @ self._Yp(d, kn, K).T)
                                tn[i] += a * mm(L, R)
                        for K in self._ks(kn, kv, km, kr):
                            a = A((('g', 'w', 'b', 'm', 'n', 0), ('g', 'n', 'm', 'v', 'r', 1), ('t', 'r', 'b')), js, (K1, K))
                            if a:
                                R = self._cg(kn, kv, km, kr, K) * np.transpose(rhoV[kn] @ self._Yp(d, km, K).T, (0, 2, 1))
                                tn[i] -= a * mm(L, R)
                    for a_ in core:
                        ka, ja = a_.kappa, _j2(a_.kappa)
                        js = _js(b=jb, a=ja, m=jm, n=jn)
                        # T1mR: g(w b m n) t(m a) g~(a n v b), x_ma / (e_m - e_a - w)
                        if reduced_c(km, ka):
                            cm = self._amp(x, a_, km) / (eE[km] - a_.eps - o)
                            Lc = None
                            for K, sg, tail in ((K, 1.0, ('g', 'a', 'n', 'v', 'b', 1)) for K in self._ks(ka, kv, kn, kb)):
                                a = A((('g', 'w', 'b', 'm', 'n', 0), ('t', 'm', 'a'), tail), js, (K1, K))
                                if a:
                                    Lc = np.einsum('wmn,m->wn', L, cm) if Lc is None else Lc
                                    Rn = self._cg(ka, kv, kn, kb, K) * (self._rho1S(sv, a_) @ self._Yc(b, kn, K).T)
                                    tn[i] += a * (Lc @ Rn.T)
                            for K in self._ks(kn, kv, ka, kb):
                                a = A((('g', 'w', 'b', 'm', 'n', 0), ('t', 'm', 'a'), ('g', 'n', 'a', 'v', 'b', 1)), js, (K1, K))
                                if a:
                                    Lc = np.einsum('wmn,m->wn', L, cm) if Lc is None else Lc
                                    Rn = self._cg(kn, kv, ka, kb, K) * (rhoV[kn] @ self._Ycc(a_, b, K))
                                    tn[i] -= a * (Lc @ Rn.T)
                        # T1nR: g(w b m n) t(n a) g~(m a v b), x_na / (e_n - e_a - w)
                        if reduced_c(kn, ka):
                            cn = self._amp(x, a_, kn) / (eE[kn] - a_.eps - o)
                            Lc = None
                            for K in self._ks(km, kv, ka, kb):
                                a = A((('g', 'w', 'b', 'm', 'n', 0), ('t', 'n', 'a'), ('g', 'm', 'a', 'v', 'b', 1)), js, (K1, K))
                                if a:
                                    Lc = L @ cn if Lc is None else Lc
                                    Rm = self._cg(km, kv, ka, kb, K) * (rhoV[km] @ self._Ycc(a_, b, K))
                                    tn[i] += a * (Lc @ Rm.T)
                            for K in self._ks(ka, kv, km, kb):
                                a = A((('g', 'w', 'b', 'm', 'n', 0), ('t', 'n', 'a'), ('g', 'a', 'm', 'v', 'b', 1)), js, (K1, K))
                                if a:
                                    Lc = L @ cn if Lc is None else Lc
                                    Rm = self._cg(ka, kv, km, kb, K) * (self._rho1S(sv, a_) @ self._Yc(b, km, K).T)
                                    tn[i] -= a * (Lc @ Rm.T)

        # Cb: - g(w b' m n) t(b b') g~(m n v b): core-core insertion, pairs of core orbitals
        pairs = [(b, bp, self._t_cc(b, bp)) for b in core for bp in core]
        pairs = [p for p in pairs if p[2] != 0.0]
        if pairs:
            for km in kaps:
                for kn in kaps:
                    Ls, Rs = {}, {}
                    for b in core:
                        den = b.eps - eE[km][:, None] - eE[kn][None, :]
                        for K in self._ks(kw, km, b.kappa, kn):
                            Ls[(id(b), K)] = (self._cg(kw, km, b.kappa, kn, K) * (rhoW[km] @ self._Yc(b, kn, K).T)
                                              / (Ew[:, None, None] + den[None]))
                        Dv = Ev[:, None, None] + den[None]
                        lst = []
                        for K in self._ks(km, kv, kn, b.kappa):
                            lst.append((K, 1.0, ('g', 'm', 'n', 'v', 'b', 1),
                                        self._cg(km, kv, kn, b.kappa, K) * (rhoV[km] @ self._Yc(b, kn, K).T) / Dv))
                        for K in self._ks(kn, kv, km, b.kappa):
                            lst.append((K, -1.0, ('g', 'n', 'm', 'v', 'b', 1), self._cg(kn, kv, km, b.kappa, K)
                                        * np.transpose(rhoV[kn] @ self._Yc(b, km, K).T, (0, 2, 1)) / Dv))
                        Rs[id(b)] = lst
                    for b, bp, T in pairs:
                        js = _js(bp=_j2(bp.kappa), b=_j2(b.kappa), m=_j2(km), n=_j2(kn))
                        for K1 in self._ks(kw, km, bp.kappa, kn):
                            for K2, sg, tail, R in Rs[id(b)]:
                                a = A((('g', 'w', 'bp', 'm', 'n', 0), ('t', 'b', 'bp'), tail), js, (K1, K2))
                                if a:
                                    base -= sg * a * T * mm(Ls[(id(bp), K1)], R)

        # ================= term 2: sum_abm g_wmab g~_abvm / (E + e_m - e_a - e_b)
        Y1V = {}

        def y1v(o, K):
            key = (id(o), K)
            if key not in Y1V:
                Y1V[key] = self._Y1S(o, sv, K)
            return Y1V[key]

        for a_ in core:
            ka, ja = a_.kappa, _j2(a_.kappa)
            rwa = self._rho1S(sw, a_)
            rav = self._rho1S(sv, a_)
            for b in core:
                kb, jb = b.kappa, _j2(b.kappa)
                for km in kaps:
                    jm = _j2(km)
                    den = eE[km] - a_.eps - b.eps
                    Dw = Ew[:, None] + den[None]
                    Dv = Ev[:, None] + den[None]
                    LDs = {K: self._cg(kw, ka, km, kb, K) * (rwa @ self._Yc(b, km, K).T) / Dw
                           for K in self._ks(kw, ka, km, kb)}
                    RDs = [(K, 1.0, ('g', 'a', 'b', 'v', 'm', 1),
                            self._cg(ka, kv, kb, km, K) * (rav @ self._Yc(b, km, K).T) / Dv)
                           for K in self._ks(ka, kv, kb, km)]
                    RDs += [(K, -1.0, ('g', 'a', 'b', 'm', 'v', 1),
                             self._cg(ka, km, kb, kv, K) * (y1v(b, K) @ self._rhoE(a_, km).T) / Dv)
                            for K in self._ks(ka, km, kb, kv)]
                    js0 = _js(a=ja, b=jb, m=jm)
                    # Mm: - g(w m' a b) t(m m') g~(a b v m)
                    for kmp in kaps:
                        T = self._t_ee(km, kmp)
                        if T is None:
                            continue
                        js = _js(a=ja, b=jb, mp=_j2(kmp), m=jm)
                        den2 = eE[kmp] - a_.eps - b.eps
                        for K1 in self._ks(kw, ka, kmp, kb):
                            L = None
                            for K2, sg, tail, R in RDs:
                                a = A((('g', 'w', 'mp', 'a', 'b', 0), ('t', 'm', 'mp'), tail), js, (K1, K2))
                                if a:
                                    if L is None:
                                        L = (self._cg(kw, ka, kmp, kb, K1) * (rwa @ self._Yc(b, kmp, K1).T)
                                             / (Ew[:, None] + den2[None]))
                                    base -= sg * a * (L @ T.T @ R.T)
                    # Ma: + g(w m a' b) t(a' a) g~(a b v m);  Mb: + g(w m a b') t(b' b) g~(a b v m)
                    for cp in core:
                        Ta = self._t_cc(cp, a_)
                        if Ta != 0.0:
                            js = _js(ap=_j2(cp.kappa), a=ja, b=jb, m=jm)
                            Dw2 = Ew[:, None] + (eE[km] - cp.eps - b.eps)[None]
                            rwc = self._rho1S(sw, cp)
                            for K1 in self._ks(kw, cp.kappa, km, kb):
                                L = None
                                for K2, sg, tail, R in RDs:
                                    a = A((('g', 'w', 'm', 'ap', 'b', 0), ('t', 'ap', 'a'), tail), js, (K1, K2))
                                    if a:
                                        if L is None:
                                            L = self._cg(kw, cp.kappa, km, kb, K1) * (rwc @ self._Yc(b, km, K1).T) / Dw2
                                        base += sg * a * Ta * (L @ R.T)
                        Tb = self._t_cc(cp, b)
                        if Tb != 0.0:
                            js = _js(bp=_j2(cp.kappa), a=ja, b=jb, m=jm)
                            Dw2 = Ew[:, None] + (eE[km] - a_.eps - cp.eps)[None]
                            for K1 in self._ks(kw, ka, km, cp.kappa):
                                L = None
                                for K2, sg, tail, R in RDs:
                                    a = A((('g', 'w', 'm', 'a', 'bp', 0), ('t', 'bp', 'b'), tail), js, (K1, K2))
                                    if a:
                                        if L is None:
                                            L = self._cg(kw, ka, km, cp.kappa, K1) * (rwa @ self._Yc(cp, km, K1).T) / Dw2
                                        base += sg * a * Tb * (L @ R.T)
                    for i, (dX, dY) in enumerate(pert):
                        x, y = sols[i]
                        o = nodes[i]
                        for kr in kaps:
                            # T2aL: g(w m r b) t(r a) g~(a b v m)      (X response of a, left ket)
                            d = dX.get((id(a_), kr))
                            if d is not None:
                                js = _js(a=ja, b=jb, r=_j2(kr), m=jm)
                                for K1 in self._ks(kw, kr, km, kb):
                                    L = None
                                    for K2, sg, tail, R in RDs:
                                        a = A((('g', 'w', 'm', 'r', 'b', 0), ('t', 'r', 'a'), tail), js, (K1, K2))
                                        if a:
                                            if L is None:
                                                L = self._cg(kw, kr, km, kb, K1) * (self._rho1S(sw, d) @ self._Yc(b, km, K1).T)
                                            tn[i] += sg * a * (L @ R.T)
                            # T2bL: g(w m a r) t(r b) g~(a b v m)
                            d = dX.get((id(b), kr))
                            if d is not None:
                                js = _js(a=ja, b=jb, r=_j2(kr), m=jm)
                                for K1 in self._ks(kw, ka, km, kr):
                                    L = None
                                    for K2, sg, tail, R in RDs:
                                        a = A((('g', 'w', 'm', 'a', 'r', 0), ('t', 'r', 'b'), tail), js, (K1, K2))
                                        if a:
                                            if L is None:
                                                L = self._cg(kw, ka, km, kr, K1) * (rwa @ self._Yp(d, km, K1).T)
                                            tn[i] += sg * a * (L @ R.T)
                            # T2aR: g(w m a b) t(a r) g~(r b v m)      (Y response of a, right bra)
                            d = dY.get((id(a_), kr))
                            if d is not None:
                                js = _js(a=ja, b=jb, r=_j2(kr), m=jm)
                                for K1, L in LDs.items():
                                    for K in self._ks(kr, kv, kb, km):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'a', 'r'), ('g', 'r', 'b', 'v', 'm', 1)), js, (K1, K))
                                        if a:
                                            R = self._cg(kr, kv, kb, km, K) * (self._rho1S(sv, d) @ self._Yc(b, km, K).T)
                                            tn[i] += a * (L @ R.T)
                                    for K in self._ks(kr, km, kb, kv):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'a', 'r'), ('g', 'r', 'b', 'm', 'v', 1)), js, (K1, K))
                                        if a:
                                            R = self._cg(kr, km, kb, kv, K) * (y1v(b, K) @ self._rhoE(d, km).T)
                                            tn[i] -= a * (L @ R.T)
                            # T2bR: g(w m a b) t(b r) g~(a r v m)
                            d = dY.get((id(b), kr))
                            if d is not None:
                                js = _js(a=ja, b=jb, r=_j2(kr), m=jm)
                                for K1, L in LDs.items():
                                    for K in self._ks(ka, kv, kr, km):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'b', 'r'), ('g', 'a', 'r', 'v', 'm', 1)), js, (K1, K))
                                        if a:
                                            R = self._cg(ka, kv, kr, km, K) * (rav @ self._Yp(d, km, K).T)
                                            tn[i] += a * (L @ R.T)
                                    for K in self._ks(ka, km, kr, kv):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'b', 'r'), ('g', 'a', 'r', 'm', 'v', 1)), js, (K1, K))
                                        if a:
                                            R = self._cg(ka, km, kr, kv, K) * (self._Y1S(d, sv, K) @ self._rhoE(a_, km).T)
                                            tn[i] -= a * (L @ R.T)
                        for c in core:
                            kc, jc = c.kappa, _j2(c.kappa)
                            js = js0 + (('c', jc),)
                            js = tuple(sorted(js))
                            # T2mL: g(w c a b) t(m c) g~(a b v m), x_mc / (e_m - e_c - w)
                            if reduced_c(km, kc):
                                cm = self._amp(x, c, km) / (eE[km] - c.eps - o)
                                for K1 in self._ks(kw, ka, kc, kb):
                                    Lsc = None
                                    for K2, sg, tail, R in RDs:
                                        a = A((('g', 'w', 'c', 'a', 'b', 0), ('t', 'm', 'c'), tail), js, (K1, K2))
                                        if a:
                                            if Lsc is None:
                                                Lsc = self._cg(kw, ka, kc, kb, K1) * (rwa @ self._Ycc(c, b, K1))
                                            tn[i] += sg * a * np.outer(Lsc, R @ cm)
                            # T2mR: g(w m a b) t(c m) g~(a b v c), y_cm / (e_m - e_c + w)
                            if reduced_c(kc, km):
                                cm = self._amp(y, c, km) / (eE[km] - c.eps + o)
                                for K1, L in LDs.items():
                                    Lc = L @ cm
                                    for K in self._ks(ka, kv, kb, kc):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'c', 'm'), ('g', 'a', 'b', 'v', 'c', 1)), js, (K1, K))
                                        if a:
                                            Rs_ = self._cg(ka, kv, kb, kc, K) * (rav @ self._Ycc(b, c, K))
                                            tn[i] += a * np.outer(Lc, Rs_)
                                    for K in self._ks(ka, kc, kb, kv):
                                        a = A((('g', 'w', 'm', 'a', 'b', 0), ('t', 'c', 'm'), ('g', 'a', 'b', 'c', 'v', 1)), js, (K1, K))
                                        if a:
                                            Rs_ = self._cg(ka, kc, kb, kv, K) * (y1v(b, K) @ (self.wq * (a_.P * c.P + a_.Q * c.Q)))
                                            tn[i] -= a * np.outer(Lc, Rs_)
        return base, tn
