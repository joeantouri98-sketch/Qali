"""Brueckner-orbital spectrum and E1 network of a one-valence atom or ion.

Levels: frozen-core DHF plus the second-order correlation potential Sigma(2)(E) (``dhf_sigma``),
solved self-consistently per level, optionally scaled per kappa to one measured level.
E1: reduced dipoles between Brueckner orbitals, dressed by the core's RPA response at the
transition frequency (``dhf_rpa``), plus structural radiation and normalisation (``dhf_sr``,
third order; scaled by the same lambda as Sigma); the core polarisability is the same RPA, so
alpha(0) has no external input either.

Accuracy (scaled, against precision data), n-max 25 runs with the third order: D lines K +0.06 /
+0.03 %, Rb +0.02 / -0.04 %, Cs -0.15 / -0.15 %, Fr +0.02 / -0.20 %, Na +0.17 %; the weak
ns-(n+1)p lines K +0.04 %, Rb -0.13 / +0.42 %, Cs 6s-7p -0.55 / -0.44 % (4-7 % low without the
third order), Cs 7s-6p -0.13 / -0.11 %, 7s-7p -0.04 %; and the ground ns-np series up to the
highest measured n within the scatter of the measurements (no spurious Cooper minimum in Cs
np1/2).  Lowest-p lifetimes of Ca+, Sr+, Ba+ within 0.5 % and their p -> d branching fractions
within ~1.5 % (checked before the third order, which moves resonance lines by ~0.1 %).
Not included: the core-valence (Pauli) term of alpha, Breit, QED.

Engine contract: inputs as arguments, results as frozen dataclasses; no files, prints or plots.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

from qali.core.constants import AU_TIME_S, HARTREE_TO_EV, HC_EV_NM
from qali.physics.atomic_structure.angular import kappa_label, kappa_lj, reduced_c
from qali.physics.atomic_structure.dhf_cp import HARTREE_TO_CM1, VALENCE_KAPPAS, Lifetime, Transition
from qali.physics.atomic_structure.dhf_rpa import CoreRPA, ValenceE1
from qali.physics.atomic_structure.dhf_sigma import Sigma2Level, brueckner_levels, setup_brueckner
from qali.physics.atomic_structure.dhf_sr import ThirdOrder

ALPHA_FS = 1 / 137.035999084


@dataclass(frozen=True)
class BruecknerResult:
    Z: int
    charge: int
    valence: str
    scaling: str                          # 'fit' or 'none'
    ionization_cm: float                  # Brueckner ionization energy (= input when s is fitted)
    scale_factors: dict[str, float]
    scale_sources: dict[str, str]
    lmax: int
    nodes_au: tuple[float, ...]
    alpha_core_au: float                  # RPA, static
    alpha_core_dhf_au: float              # same sum without the RPA dressing
    alpha0_au: float
    alpha_dynamic_au: dict[str, float]
    levels: tuple[Sigma2Level, ...]
    transitions: tuple[Transition, ...]
    lifetimes: tuple[Lifetime, ...]
    quantum_defects: dict[str, tuple[tuple[int, float], ...]]
    # Photoionization of the ground level: ((omega_au, f), ...) of the positive-energy states of the
    # basis cavity, with RPA-dressed dipoles (a quadrature of df/de); omega up to 50 Ha
    continuum: tuple[tuple[float, float], ...] = ()
    third_order: bool = False             # dipoles include structural radiation + normalisation


def compute_brueckner(Z: int, *, charge: int = 0, mass_number: float | None = None,
                      ionization_cm: float | None = None,
                      fit_levels_cm: Mapping[tuple[int, int], float] | None = None,
                      scale: bool = True, n_max: int | None = None, n_defects: int = 30, lmax: int = 5,
                      omegas_eV: Sequence[float] = (), e1: bool = True, third_order: bool = True,
                      cache: MutableMapping[str, np.ndarray] | None = None,
                      progress: Callable[[str], None] | None = None) -> BruecknerResult:
    """Brueckner levels (up to n_max; None: n_valence + 6), E1 network, lifetimes, alpha and quantum
    defects (up to n_defects).  Inputs as for ``dhf_sigma.compute_sigma2``; e1=False skips the
    dipoles (levels and defects only); third_order=False leaves out structural radiation and
    normalisation (faster; weak lines several % low)."""
    say = progress or (lambda _msg: None)
    st = setup_brueckner(Z, charge=charge, mass_number=mass_number, ionization_cm=ionization_cm,
                         fit_levels_cm=fit_levels_cm, scale=scale, n_max=n_max, lmax=lmax,
                         cache=cache, progress=progress)
    m, br = st.model, st.br
    say("Brueckner levels ...")
    levels = brueckner_levels(st)
    g_E = -levels[0].binding_eV / HARTREE_TO_EV

    defects: dict[str, tuple[tuple[int, float], ...]] = {}
    for kap in st.nlev:
        k = n_defects - st.n0(kap) + 1
        if k <= 0:
            continue
        e = br.energies(kap, k, st.lam[kap])
        defects[kappa_label(0, kap)[1:]] = tuple(
            (st.n0(kap) + i, float(st.n0(kap) + i - m.z_ion / math.sqrt(-2 * e[i])))
            for i in range(k) if e[i] < 0)

    transitions: list[Transition] = []
    lifetimes: list[Lifetime] = []
    cont: list[tuple[float, float]] = []
    alpha_core = alpha_core_dhf = alpha0 = 0.0
    alpha_dyn: dict[str, float] = {}
    if e1:
        say("RPA core response ...")
        rpa = CoreRPA(br.mb)
        alpha_core, alpha_core_dhf = rpa.alpha_core(True), rpa.alpha_core(False)
        V = ValenceE1(rpa, m.bv, m.c)
        orbs = {}
        for lv in levels:
            i = lv.n - st.n0(lv.kappa)
            E = -lv.binding_eV / HARTREE_TO_EV
            orbs[lv.label] = (lv, E, br.orbital(lv.kappa, i, E, st.lam[lv.kappa]))
        omega_max = max(max(E for _, E, _ in orbs.values()) - g_E, -g_E)
        rpa.prepare(omega_max)

        def dressed(kw, w, Ew, kv, v, Ev, wtag, vtag):
            om = min(abs(Ew - Ev), omega_max)
            rpa.at(om)
            return V.element(kw, w, kv, v, om, wtag, vtag)

        # third order: SR[(w, v)] = <w||SR||v> at omega = E_w - E_v, and dSigma/dE per level, both
        # times the level's lambda (SR: geometric mean of the two)
        sr: dict[tuple[str, str], float] = {}
        dsig: dict[str, float] = {}
        if third_order:
            say("Structural radiation and normalisation ...")
            to = ThirdOrder(br, rpa, V.on_v, m.bv)
            stacks = {}
            for kap in sorted({lv.kappa for lv, _, _ in orbs.values()}, key=abs):
                items = sorted((t for t in orbs.values() if t[0].kappa == kap), key=lambda t: t[1])
                stk = to.stack(kap, [t[2] for t in items], [t[1] for t in items])
                stacks[kap] = ([t[0].label for t in items], stk)
                for lab, ds in zip(stacks[kap][0], to.dsigma(stk)):
                    dsig[lab] = st.lam[kap] * float(ds)
            kaps = list(stacks)
            for i, k1 in enumerate(kaps):
                for k2 in kaps[i + 1:]:
                    if reduced_c(k1, k2) == 0.0:
                        continue
                    say(f"  SR {kappa_label(0, k1)[1:]} - {kappa_label(0, k2)[1:]} ...")
                    M = to.structural_radiation(stacks[k1][1], stacks[k2][1])
                    lam = math.sqrt(st.lam[k1] * st.lam[k2])
                    sgn = reduced_c(k2, k1) / reduced_c(k1, k2)       # <v||.||w> = sgn <w||.||v>
                    for a_, l1 in enumerate(stacks[k1][0]):
                        for b_, l2 in enumerate(stacks[k2][0]):
                            sr[(l1, l2)] = lam * float(M[a_, b_])
                            sr[(l2, l1)] = lam * sgn * float(M[a_, b_])

        def third(d, wlab, vlab):
            """d + SR + normalisation for <w||D||v> (unchanged without the third order)."""
            if (wlab, vlab) not in sr:
                return d
            return d + sr[(wlab, vlab)] + 0.5 * d * (dsig[wlab] + dsig[vlab])

        say("E1 network ...")
        A_tot = {lab: 0.0 for lab in orbs}
        report = sorted(orbs.values(), key=lambda t: t[1])
        for a, (lo, E_lo, s_lo) in enumerate(report):
            for up, E_up, s_up in report[a + 1:]:
                if reduced_c(up.kappa, lo.kappa) == 0.0:
                    continue
                _bare, d = dressed(up.kappa, s_up, E_up, lo.kappa, s_lo, E_lo, up.label, lo.label)
                d = third(d, up.label, lo.label)
                w = E_up - E_lo
                f = 2.0 / 3.0 * w * d * d / (kappa_lj(lo.kappa)[1] + 1)
                A = 4.0 / 3.0 * ALPHA_FS ** 3 * w ** 3 * d * d / (kappa_lj(up.kappa)[1] + 1) / AU_TIME_S
                A_tot[up.label] += A
                rho = s_lo[0] * s_up[0] + s_lo[1] * s_up[1]
                scale_ = m.bv.integrate(np.abs(rho) * m.bv.r)
                canc = abs(m.bv.integrate(rho * m.bv.r)) / scale_ if scale_ > 0 else 1.0
                transitions.append(Transition(lo.label, up.label, float(HC_EV_NM / (w * HARTREE_TO_EV)),
                                              float(abs(d)), float(f), float(A), float(canc)))
        lifetimes = [Lifetime(lab, float(1e9 / A)) for lab, A in A_tot.items() if A > 0]

        # alpha of the ground level: reported levels with their own orbitals, the rest of the
        # (discretised) spectrum from Sigma held at the top node
        say("Polarisability ...")
        g_lv, _, g_orb = orbs[kappa_label(st.nv, -1)]
        terms = []                                          # (omega_s, |D|^2)
        # cont: positive-energy states of the basis cavity, i.e. the continuum
        for kap in VALENCE_KAPPAS:
            if reduced_c(kap, -1) == 0.0 or kap not in st.nlev:
                continue
            own = {lv.n - st.n0(kap) for lv, _, _ in orbs.values() if lv.kappa == kap}
            eps, P, Q = br.spectrum(kap, max(st.nodes), st.lam[kap])
            for i in range(len(eps)):
                own_lab = None
                if i in own:
                    lv, E, s = orbs[kappa_label(st.n0(kap) + i, kap)]
                    own_lab = lv.label
                else:
                    E, s = float(eps[i]), (P[i], Q[i])
                if E - g_E > 50.0:                          # far continuum: negligible
                    break
                _b, d = dressed(kap, s, E, -1, g_orb, g_E, ('a', kap, i), 'ground')
                V.forget(('a', kap, i))
                if own_lab is not None:                     # reported levels: with the third order
                    d = third(d, own_lab, g_lv.label)
                terms.append((E - g_E, d * d))
                if E > 0:
                    cont.append((float(E - g_E), float(2.0 / 3.0 * (E - g_E) * d * d / (kappa_lj(-1)[1] + 1))))
        jg = kappa_lj(-1)[1]

        def alpha_v(omega):
            return 2.0 / (3 * (jg + 1)) * sum(d2 * w / (w * w - omega * omega) for w, d2 in terms)
        alpha0 = alpha_v(0.0) + alpha_core
        alpha_dyn = {f"{w:g}": float(alpha_v(w / HARTREE_TO_EV) + alpha_core) for w in omegas_eV}

    return BruecknerResult(
        Z=Z, charge=charge, valence=kappa_label(st.nv, -1), scaling="fit" if st.use_fit else "none",
        ionization_cm=float(-g_E * HARTREE_TO_CM1),
        scale_factors={kappa_label(0, k)[1:]: st.lam[k] for k in st.nlev},
        scale_sources={kappa_label(0, k)[1:]: st.src[k] for k in st.nlev},
        lmax=lmax, nodes_au=tuple(st.nodes),
        alpha_core_au=float(alpha_core), alpha_core_dhf_au=float(alpha_core_dhf), alpha0_au=float(alpha0),
        alpha_dynamic_au=alpha_dyn, levels=tuple(levels), transitions=tuple(transitions),
        lifetimes=tuple(lifetimes), quantum_defects=defects,
        continuum=tuple(sorted(cont)) if e1 else (), third_order=bool(e1 and third_order))
