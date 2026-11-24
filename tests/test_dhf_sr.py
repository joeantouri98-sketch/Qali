"""Third-order E1 corrections (structural radiation + normalisation, physics/atomic_structure/dhf_sr)."""
import math

import pytest

from qali.physics.atomic_structure.angular import reduced_c
from qali.physics.atomic_structure.dhf_sr import angular


@pytest.mark.parametrize("jv, jb, jm, jn", [(1, 1, 1, 1), (1, 3, 3, 5), (3, 1, 5, 3), (5, 3, 3, 7)])
def test_m_sum_angular_factor_matches_the_j_coupled_sigma(jv, jb, jm, jn):
    # Sigma(2) core-polarisation term, rank 0: the m-sum over g_wbmn g_mnvb equals the j-coupled
    # (-1)^(jm+jn-jv-jb) / ((2jv+1)(2k+1)) of sigma2_matrices, times sqrt(2jv+1) (reduced element)
    js = (('b', jb), ('m', jm), ('n', jn), ('w', jv), ('v', jv))
    spec = (('g', 'w', 'b', 'm', 'n', 0), ('g', 'm', 'n', 'v', 'b', 1))
    for k1 in range(4):
        for k2 in range(4):
            a = angular(spec, js, (k1, k2), 0)
            ok = (abs(jv - jm) <= 2 * k1 <= jv + jm and abs(jb - jn) <= 2 * k1 <= jb + jn)
            want = 0.0
            if k1 == k2 and ok:
                want = (-1) ** ((jm + jn - jv - jb) // 2) / ((jv + 1) * (2 * k1 + 1)) * math.sqrt(jv + 1)
            assert a == pytest.approx(want, abs=1e-12)


@pytest.mark.slow
def test_na_third_order_is_hermitian_and_sized_like_the_literature():
    # Na DHF orbitals in a light basis: <3p||SR||3s> = s <3s||SR||3p> (the swap with the reduced-
    # element phase); D line: SR and N nearly cancel (|SR + N| < 0.3 % of d); 3s-4p (weak, cancelling
    # radial integral): SR + N raises it, by more than it moves the D line (Na: +0.4 %); dSigma/dE < 0
    import numpy as np
    from scipy.linalg import eigh

    from qali.physics.atomic_structure.dhf_cp import _Model, one_valence_configuration
    from qali.physics.atomic_structure.dhf_rpa import CoreRPA, ValenceE1
    from qali.physics.atomic_structure.dhf_sigma import Brueckner, _nodes
    from qali.physics.atomic_structure.dhf_sr import ThirdOrder

    core, nv = one_valence_configuration(11)
    m = _Model(11, core, nv, 0.0, 0.0, None)
    m.scf_core()
    m.setup_valence()
    br = Brueckner(m, lmax=3)
    e_s = br.dhf_energies(-1, 2)
    e_p = br.dhf_energies(1, 2)
    nodes = _nodes(1.3 * e_s[0], 0.8 * e_p[-1])
    br.build(-1, nodes)
    br.build(1, nodes)
    rpa = CoreRPA(br.mb)
    V = ValenceE1(rpa, m.bv, m.c)
    to = ThirdOrder(br, rpa, V.on_v, m.bv)

    def orbs(kap):
        eps, vec = eigh(m.F0[kap], m.S[kap])
        k0 = int((eps < -m.c ** 2).sum()) + m.ncore(kap)
        P, Q, _ = m.comps[kap]
        return [(vec[:, k0 + i] @ P, vec[:, k0 + i] @ Q) for i in range(2)]
    ss = to.stack(-1, orbs(-1), e_s)
    sp = to.stack(1, orbs(1), e_p)
    M = to.structural_radiation(sp, ss)
    back = to.structural_radiation(ss, sp)
    sgn = reduced_c(-1, 1) / reduced_c(1, -1)
    assert np.allclose(back.T, sgn * M, rtol=2e-3, atol=1e-6)
    ds_s, ds_p = to.dsigma(ss), to.dsigma(sp)
    assert (ds_s < 0).all() and (ds_p < 0).all()
    for i in range(2):
        rpa.at(e_p[i] - e_s[0])
        _, d = V.element(1, orbs(1)[i], -1, orbs(-1)[0], e_p[i] - e_s[0], ('p', i), 's')
        rel = (M[i, 0] + 0.5 * d * (ds_p[i] + ds_s[0])) / d
        if i == 0:
            assert abs(rel) < 3e-3
            rel_d = rel
        else:
            assert 0.002 < rel < 0.05 and rel > abs(rel_d)
