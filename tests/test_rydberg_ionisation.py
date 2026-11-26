"""Field ionisation of Rydberg states by complex scaling (qali.physics.rydberg_stark.ionisation).

Hydrogen benchmark: Fernández & Garcia, Appl. Math. Comput. 273, 11 (2016) (arXiv:1506.05797),
Table 4, Riccati-Padé resonances |n1 n2 m> of n = 10 (Kolosov's values agree); Γ = -2 Im E.
"""
import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation.exact import MARINESCU
from qali.physics.rydberg_stark import ionisation as ion
from qali.physics.rydberg_stark.defects import Level, RydbergEnergies
from qali.physics.rydberg_stark.stark import (AU_FIELD_V_CM, CM1_GHZ, HARTREE_GHZ, build_basis, follow,
                                              stark_map)


class _Hydrogen:
    """Levels of hydrogen (infinite mass) in the RydbergEnergies interface."""
    alpha_d = 0.0

    def level(self, n, l, j2):
        return Level(n, l, j2, HARTREE_GHZ / CM1_GHZ / (2 * n * n), float(n), "hydrogen")

    def exists(self, n, l):
        return 0 <= l < n

    def defect(self, n, l, j2):
        return 0.0, "hydrogen"


def _rb():
    from qali.pipelines.dhf import nist_inputs
    ip, nist = nist_inputs("Rb", str(reference_path("NIST Levels")))
    return RydbergEnergies("Rb", nist, ip, MARINESCU["Rb"]["alphaC"])


def test_laguerre_mesh_gives_hydrogen_levels():
    x, T = ion.laguerre_mesh(150)
    h = 5.0
    for l in (0, 7):
        w = np.linalg.eigvalsh(0.5 * (T + T.T) / (2 * h * h) + np.diag(l * (l + 1) / (2 * (h * x) ** 2) - 1 / (h * x)))
        n_star = 1 / np.sqrt(-2 * w[w < 0][:20])
        assert n_star == pytest.approx(np.arange(l + 1, l + 21), abs=1e-6)


@pytest.mark.parametrize("F, re, im", [(1.4e-5, -7.212084472616482e-3, -4.007021552503371e-5),
                                       (2.2e-5, -8.660579416493959e-3, -5.329919686471733e-4)])
def test_hydrogen_red_state_matches_the_benchmark(F, re, im):
    # |0, 9, 0> of n = 10 through the whole alkali machinery: (l, j) channels at m_j = 1/2
    s = ion.build_system(_Hydrogen(), None, 10, 0, 1, 1, 8, 12, 24)
    s.E_target_au = re
    m = ion.resonance_map(s, [F * AU_FIELD_V_CM], 1.2 * abs(im) * HARTREE_GHZ)
    E = m.energies_ghz[0] / HARTREE_GHZ + re
    G = m.widths_ghz[0] / HARTREE_GHZ
    ok = ~np.isnan(E)
    j = int(np.argmin(np.abs(E[ok] - re)))
    assert E[ok][j] == pytest.approx(re, rel=1e-8)
    assert G[ok][j] == pytest.approx(-2 * im, rel=1e-6)


def test_rb_30s_resonances_match_the_bound_map_below_ionisation():
    en = _rb()
    s = ion.build_system(en, "Rb", 30, 0, 1, 1, 26, 34, 6)
    b = build_basis(en, "Rb", 30, 0, 1, 1, 26, 34, 6)
    F = [0.0, 12.0, 20.0]
    m = ion.resonance_map(s, F, 3.0)
    ref = follow(stark_map(b, F, 3.0))
    got = m.energies_ghz[np.arange(3), np.argmax(m.character, axis=1)]
    assert got[0] == pytest.approx(0.0, abs=2e-4)                        # field-free level: 0.2 MHz
    assert got[1:] == pytest.approx(ref[1:], rel=2e-3)                   # polarisation shift
    assert np.nanmax(m.widths_ghz) < 1e-3                                # no ionisation at 20 V/cm


def test_ionisation_stage_writes_widths(tmp_path):
    import json

    from helpers import make_workspace, run_pipeline
    root = make_workspace(tmp_path / "ws")
    r = run_pipeline(["stark", "Na", "--state", "12s1/2", "--ionisation", "--n-range", "1", "--points", "3",
                      "--field-min", "8000"], root)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((root / ".qali" / "data" / "Na_stark.json").read_text(encoding="utf-8"))
    io = d["ionisation"]
    assert io["classical_field_V_cm"] > 8000 and len(io["saddle_ghz"]) == 3 and io["check"]["levels"] > 0
    assert len(d["widths_ghz"]) == 3 and all(len(w) == len(e) for w, e in zip(d["widths_ghz"], d["energies_ghz"]))
    assert max(max(w) for w in d["widths_ghz"]) > 1e-3                   # something ionises at 1.2 F_class
    html = (root / ".qali" / "plots" / "Na" / "Na_stark.html").read_text(encoding="utf-8")
    assert '"ionisation":{' in html


def test_parallel_fields_match_serial():
    s = ion.build_system(_Hydrogen(), None, 10, 0, 1, 1, 8, 12, 14)
    F = np.array([1.2e-5, 1.4e-5, 1.6e-5]) * AU_FIELD_V_CM
    a = ion.resonance_map(s, F, 500.0)
    b = ion.resonance_map(s, F, 500.0, workers=2)
    assert a.energies_ghz.shape == b.energies_ghz.shape
    assert np.allclose(a.energies_ghz, b.energies_ghz, atol=1e-5, equal_nan=True)     # 0.01 MHz
    assert np.allclose(a.widths_ghz, b.widths_ghz, atol=1e-5, equal_nan=True)
