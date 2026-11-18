"""Photoionization continuum with the core-polarized dipole: measured thresholds, the Rb Cooper minimum."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qali.physics import valence_continuum as V

HA_EV = 27.211386245988
MB_PER_INV_HA = 109.76 / HA_EV          # σ (Mb) = 109.76 df/dε (eV⁻¹)


def _sigma(data_dir, sym):
    ryd = json.loads(Path(data_dir, f"{sym}_rydberg.json").read_text(encoding="utf-8"))
    eb = ryd["ionization_energy_eV"] / HA_EV
    g = ryd["ground_state"]
    lines = V.lines_for_state(sym, 0, g.get("label") or f"{g['n']}s", eb)
    eg = V.energy_grid()
    w = np.zeros(len(eg))
    d = np.diff(eg)
    w[:-1] += d / 2
    w[1:] += d / 2
    w[0] += eg[0]
    return eg * HA_EV, np.array([f for _, f in lines[:len(eg)]]) / w * MB_PER_INV_HA


@pytest.mark.parametrize("sym,meas,unc,k", [
    ("Rb", 0.076, 0.015, 2.0),      # Ciampini 2002, 296 nm (just above threshold)
    ("Na", 0.117, 0.015, 2.0),      # threshold, literature 0.105-0.129
    ("K", 0.010, 0.0025, 3.0),      # Theodosiou 2022 recommended
])
def test_threshold_cross_sections(data_dir, sym, meas, unc, k):
    e, s = _sigma(data_dir, sym)
    assert abs(np.interp(0.012, e, s) - meas) < k * unc


def test_rb_cooper_minimum_and_266nm(data_dir):
    e, s = _sigma(data_dir, "Rb")
    near = e < 2.0
    e_min = e[near][np.argmin(s[near])]
    assert 0.5 < e_min < 0.9                                  # measured near 253.7 nm: 0.71 eV above the limit
    s266 = np.interp(1239.84193 / 266 - 4.1771281, e, s)     # Lowell et al. 2002: 0.017(2) Mb
    assert 0.005 < s266 < 0.03                                # plain r gave 0.21 Mb


def test_dipole_operator_limits():
    r = np.geomspace(0.05, 50, 200)
    assert np.array_equal(V.dipole_operator("H", r, 0.2), r)          # no model parameters: plain r
    d = V.dipole_operator("Rb", r, 0.16)
    assert np.allclose(d[r > 30], r[r > 30], rtol=1e-3)               # far out: the bare dipole
    assert (d[r < 2] < r[r < 2]).all()                                # inside: screened by the core
    w1 = V._first_core_excitation("Rb")
    assert np.array_equal(V.dipole_operator("Rb", r, 1.01 * w1), r)   # above the core's first excitation
    a = V._core_fade("Rb", 0.75 * w1)
    assert 0.0 < a < 1.0 and V._core_fade("Rb", 0.4 * w1) == 1.0
