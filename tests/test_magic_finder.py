"""Magic-wavelength finder: roots between neighbouring resonances, independent of the scan grid."""
from __future__ import annotations

import pytest

from qali.core.paths import reference_path
from qali.physics.polarizability_core import build_transitions_for_state, load_element
from qali.pipelines.polarizability import find_magic_wavelengths, nm_from_omega_au


def _nets(el, gs, ex):
    d = load_element(el, str(reference_path('data_json')), verbose=False)
    d['valence_continuum'] = True
    lo = max(200.0, nm_from_omega_au(-d['state_energy_au'][ex]))
    return (build_transitions_for_state(gs, d, d['transitions']),
            build_transitions_for_state(ex, d, d['transitions']), lo)


@pytest.mark.parametrize('el,gs,ex', [('Li', '2s', '2p1/2'), ('Rb', '5s', '5p1/2')])
def test_the_list_does_not_depend_on_the_grid(el, gs, ex):
    g, e, lo = _nets(el, gs, ex)
    runs = [[m['wl_nm'] for m in find_magic_wavelengths(g, e, lo, 2000, n_scan=n)] for n in (2500, 20000)]
    assert runs[0] == runs[1]


def test_no_root_on_a_pole_and_both_sides_of_a_resonance_kept():
    """Li 2s/2p1/2: 356.811 and 367.253 nm were sign flips across a Rydberg pole (Δα undefined there);
    362.047 and 362.432 are real crossings on either side of the 362.229 nm line, 0.39 nm apart."""
    g, e, lo = _nets('Li', '2s', '2p1/2')
    wl = [m['wl_nm'] for m in find_magic_wavelengths(g, e, lo, 2000)]
    assert not any(abs(w - 356.811) < 0.01 or abs(w - 367.253) < 0.01 for w in wl)
    assert any(abs(w - 362.047) < 0.01 for w in wl) and any(abs(w - 362.432) < 0.01 for w in wl)


def test_root_close_to_a_resonance_is_found():
    """Rb 5p1/2: the crossing 0.65 nm short of the 5p1/2-5d3/2 line (762.1 nm), at the edge of its guard band."""
    g, e, lo = _nets('Rb', '5s', '5p1/2')
    assert any(abs(m['wl_nm'] - 761.456) < 0.01 for m in find_magic_wavelengths(g, e, lo, 2000))


def test_nothing_past_the_excited_threshold():
    """Fr 7p has no continuum lines; the search starts at its photoionization threshold (485.2 nm)."""
    g, e, lo = _nets('Fr', '7s', '7p1/2')
    assert lo == pytest.approx(485.17, abs=0.05)
    import json
    doc = json.loads(reference_path('data_json', 'Fr_polarizability.json').read_text(encoding='utf-8'))
    assert min(m['wl_nm'] for m in doc['magic_wavelengths']) > lo


def test_every_crossing_is_kept_even_next_to_a_weak_line():
    """Rb 5p1/2 crosses 5s 0.004 nm short of a high Rydberg line: still a magic wavelength."""
    g, e, lo = _nets('Rb', '5s', '5p1/2')
    wl = [m['wl_nm'] for m in find_magic_wavelengths(g, e, lo, 2000)]
    assert any(abs(w - 474.787) < 0.002 for w in wl)
    assert len(wl) > 80


def test_scattering_rate_has_the_two_level_limit():
    """R = (Γ/Δ) U/ħ for one line far off resonance (Grimm et al. 2000)."""
    from qali.physics.polarizability_core import AU_TIME_S, MK_OVER_HBAR_S, TrapScattering
    line = [dict(delta_E_au=0.0573, fosc=1.0, sign=1, other_label='5p1/2', source='NIST')]
    sc = TrapScattering(line, '5s', [], '5p1/2', {'5p1/2': 3.6e7})
    om = 0.0573 * 0.999
    from qali.physics.polarizability_core import alpha_real_vectorized
    import numpy as np
    a = alpha_real_vectorized(np.array([om]), line, guard=0.0)
    delta = 0.0573 * 0.001 / AU_TIME_S
    assert sc.ground(np.array([om]), a)[0] == pytest.approx(3.6e7 / delta * MK_OVER_HBAR_S, rel=2e-3)


def _scattering(el, gs, ex):
    import json
    from qali.core.paths import reference_path as rp
    from qali.pipelines.polarizability import _trapped_mass
    from qali.physics.polarizability_core import TrapScattering, state_decay_rates
    g, e, lo = _nets(el, gs, ex)
    life = json.loads(rp('data_json', f'{el}_lifetimes.json').read_text(encoding='utf-8'))
    return g, e, lo, TrapScattering(g, gs, e, ex, state_decay_rates(life), mass_amu=_trapped_mass(el)[1])


def test_recoil_heating_time_is_depth_independent_and_matches_grimm():
    """t = U / (k_B dT/dt), dT/dt = (2/3)(E_rec/k_B) Γ_sc: from the photons/s at 1 mK."""
    import numpy as np
    from qali.core.constants import AMU_KG, HBAR_SI, KB_SI
    from qali.physics.polarizability_core import NM_TO_AU_OMEGA, alpha_real_vectorized
    g, e, lo, sc = _scattering('Rb', '5s', '5p1/2')
    om = np.array([NM_TO_AU_OMEGA / 787.6])
    a = alpha_real_vectorized(om, g, guard=0.0)
    k = 2 * np.pi / 787.6e-9
    e_rec = (HBAR_SI * k) ** 2 / (2 * sc.mass_kg)
    from qali.pipelines.polarizability import _trapped_mass
    assert sc.mass_kg == pytest.approx(_trapped_mass('Rb')[1] * AMU_KG)
    t = 3 * KB_SI * 1e-3 / (2 * e_rec * sc.ground(om, a)[0])
    assert sc.heating_time(om, a)[0] == pytest.approx(t, rel=1e-9)
    assert 5 < t < 30                                       # 660 photons/s at 1 mK heat Rb out in ~12 s


def test_practical_magic_wavelengths_follow_the_physics():
    """Recoil heating rules out only crossings next to a ground-state line; the window limit is optional."""
    g, e, lo, sc = _scattering('Rb', '5s', '5p1/2')
    by = {round(m['wl_nm'], 1): m for m in find_magic_wavelengths(g, e, lo, 2000, scattering=sc)}
    assert by[787.6]['practical'] and by[474.8]['practical'] and by[1351.0]['practical']
    assert by[787.6]['heating_time_s'] < by[1351.0]['heating_time_s']
    assert by[474.8]['window_ghz'] < 1 < by[1351.0]['window_ghz']
    strict = {round(m['wl_nm'], 1): m for m in find_magic_wavelengths(g, e, lo, 2000, scattering=sc, window_min=1.0)}
    assert not strict[474.8]['practical'] and strict[1351.0]['practical']
    g, e, lo, sc = _scattering('Na', '3s', '3p1/2')
    na = {round(m['wl_nm'], 2): m for m in find_magic_wavelengths(g, e, lo, 2000, scattering=sc)}
    assert not na[589.46]['practical']                     # between the D lines: heated out in 0.1 s


def test_each_root_names_its_excited_state_line_and_the_default_n_range_keeps_the_published_ones():
    """Arora et al. 2007 (> 600 nm) and Singh et al. 2016 (Fr, 600-1500 nm) all sit by lines up to n_ground + 5."""
    from qali.pipelines.polarizability import in_n_range
    from qali.physics.polarizability_core import default_magic_n_range
    published = {'Rb': ('5s', '5p1/2', [606.2, 617.7, 727.35, 761.5, 787.6, 1350.9]),
                 'Cs': ('6s', '6p1/2', [634.3, 660.1, 759.40, 1520.0]),
                 'Fr': ('7s', '7p1/2', [621.48, 646.05, 745.6, 771.03, 838.08, 1479.49])}
    for el, (gs, ex, wls) in published.items():
        g, e, lo = _nets(el, gs, ex)
        found = find_magic_wavelengths(g, e, lo, 2000)
        rng = default_magic_n_range(gs)
        listed = [m for m in found if in_n_range(m, rng)]
        assert all(m['line_n'] is not None for m in found)
        for w in wls:                                  # each published one has a listed root nearby
            assert min(abs(m['wl_nm'] - w) for m in listed) < 10, (el, w)
        assert len(listed) < 15 < len(found)
