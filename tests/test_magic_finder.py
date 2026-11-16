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
