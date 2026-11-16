"""Published J-resolved matrix elements in the polarizability networks (load_published_lines)."""
from __future__ import annotations

import pytest

from qali.core.paths import reference_path
from qali.physics.polarizability_core import alpha0_static, build_transitions_for_state, load_element


def _net(el, label):
    data = load_element(el, str(reference_path('data_json', f'{el}_rydberg.json').parent), verbose=False)
    data['valence_continuum'] = True
    return build_transitions_for_state(label, data, data['transitions'])


def test_cs_6p_lines_follow_the_published_matrix_elements():
    tr = {t['other_label']: t for t in _net('Cs', '6p1/2')}
    # f = (2/3) dE D^2 / (2 J_lower + 1): D(6p1/2-5d3/2) = 7.1, D(6p1/2-7s) = 4.24 (Safronova et al. 2016)
    for lab, D in (('5d3/2', 7.1), ('7s', 4.24), ('6d3/2', 4.23)):
        t = tr[lab]
        assert t['fosc'] == pytest.approx(2 / 3 * t['delta_E_au'] * D * D / 2, rel=1e-3), lab
        assert t['source'] == 'literature'


@pytest.mark.parametrize('el,lab,expt,tol', [
    ('Cs', '6p1/2', 1328.4, 0.025), ('Cs', '6p3/2', 1641.0, 0.025),
    ('Rb', '5p1/2', 810.6, 0.015), ('K', '4p1/2', 606.7, 0.01), ('Na', '3p1/2', 359.2, 0.012)])
def test_excited_polarizabilities_against_stark_shift_measurements(el, lab, expt, tol):
    """Stark-shift values compiled by Arora, Safronova & Clark, PRA 76, 052509 (2007), Table II."""
    assert alpha0_static(_net(el, lab)) == pytest.approx(expt, rel=tol)


def test_the_precision_d_line_keeps_priority():
    tr = {t['other_label']: t for t in _net('Fr', '7p1/2')}
    assert tr['7s']['source'] == 'precision'


# --- "Published matrix elements" (Tables and variables) ------------------------------------------------
@pytest.fixture()
def tables(tmp_path, monkeypatch):
    from qali.core import paths
    from qali.core import user_tables as U

    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    yield U
    U.reload("published.lines")


def test_every_published_line_starts_ticked(tables):
    from qali.physics import polarizability_core as P

    for el in ('Cs', 'Rb', 'K', 'Fr'):
        rows = P.PUBLISHED_LINES[el]
        assert rows and tables.ticked_values("published.lines", P.PUBLISHED_LINES, el) == rows
        assert P.published_lines_in_use(el) == (sorted(rows), [])
    assert P.PUBLISHED_LINES['Cs']['6p1/2-5d3/2'] == 7.1
    tables.save_flags(tables.load_flags())                       # the defaults alone leave no file
    assert not tables.flags_path().exists()


def test_unticking_a_line_gives_it_back_to_nist(tables):
    from qali.physics import polarizability_core as P

    flags = tables.load_flags()
    flags["published.lines"].pop("Cs / 6p1/2-5d3/2")
    tables.save_flags(flags)
    assert ('6p1/2', '5d3/2') not in P.load_published_lines('Cs')
    assert ('6p1/2', '7s') in P.load_published_lines('Cs')
    assert P.published_lines_in_use('Cs')[1] == ['6p1/2-5d3/2']
    tr = {t['other_label']: t for t in _net('Cs', '6p1/2')}
    assert tr['5d3/2']['source'] != 'literature' and tr['7s']['source'] == 'literature'
    assert alpha0_static(_net('Cs', '6p1/2')) > 1.04 * 1328.4    # NIST's 5d f: the 6p1/2 α runs high again


def test_an_edited_value_is_the_one_used(tables):
    from qali.physics import polarizability_core as P

    tables.save_overrides({"published.lines": {"Cs / 6p1/2-5d3/2": 7.0}})
    tables.reload("published.lines")
    D, sD = P.load_published_lines('Cs')[('6p1/2', '5d3/2')]
    assert D == 7.0 and sD == pytest.approx(0.1 * 7.0 / 7.1)


def test_bundled_cs_5d_lifetimes_use_the_published_lines():
    """Cs 5d5/2: 1353(5) ns, Pucher et al., PRA 101, 042510 (2020); NIST's f gives 1281 ns."""
    import json

    doc = json.loads(reference_path('data_json', 'Cs_lifetimes.json').read_text(encoding='utf-8'))
    tau = {lt['state']: lt for lt in doc['lifetimes']}
    assert tau['5d5/2']['lifetime_ns'] == pytest.approx(1353, rel=0.02)
    assert tau['5d5/2']['channels'][0]['fosc_acc'] == 'published'
    assert '6p3/2-5d5/2' in doc['published_lines']['published'] and doc['published_lines']['unticked'] == []


def test_lifetimes_follow_the_tick(tables):
    import json
    import shutil

    from qali.core.workspace import Workspace
    from qali.pipelines import lifetimes as L

    data = Workspace.current().data_json_dir
    data.mkdir(parents=True, exist_ok=True)
    for kind in ('rydberg', 'transitions'):
        shutil.copy(reference_path('data_json', f'Cs_{kind}.json'), data)

    flags = tables.load_flags()
    flags["published.lines"].pop("Cs / 6p3/2-5d5/2")
    tables.save_flags(flags)
    assert L.main(['Cs']) == 0
    doc = json.loads((data / 'Cs_lifetimes.json').read_text(encoding='utf-8'))
    lt = {x['state']: x for x in doc['lifetimes']}['5d5/2']
    assert lt['channels'][0]['fosc_source'] == 'NIST'
    assert lt['lifetime_ns'] == pytest.approx(1281, rel=0.01)
    assert doc['published_lines']['unticked'] == ['6p3/2-5d5/2']


def test_version_is_read_from_the_package():
    import re

    import qali
    from qali._version import __version__

    assert qali.__version__ == __version__ and re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_nist_alternative_of_each_line():
    """|D| back from the network f is the published |D| when ticked; unticked, NIST's where NIST has the line."""
    from qali.physics import polarizability_core as P

    alt = P.published_line_alternatives('Cs')
    D, src = alt['6p1/2-5d3/2']
    assert src == 'NIST' and D == pytest.approx(7.29, abs=0.01)      # NIST f 0.268 against the published 7.1
    assert alt['6p1/2-7s'][1] == 'literature'                        # NIST has no J-resolved 6p1/2-7s line
    by_src = {}
    for el in ('K', 'Rb', 'Cs', 'Fr'):
        for _k, (_D, s) in P.published_line_alternatives(el).items():
            by_src[s] = by_src.get(s, 0) + 1
    assert by_src == {'NIST': 46, 'literature': 38, 'Numerov': 3}         # Rb 6p-5d: a model value when unticked


def test_line_rows_are_checked():
    from qali.core import user_tables as U

    assert U.make_key("published.lines", ["cs", "6P1/2", "7s"]) == ("Cs / 6p1/2-7s", ["Cs", "6p1/2-7s"])
    for parts, needle in ((["Cs", "6p", "7s"], "Give J"), (["Cs", "6p1/2", "7p3/2"], "differ by one"),
                          (["Cs", "6p1/2", "5d5/2"], "at most one"), (["Cs", "6p5/2", "7s"], "J of a p level"),
                          (["Cs", "6x", "7s"], "not a level")):
        with pytest.raises(ValueError, match=needle):
            U.make_key("published.lines", parts)
    assert U.key_parts("published.lines", "Cs / 6p1/2-5d3/2") == ["Cs", "6p1/2", "5d3/2"]


def test_an_added_line_reaches_both_stages(tables):
    from qali.physics import polarizability_core as P

    tables.save_additions({"published.lines": {"Na / 3p1/2-3d3/2": {"value": 6.0, "source": "my paper", "sigma": 0.05}}})
    flags = tables.load_flags()
    flags.setdefault("published.lines", {})["Na / 3p1/2-3d3/2"] = True
    tables.save_flags(flags)
    tables.reload("published.lines")
    assert P.load_published_lines('Na')[('3d3/2', '3p1/2')] == (6.0, 0.05)
    assert P.published_line_sources('Na')['3p1/2-3d3/2'] == ('my paper', 0.05)
    t = {x['other_label']: x for x in _net('Na', '3p1/2')}['3d3/2']
    assert t['source'] == 'literature' and t['fosc'] == pytest.approx(2 / 3 * t['delta_E_au'] * 36 / 2, rel=1e-3)
    tables.save_additions({})
    tables.reload("published.lines")


@pytest.mark.parametrize('state,expt,tol', [('5d3/2', 246.3, 0.04), ('5d5/2', 238.5, 0.03),
                                            ('6d3/2', 256.0, 0.02), ('6d5/2', 249.0, 0.02)])
def test_rb_nd_lifetimes(state, expt, tol):
    """Sheng et al., PRA 78, 062506 (2008) (5d); 6d values as quoted by Safronova & Safronova, PRA 83, 052508 (2011).
    They were 13-23 % short with the unscaled 2004 5p-nd matrix elements."""
    import json

    doc = json.loads(reference_path('data_json', 'Rb_lifetimes.json').read_text(encoding='utf-8'))
    tau = {lt['state']: lt['lifetime_ns'] for lt in doc['lifetimes']}
    assert tau[state] == pytest.approx(expt, rel=tol)


def test_k_6s_lifetime_and_the_nist_5p3_6s_row():
    """K 6s: 81.3 ns all-order (Safronova & Safronova, PRA 78, 052504 (2008), Table V). NIST's 5p3/2-6s gA (9.91e6)
    gives |D| = 15.5 against the recommended 12.50(2) (arXiv:1301.3181); with it 6s came out 71.3 ns."""
    import json

    from qali.physics import polarizability_core as P

    D_nist, src = P.published_line_alternatives('K')['5p3/2-6s']
    assert src == 'NIST' and D_nist == pytest.approx(15.5, abs=0.05)
    assert P.PUBLISHED_LINES['K']['5p3/2-6s'] == 12.50
    doc = json.loads(reference_path('data_json', 'K_lifetimes.json').read_text(encoding='utf-8'))
    tau = {lt['state']: lt['lifetime_ns'] for lt in doc['lifetimes']}
    assert tau['6s'] == pytest.approx(81.3, rel=0.01)
