"""Named runs of the Stark and pair stages, pages written from saved JSON, and the replot dialog."""
import json
import os

import pytest

from qali.pipelines import rydberg_runs as rr


def test_run_names_from_settings():
    base = dict(state="30s1/2", mj=0.5, fields_V_cm=[0.0, 10.0])
    assert rr.tag("stark", base) == "30s1-2_mj1-2"
    assert rr.tag("stark", {**base, "mj": -1.5, "state": "60d5/2"}) == "60d5-2_mjm3-2"
    assert rr.tag("stark", {**base, "mode": "ac", "ac": {"frequency_ghz": 2.5, "dc_V_cm": 0.0}}) == "30s1-2_mj1-2_ac2p5GHz"
    assert rr.tag("stark", {**base, "ionisation": {"theta": 0.2}}) == "30s1-2_mj1-2_ion"
    assert rr.tag("stark", {**base, "ramp": {"ramp_ns": 2.5}}) == "30s1-2_mj1-2_ramp2p5ns"
    pair = dict(state1="60s1/2", state2="60p3/2", mj1=0.5, mj2=1.5, theta_deg=90.0)
    assert rr.tag("pair", pair) == "60s1-2_60p3-2_mj1-2_3-2_th90"
    assert "ionisation" in rr.summary("stark", {**base, "ionisation": {"theta": 0.2}})
    assert "ionisation" not in rr.summary("stark", {**base, "ionisation": None})
    assert rr.title("stark", {**base, "label": "Rb", "mode": "ac", "ac": {"frequency_ghz": 2}}) == "Rb 30s1/2 AC Stark map at 2 GHz"


def test_stark_run_is_filed_by_name_and_its_page_rebuilt(tmp_path):
    from helpers import make_workspace, run_pipeline
    root = make_workspace(tmp_path / "ws")
    r = run_pipeline(["stark", "Na", "--state", "15s1/2", "--n-range", "1", "--points", "6"], root)
    assert r.returncode == 0, r.stdout + r.stderr
    data, plots = root / ".qali" / "data", root / ".qali" / "plots" / "Na"
    named = data / "Na_stark__15s1-2_mj1-2.json"
    assert named.is_file() and (data / "Na_stark.json").is_file()
    assert (plots / "Na_stark__15s1-2_mj1-2.html").is_file() and (plots / "Na_stark.html").is_file()
    os.remove(plots / "Na_stark__15s1-2_mj1-2.html")
    r = run_pipeline(["stark", "--from-json", str(named)], root)
    assert r.returncode == 0, r.stdout + r.stderr
    html = (plots / "Na_stark__15s1-2_mj1-2.html").read_text(encoding="utf-8")
    assert '"state":"15s1/2"' in html and "$!" not in html and '"_qali"' not in html
    r = run_pipeline(["stark", "--from-json", str(named), "--name", "Na 15s check"], root)
    assert r.returncode == 0 and (plots / "Na_15s_check.html").is_file(), r.stdout + r.stderr
    bad = run_pipeline(["pair", "--from-json", str(named)], root)        # a Stark run is not a pair run
    assert bad.returncode != 0 and "not a pair output" in bad.stdout


def test_replot_dialog_lists_saved_runs(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from qali.gui.replot_dialog import ReplotDialog, list_runs
    from qali.gui.stages import STAGE_BY_ID
    d = dict(state="30s1/2", mj=0.5, fields_V_cm=[0.0, 50.0], energies_ghz=[[], []])
    for name in ("Rb_stark.json", "Rb_stark__30s1-2_mj1-2.json", "Cs_stark__40s1-2_mj1-2.json", "Rb_pair.json",
                 "Rb_stark_30s_ionisation.json"):                     # a copy renamed with one underscore
        (tmp_path / name).write_text(json.dumps(d), encoding="utf-8")
    runs = list_runs("stark", tmp_path, "Rb")
    assert sorted(r.name for r in runs) == ["", "30s1-2_mj1-2", "30s_ionisation"] and all("30s1/2" in r.summary for r in runs)
    dlg = ReplotDialog(STAGE_BY_ID["stark"], tmp_path, "Rb")
    assert dlg.table.rowCount() == 3 and len(dlg.selected_paths()) == 1
    stem = os.path.splitext(os.path.basename(dlg.selected_paths()[0]))[0]
    assert dlg.name.isEnabled() and dlg.name.text() == stem
    dlg.name.setText("my page.html")
    assert dlg.page_name() == "my_page"
    dlg.all_species.setChecked(True)
    assert dlg.table.rowCount() == 4
    dlg.table.selectAll()
    assert len(dlg.selected_paths()) == 4 and not dlg.name.isEnabled() and dlg.page_name() == ""
    assert STAGE_BY_ID["pair"]["replot"]["stem"] == "pair" and "replot" not in STAGE_BY_ID["lifetimes"]
    dlg.deleteLater()
    assert app is not None


@pytest.mark.parametrize("stage", ["stark", "pair"])
def test_from_json_needs_no_element(stage):
    from qali.pipelines import pair, stark
    mod = {"stark": stark, "pair": pair}[stage]
    args = mod.build_parser().parse_args(["--from-json", "a.json", "b.json"])
    assert args.element is None and args.from_json == ["a.json", "b.json"]
