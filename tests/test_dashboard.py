"""Dashboard smoke and interaction tests with Streamlit's AppTest (skipped without processed data)."""

import pytest

from lphrs.config import PROCESSED_DIR, ROOT

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "cvd_risk.parquet").exists(),
                                reason="processed data not built")

APP = str(ROOT / "app" / "dashboard.py")


@pytest.fixture(scope="module")
def at():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    return app


def rec_titles(app):
    return [m.value for m in app.markdown if m.value.startswith("**") and ". " in m.value[:6]]


def test_loads_without_errors_and_shows_disclaimer(at):
    assert not at.exception
    assert any("Wellness guidance only. Not medical advice." in w.value for w in at.warning)
    assert len(at.tabs) == 4
    assert len(rec_titles(at)) == 5


def test_switch_patient(at):
    before = at.session_state["patient"]
    at.sidebar.selectbox(key="patient_label").select_index(5).run()
    assert not at.exception
    assert at.session_state["patient"] != before
    assert len(rec_titles(at)) == 5


def test_feedback_updates_state_and_reranks(at):
    pid = at.session_state["patient"]
    first = at.session_state["last_recs"][0]
    at.button(key=f"fb-{pid}-{first['id']}-REJECTED").click().run()
    assert not at.exception
    state = at.session_state["fb"][pid]
    assert state["by_id"][first["id"]] == (0, 1)
    assert at.session_state["fb_log"][pid][-1]["outcome"] == "REJECTED"
    assert any("Re-ranked after your feedback" in i.value for i in at.info)
    assert at.session_state["last_recs"][0]["terms"]["H"] < 0.5 or first["id"] not in \
        [r["id"] for r in at.session_state["last_recs"]]


def test_too_difficult_lowers_cap(at):
    pid = at.session_state["patient"]
    rec = at.session_state["last_recs"][0]
    before = at.session_state["fb"][pid]["caps"][rec["domain"]]
    at.button(key=f"fb-{pid}-{rec['id']}-TOO_DIFFICULT").click().run()
    after = at.session_state["fb"][pid]["caps"][rec["domain"]]
    assert after == max(1, min(before, rec["intensity"] - 1))


def test_reset_and_learned_weights(at):
    pid = at.session_state["patient"]
    at.sidebar.button(key="reset").click().run()
    assert pid not in at.session_state["fb_log"] or at.session_state["fb_log"][pid] == []
    at.sidebar.radio(key="weights").set_value("Learned (simulated feedback)").run()
    assert not at.exception
    assert len(rec_titles(at)) == 5
