"""Integration tests on the real processed data (skipped if Phases 2-5 have not been run)."""

import pytest

from lphrs.config import PROCESSED_DIR

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "cvd_risk.parquet").exists(),
                                reason="processed data not built")


@pytest.fixture(scope="module")
def data():
    from lphrs import pipeline as PL
    return PL.load_data()


def test_pipeline_returns_five_safe_explained_items(data):
    from lphrs import pipeline as PL
    from lphrs import safety as S
    pid = data.cohort_ids[0]
    res = PL.run_pipeline(pid, data)
    recs = res["recommendations"]
    assert len(recs) == 5
    assert res["minutes_used"] <= res["user_context"]["minutes_per_day"]
    assert all(r["reasons"] and r["disclaimer"] == "Wellness guidance only. Not medical advice." for r in recs)
    chosen = data.candidates[data.candidates["id"].isin({r["id"] for r in recs})]
    assert S.find_violations(chosen, res["safety_facts"]).empty
    assert res["profile"]["cvd_risk"] is not None


def test_switches_for_experiments(data):
    from lphrs import pipeline as PL
    pid = data.cohort_ids[1]
    no_safety = PL.run_pipeline(pid, data, use_safety=False)
    assert no_safety["n_after_safety"] == no_safety["n_candidates"] and no_safety["drops"].empty
    no_u = PL.run_pipeline(pid, data, use_urgency=False)
    assert (no_u["scored"]["U"] == 0).all()
    topk = PL.run_pipeline(pid, data, selection="topk")
    assert len(topk["recommendations"]) == 5
