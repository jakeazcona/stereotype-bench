import pytest

from stereotype_bench.costs.db import CallRecord, CostDB
from stereotype_bench.costs.guard import BudgetExceeded, BudgetGuard
from stereotype_bench.costs.pricing import PricingTable


def _record(db, **kwargs):
    defaults = dict(
        run_id="r1",
        provider="openrouter",
        model="m",
        purpose="generation",
        prompt_id="p",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
    )
    defaults.update(kwargs)
    db.record(CallRecord(**defaults))


def test_db_record_and_total(tmp_path):
    db = CostDB(path=tmp_path / "costs.sqlite")
    _record(db, cost_usd=0.001)
    _record(db, cost_usd=0.002, run_id="r2")
    assert db.total() == pytest.approx(0.003)
    assert db.total(run_id="r1") == pytest.approx(0.001)
    assert db.total(run_id="r2") == pytest.approx(0.002)
    assert db.total(run_id="missing") == 0.0


def test_db_by_model(tmp_path):
    db = CostDB(path=tmp_path / "costs.sqlite")
    _record(db, model="a", cost_usd=1.0)
    _record(db, model="b", cost_usd=2.0)
    _record(db, model="a", cost_usd=0.5)
    rows = db.by_model()
    assert {r["model"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["model"] == "a")
    assert a["n"] == 2
    assert a["cost"] == pytest.approx(1.5)


def test_db_to_csv(tmp_path):
    db = CostDB(path=tmp_path / "costs.sqlite")
    _record(db, cost_usd=0.123)
    csv_path = tmp_path / "out.csv"
    n = db.to_csv(csv_path)
    assert n == 1
    assert csv_path.exists()
    assert "0.123" in csv_path.read_text()


def test_guard_blocks_over_budget(tmp_path):
    db = CostDB(path=tmp_path / "costs.sqlite")
    _record(db, cost_usd=99.5)
    guard = BudgetGuard(budget_usd=100.0, db=db)
    guard.check(anticipated_usd=0.4)  # under cap, ok
    with pytest.raises(BudgetExceeded):
        guard.check(anticipated_usd=1.0)


def test_guard_force_override(tmp_path):
    db = CostDB(path=tmp_path / "costs.sqlite")
    _record(db, cost_usd=200.0)
    guard = BudgetGuard(budget_usd=100.0, db=db, force=True)
    guard.check()  # no exception even though over cap


def test_pricing_lookup():
    pt = PricingTable({"models": {"foo/bar": {"input": 3.0, "output": 15.0}}})
    cost = pt.cost_usd("foo/bar", 1000, 500)
    assert cost == pytest.approx(3.0 / 1000 + 15.0 / 2000)
    assert pt.cost_usd("unknown/model", 100, 100) is None
    assert "foo/bar" in pt.known_models()
