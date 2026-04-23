from typer.testing import CliRunner

from stereotype_bench.cli import app

runner = CliRunner()


def test_list_measures():
    result = runner.invoke(app, ["list-measures"])
    assert result.exit_code == 0
    assert "stub" in result.stdout


def test_list_tasks():
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == 0
    assert "first_impression" in result.stdout


def test_list_models():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0


def test_costs_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("STEREOTYPE_BENCH_DB_PATH", str(tmp_path / "costs.sqlite"))
    result = runner.invoke(app, ["costs"])
    assert result.exit_code == 0
    assert "Total" in result.stdout
