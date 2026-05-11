from typer.testing import CliRunner

from patent.cli import app

runner = CliRunner()


def test_cli_top_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MLOps Patent Pipeline CLI" in result.output
    assert "pipeline" in result.output
    assert "data" in result.output
    assert "model" in result.output


def test_cli_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for cmd in ["init", "update", "reserialize", "clean", "embed"]:
        assert cmd in result.output


def test_cli_model_train_help():
    result = runner.invoke(app, ["model", "train", "--help"])
    assert result.exit_code == 0
    assert "LSHiForest" in result.output
    assert "--num-trees" in result.output
    assert "--max-depth" in result.output


def test_cli_model_evaluate_help():
    result = runner.invoke(app, ["model", "evaluate", "--help"])
    assert result.exit_code == 0
    assert "evaluat" in result.output.lower()


def test_cli_pipeline_help():
    result = runner.invoke(app, ["pipeline", "--help"])
    assert result.exit_code == 0
    assert "--raw" in result.output
    assert "--force" in result.output
