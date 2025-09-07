from pathlib import Path

import pytest

from ssspx import cli

FIXTURES = Path(__file__).parent / "fixtures"


def test_help_snapshot(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    expected = (FIXTURES / "cli_help.txt").read_text()
    assert out == expected


def test_example_snapshot(capsys):
    code = cli.main(["--example"])
    assert code == 200
    out = capsys.readouterr().out
    expected = (FIXTURES / "cli_example.csv").read_text()
    assert out == expected
