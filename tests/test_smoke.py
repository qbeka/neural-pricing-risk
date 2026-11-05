"""The whole pipeline on a tiny config, through the same functions as the command line."""

import argparse

import pytest

from asianrisk.cli import smoke


@pytest.mark.slow
def test_every_stage_runs_end_to_end(capsys):
    smoke(argparse.Namespace())
    assert "smoke run passed" in capsys.readouterr().out
