# This file is part of lsst-doiutils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the LICENSE file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

"""Test the command line interface."""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

from lsst.doiutils._cli import cli

from .test_dataset_records import _make_bibtex_config


def test_create_dataset_bibs_per_type() -> None:
    """Each BibTeX entry is written to its own file in the current directory."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w") as fh:
            _make_bibtex_config().write_yaml_fh(fh)

        result = runner.invoke(cli, ["create-dataset-bibs", "config.yaml", "--create-per-type"])
        assert result.exit_code == 0, result.output

        written = sorted(p.name for p in pathlib.Path().glob("*.bib"))
        assert written == [
            "butler-object.bib",
            "butler-survey-property.bib",
            "dataset.bib",
            "misc-extra.bib",
            "tap-Object.bib",
        ]

        content = pathlib.Path("tap-Object.bib").read_text()
        assert content.startswith("@misc{10.71929/rubin/1002,\n")
        assert content.endswith("}\n")


def test_create_dataset_bibs_stdout() -> None:
    """Without the flag all entries are written to standard output."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w") as fh:
            _make_bibtex_config().write_yaml_fh(fh)

        result = runner.invoke(cli, ["create-dataset-bibs", "config.yaml"])
        assert result.exit_code == 0, result.output

        assert not list(pathlib.Path().glob("*.bib"))
        assert result.output.count("@misc{") == 5
