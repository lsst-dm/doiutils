# This file is part of lsst-doiutils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

"""Test the command line interface."""

from __future__ import annotations

import pathlib

import pytest
from click.testing import CliRunner

from lsst.doiutils import _cli
from lsst.doiutils._cli import cli
from lsst.doiutils._yaml import load_yaml_fh

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


def _write_paper_config(handle: str, doi: str, relationships: str = "") -> None:
    """Write a minimal paper configuration to the configs directory."""
    path = pathlib.Path("configs") / f"{handle.lower()}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        f"""title: Paper {handle} with a title that is long enough that the YAML writer folds it
  on to a second line
handle: {handle}
site_url: https://{handle.lower()}.lsst.io/
date: 2025-01-01
abstract: Short abstract.
doi: {doi}
authors:
- timj
{relationships}"""
    )


def test_find_internal_citations() -> None:
    """Inverse relationships are added, sorted, and never duplicated."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_paper_config(
            "TSTN-001",
            "10.71929/rubin/1",
            """relationships:
  Cites:
  - 10.71929/rubin/3
  - 10.71929/rubin/2
  - 10.71929/rubin/2
""",
        )
        # The inverse relationship is already recorded, twice.
        _write_paper_config(
            "TSTN-002",
            "10.71929/rubin/2",
            """relationships:
  IsCitedBy:
  - 10.71929/rubin/1
  - 10.71929/rubin/1
""",
        )
        # No relationships section at all.
        _write_paper_config("TSTN-003", "10.71929/rubin/3")

        result = runner.invoke(cli, ["find-internal-citations"])
        assert result.exit_code == 0, result.output

        # Existing relationships are sorted with repeated DOIs removed.
        assert "Cites:\n  - 10.71929/rubin/2\n  - 10.71929/rubin/3\n" in _read_config("TSTN-001")
        assert "IsCitedBy:\n  - 10.71929/rubin/1\n" in _read_config("TSTN-002")

        # A missing inverse relationship is added to a config that had none.
        assert "IsCitedBy:\n  - 10.71929/rubin/1\n" in _read_config("TSTN-003")


def test_find_internal_citations_no_trailing_whitespace() -> None:
    """Folded long titles are written without trailing whitespace."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_paper_config(
            "TSTN-001", "10.71929/rubin/1", "relationships:\n  Cites:\n  - 10.71929/rubin/2\n"
        )
        _write_paper_config("TSTN-002", "10.71929/rubin/2")

        result = runner.invoke(cli, ["find-internal-citations"])
        assert result.exit_code == 0, result.output

        content = _read_config("TSTN-002")
        assert not [line for line in content.splitlines() if line != line.rstrip()]

        # The title is long enough that it is folded on to a second line but
        # the folding must not change its value.
        lines = content.splitlines()
        assert lines[1].startswith("  "), lines
        with open(pathlib.Path("configs") / "tstn-002.yaml") as fh:
            model = load_yaml_fh(fh)
        assert model["title"] == (
            "Paper TSTN-002 with a title that is long enough that the YAML writer"
            " folds it on to a second line"
        )


def _read_config(handle: str) -> str:
    """Read the configuration file for the given handle."""
    return (pathlib.Path("configs") / f"{handle.lower()}.yaml").read_text()


@pytest.mark.parametrize(
    ("options", "expected"),
    [([], None), (["--update-authors"], True), (["--no-update-authors"], False)],
)
def test_update_paper_info_author_option(
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    expected: bool | None,  # noqa: FBT001
) -> None:
    """The author update option is passed on as given, with the default
    reported distinctly from an explicit choice.
    """
    calls: list[bool | None] = []
    monkeypatch.setattr(
        _cli,
        "update_paper_author_refs",
        lambda config, api, *, dry_run, update_sponsors, update_authors: calls.append(update_authors),
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_paper_config("TSTN-001", "10.71929/rubin/1")
        result = runner.invoke(cli, ["update-paper-info", "configs/tstn-001.yaml", *options])
        assert result.exit_code == 0, result.output

    assert calls == [expected]
