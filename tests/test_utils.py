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

"""Test the shared configuration helpers."""

from __future__ import annotations

import pytest

from lsst.doiutils._utils import check_self_reference


def test_check_self_reference_allows_unrelated_dois() -> None:
    """A record that only refers to other DOIs is accepted."""
    check_self_reference(
        "10.71929/rubin/1000",
        {"Cites": ["10.71929/rubin/1001", "10.5281/ZENODO.3773449"]},
    )


def test_check_self_reference_rejects_own_doi() -> None:
    """A record may not refer to its own DOI."""
    with pytest.raises(ValueError, match=r"10\.71929/rubin/1000.*Cites"):
        check_self_reference(
            "10.71929/rubin/1000",
            {"Cites": ["10.71929/rubin/1001", "10.71929/rubin/1000"]},
        )


def test_check_self_reference_ignores_doi_case() -> None:
    """DOIs are case-insensitive so a differently-cased self reference is
    still rejected.
    """
    with pytest.raises(ValueError, match=r"IsCitedBy"):
        check_self_reference("10.26624/SKGZ6035", {"IsCitedBy": ["10.26624/skgz6035"]})


def test_check_self_reference_reports_every_label() -> None:
    """All the labels containing a self reference are reported together."""
    with pytest.raises(ValueError, match=r"Cites, IsCitedBy"):
        check_self_reference(
            "10.71929/rubin/1000",
            {
                "Cites": ["10.71929/rubin/1000"],
                "IsCitedBy": ["10.71929/rubin/1000"],
                "IsPartOf": ["10.71929/rubin/1001"],
            },
        )


@pytest.mark.parametrize("doi", [None, ""])
def test_check_self_reference_allows_unassigned_doi(doi: str | None) -> None:
    """A record with no DOI of its own cannot refer to itself."""
    check_self_reference(doi, {"Cites": ["10.71929/rubin/1001"]})


def test_check_self_reference_ignores_unset_related_dois() -> None:
    """Optional related DOIs that have not been set are skipped."""
    check_self_reference("10.71929/rubin/1000", {"description_paper": [None]})
