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

"""Test DOI instrument record creation."""

from __future__ import annotations

import datetime

import pydantic
import pytest

from lsst.doiutils._instruments import InstrumentConfig


def _make_instrument_config(relationships: dict[str, list[str]]) -> InstrumentConfig:
    return InstrumentConfig(
        title="An Instrument",
        site_url="https://example.test/",  # type: ignore[arg-type]
        date=datetime.date(2026, 1, 1),
        abstract="An instrument description.",
        authors=["Rubin"],
        researching=["Rubin"],
        sponsoring=["NSF"],
        doi="10.71929/rubin/1",
        osti_id=1234,
        relationships=relationships,
    )


def test_instrument_config_accepts_other_dois() -> None:
    """An instrument related to other DOIs is accepted."""
    config = _make_instrument_config({"HasPart": ["10.71929/rubin/2"]})

    assert config.relationships == {"HasPart": ["10.71929/rubin/2"]}


def test_instrument_config_rejects_self_reference() -> None:
    """An instrument may not be related to its own DOI."""
    with pytest.raises(pydantic.ValidationError, match=r"10\.71929/rubin/1 refers to itself"):
        _make_instrument_config({"HasPart": ["10.71929/rubin/1"]})
