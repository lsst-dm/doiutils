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

from __future__ import annotations

import datetime
import logging
import typing

from pydantic import AfterValidator, AnyHttpUrl, BaseModel, field_serializer

from ._utils import strip_newlines

_LOG = logging.getLogger("lsst.doiutils")


class Paper(BaseModel):
    """Description of a paper requiring a DOI."""

    title: str
    """Title of the paper."""
    site_url: AnyHttpUrl
    """Main landing page for this paper."""
    date: datetime.date
    """Date of paper publication. This is ambiguous for tech notes which
    continually update.
    """
    abstract: typing.Annotated[str, AfterValidator(strip_newlines)]
    doi: str | None = None
    """DOI of the data release."""
    osti_id: int | None = None
    """OSTI ID of the data release (required to retrieve and edit the record
    in ELink).
    """

    @field_serializer("site_url")
    def serialize_url(self, url: AnyHttpUrl) -> str:
        return str(url)
