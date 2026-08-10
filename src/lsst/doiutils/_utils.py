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

from __future__ import annotations

__all__ = ["check_self_reference", "strip_newlines"]

from collections.abc import Iterable, Mapping


def check_self_reference(doi: str | None, related: Mapping[str, Iterable[str | None]]) -> None:
    """Check that a record does not refer to its own DOI.

    Parameters
    ----------
    doi : `str` or `None`
        The DOI of the record itself. A record with no DOI assigned yet
        cannot refer to itself and is always accepted.
    related : `~collections.abc.Mapping`
        The DOIs this record refers to, keyed by a label describing where
        they came from. The label is used in the error message so it should
        be the relationship type or the field name. Values that are `None`
        are ignored.

    Raises
    ------
    ValueError
        Raised if ``doi`` appears anywhere in ``related``.

    Notes
    -----
    DOIs are case-insensitive so the comparison ignores case.
    """
    if not doi:
        return

    normalized = doi.casefold()
    offending = sorted(
        label
        for label, related_dois in related.items()
        if any(related_doi and related_doi.casefold() == normalized for related_doi in related_dois)
    )
    if offending:
        raise ValueError(
            f"DOI {doi} refers to itself. A record can not be related to itself. "
            f"Found under: {', '.join(offending)}"
        )


def strip_newlines(text: str) -> str:
    """Replace new lines with spaces.

    All our dataset configs are single paragraphs and the YAML parser injects
    newlines that were not really there.

    Parameters
    ----------
    text : `str`
        Text to be corrected.

    Returns
    -------
    updated : `str`
        Text with newlines replaced with spaces.
    """
    return text.replace("\n", " ").strip()
