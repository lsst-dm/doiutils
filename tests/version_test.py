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

"""Test the packaging."""

from __future__ import annotations

from lsst.doiutils import __version__


def test_version() -> None:
    """Ensure that the version is set."""
    assert isinstance(__version__, str)
    # Indicates the package is not installed otherwise
    assert __version__ != "0.0.0"
