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

__all__ = ["strip_newlines"]


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
