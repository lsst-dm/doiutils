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

# YAML helper packages.

from __future__ import annotations

__all__ = ["load_yaml_fh", "prepare_block_text_for_writing", "write_to_yaml_fh"]

import io
import textwrap
import typing

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


def load_yaml_fh(fh: typing.IO[str]) -> dict[str, typing.Any]:
    """Read YAML data from file handle.

    Parameters
    ----------
    fh : `typing.IO`
        Open file handle associated with a YAML configuration.
    """
    yaml_loader = YAML()
    return yaml_loader.load(fh)


def write_to_yaml_fh(model: dict[str, typing.Any], fh: typing.IO[str]) -> None:
    """Write the supplied dictionary to the file in YAML format.

    Parameters
    ----------
    model : `dict` [ `str` , `typing.Any` ]
        Model to write.
    fh : `typing.IO`
        Open file handle to write to.

    Notes
    -----
    When a long plain scalar is folded at the output width the space that the
    line break replaced is retained at the end of the line. Those trailing
    spaces are removed before writing since they are not significant and
    trailing whitespace is not allowed in these files.
    """
    buffer = io.StringIO()
    YAML().dump(model, buffer)
    for line in buffer.getvalue().splitlines():
        print(line.rstrip(), file=fh)


def prepare_block_text_for_writing(text: str, indent: int = 2) -> str | LiteralScalarString:
    """Format text for writing to YAML file.

    Parameters
    ----------
    text : `str`
        Text to re-format.
    indent : `int`, optional
        Expected indent level in characters for this text block.

    Returns
    -------
    new_text : `str` or `LiteralScalarString`
        The potentially reformatted text or the original text. Will be
        unchanged if the original text is less than 69 characters.
    """
    if len(text) > 68:
        return LiteralScalarString(textwrap.fill(text, width=(80 - indent)))

    return text
