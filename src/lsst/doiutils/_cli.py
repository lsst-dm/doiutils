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

import logging
import sys
from typing import IO

import click
import elinkapi

from . import __version__
from ._datasets import DataReleaseConfig, make_records

_LOG = logging.getLogger("lsst.doiutils")

loglevel_choices = ["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"]


@click.group()
@click.version_option(__version__)
@click.option(
    "--log-level",
    type=click.Choice(loglevel_choices),
    help="Log level",
    default=logging.getLevelName(logging.INFO),
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log file path",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str, log_file: str | None) -> None:
    """Rubin LSST DOI utility command line tools."""
    ctx.ensure_object(dict)
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level)
    else:
        logging.basicConfig(level=log_level)


@cli.command("datasets")
@click.argument("config", type=click.File())
@click.option("--dry-run/--no-dry-run", default=False, help="Process the configuration without submitting.")
@click.option("--token", default="", type=str, help="Auth token to use for DOI submission.")
@click.option(
    "--server",
    default="https://review.osti.gov/elink2api/",
    help="Desired endpoint to use for submission. Default is to use the test server. "
    "For a final submission use https://www.osti.gov/elink2api/",
)
@click.pass_context
def datasets(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Create 'saved' DOIs for a data release.

    CONFIG is the configuration file containing a full description of all
    the datasets that are part of this data release.
    """
    dr_config = DataReleaseConfig.from_yaml_fh(config)
    records = make_records(dr_config)

    if dry_run:
        for rec in records.values():
            print(rec.model_dump_json(exclude_defaults=True, indent=2))
        return

    _LOG.info("Will be submitting %d records.", len(records))

    api = elinkapi.Elink(target=server, token=token)

    # Submit each record, recording the issued OSTI ID as we go so that
    # we can update the configuration (to prevent new uploads of the same
    # thing).

    for i, (key, record) in enumerate(records.items()):
        saved_record = api.post_new_record(record, "save")
        print(saved_record.osti_id, saved_record.doi)

        dr_config.set_saved_metadata(key, saved_record)

        if i == 2:
            break

    dr_config.write_yaml_fh(sys.stdout)
