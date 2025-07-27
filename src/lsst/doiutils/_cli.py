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
import re
import sys
from typing import IO

import click
import elinkapi

from . import __version__
from ._datasets import DataReleaseConfig, publish_records, submit_records, update_record_relationships
from ._instruments import InstrumentConfig, submit_instrument
from ._papers import PaperConfig, publish_paper, submit_paper, update_paper_author_refs

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
        logging.basicConfig(filename=log_file, level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.WARNING)
    # Force LSST logger to INFO.
    logging.getLogger("lsst").setLevel(logging.INFO)


@cli.command("save-dataset-dois")
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
def save_dataset_dois(
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
    api = elinkapi.Elink(target=server, token=token)
    n_saved = submit_records(dr_config, api, dry_run=dry_run)

    if n_saved > 0:
        dr_config.write_yaml_fh(sys.stdout)


@cli.command("update-dataset-relationships")
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
def update_dataset_relationships(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Update relationships between DOIs within a data release.

    CONFIG is the configuration file containing a full description of all
    the datasets that are part of this data release and their associated
    DOIs and OSTI IDs from a previous upload.
    """
    dr_config = DataReleaseConfig.from_yaml_fh(config)

    api = elinkapi.Elink(target=server, token=token)

    update_record_relationships(dr_config, api, dry_run=dry_run)


@cli.command("publish-dataset-dois")
@click.argument("config", type=click.File())
@click.option("--dry-run/--no-dry-run", default=False, help="Process the configuration without publishing.")
@click.option("--token", default="", type=str, help="Auth token to use for DOI release.")
@click.option(
    "--server",
    default="https://review.osti.gov/elink2api/",
    help="Desired endpoint to use for release. Default is to use the test server. "
    "For a final submission use https://www.osti.gov/elink2api/",
)
@click.pass_context
def publish_dataset_dois(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Publish the DOIs associated with this configuration.

    CONFIG is the configuration file containing a full description of all
    the datasets that are part of this data release and their associated
    DOIs and OSTI IDs from a previous upload.
    """
    dr_config = DataReleaseConfig.from_yaml_fh(config)

    api = elinkapi.Elink(target=server, token=token)

    publish_records(dr_config, api, dry_run=dry_run)


@cli.command("count-butler-datasets")
@click.argument("config", type=click.File())
@click.argument("repo", type=str)
@click.argument("collection", type=str)
@click.pass_context
def count_butler_datasets(
    ctx: click.Context,
    config: IO[str],
    repo: str,
    collection: str,
) -> None:
    """Count the number of datasets for each butler dataset type.

    CONFIG is the configuration file containing a full description of all
    the datasets that are part of this data release.

    REPO is the URI or label of a butler repository.

    COLLECTION is the Butler collection to search for datasets
    """
    # Optional butler dependency.
    from lsst.daf.butler import Butler, MissingDatasetTypeError

    dr_config = DataReleaseConfig.from_yaml_fh(config)
    butler = Butler.from_config(repo)

    collections = butler.collections.query(collection)

    updated = False
    for dtype in dr_config.dataset_types:
        if not dtype.butler:
            continue
        with butler.query() as query:
            # Calibrations is treated as a "magic" dataset type.
            dataset_types = [dtype.butler.name]
            if dtype.butler.name == "calibrations":
                all_types = butler.registry.queryDatasetTypes()
                dataset_types = [dt.name for dt in all_types if dt.isCalibration()]
            elif dtype.butler.alias:
                # Alias is an indication that we have multiple dataset types
                # in this dataset.
                dataset_types = [dt.name for dt in butler.registry.queryDatasetTypes(dtype.butler.alias)]

            count = 0
            for dataset_type in dataset_types:
                try:
                    # We want all we can find to be included in the count.
                    # This is especially important for calibrations.
                    results = query.datasets(dataset_type, collections=collections, find_first=False)
                    count += results.count(exact=True)
                except MissingDatasetTypeError:
                    _LOG.info("Dataset type %s not known to this repo.", dataset_type)

            if count > 0:
                _LOG.info("Number of datasets of type %s: %d", dtype.butler.name, count)
                dtype.butler.count = count
                updated = True

    if updated:
        dr_config.write_yaml_fh(sys.stdout)


@cli.command("generate-rst-replacements")
@click.argument("config", type=click.File())
@click.pass_context
def generate_rst_replacements(
    ctx: click.Context,
    config: IO[str],
) -> None:
    """Create the Restructured Text replacement strings required to build
    the data release documentation.

    CONFIG is the configuration file containing a full description of all
    the datasets that are part of this data release.
    """
    dr_config = DataReleaseConfig.from_yaml_fh(config)

    def _print_replacement(key: str, value: str | int | None) -> None:
        if isinstance(value, int):
            value = f"{value:,d}"  # Use comma separators.
        print(f".. |{key.replace(' ', '_')}| replace:: {value if value is not None else 'TBD'}")

    def _make_title(subtitle: str) -> str:
        return f"{dr_config.title}: {subtitle}"

    year = dr_config.date.year

    def _doi_to_rst(doi: str | None, title: str) -> str:
        """Convert the DOI to restructured text."""
        if not doi:
            return "TBD"
        return f"*Citation*: **NSF-DOE Vera C. Rubin Observatory** ({year}); {title} |doi_image| https://doi.org/{doi}"

    _print_replacement("dataset_doi", _doi_to_rst(dr_config.doi, dr_config.title))

    for dataset_type in dr_config.dataset_types:
        if butler := dataset_type.butler:
            name = butler.name
            _print_replacement(
                f"{name}_doi", _doi_to_rst(butler.doi, _make_title(butler.get_subtitle("butler")))
            )
            _print_replacement(f"{name}_butler_count", butler.count)
        if tap := dataset_type.tap:
            name = tap.name
            _print_replacement(f"{name}_doi", _doi_to_rst(tap.doi, _make_title(tap.get_subtitle("tap"))))
            _print_replacement(f"{name}_rows", tap.count)
            _print_replacement(f"{name}_columns", tap.count2)


@cli.command("save-paper-doi")
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
def save_paper_doi(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Create 'saved' DOI for a single paper.

    CONFIG is the configuration file containing a full description of
    the paper to be assigned a DOI.
    """
    paper_config = PaperConfig.from_yaml_fh(config)
    api = elinkapi.Elink(target=server, token=token)
    saved = submit_paper(paper_config, api, dry_run=dry_run)

    if saved:
        paper_config.write_yaml_fh(sys.stdout)


@cli.command("publish-paper-doi")
@click.argument("config", type=click.File())
@click.option("--dry-run/--no-dry-run", default=False, help="Process the configuration without publishing.")
@click.option("--token", default="", type=str, help="Auth token to use for DOI release.")
@click.option(
    "--server",
    default="https://review.osti.gov/elink2api/",
    help="Desired endpoint to use for release. Default is to use the test server. "
    "For a final submission use https://www.osti.gov/elink2api/",
)
@click.pass_context
def publish_paper_doi(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Publish the DOI associated with this configuration.

    CONFIG is the configuration file containing a full description of the
    the paper with DOIs and OSTI IDs from a previous upload.
    """
    paper_config = PaperConfig.from_yaml_fh(config)

    api = elinkapi.Elink(target=server, token=token)

    publish_paper(paper_config, api, dry_run=dry_run)


@cli.command("update-paper-info")
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
def update_paper_info(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Update the authors and references in the DOI record using the given
    configuration.

    CONFIG is the configuration file containing a full description of the
    the paper with DOIs and OSTI IDs from a previous upload.
    """
    paper_config = PaperConfig.from_yaml_fh(config)
    api = elinkapi.Elink(target=server, token=token)
    update_paper_author_refs(paper_config, api, dry_run=dry_run)


@cli.command("save-instrument-doi")
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
def save_instrument_doi(
    ctx: click.Context,
    config: IO[str],
    dry_run: bool,  # noqa: FBT001
    token: str,
    server: str,
) -> None:
    """Create 'saved' DOI for a single instrument.

    CONFIG is the configuration file containing a full description of
    the instrument to be assigned a DOI.
    """
    instr_config = InstrumentConfig.from_yaml_fh(config)
    api = elinkapi.Elink(target=server, token=token)
    saved = submit_instrument(instr_config, api, dry_run=dry_run)

    if saved:
        instr_config.write_yaml_fh(sys.stdout)


@cli.command("extract-references")
@click.argument("source", type=click.File())
@click.pass_context
def extract_references(
    ctx: click.Context,
    source: IO[str],
) -> None:
    """Extract references DOIs from a source file.

    SOURCE is the file to be parsed. Writes discovered DOIs to standard output.
    """
    # Since we have a file handle and not a file name with a suffix, we
    # use common heuristics to look for DOIs.
    doi_re = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"

    # Rather than doing a wide open DOI search, be a bit more intentional
    # about the variants we are looking for.
    regexes = (
        rf"doi\{{({doi_re})\}}",  # mn@doi{DOI} BBL files.
        rf"doi.org/({doi_re})",  # doi.org/DOI URLs.
        rf"doi\s+=\s*\{{({doi_re})\}}",  # doi = {DOI} bibtex entries.
        rf"doi:({doi_re})",  # doi:DOI
    )

    # Match everything in one go. Assumes the input source file is not
    # multiple gigabytes.
    matches = re.findall("|".join(regexes), source.read(), flags=re.IGNORECASE | re.MULTILINE)

    if not matches:
        print("No DOIs found in source", file=sys.stderr)

    # Because we capture within the regex in findall we get tuples of groups
    # back. Store DOIs in a dict with a case-insensitive key and the actual
    # DOI as the value.
    dois: dict[str, str] = {}
    for m in matches:
        # A DOI can't end with a "." but can have one inside it so if we
        # have matched something where the DOI is in a sentence strip the
        # final period.
        found = {doi.removesuffix(".") for doi in m if doi}
        dois.update({doi.casefold(): doi for doi in found})

    for _, doi in sorted(dois.items()):
        print(f"- {doi}")
