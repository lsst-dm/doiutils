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

__all__: list[str] = []

import datetime
import logging
import typing
from typing import IO, Self

import elinkapi
from pydantic import AfterValidator, AnyHttpUrl, BaseModel, field_serializer

from ._constants import FUNDING_ORGANIZATIONS, IDENTIFIERS, LOCATION, ORGANIZATION_AUTHORS
from ._utils import strip_newlines
from ._yaml import load_yaml_fh, prepare_block_text_for_writing, write_to_yaml_fh

"""
Aim
---

Generate instrument DOIs for an instrument.

RORs:
* Rubin: https://ror.org/048g3cy84
* NOIRLab: https://ror.org/03zmsge54
* SLAC: https://ror.org/05gzmn429

This is not generic so is mostly throw away code for LSSTCam and LATISS.

"""

_LOG = logging.getLogger("lsst.doiutils")


class InstrumentConfig(BaseModel):
    """Configuration class describing all the DOIs to be created for a
    data release.
    """

    title: str
    """Main title for the data release."""
    site_url: AnyHttpUrl
    """Main landing page for this data release."""
    date: datetime.date
    """Date the instrument went on telescope (YYYY-MM-DD)"""
    abstract: typing.Annotated[str, AfterValidator(strip_newlines)]
    """Description of this data release. Will be included with dataset type
    descriptions.
    """
    authors: list[str]
    """IDs of organizational authors."""
    researching: list[str]
    """IDs of researching contributors."""
    sponsoring: list[str]
    """IDs of sponsoring organizations."""
    relationships: dict[str, list[str]]
    """DOI relationships to this instrument where the keys are the relationship
    type and the values are DOIs.
    """
    doi: str | None = None
    """DOI of the instrument."""
    osti_id: int | None = None
    """OSTI ID of the data release (required to retrieve and edit the record
    in ELink).
    """

    @field_serializer("site_url")
    def serialize_url(self, url: AnyHttpUrl) -> str:
        return str(url)

    @classmethod
    def from_yaml_fh(cls, fh: IO[str]) -> Self:
        """Create a configuration from a file handle pointing to a YAML
        file.

        Parameters
        ----------
        fh : `typing.IO`
            Open file handle associated with a YAML configuration.
        """
        config_dict = load_yaml_fh(fh)
        return cls.model_validate(config_dict, strict=True)

    def write_yaml_fh(self, fh: IO[str]) -> None:
        """Write this configuration as YAML to the given file handle.

        Parameters
        ----------
        fh : `typing.IO`
            Open file handle associated to use for writing the YAML.
        """
        model = self.model_dump(exclude_unset=True)

        model["abstract"] = prepare_block_text_for_writing(model["abstract"])
        write_to_yaml_fh(model, fh)

    def set_saved_metadata(self, saved_record: elinkapi.Record) -> None:
        """Update the configuration to reflect that a DOI has been saved."""
        osti_id = saved_record.osti_id
        if osti_id is None:
            raise RuntimeError("This record does not correspond to a saved record.")
        doi = saved_record.doi
        if not doi:
            raise RuntimeError("No DOI associated with this record. Was it saved?")
        self.doi = doi
        self.osti_id = osti_id


def make_instrument_record(config: InstrumentConfig) -> elinkapi.Record:
    """Given a configuration, construct instrument DOI record suitable for
    submission.
    """
    related_identifiers: list[elinkapi.RelatedIdentifier] = []
    for relationship, related_dois in config.relationships.items():
        related_identifiers.extend(
            elinkapi.RelatedIdentifier(type="DOI", relation=relationship, value=related)
            for related in related_dois
        )

    # For the cameras we need to specify
    # AUTHOR
    # RESEARCHING
    # SPONSOR
    # organizations.
    organizations: list[elinkapi.Organization] = []
    organizations.extend(ORGANIZATION_AUTHORS[auth] for auth in config.authors)
    organizations.extend(FUNDING_ORGANIZATIONS[res] for res in config.researching + config.sponsoring)

    # Primary dataset.
    record_content = {
        "product_type": "DA",  # Currently this has to be changed to IN later.
        "doi_infix": "rubin",
        "site_ownership_code": "SLAC-LSST",
        "title": config.title,
        "site_url": str(config.site_url),
        "description": config.abstract,
        "related_identifiers": related_identifiers,
        "identifiers": IDENTIFIERS,
        "organizations": organizations,
        "subject_category_code": ["79"],  # "79 ASTRONOMY AND ASTROPHYSICS"
        "publication_date": config.date,
        "publisher_information": (
            # This is for internal use and won't be part of DataCite record.
            "SLAC National Accelerator Laboratory (SLAC), Menlo Park, CA (United States)"
        ),
        "access_limitations": ["UNL"],
        "geolocations": [LOCATION],
    }

    if config.osti_id:
        raise RuntimeError(f"DOI already assigned to this configuration: {config.osti_id}")

    return elinkapi.Record.model_validate(record_content, strict=True)


def get_record(config: InstrumentConfig, elink: elinkapi.Elink) -> elinkapi.Record:
    """Retrieve records associated with this configuration."""
    if not config.osti_id:
        raise ValueError("No OSTI ID found for this config. Was this configuration updated after upload?")

    record = elink.get_single_record(config.osti_id)
    if record.osti_id != config.osti_id:
        raise RuntimeError("Unexpectedly got different OSTI ID from record than requested")

    return record


def submit_instrument(config: InstrumentConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> bool:
    """Submit instrument from configuration to ELink."""
    record = make_instrument_record(config)

    if dry_run:
        print(record.model_dump_json(exclude_defaults=True, indent=2))
        return False

    saved_record = elink.post_new_record(record, "save")
    _LOG.info("Saved instrument with DOI %s", saved_record.doi)
    config.set_saved_metadata(saved_record)

    return True


def update_instrument(
    record: elinkapi.Record, elink: elinkapi.Elink, *, dry_run: bool = False, state: str = "save"
) -> None:
    """Update the given record, potentially including final submission."""
    _LOG.info("Uploading record with state %s", state)
    if dry_run:
        print(record.model_dump_json(indent=2))
    else:
        try:
            elink.update_record(record.osti_id, record, state)
        except Exception:
            _LOG.error(f"Error uploading record with OSTI ID {record.osti_id}")
            raise


def publish_instrument(config: InstrumentConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> None:
    """Publish the records.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.
    """
    _LOG.info("Retrieving records from OSTI")
    saved_record = get_record(config, elink)

    update_instrument(saved_record, elink, dry_run=dry_run, state="submit")
