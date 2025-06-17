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

import copy
import logging
import typing
from datetime import datetime
from typing import IO, Self

import elinkapi
import yaml
from pydantic import AnyHttpUrl, BaseModel, field_serializer

"""
Aim
---

Generate Dataset DOIs for a single data release.

Since we only know the relationship to the instrument 10.71929/rubin/2561361 that
is the only one we can insert: IsCollectedBy and that only for the main DOI.
All other relationships must be fixed up afterwards.

RORs:
* Rubin: https://ror.org/048g3cy84
* NOIRLab: https://ror.org/03zmsge54
* SLAC: https://ror.org/05gzmn429

* infix: Rubin
* Title: Legacy Survey of Space and Time Data Preview 1
* Date: 2025-06-30
* Abstract

* Creators
  * "NSF-DOE Vera C. Rubin Observatory" (organizational)

* Publisher Name: "SLAC National Accelerator Laboratory (SLAC), Menlo Park, CA (United States)"
* Contributors (organizational; HostingInstitution)
  * "SLAC National Accelerator Laboratory (SLAC), Menlo Park, CA (United States)"
  * "NOIRLab"

* Funding:
  * SLAC: AC02-76SF00515 "DOE Contract Number"
  * NSF: AST-1258333 and AST-1836783  "NSF Cooperative Agreement"

sizes?
formats?


"""

_LOG = logging.getLogger("lsst.doiutils")

_LOCATION = elinkapi.Geolocation(
    type="POINT",
    label="Cerro Pachón, Chile",
    points=[elinkapi.Geolocation.Point(latitude=-30.244639, longitude=-70.749417)],
)

# Affiliations can be pre-calculated
_AFFILIATIONS = {
    "Rubin": elinkapi.Affiliation(
        name="NSF-DOE Vera C. Rubin Observatory", ror_id="https://ror.org/048g3cy84"
    ),
    "NOIRLab": elinkapi.Affiliation(
        name="U.S. National Science Foundation National Optical-Infrared Astronomy Research Laboratory",
        ror_id="https://ror.org/03zmsge54",
    ),
    "SLAC": elinkapi.Affiliation(
        name="SLAC National Accelerator Laboratory", ror_id="https://ror.org/05gzmn429"
    ),
}

_ORGANIZATIONS = {
    "Rubin": elinkapi.Organization(
        type="AUTHOR", name="NSF-DOE Vera C. Rubin Observatory", ror_id="https://ror.org/048g3cy84"
    ),
    "NOIRLab": elinkapi.Organization(
        type="RESEARCHING",
        name="U.S. National Science Foundation National Optical-Infrared Astronomy Research Laboratory",
        ror_id="https://ror.org/03zmsge54",
    ),
    "SLAC": elinkapi.Organization(
        type="RESEARCHING", name="SLAC National Accelerator Laboratory", ror_id="https://ror.org/05gzmn429"
    ),
    "NSF": elinkapi.Organization(
        type="SPONSOR", name="U.S. National Science Foundation", ror_id="https://ror.org/021nxhr62"
    ),
    "DOE": elinkapi.Organization(
        type="SPONSOR", name="U.S. Department of Energy Office of Science", ror_id="https://ror.org/00mmn6b08"
    ),
}

_IDENTIFIERS = [
    elinkapi.Identifier(type="CN_DOE", value="AC02-76SF00515"),
    elinkapi.Identifier(type="CN_NONDOE", value="NSF Cooperative Agreement AST-1258333"),
    elinkapi.Identifier(type="CN_NONDOE", value="NSF Cooperative Support Agreement AST-1836783"),
]


class DatasetTypeSource(BaseModel):
    """Specific description of butler vs TAP dataset."""

    name: str
    doi: str | None = None
    osti_id: int | None = None
    count: int | None = None


class DataReleaseDatasetType(BaseModel):
    """A component dataset type found within a data release."""

    abstract: str
    path: str
    butler: DatasetTypeSource | None = None
    tap: DatasetTypeSource | None = None

    def get_record_key(self, variant: str) -> str:
        """Return a key to associate with an OSTI record prior to assigning
        an OSTI ID.

        Parameters
        ----------
        variant : `str`
            "tap" or "butler"

        Returns
        -------
        key : `str`
            Key to use in `dict` of records.
        """
        if variant == "butler":
            if not self.butler:
                raise ValueError("Butler key requested but not a butler dataset type.")
            return self.butler.name
        elif variant == "tap":
            if not self.tap:
                raise ValueError("Tap key requested but not a butler dataset type.")
            return self.tap.name
        raise RuntimeError(f"Unrecognized variant {variant} for key calculation.")

    def set_saved_metadata(self, key: str, osti_id: int, doi: str) -> bool:
        """Try to set the DOI for the given key.

        Parameters
        ----------
        key : `str`
            Key associated with this DOI information.
        ost_id : `int`
            The OSTI ID to associate with this key.
        doi : `str`
            The DOI associated with this key.

        Returns
        -------
        updated : `bool`
            `True` if the key is known to this dataset type and the record
            was updated. `False` if the key was not recognized and likely
            associated with another dataset type.
        """
        if self.butler and self.butler.name == key:
            self.butler.doi = doi
            self.butler.osti_id = osti_id
        elif self.tap and self.tap.name == key:
            self.tap.doi = doi
            self.tap.osti_id = osti_id
        else:
            return False
        return True


class DataReleaseConfig(BaseModel):
    """Configuration class describing all the DOIs to be created for a
    data release.
    """

    title: str
    site_url: AnyHttpUrl
    date: datetime
    abstract: str
    instrument_doi: str
    dataset_types: list[DataReleaseDatasetType]
    doi: str | None = None
    osti_id: int | None = None

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
        config_dict = yaml.safe_load(fh)
        return cls.model_validate(config_dict)

    def write_yaml_fh(self, fh: IO[str]) -> None:
        """Write this configuration as YAML to the given file handle.

        Parameters
        ----------
        fh : `typing.IO`
            Open file handle associated to use for writing the YAML.
        """
        yaml.safe_dump(self.model_dump(exclude_unset=True), fh)

    def set_saved_metadata(self, key: str | None, saved_record: elinkapi.Record) -> None:
        """Update the configuration to reflect that a DOI has been saved.

        Parameters
        ----------
        key : `str`
            Key associated with the dataset that has been saved. This can be
            `None` to indicating the primary DOI, or the butler or tap name
            to indicate a specific dataset type.
        """
        osti_id = saved_record.osti_id
        if osti_id is None:
            raise RuntimeError("This record does not correspond to a saved record.")
        doi = saved_record.doi
        if not doi:
            raise RuntimeError("No DOI associated with this record. Was it saved?")
        if key is None:
            self.doi = doi
            self.osti_id = osti_id
            return

        # Find the corresponding key.
        for dtype in self.dataset_types:
            if dtype.set_saved_metadata(key, osti_id, doi):
                return

        raise RuntimeError(f"Key {key} not found in this configuration. Unable to set OSTI ID.")


def _make_sub_record(
    primary_content: dict[str, typing.Any], extra_title: str, abstract: str, extra_path: str
) -> elinkapi.Record:
    """Make a sub-record for a dataset."""
    # Modify a copy.
    dtype_content = copy.deepcopy(primary_content)
    dtype_content["description"] = abstract
    dtype_content["title"] += extra_title
    dtype_content["site_url"] += extra_path

    return elinkapi.Record.model_validate(dtype_content)


def make_records(config: DataReleaseConfig) -> dict[str | None, elinkapi.Record]:
    """Given a configuration, construct DOI records suitable for submission."""
    # Remove new lines from the abstract.
    abstract = config.abstract.replace("\n", " ").strip()

    instrument_relation = elinkapi.RelatedIdentifier(
        type="DOI", relation="IsCollectedBy", value=config.instrument_doi
    )

    # Primary dataset.
    record_content = {
        "product_type": "DA",
        "doi_infix": "rubin",
        "site_ownership_code": "SLAC-LSST",
        "title": config.title,
        "site_url": str(config.site_url),
        "description": abstract,
        "related_identifiers": [instrument_relation],
        "identifiers": _IDENTIFIERS,
        "organizations": list(_ORGANIZATIONS.values()),
        "subject_category_code": ["79"],  # "79 ASTRONOMY AND ASTROPHYSICS"
        "publication_date": config.date,
        "publisher_information": (
            "SLAC National Accelerator Laboratory (SLAC), Menlo Park, CA (United States)"
        ),
        "access_limitations": ["UNL"],
        "geolocations": [_LOCATION],
    }
    records: dict[str | None, elinkapi.Record] = {}

    # Primary DOI uses a None key.
    if not config.osti_id:
        records[None] = elinkapi.Record.model_validate(record_content)
    else:
        _LOG.info("DOI already assigned for primary dataset: %d", config.osti_id)

    # Should we strip the instrument DOI from subset DOIs. We want them
    # to be solely isPartOf the full data release and not themselves be
    # isCollectedBy the instrument?
    for dataset_type in config.dataset_types:
        dtype_abstract = dataset_type.abstract.replace("\n", " ").strip()
        # Lower case the first character for grammar given that extra_text
        # below starts a sentence.
        dtype_abstract = dtype_abstract[0].lower() + dtype_abstract[1:]

        uniquify_paths = False
        if dataset_type.butler and dataset_type.tap:
            # Both datasets exist and will be given distinct DOIs but we
            # should ensure that they have different target URLs.
            uniquify_paths = True

        if dataset_type.butler:
            extra_text = (
                "This dataset is a subset of the full data release"
                f" consisting of the {dataset_type.butler.name} dataset type. These are "
            )

            abstract = typing.cast(str, record_content["description"]) + "\n\n" + extra_text + dtype_abstract
            fragment = "#butler" if uniquify_paths else ""

            if not dataset_type.butler.osti_id:
                records[dataset_type.get_record_key("butler")] = _make_sub_record(
                    record_content,
                    f": {dataset_type.butler.name} dataset type",
                    abstract,
                    dataset_type.path + fragment,
                )
            else:
                _LOG.info(
                    "DOI already assigned for %s: %d", dataset_type.butler.name, dataset_type.butler.osti_id
                )

        if dataset_type.tap:
            extra_text = (
                "This dataset is a subset of the full data release consisting of "
                f"a searchable catalog named {dataset_type.tap.name}. This catalog contains "
            )
            abstract = typing.cast(str, record_content["description"]) + "\n\n" + extra_text + dtype_abstract
            fragment = "#tap" if uniquify_paths else ""

            if not dataset_type.tap.osti_id:
                records[dataset_type.get_record_key("tap")] = _make_sub_record(
                    record_content,
                    f": {dataset_type.tap.name} searchable catalog",
                    abstract,
                    dataset_type.path + fragment,
                )
            else:
                _LOG.info("DOI already assigned for %s: %d", dataset_type.tap.name, dataset_type.tap.osti_id)

    return records
