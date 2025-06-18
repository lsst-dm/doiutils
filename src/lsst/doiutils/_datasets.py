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
import datetime
import itertools
import logging
import typing
from collections.abc import Iterable
from functools import cached_property
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
    date: datetime.date
    abstract: str
    instrument_doi: str
    dataset_types: list[DataReleaseDatasetType]
    doi: str | None = None
    osti_id: int | None = None

    @cached_property
    def osti_id_to_dataset_type(self) -> dict[int, DataReleaseDatasetType]:
        """OSTI IDs associated with each dataset type.

        It is required that every OSTI ID is specified.
        """
        osti_id_map: dict[int, DataReleaseDatasetType] = {}
        for dtype in self.dataset_types:
            found = False
            if dtype.butler and dtype.butler.osti_id:
                osti_id_map[dtype.butler.osti_id] = dtype
                found = True
            if dtype.tap and dtype.tap.osti_id:
                osti_id_map[dtype.tap.osti_id] = dtype
                found = True

            if not found:
                raise ValueError(
                    f"Could not find OSTI ID for dataset type {dtype}. Was this configuration uploaded?"
                )

        return osti_id_map

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
        return cls.model_validate(config_dict, strict=True)

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

    # Remove related identifiers. We only want the instrument to be
    # referenced from the primary DOI.
    dtype_content.pop("related_identifiers", None)

    return elinkapi.Record.model_validate(dtype_content, strict=True)


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
        records[None] = elinkapi.Record.model_validate(record_content, strict=True)
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
            if dataset_type.butler.count:
                s = "" if dataset_type.butler.count == 1 else "s"
                abstract += f" This release contains {dataset_type.butler.count} dataset{s} of this type."
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


def get_records(config: DataReleaseConfig, elink: elinkapi.Elink) -> dict[int, elinkapi.Record]:
    """Retrieve records associated with this configuration."""
    records: dict[int, elinkapi.Record] = {}

    if not config.osti_id:
        raise ValueError(
            "No OSTI ID found for this data release. Was this configuration updated after upload?"
        )

    records[config.osti_id] = elink.get_single_record(config.osti_id)
    if records[config.osti_id].osti_id != config.osti_id:
        raise RuntimeError("Unexpectedly got different OSTI ID from primary record than requested")

    for osti_id in config.osti_id_to_dataset_type:
        records[osti_id] = elink.get_single_record(osti_id)
        if records[osti_id].osti_id != osti_id:
            raise RuntimeError("Unexpectedly got different OSTI ID from record than requested")

    return records


def submit_records(config: DataReleaseConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> int:
    """Submit records from configuration to ELink.

    Returns
    -------
    n_saved : `int`
        Number of records stored. If greater than 0 the config will be updated
        with the DOI and OSTI record information.
    """
    records = make_records(config)

    if dry_run:
        for rec in records.values():
            print(rec.model_dump_json(exclude_defaults=True, indent=2))
        return 0

    _LOG.info("Will be submitting %d records.", len(records))

    # Submit each record, recording the issued OSTI ID as we go so that
    # we can update the configuration (to prevent new uploads of the same
    # thing).

    n_saved = 0
    for key, record in records.items():
        try:
            saved_record = elink.post_new_record(record, "save")
        except Exception:
            _LOG.exception("Error saving record for key %s", key)
            continue

        n_saved += 1
        _LOG.info("Saved record %s as %s", key, saved_record.doi)

        config.set_saved_metadata(key, saved_record)

    _LOG.info("Saved %d record%s out of %d", n_saved, "" if n_saved == 1 else "s", len(records))
    return n_saved


def update_record_relationships(
    config: DataReleaseConfig, elink: elinkapi.Elink, *, dry_run: bool = False
) -> None:
    """Update the relationships between records for this data release.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.
    """
    _LOG.info("Retrieving records from OSTI")
    saved_records = get_records(config, elink)

    # Set IsPartOf relationship. All entries are part of the primary
    # entry.
    primary_id = config.osti_id
    if primary_id is None:
        raise RuntimeError("Primary dataset must have a specified OSTI ID in configuration.")
    primary_doi = config.doi
    if primary_doi is None:
        raise RuntimeError("Primary dataset DOI must be defined.")
    primary_record = saved_records.pop(primary_id)

    is_part_of = elinkapi.RelatedIdentifier(type="DOI", relation="IsPartOf", value=primary_doi)

    _LOG.info("Updating relationships")
    # Mapping of OSTI ID to DatasetType configuration.
    osti_id_to_dataset_type = config.osti_id_to_dataset_type

    for dataset_osti, dataset_record in saved_records.items():
        # Record that this is part of the main collection.
        dataset_record.add(is_part_of)

        # Tell the primary that this contains the subset.
        primary_record.add(
            elinkapi.RelatedIdentifier(type="DOI", relation="HasPart", value=dataset_record.doi)
        )

        if dtype := osti_id_to_dataset_type.get(dataset_osti):
            if dtype.butler and dtype.tap:
                # We have to link the two together. Link one direction now
                # and the other direction when that OSTI ID is encountered.
                # We are only modifying the current record.
                if dtype.butler.osti_id == dataset_osti:
                    # We have verified this elsewhere but mypy cannot tell.
                    if dtype.tap.doi is None:
                        raise RuntimeError(f"Unexpectedly got null DOI for TAP dataset {dtype}")
                    dataset_record.add(
                        elinkapi.RelatedIdentifier(
                            type="DOI", relation="IsOriginalFormOf", value=dtype.tap.doi
                        )
                    )
                elif dtype.tap.osti_id == dataset_osti:
                    # We have verified this elsewhere but mypy cannot tell.
                    if dtype.butler.doi is None:
                        raise RuntimeError(f"Unexpectedly got null DOI for butler dataset {dtype}")
                    dataset_record.add(
                        elinkapi.RelatedIdentifier(
                            type="DOI", relation="IsVariantFormOf", value=dtype.butler.doi
                        )
                    )
                else:
                    raise RuntimeError("Logic error in DOI assignment")

    update_records(itertools.chain([primary_record], saved_records.values()), elink, dry_run=dry_run)


def update_records(
    records: Iterable[elinkapi.Record], elink: elinkapi.Elink, *, dry_run: bool = False
) -> None:
    """Save the given records as updates to an existing record."""
    _LOG.info("Uploading modified records")
    for record in records:
        if dry_run:
            print(record.model_dump_json(indent=2))
        else:
            elink.update_record(record.osti_id, record, "save")
