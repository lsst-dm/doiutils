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
from pydantic import AfterValidator, AnyHttpUrl, BaseModel, field_serializer

from ._constants import FUNDING_ORGANIZATIONS, IDENTIFIERS, LOCATION, ORGANIZATION_AUTHORS
from ._utils import strip_newlines
from ._yaml import load_yaml_fh, prepare_block_text_for_writing, write_to_yaml_fh

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


class DatasetTypeSource(BaseModel):
    """Specific description of butler vs TAP dataset."""

    name: str
    """Name of the dataset type to use in the DOI record."""
    alias: str | None = None
    """Aliased name that can be specific to the source this is attached to.
    For example, for butler source this is a dataset type glob that can
    correspond to multiple dataset types.
    """
    doi: str | None = None
    """DOI issued for this dataset type source."""
    osti_id: int | None = None
    """OSTI ID issued for this dataset type source."""
    count: int | None = None
    """Count to use for this dataset type source. For butler this is the
    number of datasets. For TAP this will be the number of rows in the catalog.
    """
    count2: int | None = None
    """Secondary count information. For TAP this will be the number of
    columns. For butler this could be the size in bytes.
    """
    format: str | None = None
    """Format of the dataset. For butler this will be the MIME type of file
    extension of the files. No value is used for TAP source."""

    def get_subtitle(self, variant: str) -> str:
        """Return the subtitle for this dataset type.

        Parameters
        ----------
        variant : `str`
            "tap" or "butler".

        Returns
        -------
        subtitle: `str`
            The string to add to the main dataset title. A leading colon is
            assumed to be added by the caller.
        """
        if variant == "butler":
            return f"{self.name} dataset type"
        elif variant == "tap":
            return f"{self.name} searchable catalog"
        raise RuntimeError(f"Unrecognized variant {variant} for subtitle calculation.")


class DataReleaseDatasetType(BaseModel):
    """A component dataset type found within a data release."""

    abstract: typing.Annotated[str, AfterValidator(strip_newlines)]
    """Description of this dataset type."""
    path: str
    """Path to the landing page relative to the main site URL."""
    butler: DatasetTypeSource | None = None
    """Details of butler datasets associated with this dataset type."""
    tap: DatasetTypeSource | None = None
    """Details of TAP catalogs associated with this dataset type."""

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
    """Main title for the data release."""
    site_url: AnyHttpUrl
    """Main landing page for this data release."""
    date: datetime.date
    """Date the data release was made public (YYYY-MM-DD)"""
    abstract: typing.Annotated[str, AfterValidator(strip_newlines)]
    """Description of this data release. Will be included with dataset type
    descriptions.
    """
    instrument_doi: str
    """DOI of the instrument that collected this data."""
    description_paper: str | None = None
    """The DOI of the document describing this data release."""
    product_size: str | None = None
    """Descriptive text to use in the product size part of the DOI record.
    Can be free text. Will not appear in the abstract.
    """
    dataset_types: list[DataReleaseDatasetType]
    """Dataset types associated with this data release that we wish to also
    issue DOIs for.
    """
    doi: str | None = None
    """DOI of the data release."""
    osti_id: int | None = None
    """OSTI ID of the data release (required to retrieve and edit the record
    in ELink).
    """

    @cached_property
    def osti_id_to_dataset_type(self) -> dict[int, DataReleaseDatasetType]:
        """OSTI IDs associated with each dataset type.

        It is required that every OSTI ID is specified.
        """
        osti_id_map: dict[int, DataReleaseDatasetType] = {}
        for dtype in self.dataset_types:
            if butler := dtype.butler:
                if not butler.osti_id:
                    raise ValueError(
                        f"Butler entry exists for {dtype} but no OSTI ID found. "
                        "Was this configuration uploaded?"
                    )
                osti_id_map[butler.osti_id] = dtype
            if tap := dtype.tap:
                if not tap.osti_id:
                    raise ValueError(
                        f"TAP entry exists for {dtype} but no OSTI ID found. Was this configuration uploaded?"
                    )
                osti_id_map[tap.osti_id] = dtype

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
        for dtype in model["dataset_types"]:
            dtype["abstract"] = prepare_block_text_for_writing(dtype["abstract"], indent=6)

        write_to_yaml_fh(model, fh)

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
    primary_content: dict[str, typing.Any],
    extra_title: str,
    abstract: str,
    extra_path: str,
    format_information: str | None = None,
    product_size: str | None = None,
) -> elinkapi.Record:
    """Make a sub-record for a dataset."""
    # Modify a copy.
    dtype_content = copy.deepcopy(primary_content)

    # Remove related identifiers. We only want the instrument to be
    # referenced from the primary DOI. Also remove product_size since the
    # components will have their own sizes and should not inherit the full
    # size.
    dtype_content.pop("related_identifiers", None)
    dtype_content.pop("product_size", None)

    dtype_content["description"] = abstract
    dtype_content["title"] += extra_title
    dtype_content["site_url"] += extra_path
    if format_information:
        dtype_content["format_information"] = format_information
    if product_size:
        dtype_content["product_size"] = product_size

    return elinkapi.Record.model_validate(dtype_content, strict=True)


def _make_butler_record(
    base_record: dict[str, typing.Any],
    butler: DatasetTypeSource,
    dtype_abstract: str,
    dtype_path: str,
    uniquify_paths: bool,  # noqa: FBT001
) -> tuple[str | None, elinkapi.Record | None]:
    extra_text = (
        "This dataset is a subset of the full data release"
        f" consisting of the {butler.name} dataset type. These are "
    )

    abstract = typing.cast(str, base_record["description"]) + "\n\n" + extra_text + dtype_abstract
    product_size: str | None = None
    if count := butler.count:
        s = "" if count == 1 else "s"
        abstract += f" This release contains {count:,d} dataset{s} of this type."
        product_size = f"{count:,d} file{s}"
    fragment = "#butler" if uniquify_paths else ""

    if not butler.osti_id:
        record = _make_sub_record(
            base_record,
            f": {butler.get_subtitle('butler')}",
            abstract,
            dtype_path + fragment,
            product_size=product_size,
            format_information=butler.format,
        )
        return butler.name, record
    else:
        _LOG.info("DOI already assigned for %s: %d", butler.name, butler.osti_id)
    return None, None


def _make_tap_record(
    base_record: dict[str, typing.Any],
    tap: DatasetTypeSource,
    dtype_abstract: str,
    dtype_path: str,
    uniquify_paths: bool,  # noqa: FBT001
) -> tuple[str | None, elinkapi.Record | None]:
    extra_text = (
        "This dataset is a subset of the full data release consisting of "
        f"a searchable catalog named {tap.name}. This catalog contains "
    )
    abstract = typing.cast(str, base_record["description"]) + "\n\n" + extra_text + dtype_abstract

    product_texts: list[str] = []
    count_text: str | None = None
    if count := tap.count:
        s = "" if count == 1 else "s"
        count_text = f" This catalog contains {count:,d} row{s}"
        product_texts.append(f"{count:,d} row{s}")
    if count := tap.count2:  # This is the column count.
        s = "" if count == 1 else "s"
        if count_text:
            count_text += f" with {count:,d} column{s}"
        else:
            f" This catalog contains {count:,d} column{s}"
        product_texts.append(f"{count:,d} column{s}")

    fragment = "#tap" if uniquify_paths else ""

    product_size: str | None = None
    if product_texts:
        product_size = "; ".join(product_texts)
    if count_text:
        abstract += count_text + "."

    product_format = "IVOA TAP-queryable catalog" if not tap.format else tap.format

    if not tap.osti_id:
        record = _make_sub_record(
            base_record,
            f": {tap.get_subtitle('tap')}",
            abstract,
            dtype_path + fragment,
            product_size=product_size,
            format_information=product_format,
        )
        return tap.name, record
    else:
        _LOG.info("DOI already assigned for %s: %d", tap.name, tap.osti_id)
    return None, None


def make_records(config: DataReleaseConfig) -> dict[str | None, elinkapi.Record]:
    """Given a configuration, construct DOI records suitable for submission."""
    related_identifiers: list[elinkapi.RelatedIdentifier] = []
    related_identifiers.append(
        elinkapi.RelatedIdentifier(type="DOI", relation="IsCollectedBy", value=config.instrument_doi)
    )
    if config.description_paper:
        # It's highly unlikely this will be known at DOI creation time but
        # just in case.
        related_identifiers.append(
            elinkapi.RelatedIdentifier(type="DOI", relation="IsDescribedBy", value=config.description_paper)
        )

    # Primary dataset.
    record_content = {
        "product_type": "DA",
        "doi_infix": "rubin",
        "site_ownership_code": "SLAC-LSST",
        "title": config.title,
        "site_url": str(config.site_url),
        "description": config.abstract,
        "related_identifiers": related_identifiers,
        "identifiers": IDENTIFIERS,
        "organizations": [ORGANIZATION_AUTHORS["Rubin"], *FUNDING_ORGANIZATIONS.values()],
        "subject_category_code": ["79"],  # "79 ASTRONOMY AND ASTROPHYSICS"
        "publication_date": config.date,
        "publisher_information": (
            # This is for internal use and won't be part of DataCite record.
            "SLAC National Accelerator Laboratory (SLAC), Menlo Park, CA (United States)"
        ),
        "access_limitations": ["UNL"],
        "geolocations": [LOCATION],
    }
    if config.product_size:
        record_content["product_size"] = config.product_size

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
        dtype_abstract = dataset_type.abstract
        # Lower case the first character for grammar given that extra_text
        # below starts a sentence.
        dtype_abstract = dtype_abstract[0].lower() + dtype_abstract[1:]

        uniquify_paths = False
        if dataset_type.butler and dataset_type.tap:
            # Both datasets exist and will be given distinct DOIs but we
            # should ensure that they have different target URLs.
            uniquify_paths = True

        if dataset_type.butler:
            key, record = _make_butler_record(
                record_content, dataset_type.butler, dtype_abstract, dataset_type.path, uniquify_paths
            )
            if record:
                records[key] = record

        if dataset_type.tap:
            key, record = _make_tap_record(
                record_content, dataset_type.tap, dtype_abstract, dataset_type.path, uniquify_paths
            )
            if record:
                records[key] = record

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
    records: Iterable[elinkapi.Record], elink: elinkapi.Elink, *, dry_run: bool = False, state: str = "save"
) -> None:
    """Update the given records, potentially including final submission."""
    _LOG.info("Uploading records with state %s", state)
    for record in records:
        if dry_run:
            print(record.model_dump_json(indent=2))
        else:
            try:
                elink.update_record(record.osti_id, record, state)
            except Exception:
                _LOG.error(f"Error uploading record with OSTI ID {record.osti_id}")
                raise


def publish_records(config: DataReleaseConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> None:
    """Publish the records.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.
    """
    _LOG.info("Retrieving records from OSTI")
    saved_records = get_records(config, elink)

    update_records(saved_records.values(), elink, dry_run=dry_run, state="submit")
