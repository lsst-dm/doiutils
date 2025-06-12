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
import sys
import typing
from datetime import datetime

import elinkapi
import yaml
from pydantic import AnyHttpUrl, BaseModel

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


class DataReleaseDatasetType(BaseModel):
    """A component dataset type found within a data release."""

    abstract: str
    path: str
    butler_name: str = ""
    tap_name: str = ""
    doi: str | None = None


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


def make_records(config: DataReleaseConfig) -> list[elinkapi.Record]:
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
    records: list[elinkapi.Record] = []
    records.append(elinkapi.Record.model_validate(record_content))

    # Should we strip the instrument DOI from subset DOIs. We want them
    # to be solely isPartOf the full data release and not themselves be
    # isCollectedBy the instrument?
    for dataset_type in config.dataset_types:
        dtype_abstract = dataset_type.abstract.replace("\n", " ").strip()
        # Lower case the first character for grammar given that extra_text
        # below starts a sentence.
        dtype_abstract = dtype_abstract[0].lower() + dtype_abstract[1:]

        uniquify_paths = False
        if dataset_type.butler_name and dataset_type.tap_name:
            # Both datasets exist and will be given distinct DOIs but we
            # should ensure that they have different target URLs.
            uniquify_paths = True

        if dataset_type.butler_name:
            extra_text = (
                "This dataset is a subset of the full data release"
                f" consisting of the {dataset_type.butler_name} dataset type. These are "
            )

            abstract = typing.cast(str, record_content["description"]) + "\n\n" + extra_text + dtype_abstract
            fragment = "#butler" if uniquify_paths else ""

            records.append(
                _make_sub_record(
                    record_content,
                    f": {dataset_type.butler_name} dataset type",
                    abstract,
                    dataset_type.path + fragment,
                )
            )

        if dataset_type.tap_name:
            extra_text = (
                "This dataset is a subset of the full data release consisting of "
                f"a searchable catalog named {dataset_type.tap_name}. This catalog contains "
            )
            abstract = typing.cast(str, record_content["description"]) + "\n\n" + extra_text + dtype_abstract
            fragment = "#tap" if uniquify_paths else ""

            records.append(
                _make_sub_record(
                    record_content,
                    f": {dataset_type.butler_name} searchable catalog",
                    abstract,
                    dataset_type.path + fragment,
                )
            )

    return records


def main() -> int:
    cfg_file = sys.argv[1]

    with open(cfg_file) as fd:
        config_dict = yaml.safe_load(fd)
    config = DataReleaseConfig.model_validate(config_dict)

    records = make_records(config)
    for rec in [records[5]]:
        print(rec.model_dump_json(exclude_defaults=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
