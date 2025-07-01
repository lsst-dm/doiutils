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
import os
import sys
import typing

import elinkapi
from pydantic import AfterValidator, AnyHttpUrl, BaseModel, field_serializer

from ._constants import FUNDING_ORGANIZATIONS, IDENTIFIERS, ORGANIZATION_AUTHORS
from ._utils import strip_newlines
from ._yaml import load_yaml_fh, prepare_block_text_for_writing, write_to_yaml_fh

_LOG = logging.getLogger("lsst.doiutils")


# This is copied from lsst-texmf bibtools.py
TN_SERIES = {
    "DMTN": "Data Management Technical Note",
    "RTN": "Technical Note",
    "PSTN": "Project Science Technical Note",
    "SCTR": "Commissioning Technical Report",
    "SITCOMTN": "Commissioning Technical Note",
    "SMTN": "Simulations Team Technical Note",
    "SQR": "SQuaRE Technical Note",
    "ITTN": "Information Technology Technical Note",
    "TSTN": "Telescope and Site Technical Note",
    "DMTR": "Data Management Test Report",
    "LDM": "Data Management Controlled Document",
    "LSE": "Systems Engineering Controlled Document",
    "LCA": "Camera Controlled Document",
    "LTS": "Telescope & Site Controlled Document",
    "LPM": "Project Controlled Document",
    "LEP": "Education and Public Outreach Controlled Document",
    "CTN": "Camera Technical Note",
    "RDO": "Data Management Operations Controlled Document",
    "Agreement": "Formal Construction Agreement",
    "Document": "Informal Construction Document",
    "Publication": "LSST Construction Publication",
    "Report": "Construction Report",
}


class PaperConfig(BaseModel):
    """Description of a paper requiring a DOI."""

    title: str
    """Title of the paper."""
    handle: str
    """Rubin handle for the paper."""
    site_url: AnyHttpUrl
    """Main landing page for this paper."""
    date: datetime.date
    """Date of paper publication. This is ambiguous for tech notes which
    continually update.
    """
    abstract: typing.Annotated[str, AfterValidator(strip_newlines)]
    """Abstract of the paper."""
    doi: str | None = None
    """DOI of the data release."""
    osti_id: int | None = None
    """OSTI ID of the data release (required to retrieve and edit the record
    in ELink).
    """
    authors: list[str]
    """Author information. In the future this has to be the Rubin standard
    author ID. For now special keys of Rubin, NOIRLab, and SLAC exist
    for organizational authorship.
    """
    relationships: dict[str, list[str]]
    """DOI relationships to this paper where the keys are the relationship
    type and the values are DOIs.
    """

    def get_series(self) -> str:
        """Return the document series for this paper.

        Returns
        -------
        series : `str`
            The series associated with this document. Uses the text before
            the hyphen. If no series is found returns an empty string.
        """
        prefix, _ = self.handle.split("-", 1)
        return TN_SERIES.get(prefix, "")

    @field_serializer("site_url")
    def serialize_url(self, url: AnyHttpUrl) -> str:
        return str(url)

    @classmethod
    def from_yaml_fh(cls, fh: typing.IO[str]) -> typing.Self:
        """Create a configuration from a file handle pointing to a YAML
        file.

        Parameters
        ----------
        fh : `typing.IO`
            Open file handle associated with a YAML configuration.
        """
        config_dict = load_yaml_fh(fh)
        return cls.model_validate(config_dict, strict=True)

    def write_yaml_fh(self, fh: typing.IO[str]) -> None:
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
        """Update the configuration to include the saved DOI."""
        osti_id = saved_record.osti_id
        if osti_id is None:
            raise RuntimeError("This record does not correspond to a saved record.")
        doi = saved_record.doi
        if not doi:
            raise RuntimeError("No DOI associated with this record. Was it saved?")
        self.doi = doi
        self.osti_id = osti_id


def make_paper_record(config: PaperConfig) -> elinkapi.Record:
    """Make a elink record for this paper."""
    related_identifiers: list[elinkapi.RelatedIdentifier] = []
    for relationship, related_dois in config.relationships.items():
        related_identifiers.extend(
            elinkapi.RelatedIdentifier(type="DOI", relation=relationship, value=related)
            for related in related_dois
        )

    # Even though this is not allowed to be a document DOI, we are allowed
    # to set the Dataset Product Number field. This is an Identifier with
    # type RN for Report Number.
    report_number = elinkapi.Identifier(
        type="RN", value=f"Vera C. Rubin Observatory {config.get_series()} {config.handle}"
    )

    record_content = {
        "product_type": "DA",  # Don't talk to me about this...
        "doi_infix": "rubin",
        "site_ownership_code": "SLAC-LSST",
        # Since we are not allowed to say we are a technical report this
        # becomes a bit of a disaster because we would like the handle to
        # appear somewhere.
        "title": f"{config.handle}: {config.title}",
        "site_url": str(config.site_url),
        "description": config.abstract,
        "publication_date": config.date,
        "subject_category_code": ["79"],  # "79 ASTRONOMY AND ASTROPHYSICS"
        "publisher_information": "NSF-DOE Vera C. Rubin Observatory",
        "access_limitations": ["UNL"],
        "identifiers": [report_number, *IDENTIFIERS],
        "related_identifiers": related_identifiers,
    }

    # Persons are distinct from Organizations.
    # For now no persons are authors and we would need to look up the person
    # based on author ID. Rubin as author and then the supporting
    # organizations. This is not going to be how the real system should work
    # (where we would need to keep the SPONSORING organizations but drop
    # the RESEARCHING organizations and keep only the authors). Builder
    # papers are required to have Rubin first author.
    if config.authors == ["Rubin"]:
        # Special case, sole author as organization.
        record_content["organizations"] = [ORGANIZATION_AUTHORS["Rubin"], *FUNDING_ORGANIZATIONS.values()]
    else:
        # All Rubin technotes are sponsored by NSF and DOE and have Rubin
        # publisher, but do not have SLAC/NOIRLab affiliations if there
        # are real authors.
        record_content["organizations"] = [
            FUNDING_ORGANIZATIONS[org] for org in ("RubinInstitution", "NSF", "DOE")
        ]
        # Currently only option is to use the author code from lsst-texmf.
        texmf_dir = os.getenv("LSST_TEXMF_DIR")
        if not texmf_dir:
            raise RuntimeError("Unable to find lsst-texmf dir. Please set LSST_TEXMF_DIR.")
        sys.path.append(os.path.join(texmf_dir, "bin"))
        from db2authors import AASTeX, AuthorFactory, latex2text

        with open(os.path.join(texmf_dir, "etc", "authordb.yaml")) as fh:
            authordb = load_yaml_fh(fh)
        factory = AuthorFactory.from_authordb(authordb)

        authors = [factory.get_author(authorid) for authorid in config.authors]
        affiliations: dict[str, elinkapi.Affiliation] = {}
        persons: list[elinkapi.Person] = []
        for author in authors:
            # Model doesn't allow None for orcid even though it defaults
            # to None.
            extras: dict[str, str] = {}
            if author.orcid:
                extras["orcid"] = author.orcid
            person = elinkapi.Person(
                type="AUTHOR",
                first_name=latex2text(author.given_name),
                last_name=latex2text(author.family_name),
                **extras,
            )
            for affil in author.affiliations:
                if affil not in affiliations:
                    # Need to properly organize affiliation data in authordb.
                    parsed = AASTeX.parse_affiliation(affil)
                    print(parsed)
                    # No RORs yet.
                    affiliations[affil] = elinkapi.Affiliation(name=latex2text(parsed["institute"]))
                person.add_affiliation(affiliations[affil])
            persons.append(person)
        record_content["persons"] = persons

    return elinkapi.Record.model_validate(record_content, strict=True)


def submit_paper(config: PaperConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> bool:
    """Submit paper record from configuration to ELink."""
    paper_record = make_paper_record(config)
    if dry_run:
        print(paper_record.model_dump_json(exclude_defaults=True, indent=2))
        return False

    saved_record = elink.post_new_record(paper_record, "save")
    _LOG.info("Saved paper with DOI %s", saved_record.doi)
    config.set_saved_metadata(saved_record)

    return True


def publish_paper(config: PaperConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> None:
    """Publish the paper.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.
    """
    _LOG.info("Retrieving record from OSTI")
    if not config.osti_id:
        raise ValueError("No OSTI found for this paper. Was this configuration updated after upload?")
    saved_record = elink.get_single_record(config.osti_id)
    if config.osti_id != saved_record.osti_id:
        raise AssertionError("Internal error. The requested OSTI ID does not match the one retrieved.")

    if dry_run:
        print(saved_record.model_dump_json(exclude_unset=True, indent=2))
    else:
        _LOG.info("Submitting finished DOI %s", saved_record.doi)
        elink.update_record(saved_record.osti_id, saved_record, "submit")
