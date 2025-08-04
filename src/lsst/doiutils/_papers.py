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
from itertools import zip_longest

import elinkapi
from pydantic import AfterValidator, AnyHttpUrl, BaseModel, Field, field_serializer

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
    relationships: dict[str, list[str]] = Field(default_factory=dict)
    """DOI relationships to this paper where the keys are the relationship
    type and the values are DOIs. It is allowed for there to be no
    relationships.
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


def _create_persons(config: PaperConfig) -> list[elinkapi.Person]:
    # Currently only option is to use the author code from lsst-texmf.
    texmf_dir = os.getenv("LSST_TEXMF_DIR")
    if not texmf_dir:
        raise RuntimeError("Unable to find lsst-texmf dir. Please set LSST_TEXMF_DIR.")
    sys.path.append(os.path.join(texmf_dir, "bin"))
    from db2authors import AuthorFactory, latex2text

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
                institute = affil.get_department_and_institute()
                # Pydantic model is set up incorrectly and so typing
                # rules don't let us specify it at all if it is None.
                # This is a workaround.
                affil_extras: dict[str, str] = {}
                if affil.ror_id:
                    affil_extras["ror_id"] = f"https://ror.org/{affil.ror_id}"
                affiliations[affil] = elinkapi.Affiliation(name=latex2text(institute), **affil_extras)
            person.add_affiliation(affiliations[affil])
        persons.append(person)
    return persons


def _create_related_identifiers(config: PaperConfig) -> list[elinkapi.RelatedIdentifier]:
    """Create the related identifier information from a paper configuration."""
    related_identifiers: list[elinkapi.RelatedIdentifier] = []
    for relationship, related_dois in config.relationships.items():
        related_identifiers.extend(
            elinkapi.RelatedIdentifier(type="DOI", relation=relationship, value=related)
            for related in related_dois
        )
    return related_identifiers


def make_paper_record(config: PaperConfig) -> elinkapi.Record:
    """Make a elink record for this paper."""
    related_identifiers = _create_related_identifiers(config)

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
        record_content["persons"] = _create_persons(config)

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


def _get_paper_record(config: PaperConfig, elink: elinkapi.Elink) -> elinkapi.Record:
    _LOG.info("Retrieving record from OSTI")
    if not config.osti_id:
        raise ValueError("No OSTI found for this paper. Was this configuration updated after upload?")
    saved_record = elink.get_single_record(config.osti_id)
    if config.osti_id != saved_record.osti_id:
        raise AssertionError("Internal error. The requested OSTI ID does not match the one retrieved.")
    return saved_record


def publish_paper(config: PaperConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> None:
    """Publish the paper.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.
    """
    saved_record = _get_paper_record(config, elink)

    if dry_run:
        print(saved_record.model_dump_json(exclude_unset=True, indent=2))
    else:
        _LOG.info("Submitting finished DOI %s", saved_record.doi)
        elink.update_record(saved_record.osti_id, saved_record, "submit")


def _compare_person(old: elinkapi.Person | None, new: elinkapi.Person | None) -> str:
    """Compare two Person instances and report on relevant difference."""
    if old is None and new is not None:
        return f"Added new author: {new.last_name}"
    elif new is None and old is not None:
        return f"Dropped author {old.last_name}?"
    elif old is not None and new is not None:
        if old.last_name != (last_name := new.last_name):
            return f"{old.last_name} != {last_name}"
        if old.first_name != new.first_name:
            return f"{last_name}: {old.first_name} != {new.first_name}"
        name = f"{new.first_name} {new.last_name}"
        old_orcid = old.orcid.replace("-", "") if old.orcid else ""
        new_orcid = new.orcid.replace("-", "") if new.orcid else ""
        if old_orcid != new_orcid:
            return f"{name}: {old_orcid} != {new_orcid}"

        for oldaffil, newaffil in zip_longest(old.affiliations, new.affiliations):
            changed = _compare_affiliation(oldaffil, newaffil)
            if changed:
                return changed

    return ""


def _compare_affiliation(old: elinkapi.Affiliation | None, new: elinkapi.Affiliation | None) -> str:
    """Compare two affiliations."""
    if old is not None and new is not None:
        old_ror = old.ror_id.removeprefix("https://ror.org/") if old.ror_id else None
        new_ror = new.ror_id.removeprefix("https://ror.org/") if new.ror_id else None
        if old_ror != new_ror:
            return f"ROR change: {old_ror} != {new_ror}"
        if old.name != new.name:
            return f"{old.name} != {new.name}"
    elif old is None and new is not None:
        return f"Add {new.name}?"
    elif new is None and old is not None:
        return f"Remove {old.name}?"

    return ""


def update_paper_author_refs(config: PaperConfig, elink: elinkapi.Elink, *, dry_run: bool = False) -> None:
    """Update the author and references in record for an existing DOI.

    Notes
    -----
    The configuration must correspond to one that includes saved OSTI
    identifiers.

    If authors are updated then affiliations for those authors will be updated
    to the current database entries. This raises the possibility that an
    affiliation will no longer match the affiliation that was originally
    associated with the publication.
    """
    saved_record = _get_paper_record(config, elink)

    updated = False

    # Attach entirely new set of authors to record.
    previous_persons = saved_record.persons
    saved_record.persons = _create_persons(config)

    # Order does matter as well as content.
    changes = []
    for old, new in zip_longest(previous_persons, saved_record.persons):
        if change_reason := _compare_person(old, new):
            changes.append(change_reason)

    if changes:
        _LOG.info("Detected changes to author information:\n%s\n", "\n".join(f"- {c}" for c in changes))
        updated = True

    # And any changes to references.
    previous_relations = saved_record.related_identifiers
    saved_record.related_identifiers = _create_related_identifiers(config)

    previous_refs = {f"{rel.relation}:{rel.value}" for rel in previous_relations}
    new_refs = {f"{rel.relation}:{rel.value}" for rel in saved_record.related_identifiers}
    if previous_refs != new_refs:
        updated = True
        _LOG.info("Relationships were updated.")
        if removed := (previous_refs - new_refs):
            _LOG.info("Removed references:\n%s\n", "\n".join(f"- {r}" for r in sorted(removed)))
        if added := (new_refs - previous_refs):
            _LOG.info("New references:\n%s\n", "\n".join(f"- {a}" for a in sorted(added)))

    if dry_run:
        print(saved_record.model_dump_json(exclude_unset=True, indent=2))
    elif updated:
        # Do not change state of record when updating.
        # SA == saved / R = released
        state = "save" if saved_record.workflow_status == "SA" else "submit"
        _LOG.info("Submitting updated information for %s", saved_record.doi)
        elink.update_record(saved_record.osti_id, saved_record, state)
    else:
        _LOG.info("No change in authors or references detected. Nothing to do.")
