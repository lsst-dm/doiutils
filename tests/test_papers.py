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

"""Test DOI paper record creation."""

from __future__ import annotations

import datetime
import logging
import typing

import elinkapi
import pytest

from lsst.doiutils import _papers
from lsst.doiutils._papers import PaperConfig, _compare_person, _make_person, update_paper_author_refs


def test_make_person_without_given_name() -> None:
    """An author with no given name has no first name in the record."""
    person = _make_person("", "Vera C. Rubin Observatory Team", None, [])

    assert person.first_name is None
    assert person.last_name == "Vera C. Rubin Observatory Team"
    assert "first_name" not in person.model_dump(exclude_none=True)


def test_make_person() -> None:
    """An author with a given name and ORCID retains both."""
    affiliation = elinkapi.Affiliation(name="Test Institute", ror_id="https://ror.org/048g3cy84")
    person = _make_person("Ada", "Lovelace", "0000-0002-5947-2454", [affiliation])

    assert person.first_name == "Ada"
    assert person.orcid == "0000-0002-5947-2454"
    assert person.affiliations == [affiliation]


def test_compare_person_no_given_name() -> None:
    """A stored author with no first name matches a generated author with no
    given name.
    """
    affiliation = elinkapi.Affiliation(
        name="NSF-DOE Vera C. Rubin Observatory", ror_id="https://ror.org/048g3cy84"
    )
    # Records retrieved from ELink have no first name for such authors.
    stored = elinkapi.Person(
        type="AUTHOR", last_name="Vera C. Rubin Observatory Team", affiliations=[affiliation]
    )
    generated = _make_person("", "Vera C. Rubin Observatory Team", None, [affiliation])

    assert _compare_person(stored, generated) == ""


def test_compare_person_without_affiliations() -> None:
    """Authors with no affiliations at all compare equal."""
    stored = elinkapi.Person(type="AUTHOR", first_name="Ada", last_name="Lovelace")
    generated = _make_person("Ada", "Lovelace", None, [])

    assert stored.affiliations is None
    assert generated.affiliations is None
    assert _compare_person(stored, generated) == ""


def test_compare_person_affiliation_added() -> None:
    """An affiliation appearing for a previously unaffiliated author is
    reported.
    """
    affiliation = elinkapi.Affiliation(name="Test Institute", ror_id="https://ror.org/048g3cy84")
    stored = elinkapi.Person(type="AUTHOR", first_name="Ada", last_name="Lovelace")
    generated = _make_person("Ada", "Lovelace", None, [affiliation])

    assert _compare_person(stored, generated) == "Add Test Institute?"


def test_compare_person_affiliation_removed() -> None:
    """An affiliation disappearing from an author is reported."""
    affiliation = elinkapi.Affiliation(name="Test Institute", ror_id="https://ror.org/048g3cy84")
    stored = elinkapi.Person(
        type="AUTHOR", first_name="Ada", last_name="Lovelace", affiliations=[affiliation]
    )
    generated = _make_person("Ada", "Lovelace", None, [])

    assert _compare_person(stored, generated) == "Remove Test Institute?"


def test_compare_person_ignoring_affiliations() -> None:
    """An affiliation change is not reported when only the author identity is
    being compared.
    """
    affiliation = elinkapi.Affiliation(name="Test Institute", ror_id="https://ror.org/048g3cy84")
    stored = elinkapi.Person(
        type="AUTHOR", first_name="Ada", last_name="Lovelace", affiliations=[affiliation]
    )
    renamed = elinkapi.Affiliation(name="Other Institute", ror_id="https://ror.org/048g3cy84")
    generated = _make_person("Ada", "Lovelace", None, [renamed])

    assert _compare_person(stored, generated, compare_affiliations=False) == ""
    assert _compare_person(stored, generated) == "Test Institute != Other Institute"


class _FakeElink:
    """Stand-in for the ELink API that serves one record and remembers the
    record given back to it.
    """

    def __init__(self, record: elinkapi.Record) -> None:
        self.record = record
        self.updates: list[tuple[int, elinkapi.Record, str]] = []

    def get_single_record(self, osti_id: int) -> elinkapi.Record:
        return self.record

    def update_record(self, osti_id: int, r: elinkapi.Record, state: str = "save") -> elinkapi.Record:
        self.updates.append((osti_id, r, state))
        return r


def _make_paper_config(relationships: dict[str, list[str]]) -> PaperConfig:
    return PaperConfig(
        title="A Paper",
        handle="DMTN-000",
        site_url="https://dmtn-000.lsst.io",  # type: ignore[arg-type]
        date=datetime.date(2026, 1, 1),
        abstract="An abstract.",
        doi="10.71929/rubin/1",
        osti_id=1234,
        authors=["lovelace"],
        relationships=relationships,
    )


def _make_saved_record() -> elinkapi.Record:
    """Return a record as retrieved from ELink, with the author affiliation
    that was current at publication time.
    """
    return elinkapi.Record.model_validate(
        {
            "product_type": "DA",
            "title": "DMTN-000: A Paper",
            "doi": "10.71929/rubin/1",
            "osti_id": 1234,
            "workflow_status": "R",
            "persons": [
                {
                    "type": "AUTHOR",
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "affiliations": [{"name": "Old Institute"}],
                }
            ],
            "related_identifiers": [{"type": "DOI", "relation": "Cites", "value": "10.71929/rubin/2"}],
        }
    )


@pytest.fixture
def new_affiliation(monkeypatch: pytest.MonkeyPatch) -> list[elinkapi.Person]:
    """Replace the author database lookup with an author whose affiliation
    has changed since publication.
    """
    persons = [
        _make_person("Ada", "Lovelace", None, [elinkapi.Affiliation(name="New Institute")]),
    ]
    monkeypatch.setattr(_papers, "_create_persons", lambda config: persons)
    return persons


def test_update_paper_info_retains_authors(
    new_affiliation: list[elinkapi.Person], caplog: pytest.LogCaptureFixture
) -> None:
    """By default the stored affiliations are left alone, without comment,
    but the relationships are still updated.
    """
    config = _make_paper_config({"Cites": ["10.71929/rubin/2"], "IsCitedBy": ["10.71929/rubin/3"]})
    elink = _FakeElink(_make_saved_record())

    with caplog.at_level(logging.WARNING, logger="lsst.doiutils"):
        update_paper_author_refs(config, typing.cast("elinkapi.Elink", elink))

    assert caplog.records == []
    assert len(elink.updates) == 1
    osti_id, updated, state = elink.updates[0]
    assert osti_id == 1234
    assert state == "submit"
    assert updated.persons[0].affiliations == [elinkapi.Affiliation(name="Old Institute")]
    assert {rel.value for rel in updated.related_identifiers} == {
        "10.71929/rubin/2",
        "10.71929/rubin/3",
    }


def test_update_paper_info_updates_authors(new_affiliation: list[elinkapi.Person]) -> None:
    """Requesting author updates replaces the authors with the current
    database entries.
    """
    config = _make_paper_config({"Cites": ["10.71929/rubin/2"]})
    elink = _FakeElink(_make_saved_record())

    update_paper_author_refs(config, typing.cast("elinkapi.Elink", elink), update_authors=True)

    assert len(elink.updates) == 1
    _, updated, _ = elink.updates[0]
    assert updated.persons == new_affiliation


def test_update_paper_info_no_changes(new_affiliation: list[elinkapi.Person]) -> None:
    """Nothing is submitted when the relationships match and the authors are
    not being updated.
    """
    config = _make_paper_config({"Cites": ["10.71929/rubin/2"]})
    elink = _FakeElink(_make_saved_record())

    update_paper_author_refs(config, typing.cast("elinkapi.Elink", elink), update_authors=False)

    assert elink.updates == []


def test_update_paper_info_warns_on_new_author(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A change in author membership is reported when the authors are being
    retained by default.
    """
    monkeypatch.setattr(
        _papers,
        "_create_persons",
        lambda config: [
            _make_person("Ada", "Lovelace", None, [elinkapi.Affiliation(name="New Institute")]),
            _make_person("Charles", "Babbage", None, []),
        ],
    )
    config = _make_paper_config({"Cites": ["10.71929/rubin/2"]})
    elink = _FakeElink(_make_saved_record())

    with caplog.at_level(logging.WARNING, logger="lsst.doiutils"):
        update_paper_author_refs(config, typing.cast("elinkapi.Elink", elink))

    assert "Added new author: Babbage" in caplog.text
    # The warning does not change the record.
    assert elink.updates == []
    assert elink.record.persons[0].affiliations == [elinkapi.Affiliation(name="Old Institute")]


def test_update_paper_info_no_warning_when_explicit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No warning is issued when author updates have been turned off
    explicitly.
    """
    monkeypatch.setattr(
        _papers,
        "_create_persons",
        lambda config: [_make_person("Charles", "Babbage", None, [])],
    )
    config = _make_paper_config({"Cites": ["10.71929/rubin/2"]})
    elink = _FakeElink(_make_saved_record())

    with caplog.at_level(logging.WARNING, logger="lsst.doiutils"):
        update_paper_author_refs(config, typing.cast("elinkapi.Elink", elink), update_authors=False)

    assert caplog.records == []
