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

"""Test DOI paper record creation."""

from __future__ import annotations

import elinkapi

from lsst.doiutils._papers import _compare_person, _make_person


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
