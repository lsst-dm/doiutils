"""Test DOI dataset record creation."""

from __future__ import annotations

import datetime
import os.path

from lsst.doiutils._datasets import DataReleaseConfig, make_bibtex_entries, make_records

TESTDIR = os.path.abspath(os.path.dirname(__file__))


def _make_bibtex_config() -> DataReleaseConfig:
    """Construct a small data release config with assigned DOIs."""
    return DataReleaseConfig.model_validate(
        {
            "title": "Test Data Release",
            "site_url": "https://example.test/",
            "date": datetime.date(2026, 1, 1),
            "abstract": "A test data release.",
            "instrument_doi": "10.71929/rubin/9999",
            "doi": "10.71929/rubin/1000",
            "osti_id": 1000,
            "dataset_types": [
                {
                    "abstract": "Detections from deep co-adds.",
                    "path": "products/catalogs/object.html",
                    "butler": {
                        "name": "object",
                        "doi": "10.71929/rubin/1001",
                        "osti_id": 1001,
                    },
                    "tap": {
                        "name": "Object",
                        "doi": "10.71929/rubin/1002",
                        "osti_id": 1002,
                    },
                },
                {
                    "abstract": "A dataset type that has not yet been assigned a DOI.",
                    "path": "products/catalogs/source.html",
                    "butler": {"name": "source"},
                },
            ],
        },
        strict=True,
    )


def test_make_bibtex_entries() -> None:
    """Test that a BibTeX entry is generated for every assigned DOI."""
    config = _make_bibtex_config()

    entries = make_bibtex_entries(config)

    # Primary release plus butler and tap for the first dataset type. The
    # second dataset type has no DOI so it is skipped.
    assert len(entries) == 3

    assert entries[0] == (
        "@misc{10.71929/rubin/1000,\n"
        "  doi = {10.71929/rubin/1000},\n"
        "  url = {https://www.osti.gov//servlets/purl/1000},\n"
        "  author = {{NSF-DOE Vera C. Rubin Observatory}},\n"
        "  keywords = {79 ASTRONOMY AND ASTROPHYSICS},\n"
        '  title = "{Test Data Release [Data set]}",\n'
        "  publisher = {NSF-DOE Vera C. Rubin Observatory},\n"
        "  year = {2026}\n"
        "}"
    )

    assert entries[1] == (
        "@misc{10.71929/rubin/1001,\n"
        "  doi = {10.71929/rubin/1001},\n"
        "  url = {https://www.osti.gov//servlets/purl/1001},\n"
        "  author = {{NSF-DOE Vera C. Rubin Observatory}},\n"
        "  keywords = {79 ASTRONOMY AND ASTROPHYSICS},\n"
        '  title = "{Test Data Release: object dataset type [Data set]}",\n'
        "  publisher = {NSF-DOE Vera C. Rubin Observatory},\n"
        "  year = {2026}\n"
        "}"
    )

    assert entries[2] == (
        "@misc{10.71929/rubin/1002,\n"
        "  doi = {10.71929/rubin/1002},\n"
        "  url = {https://www.osti.gov//servlets/purl/1002},\n"
        "  author = {{NSF-DOE Vera C. Rubin Observatory}},\n"
        "  keywords = {79 ASTRONOMY AND ASTROPHYSICS},\n"
        '  title = "{Test Data Release: Object searchable catalog [Data set]}",\n'
        "  publisher = {NSF-DOE Vera C. Rubin Observatory},\n"
        "  year = {2026}\n"
        "}"
    )


def test_make_bibtex_entries_skips_unassigned() -> None:
    """A dataset source without a DOI produces no BibTeX entry."""
    config = _make_bibtex_config()

    entries = make_bibtex_entries(config)

    assert not any("source dataset type" in entry for entry in entries)


def test_record_creation() -> None:
    """Test that records can be created."""
    with open(os.path.join(TESTDIR, "data", "dr.yaml")) as fh:
        config = DataReleaseConfig.from_yaml_fh(fh)

    assert len(config.dataset_types) == 17

    records = make_records(config)
    assert len(records) == 28

    primary = records[None]
    assert primary.title, "Legacy Survey of Space and Time Data Preview 1"
    assert primary.doi_infix, "rubin"
    assert len(primary.related_identifiers) == 1
    assert primary.related_identifiers[0].relation, "IsCollectedBy"
    assert primary.organizations[0].name, "NSF-DOE Vera C. Rubin Observatory"
    assert primary.site_url, "https://dp1.lsst.io/"

    raw = records["raw"]
    assert raw.title, "Legacy Survey of Space and Time Data Preview 1: raw dataset type"
    assert raw.product_size, "16125 files"
    # No related identifiers.
    assert raw.related_identifiers is None
    assert raw.site_url, "https://dp1.lsst.io/products/images/raw_exposure.html"
