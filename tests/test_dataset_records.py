"""Test DOI dataset record creation."""

from __future__ import annotations

import os.path

from lsst.doiutils._datasets import DataReleaseConfig, make_records

TESTDIR = os.path.abspath(os.path.dirname(__file__))


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
