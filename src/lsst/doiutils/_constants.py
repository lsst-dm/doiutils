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

__all__: list[str] = [
    "AFFILIATIONS",
    "DOE_IDENTIFIERS",
    "FUNDING_ORGANIZATIONS",
    "IDENTIFIERS",
    "LOCATION",
    "NSF_IDENTIFIERS",
    "ORGANIZATION_AUTHORS",
]

import elinkapi

LOCATION = elinkapi.Geolocation(
    type="POINT",
    label="Cerro Pachón, Chile",
    points=[elinkapi.Geolocation.Point(latitude=-30.244639, longitude=-70.749417)],
)

# Affiliations can be pre-calculated
AFFILIATIONS = {
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

DOE_IDENTIFIERS = [
    elinkapi.Identifier(type="CN_DOE", value="AC02-76SF00515"),
]
NSF_IDENTIFIERS = [
    elinkapi.Identifier(type="CN_NONDOE", value="NSF Cooperative Agreement AST-1258333"),
    elinkapi.Identifier(type="CN_NONDOE", value="NSF Cooperative Support Agreement AST-1836783"),
]

# To be distinct from Person authors.
ORGANIZATION_AUTHORS = {
    "Rubin": elinkapi.Organization(
        type="AUTHOR", name="NSF-DOE Vera C. Rubin Observatory", ror_id="https://ror.org/048g3cy84"
    ),
}
FUNDING_ORGANIZATIONS = {
    # The first RESEARCHING organization becomes the institution listed in
    # bib entries at DataCite so we include Rubin here in front of NOIRLab and
    # SLAC for datasets.
    "RubinInstitution": elinkapi.Organization(
        type="RESEARCHING", name="NSF-DOE Vera C. Rubin Observatory", ror_id="https://ror.org/048g3cy84"
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
        type="SPONSOR",
        name="U.S. National Science Foundation",
        ror_id="https://ror.org/021nxhr62",
        identifiers=NSF_IDENTIFIERS,
    ),
    "DOE": elinkapi.Organization(
        type="SPONSOR",
        name="U.S. Department of Energy Office of Science",
        ror_id="https://ror.org/00mmn6b08",
        identifiers=DOE_IDENTIFIERS,
    ),
}

IDENTIFIERS = DOE_IDENTIFIERS + NSF_IDENTIFIERS
