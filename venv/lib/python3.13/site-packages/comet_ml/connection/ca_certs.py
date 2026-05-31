# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
import os
import tempfile
from typing import Optional

import certifi

LOGGER = logging.getLogger(__name__)

# Global custom CA certs placeholder
merged_ca_certs_bundle: Optional[str] = None


def setup_ca_certs(custom_ca_certs: str) -> Optional[str]:
    global merged_ca_certs_bundle
    if merged_ca_certs_bundle is not None:
        return merged_ca_certs_bundle

    if not os.path.exists(custom_ca_certs):
        raise FileNotFoundError(
            f"Could not find a suitable TLS CA certificate bundle, invalid path: {custom_ca_certs}"
        )

    LOGGER.info("Setting up custom CA certificates from: %s", custom_ca_certs)

    try:
        ca_bundle = os.path.join(tempfile.mkdtemp(prefix="comet-ca-"), "ca_certs.pem")
        ca_files = [certifi.where(), custom_ca_certs]
        with open(ca_bundle, "w") as outfile:
            for filename in ca_files:
                with open(filename, "r") as infile:
                    outfile.write(infile.read())
                outfile.write("\n")

        merged_ca_certs_bundle = ca_bundle
        LOGGER.info("Merged CA certificates saved to: %s", ca_bundle)

    except Exception as e:
        LOGGER.error(
            "Could not set up custom CA certificates from: %s, reason: %r",
            custom_ca_certs,
            e,
        )

    return merged_ca_certs_bundle
