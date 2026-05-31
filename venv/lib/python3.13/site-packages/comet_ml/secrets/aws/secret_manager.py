# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
from typing import Dict, Optional, Tuple

from . import aws_api_importer

LOGGER = logging.getLogger(__name__)


class SecretManager:
    def store(
        self,
        api_key: str,
        secret_name: str,
        region_name: Optional[str],
        profile_name: Optional[str],
    ):
        ClientError = aws_api_importer.ClientError()
        client = _get_client(region_name, profile_name)

        try:
            response = client.create_secret(Name=secret_name, SecretString=api_key)
        except ClientError as error:
            if error.response["Error"]["Code"] != "ResourceExistsException":
                raise

            response = client.update_secret(SecretId=secret_name, SecretString=api_key)

        return response["VersionId"]

    def delete(
        self,
        secret_name: str,
        region_name: Optional[str],
        profile_name: Optional[str],
    ):
        client = _get_client(region_name, profile_name)

        client.delete_secret(SecretId=secret_name)

    def fetch(self, details: Dict[str, str]):
        client_kwargs, get_secret_kwargs = _separate_fetch_details(details)

        client = _get_client(**client_kwargs)
        response = client.get_secret_value(**get_secret_kwargs)

        return response["SecretString"]


def _get_client(region_name: str, profile_name: str):
    Session = aws_api_importer.Session()

    client = Session(region_name=region_name, profile_name=profile_name).client(
        "secretsmanager"
    )

    return client


def _separate_fetch_details(
    details: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    client_kwargs = {
        "region_name": details["region_name"],
        "profile_name": details["profile_name"],
    }

    get_secret_kwargs = {"SecretId": details["secret_name"]}
    if "secret_version" in details:
        get_secret_kwargs["VersionId"] = details["secret_version"]

    return client_kwargs, get_secret_kwargs
