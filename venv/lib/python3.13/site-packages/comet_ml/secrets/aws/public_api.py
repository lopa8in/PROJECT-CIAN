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
import json
import logging
from typing import Any, Dict, Optional, Tuple

import comet_ml
from comet_ml import logging_messages

from .. import codec, secret_managers_dispatch
from . import instruction_details

LOGGER = logging.getLogger(__name__)


def get_api_key_from_secret_manager(
    secret_name: str,
    secret_version: str,
    aws_region_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
) -> str:
    """
    Returns a Comet API Key Secret that can be used instead of a clear-text
    API Key when creating an Experiment or API object. The Comet API Key
    Secret is a string that represents the location of the secret in the AWS
    Secret Manager without containing the API Key. This means that this
    function don't need permission or access to AWS Secret Manager.

    Args:
        secret_name: AWS secret name.
        secret_version: AWS secret version. You can get this value
            from the
            [comet_ml.secrets.aws.store_api_key_in_secret_manager][]
            output. You can also pass `"latest"`, in that case the function will return a Comet
            API Key Secret pointing to the latest version of the AWS Secret.
        aws_region_name: AWS region name. If not specified the default region name from
            the local AWS configuration will be used.
        aws_profile_name: AWS profile name. If not specified the default profile name from
            the local AWS configuration will be used.

    Returns:
        (str) Comet API Key Secret.

    Example:
        ```python linenums="1"
        import comet_ml
        from comet_ml.secrets.aws import get_api_key_from_secret_manager

        api_key = get_api_key_from_secret_manager(
            AWS_SECRET_NAME, aws_region_name="us-east-1" ,aws_profile_name="dev"
        )
        experiment = comet_ml.start(api_key=api_key)
        ```

    """

    details = instruction_details.get(
        secret_name=secret_name,
        secret_version=secret_version,
        region_name=aws_region_name,
        profile_name=aws_profile_name,
    )
    instruction = _get_instruction(details)

    return _opaque_instruction(instruction)


def _get_instruction(details: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "type": "AWS",
        "details": dict(details),
        "comet_ml_version": comet_ml.get_comet_version(),
    }
    return result


def _opaque_instruction(instruction: Dict[str, Any]) -> str:
    return "_SECRET_-" + codec.encode(json.dumps(instruction))


def store_api_key_in_secret_manager(
    api_key: str,
    secret_name: str,
    aws_region_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Stores an API key to AWS Secret Manager as a secret. After that returns an API Key Secret and
    AWS Secret version as a string.

    Args:
        api_key: Comet API key to save.
        secret_name: AWS secret name.
        aws_region_name: AWS region name. If not specified the default region name from
            the local AWS configuration will be used.
        aws_profile_name: AWS profile name. If not specified the default profile name from
            the local AWS configuration will be used.

    Returns: (Tuple[str, str]) API Key Secret, AWS Secret version

    Example:
        ```python linenums="1"
        import comet_ml
        from comet_ml.secrets.aws import store_api_key_in_secret_manager

        api_key_secret, secret_version = store_api_key_in_secret_manager(
            COMET_API_KEY, AWS_SECRET_NAME
        )
        experiment = comet_ml.start(api_key=api_key_secret)
        ```
    """
    if aws_profile_name is not None:
        LOGGER.warning(logging_messages.AWS_SECRET_MANAGER_PROFILE_NAME_WARNING)

    details = instruction_details.get(
        secret_name=secret_name,
        region_name=aws_region_name,
        profile_name=aws_profile_name,
    )

    secret_version = secret_managers_dispatch.dispatch("AWS").store(
        api_key, details["secret_name"], details["region_name"], details["profile_name"]
    )

    details["secret_version"] = secret_version
    details["profile_name"] = None
    instruction = _get_instruction(details)

    return _opaque_instruction(instruction), secret_version
