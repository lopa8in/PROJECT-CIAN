# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
import os
from typing import Callable

from comet_ml import authhook

from requests import Session

LOGGER = logging.getLogger(__name__)

AWS_SAGEMAKER_AUTH_INITIALIZED = False


def is_in_aws_sagemaker() -> bool:
    return os.getenv("AWS_PARTNER_APP_AUTH") is not None


def _aws_sagemaker_session_hook(auth_provider) -> Callable[[Session], None]:

    def _aws_session_hook(session: Session):
        session.auth = auth_provider.get_auth()

    return _aws_session_hook


def login_aws_sagemaker() -> None:
    """
    Initializes AWS SageMaker authentication.

    If the system is not in AWS SageMaker or the authentication is already set, the function will
    return immediately without performing any action.

    Sets up SageMaker Partner SDK authentication provider, hooks HTTP session with the AWS SageMaker authentication
    provider, and marks the authentication as initialized.
    """
    global AWS_SAGEMAKER_AUTH_INITIALIZED

    if not is_in_aws_sagemaker() or AWS_SAGEMAKER_AUTH_INITIALIZED:
        return

    LOGGER.debug("AWS partner SDK authentication initialization commenced")

    # setup sagemaker partner SDK authentication provider
    from sagemaker import PartnerAppAuthProvider

    auth_provider = PartnerAppAuthProvider()
    authhook.http_session_hook = _aws_sagemaker_session_hook(auth_provider)

    AWS_SAGEMAKER_AUTH_INITIALIZED = True

    LOGGER.debug("AWS partner SDK authentication initialized")


def logout_aws_sagemaker() -> None:
    global AWS_SAGEMAKER_AUTH_INITIALIZED
    if AWS_SAGEMAKER_AUTH_INITIALIZED:
        authhook.http_session_hook = lambda session: None
        AWS_SAGEMAKER_AUTH_INITIALIZED = False
        LOGGER.debug("AWS partner SDK authentication removed")
