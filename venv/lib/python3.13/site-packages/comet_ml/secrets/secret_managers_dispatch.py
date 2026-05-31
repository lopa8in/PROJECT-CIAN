# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import types

from .._typing import Optional, Union
from .aws import secret_manager as aws_secret_manager
from .gcp import secret_manager as gcp_secret_manager

_SECRET_MANAGERS = types.MappingProxyType(
    {
        "GCP": gcp_secret_manager.SecretManager(),
        "AWS": aws_secret_manager.SecretManager(),
    }
)


def dispatch(
    key: str,
) -> Optional[
    Union[gcp_secret_manager.SecretManager, aws_secret_manager.SecretManager]
]:
    if key not in _SECRET_MANAGERS:
        return None

    return _SECRET_MANAGERS[key]
