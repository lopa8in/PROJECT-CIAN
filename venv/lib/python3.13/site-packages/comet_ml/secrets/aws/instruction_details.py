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

from typing import Optional

from . import config


def get(
    secret_name: str,
    secret_version: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
):
    region_name = region_name if region_name is not None else config.region_name()
    profile_name = profile_name if profile_name is not None else config.profile_name()

    result = {
        "secret_name": secret_name,
        "region_name": region_name,
        "profile_name": profile_name,
    }

    if secret_version is not None and secret_version != "latest":
        result["secret_version"] = secret_version

    return result
