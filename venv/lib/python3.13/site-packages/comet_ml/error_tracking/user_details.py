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

import functools
import hashlib
from typing import Optional

from comet_ml import config


def get_id() -> Optional[str]:
    """
    Hashed API key serves as a user identifier.
    It is not a strict relation, because api key might change
    or user might accidentally pass incorrect one. But
    we use it as an approximation to have better
    visibility on amount of users affected by some error.
    """
    api_key = config.get_last_returned_api_key()
    if api_key is None or not isinstance(api_key, str):
        return "undefined"

    return _compute_hash(api_key)


@functools.lru_cache()
def _compute_hash(api_key: Optional[str]) -> Optional[str]:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
