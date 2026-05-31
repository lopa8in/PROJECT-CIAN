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

import types
from typing import Optional

_COLORS = types.MappingProxyType(
    {"ERROR": "red1", "WARNING": "orange1", "INFO": "deep_sky_blue1"}
)


def get(levelname: str) -> Optional[str]:
    if levelname in _COLORS:
        return _COLORS[levelname]
    else:
        return None
