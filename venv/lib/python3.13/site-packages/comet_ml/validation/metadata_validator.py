# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
from typing import Any, Dict

from comet_ml.validation.helpers import validate_type_dict

LOGGER = logging.getLogger(__name__)


def validate_metadata(user_metadata: Any, raise_on_invalid: bool = False):
    if user_metadata is None:
        return {}

    if validate_type_dict(user_metadata) is False:
        if raise_on_invalid:
            raise ValueError("Invalid metadata, expecting dict type %r" % user_metadata)
        else:
            LOGGER.warning("Invalid metadata, expecting dict type %r", user_metadata)
            return {}

    result: Dict[Any, Any] = user_metadata

    return result
