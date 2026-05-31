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
#  This file can not be copied and/or distributed
#  without the express permission of Comet ML Inc.
# *******************************************************
import hashlib
from typing import Any, AnyStr, Dict, Mapping, Optional, Union

from ..convert_utils import convert_dict_to_string
from ..utils import generate_guid


def get_experiment_key(user_input: Optional[Union[AnyStr, Dict[str, Any]]]) -> str:
    """Generates a unique experiment key based on the provided user input. The same input consistently produces the same
    experiment key. If the user input is ``None``, a unique random key is generated each time using the UUID algorithm.

    Args:
        user_input (str, bytes, dict, optional): the user input as str, bytes or dict.
    Returns:
        the consistent experiment key string generated from ``user_input`` or unique experiment key if ``None`` is provided .
    """
    if user_input is None:
        return generate_guid()

    if isinstance(user_input, bytes) and not isinstance(user_input, str):
        user_input = user_input.decode("utf-8")
    elif isinstance(user_input, Mapping):
        user_input = convert_dict_to_string(user_input)
    elif not isinstance(user_input, str):
        user_input = str(user_input)

    return hashlib.sha1(user_input.encode("utf-8")).hexdigest()
