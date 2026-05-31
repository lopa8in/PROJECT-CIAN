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
from typing import Union

from ..logging_messages import ARTIFACT_OVERWRITE_INVALID_STRATEGY_EXCEPTION


def validate_overwrite_strategy(user_overwrite_strategy: Union[str, bool]) -> str:
    """
    Validates the user's overwrite strategy and converts it to a standardized string value.

    This function accepts a user-defined overwrite strategy, which could be in the form
    of a boolean or a string, and processes it to return one of the standardized values
    ("FAIL", "PRESERVE", or "OVERWRITE"). If the provided strategy is invalid, it raises
    a ValueError. The expected inputs allow users to specify their desired overwrite
    behavior for artifacts clearly and concisely.

    Raises:
        ValueError If the provided overwrite strategy is invalid.

    Args:
        user_overwrite_strategy :
            The overwrite strategy specified by the user. It can be given as a string or
            a boolean. Accepted values include:
            - Strings: "fail", "preserve", "overwrite" (case-insensitive)
            - Booleans: True (for overwrite), False (for fail)

    Returns:
        A standardized string representation of the overwrite strategy. Possible
        return values are:
        - "FAIL" for strategies that specify failure on conflict
        - "PRESERVE" for strategies that mandate preservation of existing artifacts
        - "OVERWRITE" for strategies that allow overwriting of artifacts
    """
    if isinstance(user_overwrite_strategy, str):
        lower_user_overwrite_strategy = user_overwrite_strategy.lower()
    else:
        lower_user_overwrite_strategy = user_overwrite_strategy

    if (
        lower_user_overwrite_strategy is False
        or lower_user_overwrite_strategy == "fail"
    ):
        return "FAIL"

    elif lower_user_overwrite_strategy == "preserve":
        return "PRESERVE"

    elif (
        lower_user_overwrite_strategy is True
        or lower_user_overwrite_strategy == "overwrite"
    ):
        return "OVERWRITE"

    else:
        raise ValueError(
            ARTIFACT_OVERWRITE_INVALID_STRATEGY_EXCEPTION % user_overwrite_strategy
        )
