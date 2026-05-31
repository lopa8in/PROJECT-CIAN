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
import sys

from ._typing import Any, Dict, List, Optional, Tuple
from .logging_messages import CLI_ARGS_PARSE_FAILED_EXCEPTION

LOGGER = logging.getLogger(__name__)


def parse_command_line_arguments():
    # type: () -> Optional[Dict[str, Any]]
    if len(sys.argv) > 1:
        try:
            return _parse_cmd_args(sys.argv[1:])

        except ValueError:
            LOGGER.debug("Failed to parse argv values. Fallback to naive parsing.")
            return _parse_cmd_args_naive(sys.argv[1:])


def _parse_cmd_args_naive(to_parse):
    # type: (List[Any]) -> Optional[Dict[str, Any]]
    vals = {}
    if len(to_parse) > 1:
        for i, arg in enumerate(to_parse):
            vals["run_arg_%s" % i] = str(arg)

    return vals


def _parse_cmd_args(argv_vals: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Parses the value of argv[1:] to a dictionary of param,value. Expects params name to start with a - or --
    and value to follow. If no value follows that param is considered to be a boolean param set to true.(e.g --test)
    Args:
        argv_vals: The sys.argv[] list without the first index (script name). Basically sys.argv[1:]

    Returns: Dictionary of param_names, param_values

    """
    results = {}

    split_argv_vals = []
    for word in argv_vals:
        if word == "--" or len(word) == 0:
            continue  # skip it
        elif "=" in word:
            key, value = _parse_arg_value_with_equal(word)
            results[key] = value
        else:
            split_argv_vals.append(word)

    current_key = None
    for word in split_argv_vals:
        word = word.strip()
        if len(word) == 0:
            continue

        # keys
        if word[0] == "-":
            prefix = 1
            if len(word) > 1 and word[1] == "-":
                prefix = 2

            if current_key is not None:
                # if we found a new key but haven't found a value to the previous
                # key it must have been a boolean argument.
                results[current_key] = True

            current_key = word[prefix:]

        # values
        else:
            word = word.strip()
            if current_key is None:
                # we failed to parse the string. We think this is a value, but we don't know what's the key.
                # fallback to naive parsing.
                raise ValueError(CLI_ARGS_PARSE_FAILED_EXCEPTION)

            else:
                word = _guess_type(word)
                results[current_key] = word
                current_key = None

    if current_key is not None:
        # last key was a boolean
        results[current_key] = True

    return results


def _parse_arg_value_with_equal(arg_val: str) -> Tuple[str, str]:
    values = arg_val.split("=", 1)
    key = values[0]

    while len(key) > 0 and key[0] == "-":
        key = key.replace("-", "", 1)

    key = key.strip()

    if not len(key) > 0:
        raise ValueError(CLI_ARGS_PARSE_FAILED_EXCEPTION)

    value = _guess_type(values[1].strip())

    return key, value


def _guess_type(s):
    import ast

    try:
        return ast.literal_eval(s)

    except (ValueError, SyntaxError):
        return str(s)
