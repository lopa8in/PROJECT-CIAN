#!/usr/bin/env python
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

"""
You can use `comet login` to  create a Comet configuration file with your API key

It is used in this manner in the terminal:

$ comet login

This will ask you for your Comet API key.

"""
import argparse
import sys

from comet_ml.config.config_init import _init as init_api_key

ADDITIONAL_ARGS = False


def get_parser_arguments(parser):
    # nothing here to add
    pass


def login(parsed_args):
    init_api_key(should_prompt_user=sys.stdin.isatty())


def main(args):
    # Called via `comet login ...`
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    get_parser_arguments(parser)
    parsed_args, _ = parser.parse_known_args(args)

    login(parsed_args)
