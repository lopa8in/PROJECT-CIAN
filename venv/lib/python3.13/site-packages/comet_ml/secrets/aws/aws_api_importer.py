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

import functools


def import_error_handler(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as exception:
            raise ImportError(
                "You are likely missing the dependency 'boto3',"
                "install it with: `python -m pip install boto3`"
            ) from exception

    return wrapper


@import_error_handler
def Session():
    import boto3

    return boto3.Session


@import_error_handler
def ClientError():
    import botocore.exceptions

    return botocore.exceptions.ClientError
