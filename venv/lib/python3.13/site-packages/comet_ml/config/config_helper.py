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
import logging
import os

from .._jupyter import _in_colab_environment
from ..file_utils import expand_user_home_path

COLAB_DRIVE_MOUNT = "/content/drive/MyDrive/"


def disable_everett_debug_logger():
    logging.getLogger("everett").setLevel(logging.WARNING)


def clean_config_path(file_path: str) -> str:
    """Apply the usual path cleaning function for config paths"""
    return os.path.abspath(expand_user_home_path(file_path))


def config_path_from_directory(directory: str) -> str:
    return clean_config_path(os.path.join(directory, ".comet.config"))


def get_default_config_path() -> str:
    config_home = os.environ.get("COMET_CONFIG")
    if config_home is not None:
        if config_home is not None and os.path.isdir(config_home):
            config_home = config_path_from_directory(config_home)

        return clean_config_path(config_home)

    elif _in_colab_environment():
        if os.path.isdir(COLAB_DRIVE_MOUNT):
            return config_path_from_directory(COLAB_DRIVE_MOUNT)
        else:
            return config_path_from_directory("~")
    else:
        return config_path_from_directory("~")
