# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import logging
import os
import shutil
from typing import Any, Dict, Optional

from ..logging_messages import (
    CONFIG_API_KEY_SAVED_INFO,
    CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING,
)
from .config_api import get_config

LOGGER = logging.getLogger(__name__)


def save(
    directory: Optional[str] = None,
    save_all: bool = False,
    force: bool = False,
    **settings: Dict[str, Any]
) -> None:
    """
    An easy way to create a config file.

    Args:
        directory: str (optional), location to save the
            .comet.config file. Typical values are "~/" (home)
            and "./" (current directory). Default is "~/" or
            COMET_CONFIG, if set
        save_all: bool (optional). If True, will create
            entries for all items that are configurable
            with their defaults. Default is False
        force: bool (optional). If True, overwrite pre-existing
            .comet.config file. If False, ask.
        settings: any valid setting and value

    Valid settings include:

    * api_key
    * disable_auto_logging
    * experiment_key
    * offline_directory
    * workspace
    * project_name
    * logging_console
    * logging_file
    * logging_file_level
    * logging_file_overwrite
    * timeout_cleaning
    * timeout_upload

    Examples:

    ```python
    >>> import comet_ml
    >>> comet_ml.config.save(api_key="...")
    >>> comet_ml.config.save(api_key="...", directory="./")
    ```
    """
    cfg = get_config()
    cfg._set_settings(settings)
    cfg.save(directory, save_all=save_all, force=force)


def api_key_save(config_path: str, api_key: str) -> None:
    """
    Low-level function to only change the api_key
    of a .comet.config file in the home directory.
    """
    from configobj import ConfigObj

    if os.path.exists(config_path):
        config = ConfigObj(config_path)
        try:
            shutil.copyfile(config_path, config_path + ".bak")
        except Exception:
            LOGGER.warning(CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING, exc_info=True)
    else:
        config = ConfigObj()
        config.filename = config_path

    if "comet" not in config:
        config["comet"] = {}

    config["comet"]["api_key"] = api_key
    config.write()
    LOGGER.info(
        CONFIG_API_KEY_SAVED_INFO,
        config_path,
    )
