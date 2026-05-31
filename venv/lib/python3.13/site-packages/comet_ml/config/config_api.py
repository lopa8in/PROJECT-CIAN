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

import logging
from typing import TYPE_CHECKING

import comet_ml.secrets.interpreter

from .. import secrets
from .._logging import _get_comet_logging_config
from .._typing import Any, Dict, Optional, Union
from ..api_key.comet_api_key import CometApiKey, parse_api_key
from ..dataclasses import experiment_info
from ..logging_messages import (
    BASE_URL_MISMATCH_CONFIG_API_KEY,
    CONFIG_GET_DISPLAY_SUMMARY_LEVEL_INVALID_WARNING,
)
from ..utils import get_root_url, sanitize_url, url_join
from .config_class import Config
from .config_constants import (
    COMET_URL_OVERRIDE_CONFIG_KEY,
    DEFAULT_COMET_URL_OVERRIDE,
    DEFAULT_COMET_URL_OVERRIDE_PATH,
)
from .config_map import CONFIG_MAP
from .streamlit_config import _in_streamlit_environment

if TYPE_CHECKING:
    from ..experiment import CometExperiment


LOGGER = logging.getLogger(__name__)

# Global experiment placeholder. Should be set by the latest call of Experiment.init()
experiment = None


def get_global_experiment() -> Optional["CometExperiment"]:
    global experiment
    return experiment


def set_global_experiment(new_experiment):
    global experiment
    experiment = new_experiment


def get_running_experiment() -> Optional["CometExperiment"]:
    """Returns a currently running experiment or None if there is no such.

    Returns:
        Returns the running experiment or None.

    Example:
        ```python linenums="1"
        import comet_ml

        experiment = comet_ml.get_running_experiment()
        ```
    """

    global experiment
    return experiment


def create_config_from_map(config_map: Dict[str, Dict[str, Any]]) -> Config:
    """
    Create a Config instance given a config
    mapping.

    Note: this may return a config for use
    in streamlit environments.
    """
    if _in_streamlit_environment():
        from .streamlit_config import StreamlitConfig

        cfg = StreamlitConfig(config_map)
    else:
        cfg = Config(config_map)

    return cfg


def get_config(setting: Any = None) -> Union[Config, Any]:
    """
    Get a config or setting from the current config
    (os.environment or .env file).

    Note: this is not cached, so every time we call it, it
    re-reads the file. This makes these values always up to date
    at the expense of re-getting the data.
    """
    cfg = create_config_from_map(CONFIG_MAP)

    if setting is None:
        return cfg
    else:
        return cfg[setting]


_last_api_key_returned = None


def get_last_returned_api_key() -> Optional[str]:
    return _last_api_key_returned


def get_api_key(api_key: Optional[str], config: Config) -> Optional[str]:
    if api_key is None:
        api_key = config["comet.api_key"]

    final_api_key = secrets.interpreter.interpret(api_key)

    parsed_api_key = parse_api_key(final_api_key)
    if parsed_api_key is None:
        return None

    # update config parameters from API key
    _update_comet_url_override(parsed_api_key, config=config)

    final_api_key = parsed_api_key.api_key
    # Hide api keys from the log
    if final_api_key and config.get_bool(None, "comet.logging.hide_api_key") is True:
        _get_comet_logging_config().redact_string(final_api_key)

    global _last_api_key_returned
    _last_api_key_returned = final_api_key

    return final_api_key


def get_comet_url_override(config: Optional[Config]) -> str:
    """This MUST always be used to get the COMET_URL_OVERRIDE value"""
    if config is None:
        config = get_config()
    url_override = config[COMET_URL_OVERRIDE_CONFIG_KEY]
    if url_override is None or url_override == "":
        return DEFAULT_COMET_URL_OVERRIDE
    else:
        return url_override


def get_backend_address(config: Optional[Config] = None) -> str:
    return sanitize_url(get_comet_url_override(config))


def get_comet_root_url(config: Optional[Config] = None) -> str:
    return get_root_url(sanitize_url(get_comet_url_override(config)))


def get_optimizer_address(config: Config) -> str:
    optimizer_url = config.get_string(None, "comet.optimizer_url")

    if optimizer_url is None:
        return url_join(get_comet_root_url(config), "optimizer/")
    else:
        return sanitize_url(optimizer_url, ending_slash=True)


def _update_comet_url_override(api_key: CometApiKey, config: Config):
    config_url_override = config[COMET_URL_OVERRIDE_CONFIG_KEY]
    if config_url_override is not None and config_url_override != "":
        config_base_url = get_root_url(config_url_override)
        if api_key.base_url is not None and api_key.base_url != config_base_url:
            LOGGER.warning(
                BASE_URL_MISMATCH_CONFIG_API_KEY, config_base_url, api_key.base_url
            )
        # do not change base url
        return

    # set the global variable with value from API key or with default one
    if api_key.base_url is not None:
        comet_url_override = url_join(api_key.base_url, DEFAULT_COMET_URL_OVERRIDE_PATH)
    else:
        comet_url_override = DEFAULT_COMET_URL_OVERRIDE

    CONFIG_MAP[COMET_URL_OVERRIDE_CONFIG_KEY]["default"] = comet_url_override


def get_project_name(project_name: str, config: Config) -> str:
    return project_name if project_name is not None else config["comet.project_name"]


def get_workspace(workspace: str, config: Config) -> str:
    return workspace if workspace is not None else config["comet.workspace"]


def get_check_tls_certificate(config: Config) -> bool:
    return config.get_bool(None, "comet.internal.check_tls_certificate")


def get_comet_timeout_http(config: Config) -> float:
    return config.get_int(None, "comet.timeout.http")


def discard_api_key(api_key: str) -> None:
    """Discards the provided API key as invalid. After this method invocation the discarded key will not be masked in
    the logger output.
    """
    if api_key is not None:
        _get_comet_logging_config().discard_string_from_redact(api_key)


def get_display_summary_level(
    display_summary_level: Optional[int], config: Config
) -> int:
    if display_summary_level is None:
        return config["comet.display_summary_level"]
    else:
        try:
            return int(display_summary_level)
        except Exception:
            LOGGER.warning(
                CONFIG_GET_DISPLAY_SUMMARY_LEVEL_INVALID_WARNING, display_summary_level
            )
            return 1


def get_previous_experiment(previous_experiment, config: Config):
    if previous_experiment is None:
        return config["comet.experiment_key"]
    else:
        return previous_experiment


def collect_experiment_info(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    workspace: Optional[str] = None,
) -> experiment_info.ExperimentInfo:
    config = get_config()
    final_project_name = config.get_string(project_name, "comet.project_name")
    final_workspace = config.get_string(workspace, "comet.workspace")
    final_api_key = get_api_key(api_key, config)

    return experiment_info.ExperimentInfo(
        api_key=final_api_key,
        project_name=final_project_name,
        workspace=final_workspace,
    )
