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

from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Optional

import comet_ml

from . import environment_details, user_details
from .error_filtering import sentry_filter_chain
from .types import Event, Hint

if TYPE_CHECKING:
    from comet_ml import CometExperiment
    from comet_ml.config_class import Config


def callback(event: Event, hint: Hint) -> Optional[Event]:
    """
    Used to filter events and provide them with extra details that could not
    be collected during Sentry client initialization.
    """
    config_ = comet_ml.get_config()

    is_valid = sentry_filter_chain.validate(event, hint)

    if not is_valid:
        return None

    if _should_drop(event, config_):
        return None

    try:
        _add_extra_details(event, config_)
    except Exception:
        return None

    return event


def _add_extra_details(event: Event, config: "Config") -> None:
    current_experiment = comet_ml.get_global_experiment()

    if "user" in event:
        event["user"]["id"] = user_details.get_id()
    else:
        event["user"] = {"id": user_details.get_id()}

    contexts = event["contexts"]

    if current_experiment is not None:
        experiment_data = {
            "experiment_id": current_experiment.id,
            "experiment_class": current_experiment.__class__.__name__,
        }
        contexts["experiment"] = experiment_data
        contexts["python-sdk-FT"] = _extract_raw_feature_toggles(current_experiment)

    contexts["python-sdk-context"][
        "cloud_provider"
    ] = environment_details.try_get_cloud_provider()
    contexts["python-sdk-context"][
        "cuda_visible_devices"
    ] = environment_details.get_cuda_visible_devices()

    comet_url = _get_config_module().get_comet_url_override(config)
    contexts["python-sdk-context"]["comet-url"] = comet_url


def _extract_raw_feature_toggles(experiment: "CometExperiment") -> Dict[str, Any]:
    if not experiment.feature_toggles:
        feature_toggles = {}
    else:
        feature_toggles = experiment.feature_toggles.raw_toggles
    return feature_toggles


def _should_drop(event: Event, config: "Config") -> bool:
    config_module = _get_config_module()

    try:
        backend_url = config_module.get_comet_url_override(config)
        CLOUD_BACKEND_ADDRESS = comet_ml.config.DEFAULT_COMET_URL_OVERRIDE

        if CLOUD_BACKEND_ADDRESS != backend_url:
            return True

        return False
    except Exception:
        return True


def _get_config_module() -> ModuleType:
    """
    Helps to avoid circular import
    """
    from comet_ml import config

    return config
