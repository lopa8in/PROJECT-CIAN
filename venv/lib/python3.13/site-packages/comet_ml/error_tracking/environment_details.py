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

import functools
import logging
import os
import sys
from typing import Any, Dict, Optional

import psutil

from .. import _jupyter, env_logging, oscontainer, utils

LOGGER = logging.getLogger(__name__)


def collect_initial_context() -> Dict[str, Any]:
    """
    Returns environment details. It is recommended to collect here only
    the information which we expect to be set before comet import.

    If you need the data which might be set during the script
    execution - consider adding it directly to sentry event dict
    inside before_send function.
    """

    result = {
        "pid": env_logging.get_pid(),
        "os": env_logging.get_os(),
        "python_version_verbose": env_logging.get_python_version_verbose(),
        "python_exe": sys.executable,
        "container_detected": _container_detected(),
        "github_actions": _is_github_actions(),
        "cpu_count": psutil.cpu_count(),
    }

    installed_packages_details = _get_installed_packages_details()
    result.update(installed_packages_details)

    return result


def collect_tags():
    """
    Tags are close to context but can be used for filtering and search
    in Sentry
    """
    result = {
        "os_type": env_logging.get_os_type(),
        "python_version": env_logging.get_python_version(),
        "release": env_logging.get_comet_version(),
        "jupyter": _jupyter._in_jupyter_environment(),
        "colab": _jupyter._in_colab_environment(),
        "aws_lambda": utils.is_aws_lambda_environment(),
        "session_id": utils.random_ascii_string(6),
    }

    return result


@functools.lru_cache(maxsize=1)
def _container_detected():
    return oscontainer.OSContainer().is_containerized()


def _is_github_actions():
    return "GITHUB_ACTIONS" in os.environ


def _get_installed_packages_details() -> Dict[str, str]:
    packages = env_logging.get_pip_packages(as_tuples=True)
    OBSERVED = set(["torch", "tensorflow", "numpy", "keras", "tensorboard"])

    result = {}
    for name, version in packages:
        if name in OBSERVED:
            result[name] = version

    return result


def try_get_cloud_provider() -> Optional[str]:
    """
    get_env_cloud_details can take time, we want to
    use it only if it was already called before and the result
    was saved in cache (e.g. after an experiment is created).

    It also means that there is no sense to call it at the beginning of the script.
    """
    try:
        cache_info = env_logging.get_env_cloud_details.cache_info()

        if cache_info.currsize > 0:
            details = env_logging.get_env_cloud_details()
            if details is not None:
                return details["provider"]
    except Exception:
        LOGGER.debug("Failed to get cloud provider data", exc_info=True)

    return None


def get_cuda_visible_devices() -> Optional[str]:
    return (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else None
    )
