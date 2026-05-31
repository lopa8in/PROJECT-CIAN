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
import random
import sys

import comet_ml
from comet_ml import _logging

import sentry_sdk

from . import before_send, environment_details, logger_setup, shutdown

LOGGER = logging.getLogger(__name__)
PYTEST_ENV = "pytest" in sys.modules

SAMPLE_RATE = 1


@functools.lru_cache(0 if PYTEST_ENV else 128)
def randomized_enabled() -> bool:
    return random.random() >= 0.5


@_logging.convert_exception_to_log_message(
    "Error setting up error tracker",
    logger=LOGGER,
    exception_info=True,
    logging_level=logging.DEBUG,
)
def setup_sentry_error_tracker() -> None:
    config = comet_ml.get_config()

    sentry_dsn = config.get_string(None, "comet.internal.sentry_dsn")
    debug = config.get_bool(None, "comet.internal.sentry_debug")

    tags = environment_details.collect_tags()

    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[],
        default_integrations=False,
        debug=debug,
        before_send=before_send.callback,
        release=tags["release"],
        sample_rate=SAMPLE_RATE,
    )

    sdk_context = environment_details.collect_initial_context()
    sentry_sdk.set_context(
        "python-sdk-context",
        sdk_context,
    )
    for key, value in tags.items():
        sentry_sdk.set_tag(key, value)

    root_logger = logging.getLogger("comet_ml")
    logger_setup.setup_sentry_error_handlers(root_logger)

    shutdown.register_flush()


@_logging.convert_exception_to_log_message(
    "Error while checking if Sentry is enabled",
    logger=LOGGER,
    exception_info=True,
    return_on_exception=False,
    logging_level=logging.DEBUG,
)
def enabled_in_config() -> bool:
    config = comet_ml.get_config()

    sentry_dsn = config.get_string(None, "comet.internal.sentry_dsn")
    if sentry_dsn is None:
        return False

    error_tracking_config = config.get_bool(
        None, "comet.error_tracking.enable", default=True
    )
    return error_tracking_config


def initialized() -> bool:
    if sentry_sdk.Hub.current.client is not None:
        return True

    return False
