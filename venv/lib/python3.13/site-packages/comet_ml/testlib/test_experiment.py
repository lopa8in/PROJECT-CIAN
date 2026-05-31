# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import types
from functools import wraps

from comet_ml import Experiment
from comet_ml._typing import Any, List
from comet_ml.config import get_backend_address
from comet_ml.connection import RestServerConnection
from comet_ml.experiment import CometExperiment
from comet_ml.feature_toggles import FeatureToggles

from .dummy import DummyHeartBeatThread, DummyOnlineStreamer, make_dummy_overrides

FLUSH_INITIAL_DATA_LOGGER_TIMEOUT = 10
SOME_API_KEY = "some api key"
SOME_RUN_ID = "some run id"
SOME_PROJECT_ID = "some project id"


def experiment_builder(
    api_key=SOME_API_KEY,
    cls=Experiment,
    streamer=None,
    allow_report=False,
    upload_web_asset_url_prefix="",
    upload_web_image_url_prefix="",
    upload_api_asset_url_prefix="",
    upload_api_image_url_prefix="",
    log_git_metadata=False,
    log_git_patch=False,
    log_env_cpu=False,
    **kwargs,
):
    class _TestingExperiment(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Override with defaults to avoid environment:
            self.config.override = make_dummy_overrides(self.config.config_map)

        def _setup_http_handler(self):
            self.http_handler = None

        def _setup_streamer(self, *args, **kwargs):
            if streamer is None:
                self.streamer = DummyOnlineStreamer(
                    api_key=SOME_API_KEY, run_id=SOME_RUN_ID, project_id=SOME_PROJECT_ID
                )
            else:
                self.streamer = streamer

            self.feature_toggles = FeatureToggles({}, self.config)
            self.upload_web_asset_url_prefix = upload_web_asset_url_prefix
            self.upload_web_image_url_prefix = upload_web_image_url_prefix
            self.upload_api_asset_url_prefix = upload_api_asset_url_prefix
            self.upload_api_image_url_prefix = upload_api_image_url_prefix

            # Create a Connection for the cases where we need to test the reports
            if allow_report:
                self.connection = RestServerConnection(
                    self.api_key,
                    self.id,
                    get_backend_address(),
                    self.config["comet.timeout.http"],
                    verify_tls=True,
                )

            self._heartbeat_thread = DummyHeartBeatThread()

            return True

        def _mark_as_started(self):
            pass

        def _mark_as_ended(self):
            pass

        def add_tag(self, *args, **kwargs):
            return CometExperiment.add_tag(self, *args, **kwargs)

        def add_tags(self, *args, **kwargs):
            return CometExperiment.add_tags(self, *args, **kwargs)

        def _report(self, *args, **kwargs):
            if not allow_report:
                return None

            return super(_TestingExperiment, self)._report(*args, **kwargs)

        def _register_callback_remotely(self, *args, **kwargs):
            pass

        def send_notification(self, title, status=None, additional_data=None):
            pass

        def _check_experiment_throttled(self):
            return False

    return _TestingExperiment(
        api_key,
        log_git_metadata=log_git_metadata,
        log_git_patch=log_git_patch,
        log_env_cpu=log_env_cpu,
        **kwargs,
    )


def build_experiment(
    flush_initial_data_logger_timeout=FLUSH_INITIAL_DATA_LOGGER_TIMEOUT,
    api_key=SOME_API_KEY,
    run_id=SOME_RUN_ID,
    project_id=SOME_PROJECT_ID,
    auto_output_logging=None,
    **kwargs,
):
    streamer = DummyOnlineStreamer(
        api_key=api_key,
        run_id=run_id,
        project_id=project_id,
    )
    if auto_output_logging is None:
        auto_output_logging = "simple"
    kwargs["log_env_cpu"] = kwargs.get("log_env_cpu", False)
    kwargs["log_env_gpu"] = kwargs.get("log_env_gpu", False)
    kwargs["log_env_network"] = kwargs.get("log_env_network", False)
    kwargs["log_env_disk"] = kwargs.get("log_env_disk", False)

    experiment = experiment_builder(
        api_key=api_key,
        streamer=streamer,
        auto_output_logging=auto_output_logging,
        **kwargs,
    )

    # flush initial data logger thread
    experiment._flush_initial_data_logger(flush_initial_data_logger_timeout)
    experiment.streamer.clean()
    return experiment


class TestExperiment:
    """
    Note: This is an alpha version of a Experiment class for use in
    external test frameworks. Checks all methods to see if there are
    stderr output after a call. If so, it raises.

    Examples:

    ```
    >>> from cometx.testlib import TestExperiment
    >>> experiment = TestExperiment()
    >>> experiment.log_image([])
    Exception: COMET ERROR: Could not convert image_data into an image; ignored
    ```
    Normally, such an improper log command would not raise, but
    would silently continue. For testing, we want the exception.
    """

    def __init__(self, log_env_details=False, log_code=False, *args, **kwargs):
        self._experiment = build_experiment(
            FLUSH_INITIAL_DATA_LOGGER_TIMEOUT,
            log_env_details=log_env_details,
            log_code=log_code,
            *args,
            **kwargs,
        )

    def end(self):
        # type: () -> None
        self._experiment.end()

    def _wrapper(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            retval = func(*args, **kwargs)
            error_messages = self._get_messages_of_type("stderr")
            self._experiment.streamer.clean()
            if error_messages:
                for message in error_messages:
                    if "COMET ERROR" in message["output"]:
                        raise Exception(message["output"]) from None
            return retval

        return wrapped

    def __getattr__(self, attr):
        # type: (str) -> Any
        attr = getattr(self._experiment, attr)
        if isinstance(attr, types.MethodType):
            attr = self._wrapper(attr)
        return attr

    def _get_all_messages_raw(self, messages, mtype):
        # type: (List[Any], str) -> List[Any]
        found_messages = []
        for message in messages:
            if getattr(message, mtype, None):
                found_messages.append(message)
        return found_messages

    def _get_messages_of_type(self, mtype=None):
        # type: (str) -> List[dict]
        if mtype is not None:
            return [
                message.to_message_dict()
                for message in self._get_all_messages_raw(
                    self._experiment.streamer.messages, mtype
                )
            ]
        else:
            return [
                message.to_message_dict()
                for message in self._experiment.streamer.messages
            ]
