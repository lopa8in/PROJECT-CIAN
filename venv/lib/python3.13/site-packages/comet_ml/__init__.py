# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

"""comet-ml"""
# flake8: noqa: F401
# Ignore unused import until we can refactor the imports
from __future__ import print_function

import logging
import sys

from . import error_tracking
from ._logging import _setup_comet_logging
from ._online import ExistingExperiment, Experiment
from ._reporting import EXPERIMENT_CREATION_FAILED
from ._ui import UI
from .annotations import Box, Layer, Polygon
from .api import API, APIExperiment
from .api_helpers.experiment_key import get_experiment_key
from .api_objects.model import Model
from .artifacts import Artifact, LoggedArtifact
from .comet_start.comet_start import CometExperiment, ExperimentBase, start
from .comet_start.config_manager import ConfigurationManager, InitParameters
from .comet_start.experiment_config import ExperimentConfig
from .config import (
    discard_api_key,
    get_api_key,
    get_backend_address,
    get_config,
    get_global_experiment,
    get_previous_experiment,
    get_running_experiment,
)
from .config.config_init import init, init_onprem, login
from .confusion_matrix import ConfusionMatrix
from .connection import RestApiClient, RestServerConnection
from .connection.connection_factory import get_comet_api_client, get_rest_api_client
from .data_structure import Embedding, Histogram
from .debug.debug_helpers import error_mode
from .exceptionhook import _create_exception_hook
from .exceptions import (
    BadCallbackArguments,
    ExperimentCleaningException,
    ExperimentDisabledException,
    ExperimentNotAlive,
)
from .experiment import BaseExperiment, end, flush
from .feature_toggles import FeatureToggles
from .file_utils import make_template_filename
from .heartbeat import HeartbeatThread
from .json_encoder import NestedEncoder
from .loggers.fastai_logger import patch as fastai_patch
from .loggers.keras_logger import patch as keras_patch
from .loggers.lightgbm_logger import patch as lgbm_patch
from .loggers.mlflow_logger import patch as mlflow_patch
from .loggers.prophet_logger import patch as prophet_patch
from .loggers.pytorch_logger import patch as pytorch_patch
from .loggers.pytorch_tensorboard.logger import patch as pytorch_tb_patch
from .loggers.shap_logger import patch as shap_patch
from .loggers.sklearn_logger import patch as sklearn_patch
from .loggers.tensorboard_logger import patch as tb_patch
from .loggers.tensorflow_logger import patch as tf_patch
from .loggers.tfma_logger import patch as tfma_patch
from .loggers.xgboost_logger import patch as xg_patch
from .logging_messages import (
    ADD_SYMLINK_ERROR,
    ADD_TAGS_ERROR,
    EXPERIMENT_LIVE,
    EXPERIMENT_THROTTLED,
    GET_ARTIFACT_VERSION_OR_ALIAS_GIVEN_TWICE,
    GET_ARTIFACT_WORKSPACE_GIVEN_TWICE,
    INTERNET_CONNECTION_ERROR,
    INVALID_API_KEY,
    REGISTER_RPC_FAILED,
    SEND_NOTIFICATION_FAILED,
)
from .monkey_patching import CometModuleFinder
from .offline import ExistingOfflineExperiment, OfflineExperiment
from .optimizer import Optimizer
from .rpc import create_remote_call, get_remote_action_definition
from .streamer import OnlineStreamer
from .upload_callback.callback import UploadCallback
from .upload_callback.s3_uploader import get_s3_uploader
from .utils import (
    create_asset_url,
    generate_guid,
    get_comet_version,
    get_time_monotonic,
    is_valid_experiment_key,
    merge_url,
    valid_ui_tabs,
)

ui = UI()

__author__ = "Gideon<Gideon@comet.ml>"
__all__ = [
    "API",
    "APIExperiment",
    "Artifact",
    "Box",
    "CometExperiment",
    "ConfusionMatrix",
    "Embedding",
    "error_mode",
    "ExistingExperiment",
    "ExistingOfflineExperiment",
    "Experiment",
    "ExperimentBase",
    "ExperimentConfig",
    "get_comet_api_client",
    "get_experiment_key",
    "get_global_experiment",
    "get_running_experiment",
    "get_s3_uploader",
    "Histogram",
    "init",
    "init_onprem",
    "Layer",
    "login",
    "Model",
    "OfflineExperiment",
    "Optimizer",
    "Polygon",
    "start",
    "UploadCallback",
    "end",
    "flush",
]
__version__ = get_comet_version()

LOGGER = logging.getLogger(__name__)

if not get_config("comet.disable_auto_logging"):
    # Activate the monkey patching
    MODULE_FINDER = CometModuleFinder()
    keras_patch(MODULE_FINDER)
    sklearn_patch(MODULE_FINDER)
    tf_patch(MODULE_FINDER)
    tb_patch(MODULE_FINDER)
    pytorch_patch(MODULE_FINDER)
    fastai_patch(MODULE_FINDER)
    mlflow_patch(MODULE_FINDER)
    xg_patch(MODULE_FINDER)
    tfma_patch(MODULE_FINDER)
    prophet_patch(MODULE_FINDER)
    shap_patch(MODULE_FINDER)
    lgbm_patch(MODULE_FINDER)
    pytorch_tb_patch(MODULE_FINDER)
    MODULE_FINDER.start()

# Configure the logging
_setup_comet_logging(get_config())

# Register exception hook to process unhandled exceptions
sys.excepthook = _create_exception_hook(sys.excepthook)

if error_tracking.randomized_enabled() and error_tracking.enabled_in_config():
    error_tracking.setup_sentry_error_tracker()
