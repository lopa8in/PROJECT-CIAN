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
Author: Boris Feld and Douglas Blank

This module contains comet base Experiment code

"""
from __future__ import print_function

import json
import logging
import numbers
import os
import os.path
import random
import sys
import tempfile
import threading
import time
import traceback
import types
from _thread import get_ident
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from threading import Thread

from ._logging import _get_comet_logging_config
from .api_helpers.experiment_key import get_experiment_key
from .common_experiment import CommonExperiment
from .system.system_metrics_types import NamedSystemMetrics, NamedSystemMetricsList

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from comet_ml import atexit_hooks

from . import env_logging, event_tracker, experiment_loggers, generalized_equality
from ._jupyter import (
    DEFAULT_COLAB_NOTEBOOK_ASSET_NAME,
    DEFAULT_JUPYTER_CODE_ASSET_NAME,
    DEFAULT_JUPYTER_INTERACTIVE_FILE_NAME,
    _get_colab_notebook_json,
    _get_colab_notebook_url,
    _in_colab_environment,
    _in_ipython_environment,
    _in_jupyter_environment,
)
from ._reporting import (
    EXPERIMENT_CREATION_FAILED,
    GIT_PATCH_GENERATION_FAILED,
    NESTED_PARAMETERS_LOGGED,
)
from ._typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AudioNormalization,
    Callable,
    Dict,
    ExperimentCleanupResults,
    ExperimentThrottledStatus,
    List,
    Optional,
    Point3D,
    Set,
    TemporaryFilePath,
    Tuple,
    Union,
    ValidFilePath,
)
from .annotations import Layer
from .api_objects.model import RemoteModel
from .artifacts import Artifact, LoggedArtifact
from .assets import preprocess as assets_preprocess
from .cli_args_parse import parse_command_line_arguments
from .cloud_storage_utils import META_ERROR_MESSAGE, META_SYNCED
from .config import (
    AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
    DEFAULT_3D_CLOUD_UPLOAD_LIMITS,
    DEFAULT_ASSET_UPLOAD_SIZE_LIMIT,
    DEFAULT_INITIAL_DATA_LOGGER_JOIN_TIMEOUT,
    DEFAULT_UPLOAD_SIZE_LIMIT,
    MAXIMAL_KEY_LENGTH,
    MAXIMAL_VALUE_LENGTH,
    get_config,
    get_display_summary_level,
    get_global_experiment,
    get_project_name,
    get_running_experiment,
    get_workspace,
    set_global_experiment,
)
from .confusion_matrix import ConfusionMatrix
from .console import get_std_logger
from .constants import (
    ASSET_TYPE_AUDIO,
    ASSET_TYPE_CONDA_INFO,
    ASSET_TYPE_CONFUSION_MATRIX,
    ASSET_TYPE_CURVE,
    ASSET_TYPE_DATAFRAME_PROFILE,
    ASSET_TYPE_DEFAULT,
    ASSET_TYPE_EMBEDDINGS,
    ASSET_TYPE_HISTOGRAM_3D,
    ASSET_TYPE_MODEL_ELEMENT,
    ASSET_TYPE_NOTEBOOK,
    ASSET_TYPE_SOURCE_CODE,
    ASSET_TYPE_TENSORFLOW_FILE,
    ASSET_TYPE_TEXT_SAMPLE,
    ASSET_TYPE_VIDEO,
    CONDA_ENV_ASSET_TYPE,
    CONDA_ENV_FILE_NAME,
    CONDA_INFO_FILE_NAME,
    CONDA_SPEC_ASSET_TYPE,
    CONDA_SPEC_FILE_NAME,
    OTHER_KEY_CREATED_FROM,
)
from .convert_utils import (
    check_is_matplotlib_figure,
    check_is_pandas_dataframe,
    convert_log_table_input_to_io,
    convert_model_to_string,
    convert_object_to_dictionary,
    convert_to_scalar,
    convert_to_string_key,
    convert_to_string_value,
    convert_user_input_to_metric_value,
    data_to_fp,
    dataset_to_sprite_image,
    fix_special_floats,
)
from .data_structure import Embedding, Histogram
from .debug import debug_helpers
from .env_logging import (
    _get_conda_env,
    _get_conda_explicit_packages,
    _get_conda_info,
    _in_conda_environment,
    _in_pydev_console,
    get_caller_file_path,
    get_env_cloud_details,
    get_env_details_message,
    get_ipython_notebook,
    get_ipython_source_code,
    get_pip_packages,
)
from .exceptions import (
    AssetIsTooBig,
    CometException,
    ExperimentAlreadyUploaded,
    InterruptedExperiment,
    LambdaUnsupported,
    MaxExperimentNumberReachedException,
    ProjectConsideredLLM,
    RPCFunctionAlreadyRegistered,
    SDKVersionIsTooOldException,
    ValidationError,
    ViewOnlyAccessException,
)
from .experiment_storage import ExperimentStorage
from .file_uploader import (
    AssetDataUploadProcessor,
    AudioUploadProcessor,
    FigureUploadProcessor,
    FileUpload,
    GitPatchUploadProcessor,
    ImageUploadProcessor,
    ObjectToConvertFileUpload,
    PreprocessedFileAsset,
    PreprocessedMemoryFileAsset,
    PreprocessedRemoteAsset,
    UserTextFileUpload,
    VideoUploadProcessor,
    dispatch_user_file_upload,
    handle_in_memory_file_upload,
    preprocess_asset_file,
    preprocess_asset_memory_file,
    preprocess_remote_asset,
)
from .file_utils import (
    expand_user_home_path,
    list_assets_folder,
    make_template_filename,
    safe_filename,
    write_file_like_to_tmp_file,
)
from .flatten_dict.exclusions import can_parameters_be_flattened
from .flatten_dict.flattener import (
    METRICS_DELIMITER,
    METRICS_MAX_DEPTH,
    PARAMETERS_DELIMITER,
    PARAMETERS_MAX_DEPTH,
    flatten_dict,
)
from .logging_messages import (
    ADD_TAGS_ERROR,
    AWS_LAMBDA_NEEDS_END,
    CODECARBON_DIR_CREATION_FAILED,
    CODECARBON_NOT_INSTALLED,
    CODECARBON_START_FAILED,
    CODECARBON_STOP_FAILED,
    CREATE_SYMLINK_ONLINE_ONLY_ERROR,
    DEPRECATED_COPY_TO_TEMP_ARGUMENT,
    DEPRECATED_WORKSPACE_MODEL_REGISTRY_ARGUMENT,
    EXPERIMENT_COMET_EXPERIMENT_KEY_INVALID_EXCEPTION,
    EXPERIMENT_CREATE_EMBEDDING_EMPTY_METADATA_ERROR,
    EXPERIMENT_CREATE_EMBEDDING_EMPTY_VECTOR_ERROR,
    EXPERIMENT_CREATE_EMBEDDING_IMAGE_FAILED_CREATING_IMAGE_ERROR,
    EXPERIMENT_CREATE_EMBEDDING_NO_IMAGE_SIZE_ERROR,
    EXPERIMENT_CREATE_EMBEDDING_VECTOR_SHAPE_ERROR,
    EXPERIMENT_GET_ARTIFACT_NOT_SUPPORTED_EXCEPTION,
    EXPERIMENT_GET_HF_DATASET_NOT_SUPPORTED_EXCEPTION,
    EXPERIMENT_GET_KERAS_CALLBACK_DEPRECATED_WARNING,
    EXPERIMENT_GET_PARAMETER_SHORT_NAME_DEPRECATION,
    EXPERIMENT_INIT_DISPLAY_SUMMARY_WARNING,
    EXPERIMENT_INITIAL_DATA_LOGGER_FLUSH_FAILED,
    EXPERIMENT_INITIAL_DATA_LOGGER_INCOMPLETE,
    EXPERIMENT_INITIAL_DATA_LOGGER_WAITING_TO_COMPLETE,
    EXPERIMENT_INVALID_EPOCH,
    EXPERIMENT_INVALID_STEP,
    EXPERIMENT_LOG_ARTIFACT_NOT_SUPPORTED_EXCEPTION,
    EXPERIMENT_LOG_ASSET_DATA_DEPRECATED_FILE_NAME_WARNING,
    EXPERIMENT_LOG_ASSET_DATA_STEP_MANDATORY_EXCEPTION,
    EXPERIMENT_LOG_ASSET_FILE_DATA_NONE_EXCEPTION,
    EXPERIMENT_LOG_ASSET_FOLDER_LOG_FILE_NAME_WARNING,
    EXPERIMENT_LOG_ASSET_UNSUPPORTED_UPLOAD_TYPE_EXCEPTION,
    EXPERIMENT_LOG_AUDIO_NAME_TRUNCATED_WARNING,
    EXPERIMENT_LOG_AUDIO_NO_DATA_EXCEPTION,
    EXPERIMENT_LOG_AUDIO_UNSUPPORTED_NORMALIZATION_ERROR,
    EXPERIMENT_LOG_CODE_CALLER_JUPYTER,
    EXPERIMENT_LOG_CODE_CALLER_NOT_FOUND,
    EXPERIMENT_LOG_CODE_FILE_NAME_FOLDER_MUTUALLY_EXCLUSIVE,
    EXPERIMENT_LOG_CODE_MISSING_CODE_NAME,
    EXPERIMENT_LOG_CODE_NOT_A_FILE_WARNING,
    EXPERIMENT_LOG_CONFUSION_MATRIX_CREATE_FAILED_ERROR,
    EXPERIMENT_LOG_CONFUSION_MATRIX_EMPTY_MATRIX_ERROR,
    EXPERIMENT_LOG_CONFUSION_MATRIX_GENERAL_ERROR,
    EXPERIMENT_LOG_CURVE_VALIDATION_ERROR,
    EXPERIMENT_LOG_DATAFRAME_PROFILE_DATAFRAME_PROFILE_ERROR,
    EXPERIMENT_LOG_DATAFRAME_PROFILE_MISSING_PANDAS_LOG_DATAFRAME,
    EXPERIMENT_LOG_DATAFRAME_PROFILE_NOT_PANDAS_DATAFRAME,
    EXPERIMENT_LOG_DATASET_HASH_WARNING,
    EXPERIMENT_LOG_DATASET_INFO_NAME_VERSION_PATH_EMPTY_WARNING,
    EXPERIMENT_LOG_EMBEDDING_NOT_IMPLEMENTED_EXCEPTION,
    EXPERIMENT_LOG_IMAGE_ANNOTATION_VALIDATION_ERROR,
    EXPERIMENT_LOG_IMAGE_MATPLOTLIB_FIGURE_WARNING,
    EXPERIMENT_LOG_IMAGE_NAME_TRUNCATED_WARNING,
    EXPERIMENT_LOG_IMAGE_NO_DATA_EXCEPTION,
    EXPERIMENT_LOG_METRIC_EXCEPTION_ERROR,
    EXPERIMENT_LOG_METRIC_NAME_TRUNCATED,
    EXPERIMENT_LOG_MODEL_NO_SUCH_FILE_OR_DIR_ERROR,
    EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_IPYNB_EXCEPTION,
    EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_STRING_EXCEPTION,
    EXPERIMENT_LOG_OTHER_EXCEPTION_ERROR,
    EXPERIMENT_LOG_OTHER_IGNORE_AUTOMATIC_INFO,
    EXPERIMENT_LOG_OTHER_KEY_VALUE_TRUNCATED,
    EXPERIMENT_LOG_OTHERS_DICT_OR_KEY_VALUE_ERROR,
    EXPERIMENT_LOG_OTHERS_FROM_ENVIRONMENT_FAILED_ERROR,
    EXPERIMENT_LOG_PARAMETER_EMPTY_NAME_ERROR,
    EXPERIMENT_LOG_PARAMETER_EXCEPTION_ERROR,
    EXPERIMENT_LOG_PARAMETER_IGNORE_AUTOMATIC_INFO,
    EXPERIMENT_LOG_PARAMETER_NAME_TRUNCATED,
    EXPERIMENT_LOG_PREPROCESSED_ASSET_NAME_TRUNCATED_WARNING,
    EXPERIMENT_LOG_PREPROCESSED_ASSET_ON_FAILED_ASSET_UPLOAD_CALL_FAILED_WARNING,
    EXPERIMENT_LOG_PREPROCESSED_ASSET_STEP_MANDATORY_ERROR,
    EXPERIMENT_LOG_PREPROCESSED_REMOTE_ASSET_NAME_TRUNCATED_WARNING,
    EXPERIMENT_LOG_TABLE_FILENAME_NOT_STRING_EXCEPTION,
    EXPERIMENT_LOG_TAG_VALIDATION_ERROR,
    EXPERIMENT_LOG_VIDEO_NAME_TRUNCATED_WARNING,
    EXPERIMENT_LOG_VIDEO_NO_DATA_EXCEPTION,
    EXPERIMENT_ON_END_CLEAN_FAILED_WARNING,
    EXPERIMENT_ON_END_STREAMER_FLUSH_FAILED_ERROR,
    EXPERIMENT_REGISTER_MODEL_NO_LOG_MODEL_CALL_EXCEPTION,
    EXPERIMENT_REGISTER_MODEL_TIMEOUT_WARNING,
    EXPERIMENT_SET_CODE_ALREADY_CALLED_WARNING,
    EXPERIMENT_SET_CODE_IGNORE_FILENAME_WARNING,
    EXPERIMENT_SET_MODEL_GRAPH_ALREADY_CALLED_WARNING,
    EXPERIMENT_SET_OS_PACKAGES_FAILED_WARNING,
    EXPERIMENT_SET_PIP_PACKAGES_FAILED_WARNING,
    EXPERIMENT_START_FAILED_CREATE_CPU_LOGER_ERROR,
    EXPERIMENT_START_FAILED_CREATE_DISK_LOGER_ERROR,
    EXPERIMENT_START_FAILED_CREATE_GPU_LOGER_ERROR,
    EXPERIMENT_START_FAILED_CREATE_NETWORK_LOGER_ERROR,
    EXPERIMENT_START_FAILED_LOG_CLOUD_DETAILS_ERROR,
    EXPERIMENT_START_FAILED_LOG_COLAB_NOTEBOOK_URL_ERROR,
    EXPERIMENT_START_FAILED_LOG_ENV_DETAILS_ERROR,
    EXPERIMENT_START_FAILED_SET_OS_PACKAGES_ERROR,
    EXPERIMENT_START_FAILED_SET_PIP_PACKAGES_ERROR,
    EXPERIMENT_START_FAILED_SET_RUN_CMD_ARGS_ERROR,
    EXPERIMENT_START_FAILED_SET_RUN_FILE_NAME_ERROR,
    EXPERIMENT_START_FAILED_SET_SOURCE_CODE_ERROR,
    EXPERIMENT_START_FAILED_SETUP_STD_LOGGER_ERROR,
    EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_ERROR,
    EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_WITH_EXCEPTION_ERROR,
    EXPERIMENT_THROTTLED,
    FAILED_LOG_EMBEDDING_GROUPS,
    FAILED_LOG_IPYTHON_NOTEBOOK,
    GIT_LOGGING_ERROR,
    GIT_REPO_NOT_FOUND,
    GO_TO_DOCS_MSG,
    INITIAL_DATA_LOGGER_FLUSHING_EXPERIMENT_INTERRUPTED_BY_USER,
    INITIAL_DATA_LOGGER_FLUSHING_FAILED,
    JUPYTER_NEEDS_END,
    LATE_IMPORT_DEGRADED_MODE,
    LOG_ASSET_FOLDER_EMPTY,
    LOG_ASSET_FOLDER_ERROR,
    LOG_COLAB_NOTEBOOK_ERROR,
    LOG_FIGURE_KALEIDO_NOT_INSTALLED,
    LOG_GIT_METADATA_ERROR,
    LOG_GIT_PATCH_ERROR,
    LOG_METRIC_AUTOMATIC_IGNORED,
    LOG_METRICS_MAX_DEPTH_REACHED,
    LOG_PARAMS_EMPTY_CONVERTED_MAPPING,
    LOG_PARAMS_EMPTY_CONVERTED_MAPPING_ERROR,
    LOG_PARAMS_EMPTY_MAPPING,
    LOG_PARAMS_MAX_DEPTH_REACHED,
    LOG_TABLE_FILENAME_AND_HEADERS,
    LOG_TABLE_NONE_VALUES,
    METRIC_NONE_WARNING,
    OPTIMIZER_COULDNT_END,
    PYDEV_NEEDS_END,
    SET_CODE_CODE_DEPRECATED,
    SET_CODE_FILENAME_DEPRECATED,
    UNEXPECTED_ERROR_WHEN_LOGGING_FIGURE,
    UNEXPECTED_LOGGING_ERROR,
    UPLOAD_ASSET_TOO_BIG,
    UPLOAD_FILE_OS_ERROR,
)
from .messages import (
    BaseMessage,
    CloudDetailsMessage,
    FileNameMessage,
    GitMetadataMessage,
    GpuStaticInfoMessage,
    HtmlMessage,
    HtmlOverrideMessage,
    InstalledPackagesMessage,
    LogDependencyMessage,
    LogOtherMessage,
    MetricMessage,
    ModelGraphMessage,
    OsPackagesMessage,
    ParameterMessage,
    RegisterModelMessage,
    SystemInfoMessage,
)
from .monkey_patching import ALREADY_IMPORTED_MODULES
from .rpc import RemoteCall, call_remote_function
from .streamer import BaseStreamer
from .summary import Summary
from .synchronization.model_upload import status_handler
from .system.cpu.cpu_metrics_data_logger import CPUMetricsDataLogger
from .system.disk.disk_metrics_data_logger import DiskMetricsDataLogger
from .system.gpu.gpu_logging import (
    convert_gpu_details_to_metrics,
    get_gpu_static_info,
    get_initial_gpu_metric,
    is_gpu_details_available,
)
from .system.gpu.gpu_metrics_data_logger import GPUMetricsDataLogger
from .system.network.network_metrics_data_logger import NetworkMetricsDataLogger
from .system.system_metrics_logging_thread import SystemMetricsLoggingThread
from .upload_callback.callback import UploadCallback
from .utils import (
    compress_git_patch,
    create_asset_url,
    find_logger_spec,
    generate_guid,
    get_dataframe_profile_html,
    get_user,
    is_aws_lambda_environment,
    is_list_like,
    is_valid_experiment_key,
    local_timestamp,
    metric_name,
    read_unix_packages,
    shape,
    truncate_to_length,
    truncate_to_length_or_raise,
)
from .validation.curve_data_validator import CurveDataValidator
from .validation.image.annotation_validator import ImageAnnotationValidator
from .validation.method_parameters_validator import MethodParametersTypeValidator
from .validation.tag_validator import TagsValidator, TagValidator

LOGGER = logging.getLogger(__name__)
LOG_ONCE_CACHE: Set[str] = set()

if TYPE_CHECKING:
    import dulwich.repo  # pragma: no-cover


class CometExperiment(CommonExperiment):
    """
    Experiment is a unit of measurable research that defines a single run with some data/parameters/code/results.

    Creating an Experiment object in your code will report a new experiment to your Comet.ml project. Your Experiment
    will automatically track and collect many things and will also allow you to manually report anything.

    You can create multiple objects in one script (such as when looping over multiple hyper-parameters).
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        workspace: Optional[str] = None,
        log_code: Optional[bool] = True,
        log_graph: Optional[bool] = True,
        auto_param_logging: Optional[bool] = True,
        auto_metric_logging: Optional[bool] = True,
        parse_args: Optional[bool] = True,
        auto_output_logging: Optional[str] = AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
        log_env_details: Optional[bool] = True,
        log_git_metadata: Optional[bool] = True,
        log_git_patch: Optional[bool] = True,
        disabled: Optional[bool] = False,
        log_env_gpu: Optional[bool] = True,
        log_env_host: Optional[bool] = True,
        display_summary: Optional[bool] = None,
        log_env_cpu: Optional[bool] = True,
        log_env_network: Optional[bool] = True,
        log_env_disk: Optional[bool] = True,
        display_summary_level: Optional[int] = None,
        optimizer_data: Optional[Dict[str, Any]] = None,
        auto_weight_logging: Optional[bool] = None,
        auto_log_co2: Optional[bool] = True,
        auto_metric_step_rate: Optional[int] = 10,
        auto_histogram_tensorboard_logging: Optional[bool] = False,
        auto_histogram_epoch_rate: Optional[int] = 1,
        auto_histogram_weight_logging: Optional[bool] = False,
        auto_histogram_gradient_logging: Optional[bool] = False,
        auto_histogram_activation_logging: Optional[bool] = False,
        experiment_key: Optional[str] = None,
        distributed_node_identifier: Optional[str] = None,
    ):
        """
        Base class for all experiment classes.
        """
        self._start_time_local = time.time()

        self.tmpdir = tempfile.mkdtemp()
        self.config = get_config()

        LOGGER.debug("Experiment's temporary dir: %r", self.tmpdir)

        if self.config.get_bool(display_summary, "comet.display_summary") is not None:
            LOGGER.warning(EXPERIMENT_INIT_DISPLAY_SUMMARY_WARNING)

        self.display_summary_level = get_display_summary_level(
            display_summary_level, self.config
        )
        self._summary = Summary(self.__class__.__name__)

        self.project_name = get_project_name(project_name, self.config)
        self.workspace = get_workspace(workspace, self.config)
        self.name = None

        self.params = {}
        self.deprecated_params = (
            {}
        )  # holds parameters with short name to check for deprecated
        self.metrics = {}
        self.others = {}
        self.tags: Set[str] = set()

        self.model_upload_synchronizer = status_handler.StatusHandler()
        self.model_register_synchronizer = status_handler.StatusHandler()

        # Get parameters:

        self.distributed_node_identifier = self.config.get_string(
            distributed_node_identifier, "comet.distributed_node_identifier"
        )
        self._log_code = self.config.get_bool(
            log_code, "comet.auto_log.code", True, not_set_value=True
        )
        self.log_graph = self.config.get_bool(
            log_graph, "comet.auto_log.graph", True, not_set_value=True
        )
        self.auto_param_logging = self.config.get_bool(
            auto_param_logging, "comet.auto_log.parameters", True, not_set_value=True
        )
        self.auto_metric_logging = self.config.get_bool(
            auto_metric_logging, "comet.auto_log.metrics", True, not_set_value=True
        )
        self.parse_args = self.config.get_bool(
            parse_args, "comet.auto_log.cli_arguments", True, not_set_value=True
        )
        self.log_env_details = self.config.get_bool(
            log_env_details, "comet.auto_log.env_details", True, not_set_value=True
        )
        self.log_git_metadata = self.config.get_bool(
            log_git_metadata, "comet.auto_log.git_metadata", True, not_set_value=True
        )
        self.log_git_patch = self.config.get_bool(
            log_git_patch, "comet.auto_log.git_patch", True, not_set_value=True
        )
        self.disabled = self.config.get_bool(
            disabled, "comet.auto_log.disable", False, not_set_value=False
        )
        self.log_env_gpu = self.config.get_bool(
            log_env_gpu, "comet.auto_log.env_gpu", True, not_set_value=True
        )
        self.log_env_host = self.config.get_bool(
            log_env_host, "comet.auto_log.env_host", True, not_set_value=True
        )
        self.log_env_cpu = self.config.get_bool(
            log_env_cpu, "comet.auto_log.env_cpu", True, not_set_value=True
        )
        self.log_env_network = self.config.get_bool(
            log_env_network, "comet.auto_log.env_network", True, not_set_value=True
        )
        self.log_env_disk = self.config.get_bool(
            log_env_disk, "comet.auto_log.env_disk", True, not_set_value=True
        )
        self.auto_log_co2 = self.config.get_bool(
            auto_log_co2, "comet.auto_log.co2", True, not_set_value=True
        )
        self.auto_histogram_tensorboard_logging = self.config.get_bool(
            auto_histogram_tensorboard_logging,
            "comet.auto_log.histogram_tensorboard",
            not_set_value=False,
        )
        self.auto_histogram_weight_logging = self.config.get_deprecated_bool(
            auto_weight_logging,
            "comet.auto_log.weights",
            auto_histogram_weight_logging,
            "comet.auto_log.histogram_weights",
            new_not_set_value=False,
        )
        self.auto_histogram_gradient_logging = self.config.get_bool(
            auto_histogram_gradient_logging,
            "comet.auto_log.histogram_gradients",
            not_set_value=False,
        )
        self.auto_histogram_activation_logging = self.config.get_bool(
            auto_histogram_activation_logging,
            "comet.auto_log.histogram_activations",
            not_set_value=False,
        )
        self.auto_metric_step_rate = self.config.get_int(
            auto_metric_step_rate,
            "comet.auto_log.metric_step_rate",
            not_set_value=10,
        )
        self.auto_histogram_epoch_rate = self.config.get_int(
            auto_histogram_epoch_rate,
            "comet.auto_log.histogram_epoch_rate",
            not_set_value=1,
        )
        # Default is "native" for regular environments, "simple" for IPython:
        auto_output_logging = self.config.get_raw(
            auto_output_logging,
            "comet.auto_log.output_logger",
            AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
            not_set_value=AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
        )
        self.auto_output_logging = find_logger_spec(auto_output_logging)

        # Deactivate git logging in case the user disabled logging code
        if not self._log_code:
            self.log_git_patch = False

        # Disable some logging if log_env_details is False
        if not self.log_env_details:
            self.log_env_gpu = False
            self.log_env_cpu = False
            self.log_env_host = False

        self.autolog_others_ignore = set(self.config["comet.logging.others_ignore"])
        self.autolog_metrics_ignore = set(self.config["comet.logging.metrics_ignore"])
        self.autolog_parameters_ignore = set(
            self.config["comet.logging.parameters_ignore"]
        )

        # Keep track of which auto-loggers data came from
        self._frameworks: Set[str] = set()

        if not self.disabled:
            if len(ALREADY_IMPORTED_MODULES) > 0:
                LOGGER.warning(
                    LATE_IMPORT_DEGRADED_MODE, ", ".join(ALREADY_IMPORTED_MODULES)
                )

        # Generate a unique identifier for this experiment.
        self.id = self._get_experiment_key(experiment_key)

        self.alive = False
        self.ended = False
        self.streamer_wait_completed_and_closed = False
        self.is_github = False
        self.focus_link = None
        self.upload_limit = DEFAULT_UPLOAD_SIZE_LIMIT
        self.video_upload_limit = DEFAULT_UPLOAD_SIZE_LIMIT
        self.asset_upload_limit = DEFAULT_ASSET_UPLOAD_SIZE_LIMIT
        self.points_3d_upload_limits = DEFAULT_3D_CLOUD_UPLOAD_LIMITS
        self.upload_web_asset_url_prefix: Optional[str] = None
        self.upload_web_image_url_prefix: Optional[str] = None
        self.upload_api_asset_url_prefix: Optional[str] = None
        self.upload_api_image_url_prefix: Optional[str] = None
        self.artifact_name_max_length: Optional[int] = None
        self.asset_name_max_length: Optional[int] = None
        self.image_name_max_length: Optional[int] = None

        self.streamer: Optional[BaseStreamer] = None
        self.logger = None
        self.system_metrics_thread: Optional[SystemMetricsLoggingThread] = None

        self.initial_data_logger_thread: Optional[Thread] = None
        self.run_id = None
        self.project_id = None
        self.optimizer = None

        self.main_thread_id = get_ident()

        # If set to True, wrappers should only run the original code
        self.disabled_monkey_patching = False

        # Experiment state
        self.context = None
        self.curr_step = None
        self.curr_epoch = None
        self.filename = None

        self.figure_counter = 0

        self.feature_toggles = {}

        # Storage area for use by loggers
        self._storage = ExperimentStorage(
            {
                "keras": {"json_model": {}},
                "torch": {"model": None},
                "xgboost": {
                    "env_model_parameter_set": False,
                    "env_parameter_set": False,
                    "model_graph_set": False,
                    "train_parameter_set": False,
                },
                "shap": {"counter": 0},
                "prophet": {"counter": 0, "internal": False},
            }
        )
        self._localstorage = threading.local()

        self._atexit_hook: Optional[atexit_hooks.ConditionalCallable] = None

        self._graph_set = False
        self._code_set = False
        self._notebook_logged = False
        self._pending_calls: List[RemoteCall] = []
        self._embedding_groups = defaultdict(list)

        self._co2_tracker_directory = os.path.join(self.tmpdir, "codecarbon")
        self._co2_tracker: Any = None

        # Cleanup old experiment before replace it
        previous_experiment = get_global_experiment()
        if previous_experiment is not None and previous_experiment is not self:
            try:
                success = previous_experiment._on_end(wait=False)
                LOGGER.debug(
                    "Previous Experiment was cleaned. Successfully: %r", success
                )
            except Exception:
                LOGGER.debug(
                    "Failed to clean Experiment: [%s]",
                    previous_experiment.id,
                    exc_info=True,
                )

        set_global_experiment(self)

        self.rpc_callbacks = {}

        if optimizer_data is not None:
            self._set_optimizer_from_data(optimizer_data)

        self._print_environment_warnings()

    @staticmethod
    def _print_environment_warnings():
        if _in_jupyter_environment():
            LOGGER.warning(JUPYTER_NEEDS_END)

        if _in_pydev_console():
            LOGGER.warning(PYDEV_NEEDS_END)

        if is_aws_lambda_environment():
            LOGGER.warning(AWS_LAMBDA_NEEDS_END)

    @property
    def url(self):
        """
        Get the url of the experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()
            print(exp.url)
            ```
        """
        return None

    def clean(self) -> bool:
        """Clean the experiment loggers, useful in case you want to debug
        your scripts with IPDB.

        Returns:
            Returns a Boolean indicating whether the cleaning was successful or not.
        """
        return self._on_end(wait=False)

    def end(self) -> None:
        """The `end()` method finalizes an experiment by ensuring all data
        (i.e., parameters, metrics, asset, etc.) are uploaded to Comet before
        it returns. Unless you are running in a Jupyter Notebook, using the
        `end()` method is optional as it will be called automatically by
        Comet when the script ends.

        Note:
            Only one Experiment can be alive in a Python process, if you wish
            to start a new Experiment you should first end the previous Experiment
            by calling the `end()` method and then start the new Experiment.

        Example:
            ```python linenums="1"
            import comet_ml
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # Initialize Comet
            comet_ml.login()
            exp = comet_ml.start(project_name="comet-docs")

            # Run an experiment
            data = load_iris()
            X = data.data
            y = data.target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)

            acc = accuracy_score(y_train, y_pred_train)

            # Log accuracy metric to Comet
            exp.log_metric("accuracy", acc)

            # End the experiment
            exp.end()
            ```
        """
        self._on_end(wait=True)

    def flush(self, timeout: Optional[int] = None) -> bool:
        """
        Flush all pending data to the Comet server. It works similar to
        [end][comet_ml.CometExperiment.end] but without ending the run. This is a
        blocking operation that will wait for all the data logged so far to
        be delivered to the Comet server.

        Args:
            timeout: the timeout to wait for flushing data or None for default timeout (seconds).

        Returns:
            Returns a Boolean indicating whether the flush was successful or
                not.
        """
        LOGGER.debug("Flushing initial data logger")
        if self._flush_initial_data_logger(DEFAULT_INITIAL_DATA_LOGGER_JOIN_TIMEOUT):
            LOGGER.debug("Initial data logger flushed successfully")
        else:
            LOGGER.error("Failed to flush initial data logger")

        LOGGER.debug("Flushing streamer")
        if self.streamer is not None:
            if self.streamer.flush(timeout=timeout):
                LOGGER.debug("Streamer flushed successfully")
                return True
            else:
                LOGGER.debug("Failed to flush streamer")

        return False

    def _check_metric_step_report_rate(self, value: int) -> bool:
        """
        Check to see we should report at the current batch and value.
        """
        if value is not None and isinstance(value, int):
            if self.auto_metric_step_rate == 0:
                return False
            else:
                return (value % self.auto_metric_step_rate) == 0
        else:
            LOGGER.debug(
                "report rate value is not an integer: %r; logging anyway", value
            )
            return True

    def _check_histogram_epoch_report_rate(self, value: int) -> bool:
        """
        Check to see we should report at the current epoch and value.

        Note: If unknown level or invalid value, returns True
        """
        if value is not None and isinstance(value, int):
            if self.auto_histogram_epoch_rate == 0:
                return False
            else:
                return (value % self.auto_histogram_epoch_rate) == 0
        else:
            LOGGER.debug(
                "report rate value is not an integer: %r; logging anyway", value
            )
            return True

    def _get_experiment_key(self, user_experiment_key: Optional[str]) -> str:
        experiment_key = self.config.get_string(
            user_experiment_key, "comet.experiment_key"
        )
        if experiment_key is not None:
            if is_valid_experiment_key(experiment_key):
                return experiment_key
            else:
                raise ValueError(
                    EXPERIMENT_COMET_EXPERIMENT_KEY_INVALID_EXCEPTION % experiment_key
                )
        else:
            return get_experiment_key(user_input=None)

    def _on_start(self) -> None:
        """Called when the Experiment is started"""
        self._mark_as_started()

    def _mark_as_started(self) -> None:
        pass

    def _mark_as_ended(self) -> None:
        pass

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        pass

    def __internal_api__announce__(self) -> None:
        pass

    def _report_summary(self) -> None:
        """
        Display to logger a summary of experiment if there
        is anything to report. If not, no summary will be
        shown.
        """
        # We wait to set this until now:
        self._summary.set("data", "url", self._get_experiment_url())
        self._summary.set("data", "display_summary_level", self.display_summary_level)
        if self.name is not None:
            self._summary.set("data", "name", self.name)

        summary = self._summary.generate_summary(self.display_summary_level)

        if self.get_name() is None:
            notification_title = "%s Experiment summary" % get_user()
        else:
            notification_title = "%s - %s Experiment summary" % (
                get_user(),
                self.get_name(),
            )

        self.send_notification(notification_title, "finished", summary)

    def _log_once_at_level(
        self, logging_level: int, message: str, *args: Any, **kwargs: Any
    ) -> None:
        """Log the given message once at the given level then at the DEBUG level on further calls"""
        global LOG_ONCE_CACHE

        if message not in LOG_ONCE_CACHE:
            LOG_ONCE_CACHE.add(message)
            LOGGER.log(logging_level, message, *args, **kwargs)
        else:
            LOGGER.debug(message, *args, **kwargs)

    def _clear_storage(self) -> None:
        """Called to clear storage associated with this experiment to avoid memory leaks."""
        self._storage.clear()
        self._localstorage.__dict__.clear()

    def _streamer_wait_for_finish(self) -> bool:
        """Called to wait for experiment streamer's cleanup procedures"""
        return self.streamer.wait_for_finish()

    def _streamer_has_connection(self) -> bool:
        if self.streamer is None:
            return False
        return self.streamer.has_connection_to_server()

    def _experiment_fully_ended(self) -> bool:
        """Allows to check if experiment already ended fully, i.e., all associated resources was properly released"""
        return self.ended and self.streamer_wait_completed_and_closed

    def _on_end(self, wait=True) -> bool:
        """Called when the Experiment is replaced by another one or at the
        end of the script
        """
        LOGGER.debug(
            "Experiment on_end called, wait: %s, experiment ID: %s, fully ended: %r",
            wait,
            self.id,
            self._experiment_fully_ended(),
        )

        if self._atexit_hook is not None and wait:
            self._atexit_hook.disable()

        if self._experiment_fully_ended() is True:
            # already ended, no need to process it twice
            return True

        cleaned_successfully = False
        try:
            clean_result = self._clean_experiment(wait)
            cleaned_successfully = clean_result.is_experiment_cleaned_successfully()
            if not cleaned_successfully:
                # https://comet-ml.atlassian.net/browse/CM-10746?focusedCommentId=34445
                LOGGER.debug("Failed to clean up the experiment: %s", clean_result)

        except Exception as e:
            LOGGER.warning(EXPERIMENT_ON_END_CLEAN_FAILED_WARNING, e, exc_info=True)
        finally:
            self._finalize()

        return cleaned_successfully

    def _finalize(self) -> None:
        # Mark the experiment as not alive anymore to avoid future new messages
        self.alive = False

        # Mark also the experiment as ended as some experiments might never be alive
        self.ended = True

        # Clear associated storage only after unsetting the experiment.alive to make sure
        # that storage consumers can check alive flag before trying to access storage
        self._clear_storage()

    def _clean_experiment(self, wait: bool) -> ExperimentCleanupResults:
        initial_thread_cleaned = True

        # flush initial data logger first to have summary populated with GIT patch data
        if wait:
            initial_thread_cleaned = self._flush_initial_data_logger(
                DEFAULT_INITIAL_DATA_LOGGER_JOIN_TIMEOUT
            )
            if not initial_thread_cleaned:
                LOGGER.warning(EXPERIMENT_INITIAL_DATA_LOGGER_FLUSH_FAILED)
        elif (
            self.initial_data_logger_thread is not None
            and self.initial_data_logger_thread.is_alive()
        ):
            LOGGER.warning(EXPERIMENT_INITIAL_DATA_LOGGER_INCOMPLETE, self.id)

        if self.alive:
            if self.optimizer is not None:
                LOGGER.debug("optimizer.end() called")
                try:
                    force_wait = self.optimizer["optimizer"].end(self)
                    if force_wait:
                        # Force wait to be true causes all uploads to finish:
                        LOGGER.debug("forcing wait to be True for optimizer")
                        wait = True
                except Exception:
                    LOGGER.error(OPTIMIZER_COULDNT_END, exc_info=True)

            if len(self._embedding_groups) > 0:
                try:
                    self._log_embedding_groups()
                except Exception:
                    LOGGER.error(FAILED_LOG_EMBEDDING_GROUPS, exc_info=True)

            # Co2 Tracking
            if self._co2_tracker is not None:
                try:
                    self._co2_tracker.stop()
                except Exception:
                    LOGGER.debug(CODECARBON_STOP_FAILED, exc_info=True)

                if os.path.isdir(self._co2_tracker_directory) and os.listdir(
                    self._co2_tracker_directory
                ):
                    self._log_asset_folder(
                        self._co2_tracker_directory, folder_name="co2-tracking"
                    )

            # Try to log IPython notebook code
            if _in_ipython_environment() and self._log_code:
                self._log_ipython_notebook_code()

            # Try to log Google Colab notebook code
            if _in_colab_environment() and self._log_code and not self._notebook_logged:
                self._log_colab_notebook_code()

            # Logs the source of the data
            for framework in sorted(self._frameworks):
                # Temporary way of logging multiple frameworks. Pass `framework="comet"` so it won't
                # show in the Experiment summary by default
                self._log_other(
                    OTHER_KEY_CREATED_FROM,
                    framework,
                    framework="comet",
                    include_context=False,
                )

            try:
                self._report_summary()
            except Exception:
                LOGGER.debug("Summary not reported", exc_info=True)

            self.__internal_api__announce__()

            if len(ALREADY_IMPORTED_MODULES) > 0:
                LOGGER.warning(
                    LATE_IMPORT_DEGRADED_MODE, ", ".join(ALREADY_IMPORTED_MODULES)
                )

                self._log_other(
                    "sys.already_imported_modules",
                    json.dumps(list(ALREADY_IMPORTED_MODULES)),
                    include_context=False,
                )

            # Display throttling message
            try:
                throttled, message, reasons = self._check_experiment_throttled()
                if throttled:
                    LOGGER.warning(EXPERIMENT_THROTTLED, message)
            except Exception:
                LOGGER.debug(
                    "Failed to check if experiment has been throttled", exc_info=True
                )

            if "pytest" not in sys.modules:
                # It's not sent in unit test to avoid old tests failures related to message indices and message amount
                # The solution is supposed to be temporary.
                self._log_metric(
                    "process duration",
                    round(time.time() - self._start_time_local),
                    step=self.curr_step,
                    epoch=self.curr_epoch,
                    include_context=False,
                )

        system_metrics_cleaned = self._close_system_metrics_logging_threads(wait)

        streamer_cleaned_successfully = True
        if self.streamer is not None:
            LOGGER.debug("Closing streamer: %r", self.streamer)
            self.streamer.close()
            if wait:
                try:
                    if self._streamer_wait_for_finish():
                        LOGGER.debug("Streamer cleaned successfully")
                    else:
                        LOGGER.debug("Streamer DIDN'T clean successfully")
                        streamer_cleaned_successfully = False
                except Exception as e:
                    LOGGER.error(
                        EXPERIMENT_ON_END_STREAMER_FLUSH_FAILED_ERROR,
                        self.streamer,
                        e,
                        exc_info=True,
                    )
                    streamer_cleaned_successfully = False

                # set flag that streamer was closed in synchronous mode
                self.streamer_wait_completed_and_closed = True
            else:
                # mark to eventually terminate the message loop
                self.streamer.stop()

        self._mark_as_ended()

        # flush log records collected through callbacks
        LOGGER.debug("Flushing collected logs about throttling ")
        _get_comet_logging_config().flush_logging_callback()

        # clean logging
        std_logger_cleaned = self._clean_logging()

        return ExperimentCleanupResults(
            initial_data_logger_cleaned=initial_thread_cleaned,
            system_metrics_thread_cleaned=system_metrics_cleaned,
            streamer_cleaned=streamer_cleaned_successfully,
            std_logger_cleaned=std_logger_cleaned,
        )

    def _close_system_metrics_logging_threads(self, wait: bool) -> bool:
        if self.system_metrics_thread is None:
            return True

        self.system_metrics_thread.close()
        successful_clean = True
        if wait is True:
            LOGGER.debug(
                "SystemMetricsLoggingThread before join; isAlive = %s",
                self.system_metrics_thread.is_alive(),
            )
            self.system_metrics_thread.join(2)
            successful_clean = not self.system_metrics_thread.is_alive()

            if not successful_clean:
                LOGGER.debug(
                    "SystemMetricsLoggingThread didn't clean successfully after 2s"
                )
            else:
                LOGGER.debug("SystemMetricsLoggingThread cleaned successfully")

        return successful_clean

    def _check_experiment_throttled(
        self,
    ) -> ExperimentThrottledStatus:
        return ExperimentThrottledStatus(False, None, None)

    def _clean_logging(self) -> bool:
        if self.logger is not None:
            LOGGER.debug("Cleaning STDLogger")
            try:
                self.logger.clean()
            except Exception:
                LOGGER.debug("Failed to clean std logger", exc_info=True)
                return False
            self.logger = None

        return True

    def _start(self, log_initial_data_asynchronously: bool = True) -> None:
        """Starts the current experiment by preparing all relevant routines and data loggers.
        Args:
            log_initial_data_asynchronously: Optional, bool - allows to switch ON/OFF initial data logging in the
            separate thread. It is useful for testing purposes to quickly switch between logging modes.
        """
        try:
            self.alive = self._setup_streamer()

        except (
            ExperimentAlreadyUploaded,
            ViewOnlyAccessException,
            MaxExperimentNumberReachedException,
        ) as ex:
            LOGGER.error(
                EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_WITH_EXCEPTION_ERROR % ex
            )
        except SDKVersionIsTooOldException as ex:
            LOGGER.error(
                EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_WITH_EXCEPTION_ERROR % ex
            )
            raise ex
        except ProjectConsideredLLM as ex:
            LOGGER.error(str(ex))
        except CometException as exception:
            tb = traceback.format_exc()

            exc_log_message = getattr(exception, "log_message", None)
            exc_args = getattr(exception, "args", None)

            if exc_log_message is not None:
                if exc_args is not None:
                    LOGGER.error(exc_log_message, *exc_args, exc_info=True)
                else:
                    # We don't have any log args
                    LOGGER.error(exc_log_message, exc_info=True)
            else:
                LOGGER.error(
                    EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_ERROR + GO_TO_DOCS_MSG,
                    exc_info=True,
                )

            self._report(event_name=EXPERIMENT_CREATION_FAILED, err_msg=tb)
        except Exception:
            tb = traceback.format_exc()
            err_msg = EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_ERROR + GO_TO_DOCS_MSG
            LOGGER.error(err_msg, exc_info=True, extra={"show_traceback": True})
            self._report(event_name=EXPERIMENT_CREATION_FAILED, err_msg=tb)

        # no need to go any further if not alive
        if not self.alive:
            LOGGER.debug("Experiment is not alive, terminating")
            return

        # Register the cleaning method to be called when the script ends
        self._atexit_hook = atexit_hooks.register(self._on_end)

        # After the handshake is done, mark the experiment as alive
        self._on_start()

        try:
            self._setup_std_logger()
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_SETUP_STD_LOGGER_ERROR, exc_info=True)

        ##############################################################
        # log_co2:
        ##############################################################
        if self.auto_log_co2:
            self._start_log_co2_tracker()

        ##############################################################
        # log_code:
        ##############################################################
        if self._log_code:
            self._log_experiment_source_code()

        ##############################################################
        # log_env_details: CPU and GPU logging threads
        ##############################################################
        if self.log_env_details:
            self._start_env_details_logging()

        ##############################################################
        # parse_args:
        ##############################################################
        if self.parse_args:
            try:
                self.set_cmd_args()
            except Exception:
                LOGGER.error(
                    EXPERIMENT_START_FAILED_SET_RUN_CMD_ARGS_ERROR, exc_info=True
                )

        ##############################################################
        # _log_initial_data: Start thread to log the rest of the data
        ##############################################################
        if log_initial_data_asynchronously:
            self.initial_data_logger_thread = Thread(
                group=None, target=self._log_initial_data
            )
            self.initial_data_logger_thread.daemon = True
            self.initial_data_logger_thread.start()
        else:
            self._log_initial_data()  # pragma: no-cover

    def _start_log_co2_tracker(self):
        try:
            import codecarbon

            # Ensure the codecarbon directory exists
            if not os.path.isdir(self._co2_tracker_directory):
                try:
                    os.makedirs(self._co2_tracker_directory)
                except OSError as exc:
                    LOGGER.warning(
                        CODECARBON_DIR_CREATION_FAILED,
                        self._co2_tracker_directory,
                        exc,
                        exc_info=True,
                    )

            self._co2_tracker = codecarbon.EmissionsTracker(
                project_name=self.project_name,
                output_dir=self._co2_tracker_directory,
            )
            self._co2_tracker.start()
        except ImportError:
            LOGGER.debug(CODECARBON_NOT_INSTALLED)

        except Exception:
            LOGGER.debug(CODECARBON_START_FAILED, exc_info=True)

    def _log_experiment_source_code(self):
        try:
            filename = self._get_filename()
            self.set_filename(filename)
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_SET_RUN_FILE_NAME_ERROR, exc_info=True)

        try:
            # Do not log ipython related files
            if not _in_ipython_environment():
                caller = get_caller_file_path()
                if caller is not None:
                    (
                        caller_module_name,
                        experiment_creation_file,
                    ) = caller
                    self._log_code_asset(
                        "experiment_creation", file_name=experiment_creation_file
                    )

                    script_name = sys.argv[0]
                    if caller_module_name != "__main__" and os.path.isfile(script_name):
                        self._log_code_asset(
                            "python_script_name", file_name=script_name
                        )
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_SET_SOURCE_CODE_ERROR, exc_info=True)

    def _start_env_details_logging(self):
        if _in_colab_environment():
            try:
                notebook_url = _get_colab_notebook_url()
                self.log_others({"notebook_url": notebook_url})
            except Exception as ex:
                LOGGER.warning(
                    EXPERIMENT_START_FAILED_LOG_COLAB_NOTEBOOK_URL_ERROR,
                    ex,
                    exc_info=True,
                )

        try:
            self.set_pip_packages()
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_SET_PIP_PACKAGES_ERROR, exc_info=True)

        try:
            self.set_os_packages()
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_SET_OS_PACKAGES_ERROR, exc_info=True)

        try:
            self._log_cloud_details()
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_LOG_CLOUD_DETAILS_ERROR, exc_info=True)

        try:
            if self.log_env_host:
                self._log_env_details()
        except Exception:
            LOGGER.error(EXPERIMENT_START_FAILED_LOG_ENV_DETAILS_ERROR, exc_info=True)

        self._log_others_from_environment()
        self._set_name_from_environment()

        self._start_system_metrics_data_logger()

    def _start_system_metrics_data_logger(self) -> None:
        if not self.alive:
            return

        enabled_data_loggers = []
        if self.log_env_gpu and is_gpu_details_available():
            try:
                gpu_data_logger = self._create_gpu_data_logger()
                enabled_data_loggers.append(gpu_data_logger)
            except Exception:
                LOGGER.error(
                    EXPERIMENT_START_FAILED_CREATE_GPU_LOGER_ERROR, exc_info=True
                )

        if self.log_env_cpu:
            try:
                cpu_data_logger = CPUMetricsDataLogger(
                    initial_interval=self.config["comet.system_cpu_interval"],
                    callback=self._log_cpu_details,
                    include_compute_metrics=self._is_compute_metric_included(),
                    include_cpu_per_core=self.config["comet.auto_log.env_cpu_per_core"],
                )
                enabled_data_loggers.append(cpu_data_logger)
            except Exception:
                LOGGER.error(
                    EXPERIMENT_START_FAILED_CREATE_CPU_LOGER_ERROR, exc_info=True
                )

        if self.log_env_network:
            try:
                network_data_logger = NetworkMetricsDataLogger(
                    initial_interval=self.config["comet.system_network_interval"],
                    callback=self._log_network_rate,
                )
                enabled_data_loggers.append(network_data_logger)
            except Exception:
                LOGGER.error(
                    EXPERIMENT_START_FAILED_CREATE_NETWORK_LOGER_ERROR, exc_info=True
                )

        if self.log_env_disk:
            try:
                disk_utilization_data_logger = DiskMetricsDataLogger(
                    initial_interval=self.config["comet.system_disk_interval"],
                    callback=self._log_disk_utilization,
                )
                enabled_data_loggers.append(disk_utilization_data_logger)
            except Exception:
                LOGGER.error(
                    EXPERIMENT_START_FAILED_CREATE_DISK_LOGER_ERROR,
                    exc_info=True,
                )

        available_data_loggers = []
        for logger in enabled_data_loggers:
            if logger.available():
                available_data_loggers.append(logger)

        # start logging thread if appropriate
        if len(available_data_loggers) == 0:
            return

        self.system_metrics_thread = SystemMetricsLoggingThread(
            metric_data_loggers=available_data_loggers
        )

        self.system_metrics_thread.start()

    def _flush_initial_data_logger(self, timeout: Optional[float] = None) -> bool:
        """Invoked to flush the initial data logger thread. Thus, all collected data will be put into message queue."""
        if self.initial_data_logger_thread is not None:
            if self.initial_data_logger_thread.is_alive():
                LOGGER.info(EXPERIMENT_INITIAL_DATA_LOGGER_WAITING_TO_COMPLETE)
            try:
                self.initial_data_logger_thread.join(timeout)
                return not self.initial_data_logger_thread.is_alive()
            except KeyboardInterrupt:
                LOGGER.warning(
                    INITIAL_DATA_LOGGER_FLUSHING_EXPERIMENT_INTERRUPTED_BY_USER
                )
                return False
            except Exception:
                LOGGER.warning(INITIAL_DATA_LOGGER_FLUSHING_FAILED)
                return False

        return True

    def _log_initial_data(self) -> None:
        """Defines all data to be logged during experiment initialization in the separate thread"""
        ##############################################################
        # log_code:
        ##############################################################
        try:
            self._log_git_information()
        except Exception:
            LOGGER.error(GIT_LOGGING_ERROR, exc_info=True)

        if self.log_env_details:
            # Must be last as in Python 2 subprocess don't have any timeout
            try:
                self._log_conda_packages()
            except Exception:
                LOGGER.debug("Failing to collect conda information", exc_info=True)

    def _report(self, *args: Any, **kwargs: Any) -> None:
        """Do nothing, could be overridden by subclasses"""
        pass

    def __internal_api__report__(self, *args: Any, **kwargs: Any) -> None:
        self._report(*args, **kwargs)

    def _setup_streamer(self) -> bool:
        """
        Do the necessary work to create mandatory objects, like the streamer
        and feature flags
        """
        raise NotImplementedError()

    def _setup_std_logger(self) -> None:
        # Override default sys.stdout and feed to streamer.
        self.logger = get_std_logger(self.auto_output_logging, self.streamer)
        if self.logger is not None:
            self.logger.set_experiment(self)

    def _enqueue_message(self, message: BaseMessage) -> None:
        """Queue a single message in the streamer"""
        # First check for pending callbacks call.
        # We do the check in _enqueue_message as it is the most central code
        if get_ident() == self.main_thread_id:
            self._check_rpc_callbacks()
        self.streamer.put_message_in_q(message)

    def _get_asset_upload_step(self) -> Optional[int]:
        """
        Get the current step of the experiment.
        """
        if self.curr_step is not None:
            # Temporary fix to force step to be integer:
            return int(self.curr_step)

        return None

    def _get_asset_upload_epoch(self) -> Optional[int]:
        """
        Get the current epoch of the experiment.
        """
        if self.curr_epoch is not None:
            # Temporary fix to force epoch to be integer:
            return int(self.curr_epoch)

        return None

    def get_name(self) -> str:
        """
        Get the name of the experiment, if one.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start(project_name="comet-docs")

            exp.set_name("My Name")

            print(exp.get_name())
            ```
        """
        return self.name

    def get_metric(self, name: str) -> Any:
        """
        Get the last value for a metric from those logged.

        Args:
            name (str): The name of the metric to get. This method will return the
                last value logged to this metric.
        """
        name = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH).value

        return self.metrics[name]

    def get_parameter(self, name: str) -> Any:
        """Get a parameter that was logged previously in an Experiment.

        Args:
            name (str): The name of the parameter to get.

        Returns:
            str: Parameter value

        Note:
            If this method is called inside a context, like [test][comet_ml.Experiment.test],
            [train][comet_ml.Experiment.train], [validate][comet_ml.Experiment.validate] or
            [context_manager][comet_ml.Experiment.context_manager], the current context name will be
            automatically added at the front of parameter name.

            Raises a KeyError if parameter with given name not found.

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start(project_name="comet-docs")

            exp.log_parameter("learning_rate", 0.0001)
            exp.log_parameter("batch_size", 64)

            learning_rate = exp.get_parameter("learning_rate")
            print(f"Learning Rate: {learning_rate}")
            assert float(learning_rate) == 0.0001

            batch_size = exp.get_parameter("batch_size")
            print(f"Batch Size: {batch_size}")
            assert int(batch_size) == 64

            exp.end()
            ```
        """
        name = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH).value

        # first: assume the call is within context
        full_name = self._fullname(name)
        if full_name in self.params:
            return self.params[full_name]

        # second: assume the full name provided
        if name in self.params:
            return self.params[name]

        # TODO: current implementation allows cross-context parameter retrieval which should be fixed
        #  in the next major release, see test_get_parameter_cross_context_retrieval_allowed_bug() for details

        # third: check in the deprecated parameters
        if name in self.deprecated_params:
            potential = [k for k in self.params.keys() if k.endswith(name)]
            LOGGER.warning(
                EXPERIMENT_GET_PARAMETER_SHORT_NAME_DEPRECATION % (name, potential)
            )
            return self.deprecated_params[name]

        # not found anywhere - raise KeyError
        raise KeyError(name)

    def get_other(self, name: str) -> Any:
        """
        Get the latest value logged for an `other` parameter from those previously logged.

        Args:
            name (str): The name of the other parameter to get.
        """
        name = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH).value
        return self.others[name]

    def get_key(self) -> str:
        """Returns the experiment key, useful for resume logging data later

        Returns:
            str: Experiment Key (String)

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start(project_name="comet-docs")

            exp.end()

            prev_exp = comet_ml.start(mode="get", experiment_key=exp.get_key())
            ```
        """
        return self.id

    def log_other(self, key: Any, value: Any) -> None:
        """
        Reports a key and value to the `Other` tab on
        Comet.ml. Useful for reporting datasets attributes, datasets
        path, unique identifiers etc.

        See related methods: [log_parameter][comet_ml.Experiment.log_parameter] and
            [log_metric][comet_ml.Experiment.log_metric]

        `Other` metadata can also be defined as environment variables. All environment variables
        that starts with `COMET_LOG_OTHER_` will be logged automatically to the Experiment.

        Args:
            key: Any type of key (str,int,float..)
            value: Any type of value (str,int,float..)
        """
        try:
            return self._log_other(key, value)
        except Exception:
            LOGGER.error(
                EXPERIMENT_LOG_OTHER_EXCEPTION_ERROR,
                exc_info=True,
            )

    def _log_other(
        self,
        key: Any,
        value: Any,
        framework: Optional[str] = None,
        include_context: bool = True,
    ) -> None:
        # Internal logging handler with option to ignore auto-logged keys
        if not self.alive:
            return

        result = convert_to_string_key(key, max_length=MAXIMAL_KEY_LENGTH)
        if result.truncated:
            LOGGER.warning(
                EXPERIMENT_LOG_OTHER_KEY_VALUE_TRUNCATED, MAXIMAL_KEY_LENGTH, key
            )

        key = result.value

        if framework:
            if ("%s:%s" % (framework, key)) in self.autolog_others_ignore:
                # Use % in this message to cache specific string:
                self._log_once_at_level(
                    logging.INFO,
                    EXPERIMENT_LOG_OTHER_IGNORE_AUTOMATIC_INFO % (key, framework, key),
                )
                return
            else:
                self._track_framework_usage(framework)

        if not (isinstance(value, numbers.Number) or value is None):
            value = convert_to_string_value(value, max_length=MAXIMAL_VALUE_LENGTH)

        current_context = self.context
        if include_context is False:
            current_context = None

        message = LogOtherMessage.create(
            context=current_context,
            key=key,
            value=value,
        )

        self._enqueue_message(message)
        self._summary.set(
            "others", self._fullname(key, include_context), value, framework=framework
        )
        self.others[key] = value

    def log_others(self, dictionary: Dict) -> None:
        """
        Reports dictionary of key/values to the `Other` tab on
        Comet.ml. Useful for reporting datasets attributes, datasets
        path, unique identifiers etc.

        See [log_other][comet_ml.Experiment.log_others]

        Args:
            dictionary (dict): dict of key/values where value is any type of
                value (str,int,float..)
        """
        if self.alive:
            if not isinstance(dictionary, Mapping):
                LOGGER.error(
                    EXPERIMENT_LOG_OTHERS_DICT_OR_KEY_VALUE_ERROR, exc_info=True
                )
                return
            else:
                for other in dictionary:
                    self.log_other(other, dictionary[other])

    def _log_others(self, dictionary: Dict, include_context: bool = True) -> None:
        if self.alive:
            if not isinstance(dictionary, Mapping):
                LOGGER.error(
                    EXPERIMENT_LOG_OTHERS_DICT_OR_KEY_VALUE_ERROR, exc_info=True
                )
                return
            else:
                for other in dictionary:
                    self._log_other(
                        other, dictionary[other], include_context=include_context
                    )

    def log_dependency(self, name: Any, version: Any) -> None:
        """
        Reports name,version to the `Installed Packages` tab on Comet. Useful to track dependencies.

        Args:
            name: Any type of key (str,int,float...)
            version: Any type of value (str,int,float...)
        """
        if not self.alive:
            return

        message = LogDependencyMessage.create(
            context=self.context,
            name=name,
            version=version,
        )

        self._enqueue_message(message)
        self._summary.increment_section("uploads", "dependency")

    def log_system_info(self, key: Any, value: Any) -> None:
        """Reports the key and value to the `System metrics` tab in the single experiments view of the
        Comet UI.

        Args:
            key: Any type of key.
            value: Any type of value.

        Note:
            Useful to track general system information. This information can be added to the
            table on the Project view. You can retrieve this information via the Python API.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_system_info(key="info-about-system", value="debian-based")
            exp.end()
            ```
        """
        if not self.alive:
            return

        message = SystemInfoMessage.create(
            context=self.context,
            key=key,
            value=value,
        )
        self._enqueue_message(message)
        self._summary.set("system-info", key, value)

    def log_html(self, html: str, clear: bool = False) -> None:
        """
        Reports any HTML blob to the `HTML` tab on Comet.ml. Useful for creating your own rich reports.
        The HTML will be rendered as an Iframe. Inline CSS/JS supported.

        Args:
            html: Any html string.
            clear: When setting clear=True it will remove all previous html.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_html('<a href="www.comet.com"> I love Comet</a>')
            exp.end()
            ```

        """
        if not self.alive:
            return

        if clear:
            message = HtmlOverrideMessage.create(
                context=self.context,
                htmlOverride=html,
            )
        else:
            message = HtmlMessage.create(
                context=self.context,
                html=html,
            )

        self._enqueue_message(message)
        self._summary.increment_section("uploads", "html")

    def log_html_url(
        self, url: str, text: Optional[str] = None, label: Optional[str] = None
    ) -> None:
        """
        Easy to use method to add a link to a URL in the `HTML` tab
        on Comet.

        Args:
            url: a link to a file or notebook, for example
            text: text to use a clickable word or phrase (optional; uses url if not given)
            label: text that precedes the link

        Example:
            Logging a link:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_html_url("https://my-company.com/file.txt")
            exp.end()
            ```

            Adds html similar to:

            ```html
            <a href="https://my-company.com/file.txt">
            https://my-company.com/file.txt
            </a>
            ```

            Logging a link with text:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_html_url("https://my-company.com/file.txt",
                                        "File")
            exp.end()
            ```

            Adds html similar to:

            ```html
            <a href="https://my-company.com/file.txt">File</a>
            ```

            Logging a link with text and a label:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()
            exp.log_html_url("https://my-company.com/file.txt", "File", "Label")
            ```

            Adds html similar to:

            ```
            Label: <a href="https://my-company.com/file.txt">File</a>
            ```
        """
        text = text if text is not None else url
        if label:
            self.log_html(
                """<div><b>%s</b>: <a href="%s" target="_blank">%s</a></div>"""
                % (label, url, text)
            )
        else:
            self.log_html(
                """<div><a href="%s" target="_blank">%s</a></div>""" % (url, text)
            )

    def set_step(self, step: Optional[int]) -> None:
        """Sets the current step in a training process. In Deep Learning, each
        step is after feeding a single batch into the network. This is used to
        generate correct plots on Comet.

        Args:
            step: The current step number.

        Note:
            You can also pass the step directly
            when reporting [comet_ml.Experiment.log_metric][], and
            [comet_ml.Experiment.log_parameter][].

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            # Initialize Comet experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Create a simple neural network model
            model = Sequential([
                Dense(64, activation='relu', input_shape=(10,)),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Generate dummy data
            X_train = np.random.random((1000, 10))
            y_train = np.random.randint(2, size=(1000, 1))

            # Train the model with detailed step tracking
            for epoch in range(10):
                for batch in range(100):
                    step = epoch * 100 + batch
                    exp.set_step(step)  # Set step for detailed tracking
                    model.train_on_batch(X_train, y_train)

            # End the experiment
            exp.end()
            ```
        """
        self._set_step(step, source=ParameterMessage.source_manual)

    def _set_step(
        self, step: Optional[int], source: str = ParameterMessage.source_autologger
    ) -> None:
        if step is not None:
            step = convert_to_scalar(step)

            if isinstance(step, numbers.Number):
                self.curr_step = step
                self._log_parameter(
                    name="curr_step",
                    value=step,
                    framework="comet",
                    source=source,
                    include_context=False,
                )
            else:
                debug_helpers.log_warning_or_raise(
                    EXPERIMENT_INVALID_STEP, step, logger=LOGGER
                )

    def set_epoch(self, epoch: Optional[int]) -> None:
        """Sets the current epoch in the training process.

        Args:
            epoch: Integer value

        Note:
            You can also pass the epoch directly when reporting
            [comet_ml.Experiment.log_metric][].

        Example:
            ```python linenums="1"
            import comet_ml
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            # Initialize Comet.ml experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Create a dummy dataset, model, and dataloader
            x = torch.randn(100, 10)
            y = torch.randn(100, 1)
            dataset = TensorDataset(x, y)
            data_loader = DataLoader(dataset, batch_size=10)
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
            loss_function = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Training loop
            for epoch in range(10):
                exp.set_epoch(epoch)

                for data, target in data_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_function(output, target)
                    loss.backward()
                    optimizer.step()

                    # Log loss every epoch
                    exp.log_metric("loss", loss.item(), epoch=epoch)

            # End the experiment
            exp.end()
            ```
        """
        self._set_epoch(epoch, source=ParameterMessage.source_manual)

    def _set_epoch(
        self, epoch: Optional[int], source: str = ParameterMessage.source_autologger
    ) -> None:
        if epoch is not None:
            epoch = convert_to_scalar(epoch)

            if isinstance(epoch, numbers.Number):
                self.curr_epoch = epoch
                self._log_parameter(
                    "curr_epoch",
                    epoch,
                    framework="comet",
                    source=source,
                    include_context=False,
                )
            else:
                debug_helpers.log_warning_or_raise(
                    EXPERIMENT_INVALID_EPOCH, epoch, logger=LOGGER
                )

    def log_epoch_end(self, epoch_cnt: int, step: Optional[int] = None) -> None:
        """Logs that the epoch finished. Required for progress bars.

        Args:
            epoch_cnt (int): The current count of completed epochs, indicating
                how many epochs have finished in the training cycle.
            step (int): The step count at which the epoch ends.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            num_epochs = 5
            exp.log_epoch_end(num_epochs)

            exp.end()
            ```
        """
        self._set_step(step)

        if self.alive:
            self._set_epoch(epoch_cnt)

    def log_metric(
        self,
        name: Any,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        """
        Logs a general metric (i.e. accuracy, f1).

        See also [log_metrics][comet_ml.Experiment.log_metrics].


        Args:
            name: Name of your metric.
            value: The value of your metric, we recommend that this is a numerical value.
            step: Used as the X axis when plotting on Comet.
            epoch: Used as the X axis when plotting on Comet.
            include_context: If set to True, the
                current context will be logged along the metric.

        Note:
            Comet guarantees to store 15,000 data points for each metric. If more than 15,000 data points are reported we
            perform a form of [reservoir sub-sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) and download
            sample the metric values.

        Example:
            ```python linenums="1"
            y_pred_train = model.predict(X_train)
            acc = compute_accuracy(y_pred_train, y_train)
            experiment.log_metric("accuracy", acc)
            ```
        """
        with debug_helpers.debug_error_handler(
            message=EXPERIMENT_LOG_METRIC_EXCEPTION_ERROR,
            exc_info=True,
            logger=LOGGER,
        ) as handle:
            handle(
                self._log_metric,
                name=name,
                value=value,
                step=step,
                epoch=epoch,
                include_context=include_context,
            )

    def __internal_api__log_metric__(
        self,
        name: Any,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
        framework: Optional[str] = None,
    ) -> None:
        self._log_metric(
            name=name,
            value=value,
            step=step,
            epoch=epoch,
            include_context=include_context,
            framework=framework,
        )

    def _log_metric(
        self,
        name: Any,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
        framework: Optional[str] = None,
    ) -> None:
        result = convert_to_string_key(
            name,
            max_length=MAXIMAL_KEY_LENGTH,
            raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
        )
        if result.truncated:
            LOGGER.warning(
                EXPERIMENT_LOG_METRIC_NAME_TRUNCATED, MAXIMAL_KEY_LENGTH, name
            )

        name = result.value
        # Internal logging handler with option to ignore auto-logged names
        if framework:
            if ("%s:%s" % (framework, name)) in self.autolog_metrics_ignore:
                # Use % in this message to cache specific string:
                self._log_once_at_level(
                    logging.INFO, LOG_METRIC_AUTOMATIC_IGNORED, name, framework, name
                )
                return

        LOGGER.debug("Log metric: %s %r %r", name, value, step)

        self._set_step(step)
        self._set_epoch(epoch)

        if self.alive:
            if value is None:
                debug_helpers.log_warning_or_raise(
                    METRIC_NONE_WARNING,
                    name,
                    logger=LOGGER,
                )
                return

            self._track_framework_usage(framework)

            if include_context is True:
                message = MetricMessage(context=self.context)
            else:
                message = MetricMessage()

            value = convert_user_input_to_metric_value(
                value,
                max_length=MAXIMAL_VALUE_LENGTH,
                raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
            )

            message.set_metric(
                name, value=value, step=self.curr_step, epoch=self.curr_epoch
            )
            self._enqueue_message(message)
            self._summary.set(
                "metrics",
                self._fullname(name, include_context=include_context),
                value,
                framework=framework,
            )

        # save state.
        self.metrics[name] = value

    def _fullname(self, name: str, include_context: bool = True) -> str:
        """
        If in a context manager, add the context name.
        """
        if self.context is not None and include_context is True:
            return "%s_%s" % (self.context, name)
        else:
            return name

    def log_parameter(self, name: Any, value: Any, step: Optional[int] = None) -> None:
        """Logs a parameter.

        Args:
            name: Name of your parameter.
            value: The value of your parameter.
            step: Used as the x-axis when plotting on Comet.

        Note:
            It's encouraged to use [log_other][comet_ml.Experiment.log_other] for additional
                values that are not hyper parameters.

            If the same key is reported multiple times only the last reported value
                will be saved.

            If this method is called inside a context, like [test]
            [comet_ml.Experiment.test],[train][comet_ml.Experiment.train],
            [validate][comet_ml.Experiment.validate] or [context_manager]
            [comet_ml.Experiment.context_manager], the parameter will be
            stored with the current context name as a prefix.

            See also [log_parameters][comet_ml.Experiment.log_parameters].

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize an experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Log a parameter
            exp.log_parameter("batch_size", 64)

            # End the experiment
            exp.end()
            ```
        """
        with debug_helpers.debug_error_handler(
            message=EXPERIMENT_LOG_PARAMETER_EXCEPTION_ERROR,
            exc_info=True,
            logger=LOGGER,
        ) as handle:
            handle(
                self._log_parameter,
                name=name,
                value=value,
                step=step,
                source=ParameterMessage.source_manual,
            )

    def _log_parameter(
        self,
        name: Any,
        value: Any,
        step: Optional[int] = None,
        framework: Optional[str] = None,
        source: Optional[str] = None,
        include_context: bool = True,
    ) -> None:
        if name is None or name == "":
            debug_helpers.log_warning_or_raise(
                EXPERIMENT_LOG_PARAMETER_EMPTY_NAME_ERROR, logger=LOGGER
            )
            return None

        # show warning only if truncated and not in debug mode,
        # otherwise raise an ValueTruncatedException if truncated
        result = convert_to_string_key(
            name,
            max_length=MAXIMAL_KEY_LENGTH,
            raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
        )
        if result.truncated:
            LOGGER.warning(
                EXPERIMENT_LOG_PARAMETER_NAME_TRUNCATED, MAXIMAL_KEY_LENGTH, name
            )

        name = result.value

        if framework:
            if ("%s:%s" % (framework, name)) in self.autolog_parameters_ignore:
                # Use % in this message to cache specific string:
                self._log_once_at_level(
                    logging.INFO,
                    EXPERIMENT_LOG_PARAMETER_IGNORE_AUTOMATIC_INFO
                    % (name, framework, name),
                )
                return None
            else:
                self._track_framework_usage(framework)

        self._set_step(step)

        # de-duplicate using parameter's full name, i.e., including the context name
        param_full_name = self._fullname(name, include_context=include_context)
        if param_full_name in self.params:
            saved_value = self.params[param_full_name]
            seen_before = generalized_equality.equal(value, saved_value)
            if seen_before:
                return None

        if self.alive:
            # create message
            if include_context:
                message = ParameterMessage(context=self.context)
            else:
                message = ParameterMessage(context=None)
            # todo check "value" conversion, it will be converted only "if self.alive",
            #  but then will be saved to self.params in any case
            value = convert_to_scalar(value)

            if source is None:
                source = ParameterMessage.source_autologger

            # Check if we have a list-like, dict, number, bool, None, or a string
            if isinstance(value, Mapping):
                value = convert_to_string_value(
                    value,
                    max_length=MAXIMAL_VALUE_LENGTH,
                    raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
                )
                message.set_param(name, value, step=self.curr_step, source=source)
            elif is_list_like(value):
                message.set_params(name, value, step=self.curr_step, source=source)
            elif (
                isinstance(value, numbers.Number) or value is None
            ):  # booleans are Numbers
                message.set_param(name, value, step=self.curr_step, source=source)
            else:
                value = convert_to_string_value(
                    value,
                    max_length=MAXIMAL_VALUE_LENGTH,
                    raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
                )
                message.set_param(name, value, step=self.curr_step, source=source)

            self._enqueue_message(message)
            self._summary.set("parameters", param_full_name, value, framework=framework)

        self.params[param_full_name] = value
        # save with the short name to allow check for deprecated usage of the get_parameter()
        self.deprecated_params[name] = value

    def log_figure(
        self,
        figure_name: Optional[str] = None,
        figure: Any = None,
        overwrite: bool = False,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "auto",
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Log a visual representation of the provided figure to Comet in SVG format.

        Args:
            figure_name: A descriptive name for the figure.
            figure: The figure to be logged. Accepts Plotly Figures, Matplotlib Figures
                or Seaborn simple plots. If not provided, the function will log the current global
                Matplotlib Pyplot figure.
            overwrite: Determines whether to overwrite an existing figure with the same name.
            step: Associates the figure asset with a specific step in the Comet experiment.
            metadata: Additional metadata.
            format: The format of the figure to be logged. Accepts "auto" (default), "png" or "svg".
                If "auto" is selected, the format will be inferred from the resulting figure size.

        Note:
            Kaleido is required to be installed to log plotly figures.
        """
        return self._log_figure(
            figure_name=figure_name,
            figure=figure,
            overwrite=overwrite,
            step=step,
            metadata=metadata,
            format=format,
        )

    def _log_figure(
        self,
        figure_name: Optional[str] = None,
        figure: Any = None,
        overwrite: bool = False,
        step: Optional[int] = None,
        figure_type: Optional[str] = None,
        framework: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "svg",
    ) -> Optional[Dict[str, Optional[str]]]:
        try:
            if hasattr(figure, "to_plotly_json"):
                import kaleido
        except ImportError:
            LOGGER.error(LOG_FIGURE_KALEIDO_NOT_INSTALLED)
            return None

        if not self.alive:
            return None

        self._set_step(step)

        # Pass additional url params
        figure_number = self.figure_counter
        figure_id = generate_guid()
        url_params = {
            "step": self._get_asset_upload_step(),
            "figCounter": figure_number,
            "context": self.context,
            "runId": self.run_id,
            "overwrite": overwrite,
            "imageId": figure_id,
        }

        if figure_name is not None:
            url_params["figName"] = figure_name

        processor = FigureUploadProcessor(
            figure,
            self.upload_limit,
            url_params,
            metadata=metadata,
            copy_to_tmp=False,
            error_message_identifier=figure_number,
            tmp_dir=self.tmpdir,
            upload_type=figure_type,
            format=format,
            critical=False,
            svg_size_limit=self.config.get_int(
                None, "comet.internal.max_svg_figure_size"
            ),
        )
        upload_message = processor.process()

        if upload_message is None:
            self._report_experiment_error(UNEXPECTED_ERROR_WHEN_LOGGING_FIGURE)
            return None

        self._enqueue_message(upload_message)

        self._summary.increment_section("uploads", "figures")
        self.figure_counter += 1

        return self._get_uploaded_figure_url(figure_id)

    def log_tensorboard_folder(
        self, folder: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Logs a TensorBoard folder for monitoring and visualization.

        This method is used to log a folder containing TensorBoard data. If metadata
        is provided, it is associated with the logged folder. The data in the folder
        is assumed to be prepared for TensorFlow visualization.

        Args:
            folder: The path to the folder containing TensorBoard data files.
            metadata: Additional metadata to associate with the logged folder.

        Note:
            Use [comet_ml.APIExperiment.download_tensorflow_folder][] to get the contents
            There is currently no separate download method for TensorBoard data. Use
            [comet_ml.APIExperiment.download_tensorflow_folder][] to get the contents
            of a previously logged TensorBoard folder.

        Example:
            ```python linenums="1"
            import comet_ml
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.profiler import profile, record_function, ProfilerActivity
            from torch.utils.tensorboard import SummaryWriter

            # Initialize Comet.ml
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Simple model
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(784, 256)
                    self.fc2 = nn.Linear(256, 10)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    return self.fc2(x)

            # Dummy dataset
            data = torch.randn(64, 784)
            target = torch.randint(0, 10, (64, ))

            # Model, loss, optimizer
            model = SimpleNet()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # TensorBoard writer
            tb_log_dir="runs/profiler_example"
            writer = SummaryWriter(log_dir=tb_log_dir)

            # Profiler setup
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for step in range(8):
                    # Forward + backward + optimize
                    optimizer.zero_grad()
                    with record_function("forward_pass"):
                        outputs = model(data)
                        loss = criterion(outputs, target)

                    with record_function("backward_pass"):
                        loss.backward()

                    optimizer.step()

                    # Step profiler
                    prof.step()

            # Log TensorBoard logs folder to Comet
            exp.log_tensorboard_folder(tb_log_dir)

            # End the experiment
            exp.end()
            ```
        """
        self.log_tensorflow_folder(folder, metadata=metadata)

    def log_tensorflow_folder(
        self, folder: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Logs all the tensorflow log files located in the given folder as assets.

        Args:
            folder (str): The path to the folder you want to log.
            metadata (dict): Some additional data to attach to the folder's assets.

        Note:
            Use [comet_ml.APIExperiment.download_tensorflow_folder][] to get the contents
            of a previously logged folder.

        Example:
            ```python linenums="1"
            import comet_ml
            import tensorflow as tf
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.models import Sequential
            import json

            # Initialize Comet.ml
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Setup TensorFlow model
            model = Sequential([
                Dense(64, activation='relu', input_shape=(10,)),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Create a directory and save the TensorFlow model there
            model_dir = "saved_model"
            model.save(model_dir)

            # Save hyperparameters
            params = {
                'batch_size': 32,
                'epochs': 10,
                'layers': [64, 1],
                'activation': ['relu', 'linear'],
                'optimizer': 'adam',
                'loss': 'mean_squared_error'
            }
            with open(f"{model_dir}/params.json", 'w') as f:
                json.dump(params, f)

            # Log TensorFlow model folder to Comet.ml
            exp.log_tensorflow_folder(model_dir)

            # End the experiment
            exp.end()
            ```
        """
        if not self.alive:
            return

        if folder.endswith(os.sep):
            folder = folder[:-1]

        self._log_asset_folder(
            folder,
            log_file_name=True,
            recursive=True,
            asset_type=ASSET_TYPE_TENSORFLOW_FILE,
            metadata=metadata,
        )

    def log_text(
        self,
        text: str,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Logs the text. These strings appear on the Text Tab in the Comet UI.

        Args:
            text: String of text to be stored.
            step: Used to associate the asset to a specific step.
            metadata: Some additional data to attach to the text. Must be a
                JSON-encodable dict.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize an experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Example of logging text at different steps of an experiment
            exp.log_text("Starting an experiment", step=0)

            # End the experiment
            exp.end()
            ```
        """
        # Send fake file_name, which is replaced on the backend:
        return self._log_asset_data(
            text,
            file_name="auto-generated-in-the-backend",
            asset_type=ASSET_TYPE_TEXT_SAMPLE,
            step=step,
            metadata=metadata,
        )

    def _deprecate_copy_to_tmp(self, copy_to_tmp: bool, function_name: str) -> bool:
        if not copy_to_tmp:
            debug_helpers.log_warning_or_raise(
                DEPRECATED_COPY_TO_TEMP_ARGUMENT, function_name, logger=LOGGER
            )
            return True

        return copy_to_tmp

    def log_model(
        self,
        name: str,
        file_or_folder: Union[str, IO],
        file_name: Optional[str] = None,  # does not apply to folders
        overwrite: bool = False,  # does not apply to folders
        metadata: Optional[Dict[str, Any]] = None,
        copy_to_tmp: bool = True,  # if data is a file pointer
        prepend_folder_name: bool = True,
    ) -> Optional[Dict[str, str]]:
        """
        Logs the model data under the name. Data can be a file path, a folder
        path or a file-like object.

        Args:
            name (str): The name of the model
            file_or_folder (str|IO): The model data; can be a file path, a
                folder path or a file-like object.
            file_name (str): The name of the model data. Used with file-like
                objects or files only.
            overwrite (bool): If True, then overwrite previous versions
                Does not apply to folders.
            metadata (dict): Some additional data to attach to the the data.
                Must be a JSON-encodable dict.
            copy_to_tmp (bool): Deprecated. For file name or file-like; if True copy to
                temporary location before uploading; if False, then
                upload from current location
            prepend_folder_name (bool): If True and logging a folder, prepend file
                path by the folder name.

        Returns:
            dict: Dictionary of model URLs
        """
        copy_to_tmp = self._deprecate_copy_to_tmp(copy_to_tmp, "log_model")
        event_tracker.register("experiment.log_model-called", self.id)
        return self._log_model(
            model_name=name,
            file_or_folder=file_or_folder,
            file_name=file_name,
            overwrite=overwrite,
            metadata=metadata,
            copy_to_tmp=copy_to_tmp,
            prepend_folder_name=prepend_folder_name,
        )

    def _log_model(
        self,
        model_name: str,
        file_or_folder: Union[str, IO],
        file_name: Optional[str] = None,  # does not apply to folders
        overwrite: bool = False,  # does not apply to folders
        metadata: Optional[Dict[str, Any]] = None,
        copy_to_tmp: bool = True,  # if data is a file pointer
        folder_name: Optional[str] = None,  # if data is a folder
        prepend_folder_name: bool = True,  # if data is a folder
        critical: bool = False,
        on_model_upload: Optional[Callable] = None,
        on_failed_model_upload: Optional[Callable] = None,
    ):
        if isinstance(file_or_folder, str):
            file_or_folder = expand_user_home_path(file_or_folder)
            if os.path.isfile(file_or_folder):
                return self._log_asset(
                    file_or_folder,  # file name
                    file_name=file_name,
                    overwrite=overwrite,
                    copy_to_tmp=copy_to_tmp,
                    asset_type=ASSET_TYPE_MODEL_ELEMENT,
                    metadata=metadata,
                    grouping_name=model_name,  # model name
                    critical=critical,
                    on_asset_upload=on_model_upload,
                    on_failed_asset_upload=on_failed_model_upload,
                )
            elif os.path.isdir(file_or_folder):
                return self._log_asset_folder(
                    file_or_folder,  # folder name
                    recursive=True,
                    log_file_name=True,
                    asset_type=ASSET_TYPE_MODEL_ELEMENT,
                    metadata=metadata,
                    grouping_name=model_name,  # model name
                    folder_name=folder_name,
                    prepend_folder_name=prepend_folder_name,
                )
        elif hasattr(file_or_folder, "read"):  # file-like object
            return self._log_asset(
                file_or_folder,  # file-like object
                file_name=file_name,  # filename
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
                asset_type=ASSET_TYPE_MODEL_ELEMENT,
                metadata=metadata,
                grouping_name=model_name,  # model name
                critical=critical,
                on_asset_upload=on_model_upload,
                on_failed_asset_upload=on_failed_model_upload,
            )

        debug_helpers.log_error_or_raise(
            EXPERIMENT_LOG_MODEL_NO_SUCH_FILE_OR_DIR_ERROR,
            model_name,
            file_or_folder,
            logger=LOGGER,
        )
        return None

    def _log_model_synchronization_callbacks(
        self,
        model_name: str,
        on_upload_original: Optional[Callable],
        on_failed_original: Optional[Callable],
    ) -> Tuple[Callable, Callable]:
        (
            on_completed_sync,
            on_failed_sync,
        ) = self.model_upload_synchronizer.start_processing(model_name)

        if on_upload_original is None:
            on_upload = lambda response: on_completed_sync()  # noqa: E731
        else:
            on_upload = lambda response: (  # noqa: E731
                on_upload_original(response),
                on_completed_sync(),
            )

        if on_failed_original is None:
            on_failed_upload = lambda response: on_failed_sync()  # noqa: E731
        else:
            on_failed_upload = lambda response: (  # noqa: E731
                on_failed_original(response),
                on_failed_sync(),
            )

        return on_upload, on_failed_upload

    def log_remote_model(
        self,
        model_name: str,
        uri: str,
        metadata: Optional[Dict[str, Any]] = None,
        sync_mode: bool = True,
        max_synced_objects: int = 10000,
    ) -> None:
        """Logs metadata about a model that is stored elsewhere, such as
        remote storage solutions like AWS S3, Google Cloud Storage, Azure Blob
        Storage, etc. It allows you to keep track of model metadata without
        moving the actual model files into Comet.

        Args:
            model_name (str): The name of the model.
            uri (str): The remote model location, there is no imposed format and it could be a
                private link. Can log a single file and a folder.
            metadata (dict): some additional data to attach to the data.
                Must be a JSON-encodable dict.
            sync_mode (bool): If True and the URI begins with s3:// or gs://, Comet attempts to list all
                objects in the given bucket and path. All the objects will be logged under the given model name.
                If False, Comet just logs a single remote model with the provided URI as the remote URI.
            max_synced_objects (int): When sync_mode is True and the URI begins with s3:// or gs://, set the
                maximum number of S3/GCP objects to log. If there are more matching S3/GCP objects than
                max_synced_objects, a warning will be displayed and the provided URI will be logged
                as a single remote model.

        Note:
            If the URI begins with s3:// or gs://, Comet attempts to list all objects in the
            given bucket and path and logs them individually.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            model_name = 'remote_model'
            model_uri = 'gs://demo-remote-model/'

            exp.log_remote_model(model_name=model_name, uri=model_uri, sync_mode=True)

            exp.end()
            ```
        """

        if not self.alive:
            return

        self._log_remote_model(model_name, uri, metadata, sync_mode, max_synced_objects)

    def _log_remote_model(
        self,
        model_name: str,
        uri: str,
        metadata: Optional[Dict[str, Any]],
        sync_mode: bool,
        max_synced_objects: int,
        on_model_upload: Optional[Callable] = None,
        on_failed_model_upload: Optional[Callable] = None,
    ) -> None:
        remote_model = RemoteModel(
            workspace=self.workspace,
            model_name=model_name,
        )

        (
            on_model_upload,
            on_failed_model_upload,
        ) = self._log_model_synchronization_callbacks(
            model_name, on_model_upload, on_failed_model_upload
        )

        if sync_mode is True:
            (
                success,
                error_message,
                processed_assets,
            ) = assets_preprocess.remote_model(
                uri=uri,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
                raise_on_error=debug_helpers.has_enabled_debug_exception_raising(),
            )

            if success:
                for asset in processed_assets:
                    remote_model.append_asset(asset)

                self._enqueue_message(
                    remote_model.to_message(on_model_upload, on_failed_model_upload)
                )
                return

            if error_message is not None:
                if metadata is None:
                    metadata = dict()
                metadata[META_ERROR_MESSAGE] = error_message
                metadata[META_SYNCED] = False

        UNUSED = None
        ALWAYS_OVERWRITE = True

        remote_model_asset = preprocess_remote_asset(
            remote_uri=uri,
            logical_path=UNUSED,
            overwrite=ALWAYS_OVERWRITE,
            upload_type=UNUSED,
            metadata=metadata,
        )

        remote_model.append_asset(remote_model_asset)

        self._enqueue_message(
            remote_model.to_message(on_model_upload, on_failed_model_upload)
        )

    def register_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        workspace: Optional[str] = None,
        registry_name: Optional[str] = None,
        public: Optional[bool] = None,
        description: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[list] = None,
        stages: Optional[list] = None,
        status: Optional[str] = None,
        sync: Optional[bool] = False,
        timeout: float = 10.0,
    ) -> None:
        """Register an experiment's model to the registry.

        Args:
            model_name (str): The name of the experiment model.
            workspace (str): This argument is deprecated and ignored. Models are registered to the workspace the experiment belongs to.
            version (str): A proper semantic version string; defaults to "1.0.0".
            registry_name (str): The name of the registered workspace model, if not provided the
                model_name will be used instead.
            public (bool): If True, then the model will be publicly viewable.
            description (str): A textual description of the model.
            comment (str): A textual comment about the model
            tags (list): A list of textual tags such as ["tag1", "tag2"], etc.
            stages (list): This argument is deprecated and will be ignored. Please use `tags` instead.
            status (str): A string describing the status of this model version.
            sync (bool): Whether this function is synchronous and will be finished only once the model was registered.
            timeout (float): Maximum time (In seconds) before the function would end if called with sync = True.

        Example:
            ```python linenums="1"
            import comet_ml
            import pandas as pd
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split

            comet_ml.login()
            exp = comet_ml.start()

            data = pd.DataFrame({
                'Feature1': np.random.rand(100),
                'Feature2': np.random.rand(100),
                'Target': np.random.rand(100)
            })
            X = data[['Feature1', 'Feature2']]
            y = data['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            from joblib import dump
            dump(model, "ols_model.pickle")
            exp.log_model("ols_model", "ols_model.pickle")

            exp.register_model(model_name="ols_model", version="1.0.1")

            exp.end()
            ```
        """
        if not self.alive:
            return

        validator = MethodParametersTypeValidator(
            method_name=self.register_model.__name__, class_name=self.__class__.__name__
        )
        validator.add_str_parameter(model_name, name="model_name", allow_empty=False)
        validator.add_str_parameter(version, name="version")
        validator.add_str_parameter(workspace, name="workspace")
        validator.add_str_parameter(registry_name, name="registry_name")
        validator.add_bool_parameter(public, name="public")
        validator.add_str_parameter(description, name="description")
        validator.add_str_parameter(comment, name="comment")
        validator.add_list_parameter(tags, name="tags")
        validator.add_list_parameter(stages, name="stages")
        validator.add_str_parameter(status, name="status")
        validator.add_bool_parameter(sync, name="sync")
        validator.add_numeric_parameter(timeout, name="timeout")

        if not validator.validate():
            validator.print_result(LOGGER)
            return

        if workspace:
            LOGGER.warning(DEPRECATED_WORKSPACE_MODEL_REGISTRY_ARGUMENT)

        if registry_name is None:
            registry_name = model_name

        try:
            upload_status_observer_callback = self.model_upload_synchronizer.observer(
                model_name
            )
        except KeyError as exception:
            raise ValueError(
                EXPERIMENT_REGISTER_MODEL_NO_LOG_MODEL_CALL_EXCEPTION.format(model_name)
            ) from exception

        (
            on_model_register,
            on_failed_model_register,
        ) = self.model_register_synchronizer.start_processing(model_name)

        message = RegisterModelMessage(
            experiment_id=self.id,
            model_name=model_name,
            version=version,
            workspace=self.workspace,
            registry_name=registry_name,
            public=public,
            description=description,
            comment=comment,
            tags=tags,
            status=status,
            stages=stages,
            upload_status_observer_callback=upload_status_observer_callback,
            on_model_register=on_model_register,
            on_failed_model_register=on_failed_model_register,
        )

        self._enqueue_message(message)

        if not sync:
            return

        start_time = time.time()
        model_register_observer = self.model_register_synchronizer.observer(model_name)
        while model_register_observer() == "IN_PROGRESS":
            time.sleep(0.5)
            if time.time() - start_time >= timeout:
                LOGGER.warning(EXPERIMENT_REGISTER_MODEL_TIMEOUT_WARNING % timeout)
                return

    def _log_ipython_notebook_code(self) -> Optional[Dict[str, str]]:
        """Invoked to log the IPython notebook code if appropriate"""
        try:
            if not (_in_ipython_environment() and self._log_code):
                # safeguard check
                return None

            source_code = get_ipython_source_code()
            if source_code == "":
                # We might be running script directly:
                caller = get_caller_file_path()
                if caller is not None:
                    self._log_code_asset("experiment_creation", file_name=caller[1])
            else:
                self._log_code_asset(
                    "experiment_creation",
                    code=source_code,
                    code_name=DEFAULT_JUPYTER_INTERACTIVE_FILE_NAME,
                )

            # Now attempt to log the code as a notebook:
            if self.alive:
                notebook_json = get_ipython_notebook()
                name = DEFAULT_JUPYTER_CODE_ASSET_NAME
                return self._log_asset_data(
                    notebook_json,
                    file_name=name,
                    overwrite=True,
                    asset_type=ASSET_TYPE_NOTEBOOK,
                )
        except Exception:
            LOGGER.error(FAILED_LOG_IPYTHON_NOTEBOOK, exc_info=True)

    def _log_colab_notebook_code(self) -> Optional[Dict[str, str]]:
        """Invoked to log the Google Colab notebook code if appropriate"""
        try:
            if self.alive:
                notebook_json = _get_colab_notebook_json()

                name = DEFAULT_COLAB_NOTEBOOK_ASSET_NAME
                return self._log_asset_data(
                    notebook_json,
                    file_name=name,
                    overwrite=True,
                    asset_type=ASSET_TYPE_NOTEBOOK,
                )
        except Exception as ex:
            LOGGER.warning(LOG_COLAB_NOTEBOOK_ERROR, ex, exc_info=True)
            return None

    def log_notebook(
        self, filename: str, overwrite: bool = False
    ) -> Optional[Dict[str, str]]:
        """Log a Jupyter Notebook file as an asset.

        Args:
            filename (str): The path and name of the notebook to be logged.
            overwrite (bool): If True, overwrites
                the previously logged notebook asset.

        Example:
            ```python linenums="1"
            import comet_ml
            import nbformat
            from nbformat.v4 import new_notebook

            # Create a dummy notebook to log
            nb = new_notebook()

            with open('example_notebook.ipynb', 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            # Log the new notebook to Comet
            comet_ml.login()
            exp = comet_ml.start()

            exp.log_notebook("./example_notebook.ipynb")

            exp.end()
            ```
        """
        if not isinstance(filename, str):
            raise ValueError(EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_STRING_EXCEPTION)

        if not filename.endswith(".ipynb"):
            raise ValueError(EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_IPYNB_EXCEPTION)

        filename = expand_user_home_path(filename)
        self._notebook_logged = True
        return self._log_asset(
            filename, asset_type=ASSET_TYPE_NOTEBOOK, overwrite=overwrite
        )

    def log_curve(
        self,
        name: str,
        x: Any,
        y: Any,
        overwrite: bool = False,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Log timeseries data.

        Args:
            name: Name of data.
            x: list of x-axis values.
            y: list of y-axis values.
            overwrite: If True, overwrite previous log.
            step: The step value.
            metadata: The additional metadata to associate with logged curves.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_curve("my curve", x=[1, 2, 3, 4, 5],
                                      y=[10, 20, 30, 40, 50])
            exp.end()
            ```
        """
        if self.alive:
            validator = CurveDataValidator(name=name, x=x, y=y)
            result = validator.validate()
            if result.failed():
                LOGGER.error(
                    EXPERIMENT_LOG_CURVE_VALIDATION_ERROR % result.failure_reasons
                )
                return None

            data = {"x": list(x), "y": list(y), "name": name}
            return self._log_asset_data(
                data,
                file_name=name,
                overwrite=overwrite,
                asset_type=ASSET_TYPE_CURVE,
                step=step,
                metadata=metadata,
            )

    def log_asset_data(
        self,
        data: Any,
        name: Optional[str] = None,
        overwrite: bool = False,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_name: Optional[str] = None,
        epoch: Optional[int] = None,
    ):
        """Logs the data given (str, binary, or JSON).

        Args:
            data: data to be saved as asset.
            name:  A custom file name to be displayed.
                If not provided the filename from the temporary saved file will be used.
            overwrite: If True will overwrite all existing
                assets with the same name.
            step: Used to associate the asset to a specific step.
            epoch: Used to associate the asset to a specific epoch.
            metadata: Some additional data to attach to the asset data.
                Must be a JSON-encodable dict.
            file_name: A custom file name to be displayed [Deprecated: use ``name`` instead].

        Note:
            See also [comet_ml.APIExperiment.get_asset_list][], [comet_ml.APIExperiment.get_asset][], and
            [comet_ml.APIExperiment.get_asset_by_name][].

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            data = {"key": "value", "number": 123}

            exp.log_asset_data(data, file_name = "sample_data.json")

            exp.end()
            ```
        """
        if file_name is not None:
            LOGGER.warning(EXPERIMENT_LOG_ASSET_DATA_DEPRECATED_FILE_NAME_WARNING)
            name = file_name

        if name is None:
            name = "data"

        return self._log_asset_data(
            data,
            file_name=name,
            overwrite=overwrite,
            asset_type=ASSET_TYPE_DEFAULT,
            step=step,
            epoch=epoch,
            metadata=metadata,
        )

    def _log_asset_data(
        self,
        data: Any,
        file_name: Optional[str] = None,
        overwrite: bool = False,
        asset_type: str = ASSET_TYPE_DEFAULT,
        step: Optional[int] = None,
        require_step: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        grouping_name: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> Optional[Dict[str, str]]:

        if not self.alive:
            return None

        self._set_step(step)
        self._set_epoch(epoch)

        if require_step:
            if self.curr_step is None:
                raise TypeError(EXPERIMENT_LOG_ASSET_DATA_STEP_MANDATORY_EXCEPTION)

        asset_id = generate_guid()
        url_params = {
            "assetId": asset_id,
            "context": self.context,
            "fileName": file_name,
            "overwrite": overwrite,
            "runId": self.run_id,
            "step": self._get_asset_upload_step(),
            "epoch": self._get_asset_upload_epoch(),
        }

        # If the asset type is more specific, include the
        # asset type as "type" in query parameters:
        if asset_type != ASSET_TYPE_DEFAULT:
            url_params["type"] = asset_type

        processor = AssetDataUploadProcessor(
            user_input=data,
            upload_type=asset_type,
            url_params=url_params,
            metadata=metadata,
            upload_limit=self.asset_upload_limit,
            copy_to_tmp=False,
            error_message_identifier=None,
            tmp_dir=self.tmpdir,
            critical=False,
        )
        upload_message = processor.process()

        if not upload_message:
            return None

        asset_name, truncated = truncate_to_length(
            upload_message.additional_params.get("fileName", None),
            self.asset_name_max_length,
        )
        if truncated:
            upload_message.additional_params["fileName"] = asset_name
            LOGGER.warning(
                EXPERIMENT_LOG_PREPROCESSED_ASSET_NAME_TRUNCATED_WARNING,
                self.asset_name_max_length,
            )

        self._enqueue_message(upload_message)

        self._summary.increment_section("uploads", asset_type)
        return self._get_uploaded_asset_url(asset_id)

    def log_asset_folder(
        self,
        folder: str,
        step: Optional[int] = None,
        log_file_name: Optional[bool] = None,
        recursive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[None, List[Tuple[str, Dict[str, str]]]]:
        """Logs all the files located in the given folder as assets.

        Args:
            folder (str): the path to the folder you want to log.
            step (int): used to associate the asset to a specific step.
            log_file_name (bool): if True, log the file path with each file.
            recursive (bool): if True, recurse folder and save file names.
            metadata: Some additional data to attach to the asset data.

        Note:
            If log_file_name is set to True, each file in the given folder will be
            logged with the following name schema:
            `FOLDER_NAME/RELPATH_INSIDE_FOLDER`. Where `FOLDER_NAME` is the basename
            of the given folder and `RELPATH_INSIDE_FOLDER` is the file path
            relative to the folder itself.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize Comet ML Experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Path to the folder containing assets you want to log
            folder_path = "./"

            # Log the entire folder as an asset
            exp.log_asset_folder(folder_path)

            # End the experiment
            exp.end()
            ```
        """

        # Current default is False, we want to move it to True in a future release
        if log_file_name is None:
            debug_helpers.log_warning_or_raise(
                EXPERIMENT_LOG_ASSET_FOLDER_LOG_FILE_NAME_WARNING, logger=LOGGER
            )
            log_file_name = False

        return self._log_asset_folder(
            folder,
            step=step,
            log_file_name=log_file_name,
            recursive=recursive,
            metadata=metadata,
        )

    def _log_asset_folder(
        self,
        folder: str,
        step: Optional[int] = None,
        log_file_name: bool = False,
        recursive: bool = False,
        asset_type: str = ASSET_TYPE_DEFAULT,
        metadata: Optional[Dict[str, Any]] = None,
        grouping_name: Optional[str] = None,
        folder_name: Optional[str] = None,
        extension_filter: Optional[List[str]] = None,
        prepend_folder_name: bool = True,
        overwrite: bool = False,
    ) -> Optional[List[Tuple[str, Dict[str, str]]]]:
        self._set_step(step)

        urls = []

        folder = expand_user_home_path(folder)
        if not os.path.isdir(folder):
            debug_helpers.log_error_or_raise(
                LOG_ASSET_FOLDER_ERROR, folder, logger=LOGGER
            )
            return None

        folder_abs_path = os.path.abspath(folder)
        if folder_name is None:
            folder_name = os.path.basename(folder)

        try:
            for file_name, file_path in list_assets_folder(
                folder_abs_path, recursive, extension_filter
            ):
                # The file path should be absolute as we are passing the folder
                # path as an absolute path
                if log_file_name:
                    if prepend_folder_name is True:
                        asset_file_name = os.path.join(
                            folder_name, os.path.relpath(file_path, folder_abs_path)
                        )
                    else:
                        asset_file_name = os.path.relpath(file_path, folder_abs_path)

                    asset_url = self._log_asset(
                        file_data=file_path,
                        file_name=asset_file_name,
                        asset_type=asset_type,
                        metadata=metadata,
                        grouping_name=grouping_name,
                        overwrite=overwrite,
                    )
                else:
                    asset_url = self._log_asset(
                        file_data=file_path,
                        asset_type=asset_type,
                        metadata=metadata,
                        grouping_name=grouping_name,
                        overwrite=overwrite,
                    )

                # Ignore files that has failed to be logged
                if asset_url:
                    urls.append((file_name, asset_url))
        except Exception as e:
            debug_helpers.log_error_or_raise(
                LOG_ASSET_FOLDER_ERROR, folder, original_exception=e
            )
            return None

        if not urls:
            debug_helpers.log_warning_or_raise(
                LOG_ASSET_FOLDER_EMPTY, folder, logger=LOGGER
            )
            return None

        return urls

    def log_asset(
        self,
        file_data: Any,
        file_name: Optional[str] = None,
        overwrite: bool = False,
        copy_to_tmp: bool = True,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Logs the Asset determined by file_data.

        Args:
            file_data (Any): Either the file path of the file you want
                to log, or a file-like asset.
            file_name (str): A custom file name to be displayed. If not
                provided the filename from the `file_data` argument will be used.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            copy_to_tmp (bool): (Deprecated) If `file_data` is a file-like object, then this flag determines
                if the file is first copied to a temporary file before upload. If
                `copy_to_tmp` is False, then it is sent directly to the cloud.
            step (int): Used to associate the asset to a specific step.
            metadata (Any): Some additional data to attach to the asset. Must be a
                JSON-encodable dict.

        Example:
            ```python linenums="1"
            import comet_ml
            import pandas as pd

            data = {
                'Name': ['Alice', 'Bob', 'Charlie', 'David'],
                'Age': [25, 30, 35, 40],
                'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
            }
            df = pd.DataFrame(data)
            df.to_csv('sample_data.csv', index=False)

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_asset(file_data="sample_data.csv", file_name="sample_data.csv")

            exp.end()
            ```
        """
        copy_to_tmp = self._deprecate_copy_to_tmp(copy_to_tmp, "log_asset")

        return self._log_asset(
            file_data,
            file_name=file_name,
            overwrite=overwrite,
            copy_to_tmp=copy_to_tmp,
            asset_type=ASSET_TYPE_DEFAULT,
            step=step,
            metadata=metadata,
        )

    def _log_asset(
        self,
        file_data: Any,
        file_name: Optional[str] = None,
        overwrite: bool = False,
        copy_to_tmp: bool = True,
        asset_type: str = ASSET_TYPE_DEFAULT,
        step: Optional[int] = None,
        require_step: bool = False,
        grouping_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_version_id: Optional[str] = None,
        critical: bool = False,
        asset_id: Optional[str] = None,
        return_url: bool = True,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
        framework: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:

        # Keep to speed up the case where Experiment is not alive
        if not self.alive:
            return None

        if file_data is None:
            raise TypeError(EXPERIMENT_LOG_ASSET_FILE_DATA_NONE_EXCEPTION)

        if file_name is not None:
            file_name = str(file_name)

        self._set_step(step)

        self._track_framework_usage(framework)

        if asset_type == ASSET_TYPE_MODEL_ELEMENT:
            (
                on_asset_upload,
                on_failed_asset_upload,
            ) = self._log_model_synchronization_callbacks(
                grouping_name, on_asset_upload, on_failed_asset_upload
            )

        dispatched = dispatch_user_file_upload(file_data)

        if isinstance(dispatched, UserTextFileUpload):
            debug_helpers.log_error_or_raise(
                UPLOAD_FILE_OS_ERROR, dispatched.user_input, logger=LOGGER
            )
            return None
        elif isinstance(dispatched, ObjectToConvertFileUpload):
            raise TypeError(
                EXPERIMENT_LOG_ASSET_UNSUPPORTED_UPLOAD_TYPE_EXCEPTION
                % type(dispatched.user_input)
            )
        elif isinstance(dispatched, FileUpload):
            preprocessed = preprocess_asset_file(
                dispatched=dispatched,
                upload_type=asset_type,
                file_name=file_name,
                metadata=metadata,
                overwrite=overwrite,
                asset_id=asset_id,
                copy_to_tmp=copy_to_tmp,
                grouping_name=grouping_name,
                step=self._get_asset_upload_step(),
            )
        else:
            preprocessed = preprocess_asset_memory_file(
                dispatched=dispatched,
                upload_type=asset_type,
                file_name=file_name,
                metadata=metadata,
                overwrite=overwrite,
                asset_id=asset_id,
                copy_to_tmp=copy_to_tmp,
                grouping_name=grouping_name,
                step=self._get_asset_upload_step(),
            )

        try:
            return self._log_preprocessed_asset(
                preprocessed,
                artifact_version_id=artifact_version_id,
                critical=critical,
                return_url=return_url,
                require_step=require_step,
                on_asset_upload=on_asset_upload,
                on_failed_asset_upload=on_failed_asset_upload,
            )
        except AssetIsTooBig as e:
            debug_helpers.log_error_or_raise(
                UPLOAD_ASSET_TOO_BIG,
                e.file_path,
                e.file_size,
                e.max_size,
                logger=LOGGER,
                original_exception=e,
                exc_info=True,
            )
            return None

    def _log_preprocessed_asset(
        self,
        preprocessed_asset: Union[PreprocessedFileAsset, PreprocessedMemoryFileAsset],
        artifact_version_id: Optional[str] = None,
        critical: bool = False,
        return_url: bool = True,
        require_step: bool = False,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> Optional[Dict[str, str]]:
        if not self.alive:
            return None

        self._set_step(preprocessed_asset.step)

        if require_step:
            if self.curr_step is None:
                if on_failed_asset_upload is not None:
                    try:
                        on_failed_asset_upload((None, None, None))
                    except Exception as e:
                        debug_helpers.log_warning_or_raise(
                            EXPERIMENT_LOG_PREPROCESSED_ASSET_ON_FAILED_ASSET_UPLOAD_CALL_FAILED_WARNING,
                            logger=LOGGER,
                            original_exception=e,
                            exc_info=True,
                        )

                raise TypeError(EXPERIMENT_LOG_PREPROCESSED_ASSET_STEP_MANDATORY_ERROR)

        if preprocessed_asset.size > self.asset_upload_limit:
            error_message_identifier = preprocessed_asset.local_path_or_data

            raise AssetIsTooBig(
                error_message_identifier,
                preprocessed_asset.size,
                self.asset_upload_limit,
            )

        if preprocessed_asset.copy_to_tmp is True:
            if isinstance(preprocessed_asset, PreprocessedFileAsset):
                new_upload_filepath = handle_in_memory_file_upload(
                    self.tmpdir,
                    upload_filepath=ValidFilePath(
                        preprocessed_asset.local_path_or_data
                    ),
                    raise_on_error=debug_helpers.has_enabled_debug_exception_raising(),
                )

                # If we failed to copy the file, abort
                if not new_upload_filepath:
                    return None

                preprocessed_asset = preprocessed_asset.copy(
                    new_upload_filepath, new_copy_to_tmp=False
                )
            else:
                new_upload_tmp_file = write_file_like_to_tmp_file(
                    preprocessed_asset.local_path_or_data, self.tmpdir
                )

                preprocessed_asset = preprocessed_asset.to_preprocessed_file_asset(
                    new_upload_tmp_file, new_copy_to_tmp=False
                )

                # # TODO it would be easier to use the same field name for a file or a figure upload
                # if "fileName" in self.url_params and self.url_params["fileName"] is None:
                #     self.url_params["fileName"] = os.path.basename(file_path)

                # if "figName" in self.url_params and self.url_params["figName"] is None:
                #     self.url_params["figName"] = os.path.basename(file_path)

        # Clean only temporary files
        if isinstance(preprocessed_asset.local_path_or_data, TemporaryFilePath):
            clean = True
        else:
            clean = False

        upload_message = preprocessed_asset.to_message(
            critical,
            on_asset_upload,
            on_failed_asset_upload,
            clean=clean,
            experiment_url_params=self._get_asset_url_params(artifact_version_id),
        )

        asset_name, truncated = truncate_to_length_or_raise(
            upload_message.additional_params.get("fileName", None),
            length=self.asset_name_max_length,
            raise_on_truncate=debug_helpers.has_enabled_debug_exception_raising(),
        )
        if truncated:
            upload_message.additional_params["fileName"] = asset_name
            LOGGER.warning(
                EXPERIMENT_LOG_PREPROCESSED_ASSET_NAME_TRUNCATED_WARNING,
                self.asset_name_max_length,
            )

        self._enqueue_message(upload_message)

        if artifact_version_id is None:
            self._summary.increment_section(
                "uploads", preprocessed_asset.upload_type, size=preprocessed_asset.size
            )
        else:
            self._summary.increment_section(
                "uploads", "artifact assets", size=preprocessed_asset.size
            )

        if return_url:
            return self._get_uploaded_asset_url(preprocessed_asset.asset_id)
        else:
            return None

    def log_remote_asset(
        self,
        uri: str,
        remote_file_name: Optional[str] = None,
        overwrite: bool = False,
        asset_type: str = ASSET_TYPE_DEFAULT,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Logs a Remote Asset identified by a URI. A Remote Asset is an asset but its content is not
            uploaded and stored on Comet. Rather a link for its location is stored, so you can identify
            and distinguish between two experiment using different version of a dataset stored somewhere
            else.

        Args:
            uri (str): The remote asset location, there is no imposed format, and it could be a
                private link.
            remote_file_name (str): The "name" of the remote asset, could be a dataset
                name, a model file name.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            asset_type (str): Specifies the type of the asset being logged.
            step (int): Used to associate the asset to a specific step.
            metadata (dict): Some additional data to attach to the remote asset.
                Must be a JSON-encodable dict.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_remote_asset("s3://bucket/folder/file")
            exp.log_remote_asset("dataset:701bd06b43b7423296fb626027d02198")

            exp.end()
            ```
        """
        return self._log_remote_asset(
            uri, remote_file_name, overwrite, asset_type, step, metadata
        )

    def _log_remote_asset(
        self,
        uri: str,
        remote_file_name: Optional[str] = None,
        overwrite: bool = False,
        asset_type: str = ASSET_TYPE_DEFAULT,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_version_id: Optional[str] = None,
        critical: bool = False,
        return_url: bool = True,
        asset_id: Optional[str] = None,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> Optional[Dict[str, str]]:
        self._set_step(step)

        preprocessed = preprocess_remote_asset(
            remote_uri=uri,
            logical_path=remote_file_name,
            overwrite=overwrite,
            upload_type=asset_type,
            asset_id=asset_id,
            metadata=metadata,
            step=self._get_asset_upload_step(),
        )

        return self._log_preprocessed_remote_asset(
            preprocessed,
            artifact_version_id=artifact_version_id,
            critical=critical,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            return_url=return_url,
        )

    def _get_asset_url_params(self, artifact_version_id):
        exp_url_params = {
            "context": self.context,
            "runId": self.run_id,
            "step": self._get_asset_upload_step(),
        }

        if artifact_version_id:
            exp_url_params.update({"artifactVersionId": artifact_version_id})

        return exp_url_params

    def _log_preprocessed_remote_asset(
        self,
        preprocessed_remote_asset: PreprocessedRemoteAsset,
        artifact_version_id: Optional[str] = None,
        critical: bool = False,
        return_url: bool = True,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> Optional[Dict[str, str]]:
        if not self.alive:
            return None

        self._set_step(preprocessed_remote_asset.step)

        upload_message = preprocessed_remote_asset.to_message(
            critical,
            on_asset_upload,
            on_failed_asset_upload,
            self._get_asset_url_params(artifact_version_id),
        )

        asset_name, truncated = truncate_to_length_or_raise(
            upload_message.additional_params.get("fileName", None),
            length=self.asset_name_max_length,
            raise_on_truncate=debug_helpers.has_enabled_debug_exception_raising(),
        )
        if truncated:
            upload_message.additional_params["fileName"] = asset_name
            LOGGER.warning(
                EXPERIMENT_LOG_PREPROCESSED_REMOTE_ASSET_NAME_TRUNCATED_WARNING,
                self.asset_name_max_length,
            )

        self._enqueue_message(upload_message)

        if artifact_version_id is None:
            self._summary.increment_section(
                "uploads",
                upload_message.upload_type,
                size=preprocessed_remote_asset.size,
            )
        else:
            self._summary.increment_section(
                "uploads", "artifact assets", size=preprocessed_remote_asset.size
            )

        if return_url:
            return self._get_uploaded_asset_url(preprocessed_remote_asset.asset_id)
        else:
            return None

    def log_audio(
        self,
        audio_data: Any,
        sample_rate: Optional[int] = None,
        normalize_data: str = "PCM_16_BIT",
        file_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        copy_to_tmp: bool = True,
        step: Optional[int] = None,
    ) -> Optional[Dict[str, Optional[str]]]:
        """Logs the audio Asset determined by audio data.

        Args:
            audio_data (str | numpy.array): Either the file path of the file you want
                to log, or a numpy array given to `scipy.io.wavfile.write` for wav conversion.
                Note that scipy assumes that stereo audio is in (n_samples, n_channels) order.
            sample_rate (int): The sampling rate given to
                `scipy.io.wavfile.write` for creating the wav file.
            normalize_data (str): The parameter that determines whether `audio_data` should be
                normalized to fit within the specific numerical bounds (e.g. 16-bit PCM).
                If you set this parameter to 'OFF', it is your responsibility to ensure
                that `audio_data` is compatible with the format supported by `scipy.io.wavfile.write`.
                Supported values: 'PCM_16_BIT', 'OFF'.
            file_name (str): A custom file name to be displayed.
                If not provided, the filename from the `audio_data` argument
                will be used.
            metadata (dict): Some additional data to attach to the audio asset.
                Must be a JSON-encodable dict.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            copy_to_tmp (bool): (Deprecated) If `audio_data` is a numpy array, then this flag
                determines if the WAV file is first copied to a temporary file
                before upload. If `copy_to_tmp` is False, then it is sent
                directly to the cloud.
            step (int):  Used to associate the audio asset to a specific step.

        Note:
            See also [Log audio](/docs/v2/guides/experiment-management/log-data/audio/) and
            [Audio tab](/docs/v2/guides/comet-ui/experiment-management/single-experiment-page/#audio-tab).

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np

            # Initialize Comet ML Experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Create an audio sample
            fs = 100
            rate = 44100
            t = np.linspace(0., 1., rate)
            amplitude = np.iinfo(np.int16).max
            audio_sample = amplitude * np.sin(2. * np.pi * fs * t)

            # Log the audio to Comet ML
            exp.log_audio(
                audio_data=audio_sample,
                sample_rate=rate,
                file_name="Example Audio",
                step=0,
            )

            # End the experiment
            exp.end()
            ```
        """

        copy_to_tmp = self._deprecate_copy_to_tmp(copy_to_tmp, "log_audio")

        if not self.alive:
            return None

        if audio_data is None:
            raise TypeError(EXPERIMENT_LOG_AUDIO_NO_DATA_EXCEPTION)

        try:
            normalize_data_enum = AudioNormalization[normalize_data]
        except KeyError as ex:
            raise ValueError(
                EXPERIMENT_LOG_AUDIO_UNSUPPORTED_NORMALIZATION_ERROR
                % (normalize_data, AudioNormalization.list_names())
            ) from ex

        self._set_step(step)

        asset_id = generate_guid()
        url_params = {
            "step": self._get_asset_upload_step(),
            "context": self.context,
            "fileName": file_name,
            "runId": self.run_id,
            "overwrite": overwrite,
            "assetId": asset_id,
            "type": ASSET_TYPE_AUDIO,
        }

        audio_data = fix_special_floats(audio_data)

        processor = AudioUploadProcessor(
            user_input=audio_data,
            sample_rate=sample_rate,
            normalize_data=normalize_data_enum,
            upload_limit=self.asset_upload_limit,
            url_params=url_params,
            metadata=metadata,
            copy_to_tmp=copy_to_tmp,
            error_message_identifier=None,
            tmp_dir=self.tmpdir,
            critical=False,
        )
        upload_message = processor.process()

        if upload_message is None:
            self._report_experiment_error(UNEXPECTED_LOGGING_ERROR % "audio")
            return None

        asset_name, truncated = truncate_to_length(
            upload_message.additional_params.get("fileName", None),
            self.asset_name_max_length,
        )
        if truncated:
            upload_message.additional_params["fileName"] = asset_name
            LOGGER.warning(
                EXPERIMENT_LOG_AUDIO_NAME_TRUNCATED_WARNING,
                self.asset_name_max_length,
            )

        self._enqueue_message(upload_message)
        self._summary.increment_section("uploads", "audio")
        return self._get_uploaded_audio_url(asset_id)

    def create_confusion_matrix(
        self,
        y_true: Optional[List] = None,
        y_predicted: Optional[List] = None,
        labels: Optional[List[str]] = None,
        matrix: Optional[List[List]] = None,
        title: str = "Confusion Matrix",
        row_label: str = "Actual Category",
        column_label: str = "Predicted Category",
        max_examples_per_cell: int = 25,
        max_categories: int = 25,
        winner_function: Optional[Callable] = None,
        index_to_example_function: Optional[Callable] = None,
        cache: bool = True,
        selected: Optional[List[int]] = None,
        images: Optional[List] = None,
        images_annotations: Optional[List[List[Layer]]] = None,
        **kwargs: Any,  # keyword args for index_to_example_function
    ):
        """Create a confusion matrix for use over multiple epochs.

        Args:
            y_true (list): List of vectors representing the targets, or a list
                of integers representing the correct label. If
                not provided, then matrix may be provided.
            y_predicted (list): List of vectors representing predicted
                values, or a list of integers representing the output. If
                not provided, then matrix may be provided.
            labels (list): A list of strings that name of the
                columns and rows, in order.
            matrix (list): The confusion matrix (list of lists).
                Must be square, if given. If not given, then it is
                possible to provide y_true and y_predicted.
            title (str): A custom name to be displayed. By
                default, it is "Confusion Matrix".
            row_label (str): Label for rows.
            column_label (str): Label for columns.
            max_examples_per_cell (int): Maximum number of
                examples per cell.
            max_categories (int): Max number of columns and rows to
                use.
            winner_function (callable): A function that takes in an
                entire list of rows of patterns, and returns
                the winning category for each row.
            index_to_example_function (callable): A function
                that takes an index and returns either
                a number, a string, a URL, or a {"sample": str,
                "assetId": str} dictionary. See below for more info. If left blank, the function
                returns a number representing the index of the example, and logs images if parameter is not None.
            cache (bool): Should the results of index_to_example_function
                be cached and reused?
            selected (list): A list of selected category
                indices. These are the rows/columns that will be shown. By
                default, select is None. If the number of categories is
                greater than max_categories, and selected is not provided,
                then selected will be computed automatically by selecting
                the most confused categories.
            images (list): A list of data that can be passed to
                Experiment.log_image().
            images_annotations (list, optional): Specifies annotation layers for images.
                This parameter expects a list of lists, where each inner list defines an
                annotation layer, potentially containing bounding boxes and/or polygons.

                For a comprehensive understanding of how annotations are utilized and structured,
                please consult the documentation for `Experiment.log_image()`.

                **Important Note for Custom `index_to_example_function` Implementations:**
                If you provide your own implementation for `index_to_example_function` (bypassing the default),
                it is critical to ensure that every `images_annotations` entry is consistently passed alongside its corresponding image within each `log_image()` call.
                Furthermore, the metadata associated with these annotations must precisely align with the expectations and requirements of the `Experiment.log_image()` call.
            kwargs (optional): Any extra keywords and their values will
                be passed onto the index_to_example_function.

        Note:
            For more details and example uses, please see:
            [Log confusion matrices](/docs/v2/guides/experiment-management/log-data/confusion-matrix/).
            Uses winner_function to compute winning categories for
            y_true and y_predicted, if they are vectors.

            Also, for more low-level information, see comet_ml.utils.ConfusionMatrix

        Example:
            ```python linenums="1"
            import comet_ml
            from sklearn.metrics import confusion_matrix

            comet_ml.login()
            exp = comet_ml.start()

            y_true = [2, 0, 2, 2, 0, 1]
            y_pred = [0, 0, 2, 2, 0, 2]

            confusion_matrix = exp.create_confusion_matrix(y_true, y_pred)

            exp.end()
            ```
        """
        confusion_matrix = ConfusionMatrix(
            y_true=y_true,
            y_predicted=y_predicted,
            labels=labels,
            matrix=matrix,
            title=title,
            row_label=row_label,
            column_label=column_label,
            max_examples_per_cell=max_examples_per_cell,
            max_categories=max_categories,
            winner_function=winner_function,
            index_to_example_function=index_to_example_function,
            cache=cache,
            selected=selected,
            images=images,
            images_annotations=images_annotations,
            experiment=self,
            **kwargs,
        )
        return confusion_matrix

    def log_confusion_matrix(
        self,
        y_true: Optional[List] = None,
        y_predicted: Optional[List] = None,
        matrix: Optional[List[List]] = None,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        row_label: str = "Actual Category",
        column_label: str = "Predicted Category",
        max_examples_per_cell: int = 25,
        max_categories: int = 25,
        winner_function: Optional[Callable] = None,
        index_to_example_function: Optional[Callable] = None,
        cache: bool = True,
        # Logging options:
        file_name: str = "confusion-matrix.json",
        overwrite: bool = False,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        selected: Optional[List[int]] = None,
        images_annotations: Optional[List[List[Layer]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Logs a confusion matrix.

        Args:
            y_true (list): List of vectors representing the targets, or a list
                of integers representing the correct label. If
                not provided, then matrix may be provided.
            y_predicted (list): List of vectors representing predicted
                values, or a list of integers representing the output. If
                not provided, then matrix may be provided.
            labels (list): A list of strings that name of the
                columns and rows, in order.
            matrix (list): The confusion matrix (list of lists).
                Must be square, if given. If not given, then it is
                possible to provide y_true and y_predicted.
            title (str): A custom name to be displayed. By
                default, it is "Confusion Matrix".
            row_label (str, optional): Label for rows.
            column_label (str, optional): Label for columns.
            max_examples_per_cell (int, optional): Maximum number of
                examples per cell.
            max_categories (int, optional): Max number of columns and rows to
                use.
            winner_function (callable): A function that takes in an
                entire list of rows of patterns, and returns
                the winning category for each row.
            index_to_example_function (callable): A function
                that takes an index and returns either
                a number, a string, a URL, or a {"sample": str,
                "assetId": str} dictionary. See below for more info. If left blank, the function
                returns a number representing the index of the example.
            cache (bool): Should the results of index_to_example_function
                be cached and reused?
            file_name (str): The name of the file that
                the confusion matrix will be saved as when logged.
            overwrite (bool): If set to True, the existing confusion matrix with
                the same file name will be overwritten. If False, a new entry is created, preserving the
                previous versions.
            step (int): The step parameter can be used to specify the particular
                step or iteration in the training process at which the confusion matrix is logged.
            epoch (int): Similar to the step parameter, epoch specifically denotes
                the training epoch.
            images (list): A list of data that can be passed to
                Experiment.log_image().
            selected (list): A list of selected category
                indices. These are the rows/columns that will be shown. By
                default, select is None. If the number of categories is
                greater than max_categories, and selected is not provided,
                then selected will be computed automatically by selecting
                the most confused categories.
            metadata (dict): Some additional data to attach to the confusion matrix asset.
            images_annotations (list, optional): Specifies annotation layers for images.
                This parameter expects a list of lists, where each inner list defines an
                annotation layer, potentially containing bounding boxes and/or polygons.

                For a comprehensive understanding of how annotations are utilized and structured,
                please consult the documentation for `Experiment.log_image()`.

                **Important Note for Custom `index_to_example_function` Implementations:**
                If you provide your own implementation for `index_to_example_function` (bypassing the default),
                it is critical to ensure that every `images_annotations` entry is consistently passed alongside its corresponding image within each `log_image()` call.
                Furthermore, the metadata associated with these annotations must precisely align with the expectations and requirements of the `Experiment.log_image()` call.
            kwargs (Any): Any extra keywords and their values will
                be passed onto the index_to_example_function.


        Note:
            For more details and example uses, please see:
            [Log confusion matrices](/docs/v2/guides/experiment-management/log-data/confusion-matrix/).
            Uses winner_function to compute winning categories for
            y_true and y_predicted, if they are vectors.

            Also, for more low-level information, see comet_ml.utils.ConfusionMatrix

        Example:
            ```python linenums="1"

            import comet_ml
            from sklearn.metrics import confusion_matrix

            comet_ml.login()
            exp = comet_ml.start()

            y_true = [2, 0, 2, 2, 0, 1]
            y_pred = [0, 0, 2, 2, 0, 2]

            confusion_matrix = exp.create_confusion_matrix(y_true, y_pred)
            exp.log_confusion_matrix(matrix=confusion_matrix, step=0, file_name="confusion_matrix.json")

            exp.end()
            ```
        """

        if images_annotations is not None:
            if not isinstance(images_annotations, list):
                LOGGER.error(
                    EXPERIMENT_LOG_IMAGE_ANNOTATION_VALIDATION_ERROR
                    % f"images_annotations is not a list as expected, has a type of: {type(images_annotations)}"
                )
                return None

            for image_annotations in images_annotations:
                validator = ImageAnnotationValidator(annotations=image_annotations)
                result = validator.validate()
                if not result:
                    LOGGER.error(
                        EXPERIMENT_LOG_IMAGE_ANNOTATION_VALIDATION_ERROR
                        % result.failure_reasons
                    )
                    return None

        if isinstance(matrix, ConfusionMatrix):
            confusion_matrix = matrix
            confusion_matrix.need_init()
        else:
            try:
                confusion_matrix = self.create_confusion_matrix(
                    y_true=y_true,
                    y_predicted=y_predicted,
                    matrix=matrix,
                    labels=labels,
                    title=title,
                    row_label=row_label,
                    column_label=column_label,
                    max_examples_per_cell=max_examples_per_cell,
                    max_categories=max_categories,
                    winner_function=winner_function,
                    index_to_example_function=index_to_example_function,
                    cache=cache,
                    images=images,
                    images_annotations=images_annotations,
                    selected=selected,
                    **kwargs,
                )
            except Exception as exc:
                LOGGER.error(
                    EXPERIMENT_LOG_CONFUSION_MATRIX_CREATE_FAILED_ERROR,
                    exc,
                    exc_info=True,
                )
                self._report_experiment_error(
                    EXPERIMENT_LOG_CONFUSION_MATRIX_CREATE_FAILED_ERROR
                )
                return

        try:
            confusion_matrix_json = confusion_matrix.to_json()
            if confusion_matrix_json["matrix"] is None:
                LOGGER.error(EXPERIMENT_LOG_CONFUSION_MATRIX_EMPTY_MATRIX_ERROR)
                return
            return self._log_asset_data(
                confusion_matrix_json,
                file_name=file_name,
                overwrite=overwrite,
                asset_type=ASSET_TYPE_CONFUSION_MATRIX,
                step=step,
                epoch=epoch,
                metadata=metadata,
            )
        except Exception:
            LOGGER.error(EXPERIMENT_LOG_CONFUSION_MATRIX_GENERAL_ERROR, exc_info=True)
            self._report_experiment_error(EXPERIMENT_LOG_CONFUSION_MATRIX_GENERAL_ERROR)
            return

    def log_histogram_3d(
        self,
        values: Any,
        name: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, str]]:
        """Logs a histogram of values for a 3D chart as an asset for this
        experiment. Calling this method multiple times with the same
        name and incremented steps will add additional histograms to
        the 3D chart on Comet.ml.

        Args:
            values: A list, tuple, array (any shape) to summarize, or a
                Histogram object.
            name: Name of summary.
            step: Used as the Z axis when plotting on Comet.
            epoch: Used as the Z axis when plotting on Comet.
            metadata: Used for items like prefix for histogram name.
            kwargs: Additional keyword arguments for histogram.

        Note:
            This method requires that step is either given here, or has
            been set elsewhere. For example, if you are using an auto-
            logger that sets step then you don't need to set it here.

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np

            comet_ml.login()
            exp = comet_ml.start()

            values = np.random.normal(0, 1, (1000, 3))

            exp.log_histogram_3d(
                values=values,
                name="Histogram of Randomly Distributed Data",
                step=1,
                epoch=1,
                metadata={'description': 'This is a 3D histogram of three normally distributed random variables'}
            )

            exp.end()
            ```
        """
        if isinstance(values, Histogram):
            histogram = values
        else:
            histogram = Histogram(**kwargs)
            histogram.add(values)
        if name is None:
            name = "histogram_3d.json"

        if histogram.is_empty():
            LOGGER.warning("ignoring empty histogram")
            return None

        try:
            histogram_json = histogram.to_json()
        except Exception:
            msg = UNEXPECTED_LOGGING_ERROR % "histogram"
            LOGGER.error(msg, exc_info=True)
            self._report_experiment_error(msg)
            return None

        return self._log_asset_data(
            histogram_json,
            file_name=name,
            overwrite=False,
            asset_type=ASSET_TYPE_HISTOGRAM_3D,
            step=step,
            epoch=epoch,
            require_step=True,
            metadata=metadata,
        )

    def log_image(
        self,
        image_data: Any,
        name: Optional[str] = None,
        overwrite: bool = False,
        image_format: str = "png",
        image_scale: float = 1.0,
        image_shape: Optional[Tuple[int, int]] = None,
        image_colormap: Optional[str] = None,
        image_minmax: Optional[Tuple[int, int]] = None,
        image_channels: str = "last",
        copy_to_tmp: bool = True,
        step: Optional[int] = None,
        annotations: Optional[List[Layer]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Logs the image. Images are displayed on the Graphics tab in Comet.

        Args:
            image_data (Any): Is one of the following:
                - a path (string) to an image
                - a file-like object containing an image
                - a numpy matrix
                - a TensorFlow tensor
                - a PyTorch tensor
                - a list or tuple of values
                - a PIL Image
            name (str): A custom name to be displayed on the dashboard.
                If not provided the filename from the `image_data` argument will be
                used if it is a path.
            overwrite (bool): If another image with
                the same name exists, it will be overwritten if overwrite is set to
                True.
            image_format (str): If the image_data is
                actually something that can be turned into an image, this is the
                format used. Typical values include 'png' and 'jpg'.
            image_scale (float): If the image_data is
                actually something that can be turned into an image, this will be the
                new scale of the image.
            image_shape (tuple): If the image_data is actually
                something that can be turned into an image, this is the new shape
                of the array. Dimensions are (width, height) or (width, height, colors)
                where `colors` is 3 (RGB) or 1 (grayscale).
            image_colormap (str): If the image_data is actually something
                that can be turned into an image, this is the colormap used to
                colorize the matrix.
            image_minmax (tuple): If the image_data is actually
                something that can be turned into an image, this is the (min, max)
                used to scale the values. Otherwise, the image is autoscaled between
                (array.min, array.max).
            image_channels (str): If the image_data is
                actually something that can be turned into an image, this is the
                setting that indicates where the color information is in the format
                of the 2D data. 'last' indicates that the data is in (rows, columns,
                channels) where 'first' indicates (channels, rows, columns).
            copy_to_tmp (bool): (Deprecated) If `image_data` is not a file path, then this flag determines
                if the image is first copied to a temporary file before upload. If
                `copy_to_tmp` is False, then it is sent directly to the cloud.
            step (int, optional): Used to associate the image asset to a specific step.
            annotations (list, optional): Annotations contains a list of annotation
                layers. Each annotation layer can contain bounding boxes and/or polygons. See
                example below.
            metadata (dict, optional): Additional metadata to be associated with logged image.
        Returns: None

        Example:
            ```python linenums="1"
            import comet_ml
            from comet_ml.annotations import Layer, Box, Polygon
            import numpy as np
            from PIL import Image

            comet_ml.login()
            exp = comet_ml.start()


            # Create a dummy image using numpy
            width, height = 256, 256
            array = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
            image = Image.fromarray(array)

            # Convert image to grayscale
            image = image.convert("L")

            # One layer with a box and polygon:
            annotations = [
                Layer(
                    boxes=[Box([x, y, w, h], label="dog")],
                    polygons=[Polygon([(x1, y1), ...], label="cat")],
                    name="Predictions",
                )
            ]

            # Log the image
            exp.log_image(
                image_data=image,
                name="Gradient.png",
                image_format="png",
                image_scale=0.5,
                image_shape=(128, 128),
                image_colormap='gray',
                step=1,
                annotations=annotations,
                metadata={"description": "A simple gradient image from black to white"}
            )

            # End the experiment
            exp.end()
            ```
        """

        copy_to_tmp = self._deprecate_copy_to_tmp(copy_to_tmp, "log_image")

        if not self.alive:
            return None

        if check_is_matplotlib_figure(image_data):
            LOGGER.warning(EXPERIMENT_LOG_IMAGE_MATPLOTLIB_FIGURE_WARNING)
            return None

        if annotations is not None:
            validator = ImageAnnotationValidator(annotations)
            result = validator.validate()
            if not result:
                LOGGER.error(
                    EXPERIMENT_LOG_IMAGE_ANNOTATION_VALIDATION_ERROR
                    % result.failure_reasons
                )
                return None

        self._set_step(step)

        if image_data is None:
            raise TypeError(EXPERIMENT_LOG_IMAGE_NO_DATA_EXCEPTION)

        # Prepare parameters
        figure_number = self.figure_counter

        image_id = generate_guid()
        url_params = {
            "step": self._get_asset_upload_step(),
            "context": self.context,
            "runId": self.run_id,
            "figName": expand_user_home_path(name),
            "figCounter": figure_number,
            "overwrite": overwrite,
            "imageId": image_id,
        }

        # copy metadata into separate dictionary to avoid its change by reference in user script
        # while message processing is still in progress
        if metadata is not None:
            image_metadata = deepcopy(metadata)
        else:
            image_metadata = {}

        if annotations is not None and len(annotations) > 0:
            image_metadata["annotations"] = deepcopy(annotations)

        processor = ImageUploadProcessor(
            image_data,
            name=expand_user_home_path(name),
            overwrite=overwrite,
            image_format=image_format,
            image_scale=image_scale,
            image_shape=image_shape,
            image_colormap=image_colormap,
            image_minmax=image_minmax,
            image_channels=image_channels,
            upload_limit=self.upload_limit,
            url_params=url_params,
            metadata=image_metadata,
            copy_to_tmp=copy_to_tmp,
            error_message_identifier=None,
            tmp_dir=self.tmpdir,
            critical=False,
        )
        upload_message = processor.process()

        if upload_message is None:
            self._report_experiment_error(UNEXPECTED_LOGGING_ERROR % "image")
            return None

        fig_name, truncated = truncate_to_length(
            upload_message.additional_params.get("figName", None),
            self.image_name_max_length,
        )
        if truncated:
            upload_message.additional_params["figName"] = fig_name
            LOGGER.warning(
                EXPERIMENT_LOG_IMAGE_NAME_TRUNCATED_WARNING, self.image_name_max_length
            )

        self._enqueue_message(upload_message)
        self._summary.increment_section("uploads", "images")
        self.figure_counter += 1

        return self._get_uploaded_image_url(image_id)

    def log_video(
        self,
        file: Union[str, IO],
        name: Optional[str] = None,
        overwrite: bool = False,
        format: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """Logs a video to Comet. Videos are displayed on the assets tab in Comet and support the
        following formats: MP4, MOV, WMV, and GIF.

        Args:
            file (str): The path to the video or a file-like object that contains the
                video.
            name (str): A custom name can be provided to be displayed on the assets
                tab. If not provided, the filename from the file argument will be used if it is a path.
            overwrite (bool): If another video with the same name exists, it will be
                overwritten if overwrite is set to True.
            format (str): The extension of the file is used to check if the asset is of a
                supported format. If the extension of the file is not correct or if you are uploading a
                file-like object, you can indicate the format of the video through this parameter.
            step (int): This is used to associate the video asset with a specific step.
            epoch (int): Used to associate the asset to a specific epoch.
            metadata (dict): additional custom metadata can be associated with the logged
                video.

        Example:
            ```python linenums="1"
            import comet_ml
            import cv2
            import numpy as np

            # Initialize Comet.ml
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Create a dummy video file in MP4 format
            video_filename = 'output.mp4'
            frame_count = 60  # Number of frames in the video
            frame_width = 640
            frame_height = 480
            frame_rate = 10  # Frames per second

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))

            for i in range(frame_count):
                frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
                out.write(frame)

            out.release()

            # Log video file to Comet
            exp.log_video(video_filename, overwrite=True)

            # End the experiment
            exp.end()
            ```
        """
        if not self.alive:
            return None

        self._set_step(step)
        self._set_epoch(epoch)

        if file is None:
            raise TypeError(EXPERIMENT_LOG_VIDEO_NO_DATA_EXCEPTION)

        video_id = generate_guid()

        url_params = {
            "step": self._get_asset_upload_step(),
            "epoch": self._get_asset_upload_epoch(),
            "context": self.context,
            "fileName": name,
            "runId": self.run_id,
            "overwrite": overwrite,
            "assetId": video_id,
            "type": ASSET_TYPE_VIDEO,
        }

        processor = VideoUploadProcessor(
            user_input=file,
            name=name,
            format=format,
            upload_limit=self.video_upload_limit,
            url_params=url_params,
            metadata=metadata,
            copy_to_tmp=True,
            error_message_identifier=None,
            tmp_dir=self.tmpdir,
            critical=False,
        )
        upload_message = processor.process()

        if upload_message is None:
            self._report_experiment_error(UNEXPECTED_LOGGING_ERROR % ASSET_TYPE_VIDEO)
            return None

        asset_name, truncated = truncate_to_length(
            upload_message.additional_params.get("fileName", None),
            self.asset_name_max_length,
        )
        if truncated:
            upload_message.additional_params["fileName"] = asset_name
            LOGGER.warning(
                EXPERIMENT_LOG_VIDEO_NAME_TRUNCATED_WARNING,
                self.asset_name_max_length,
            )

        self._enqueue_message(upload_message)
        self._summary.increment_section("uploads", "videos")

        return self._get_uploaded_asset_url(video_id)

    def log_current_epoch(self, value: int) -> None:
        """
        Deprecated.
        """
        if self.alive:
            message = MetricMessage(context=None)
            message.set_metric("curr_epoch", value)
            self._enqueue_message(message)
            self._summary.set("metrics", "curr_epoch", value)

    def log_parameters(
        self,
        parameters: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        nested_support: bool = True,
    ) -> None:
        """Logs a dictionary (or dictionary-like object) of multiple parameters.

        Args:
            parameters (dict): This is a dictionary where each key is a string
                representing the name of the parameter, and the value is the
                parameter value itself, which can be of any data type that is
                serializable. The method logs each key-value pair as a parameter
                in the experiment.
            prefix (str): This optional parameter allows you to add
                a prefix to the keys in the parameters dictionary when they are
                logged. This can be helpful for organizing or grouping parameters
                under a common namespace, making them easier to identify and analyze
                in the Comet UI.
            step (int): This optional parameter can be used to associate
                the logged parameters with a particular step or iteration in an
                experiment. This is useful for tracking how parameters change over time,
                particularly in iterative processes like training a machine learning
                model over multiple epochs.
            nested_support (bool): This parameter controls whether
                the method should support nested dictionaries. If set to True, the method
                can handle parameters that are dictionaries themselves and will flatten
                these into a format suitable for logging. Each key in a nested dictionary
                will be combined with its parent keys to create a single, flat key.

        Note:
            See also [log_parameter][comet_ml.Experiment.log_parameter].

            If you call this method multiple times with the same
            keys your values would be overwritten.

            If this method is called inside a context, like [test]
            [comet_ml.Experiment.test],[train][comet_ml.Experiment.train],
            [validate][comet_ml.Experiment.validate] or [context_manager]
            [comet_ml.Experiment.context_manager], the parameter will be
            stored with the current context name as a prefix.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            params = {
                "batch_size": 64,
                "layer1": "LSTM(128)",
                "layer2": "LSTM(128)",
                "MAX_LEN":200
            }

            exp.log_parameters(params)

            exp.end()
            ```
        """
        self._log_parameters(
            parameters,
            prefix,
            step=step,
            source=ParameterMessage.source_manual,
            flatten_nested=nested_support,
        )

    def _log_parameters(
        self,
        parameters: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        framework: Optional[str] = None,
        source: Optional[str] = None,
        flatten_nested: bool = False,
    ) -> None:
        # Internal logging handler with option to ignore auto-logged keys
        if isinstance(parameters, Mapping):
            if len(parameters) == 0:
                LOGGER.debug(LOG_PARAMS_EMPTY_MAPPING, parameters)

                return None

            dic = parameters
        else:
            dic = convert_object_to_dictionary(parameters)

            if len(dic) == 0:
                if not debug_helpers.has_enabled_debug_exception_raising():
                    self._log_once_at_level(
                        logging.WARNING, LOG_PARAMS_EMPTY_CONVERTED_MAPPING, parameters
                    )
                else:
                    # raise CometDebugException if comet_ml.error_mode() is enabled
                    debug_helpers.raise_debug_exception(
                        LOG_PARAMS_EMPTY_CONVERTED_MAPPING_ERROR, parameters
                    )

                return None

        if flatten_nested and can_parameters_be_flattened(dic, source=source):
            flatten_op_result = flatten_dict(
                d=dic, separator=PARAMETERS_DELIMITER, max_depth=PARAMETERS_MAX_DEPTH
            )
            dic = flatten_op_result.flattened
            if flatten_op_result.max_depth_limit_reached:
                debug_helpers.log_warning_or_raise(
                    LOG_PARAMS_MAX_DEPTH_REACHED,
                    parameters,
                    PARAMETERS_MAX_DEPTH,
                    logger=LOGGER,
                )

            if flatten_op_result.has_nested_dictionary():
                self._log_other("hasNestedParams", True)
            self._report(event_name=NESTED_PARAMETERS_LOGGED, err_msg=str(len(dic)))

        self._set_step(step)

        for k in sorted(dic):
            delimiter = "_"  # default prefix_name delimiter
            delimiters = ["_", ":", "-", "+", ".", "/", "|"]
            if prefix is not None:
                if any(prefix.endswith(d) for d in delimiters):
                    # prefix already has a delimiter
                    delimiter = ""
                self._log_parameter(
                    "%s%s%s" % (prefix, delimiter, k),
                    dic[k],
                    framework=framework,
                    source=source,
                )
            else:
                self._log_parameter(k, dic[k], framework=framework, source=source)

    def __internal_api__log_parameters__(
        self,
        parameters: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        framework: Optional[str] = None,
        source: Optional[str] = None,
        flatten_nested: bool = False,
    ) -> None:
        self._log_parameters(
            parameters=parameters,
            prefix=prefix,
            step=step,
            framework=framework,
            source=source,
            flatten_nested=flatten_nested,
        )

    def log_metrics(
        self,
        dic: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        """
        Logs a key,value dictionary of metrics.
        See also [log_metric][comet_ml.Experiment.log_metric]
        """
        self._log_metrics(
            dic, prefix=prefix, step=step, epoch=epoch, include_context=include_context
        )

    def _log_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        framework: Optional[str] = None,
        include_context: bool = True,
    ) -> None:
        # Internal logging handler with option to ignore auto-logged names
        self._set_step(step)
        self._set_epoch(epoch)

        # flatten dictionary if appropriate
        flatten_op_result = flatten_dict(
            d=metrics, separator=METRICS_DELIMITER, max_depth=METRICS_MAX_DEPTH
        )
        dic = flatten_op_result.flattened
        if flatten_op_result.max_depth_limit_reached:
            debug_helpers.log_warning_or_raise(
                LOG_METRICS_MAX_DEPTH_REACHED,
                metrics,
                METRICS_MAX_DEPTH,
                logger=LOGGER,
            )

        if self.alive:
            for k in sorted(dic):
                if prefix is not None:
                    self._log_metric(
                        prefix + "_" + str(k),
                        dic[k],
                        framework=framework,
                        include_context=include_context,
                    )
                else:
                    self._log_metric(
                        k, dic[k], framework=framework, include_context=include_context
                    )

    def __internal_api__log_metrics__(
        self,
        dic: Dict[str, Any],
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        framework: Optional[str] = None,
        include_context: bool = True,
    ) -> None:
        self._log_metrics(
            metrics=dic,
            prefix=prefix,
            step=step,
            epoch=epoch,
            framework=framework,
            include_context=include_context,
        )

    def log_dataset_info(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """Used to log information about your dataset.

        Args:
            name (str): A string representing the name of the dataset.
            version (str): A string representing a version identifier.
            path (str): A string that represents the path to the dataset.
                Potential values could be a file system path, S3 path, or Database query.

        Note:
            At least one argument should be included when calling this method. The logged
            values will show up in the `Other` tab of the Comet UI.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.log_dataset_info(name="Your dataset", version="1.0", path="./data.csv")

            exp.end()
            ```
        """
        if name is None and version is None and path is None:
            LOGGER.warning(EXPERIMENT_LOG_DATASET_INFO_NAME_VERSION_PATH_EMPTY_WARNING)
            return
        info = ""
        if name is not None:
            info += str(name)
        if version is not None:
            if info:
                info += "-"
            info += str(version)
        if path is not None:
            if info:
                info += ", "
            info += str(path)
        self.log_other("dataset_info", info)

    def log_dataset_hash(self, data: Any) -> None:
        """Used to log the hash of the provided object.

        Args:
            data: Any object that when casted to string (e.g. str(data)) returns a value that represents the underlying data.

        Note:
            This is a best-effort hash computation which is based on the md5
            hash of the underlying string representation of the object data. Developers are encouraged to implement their
            own hash computation that's tailored to their underlying data source. That could be reported as
            `experiment.log_parameter("dataset_hash", your_hash)`.

        Example:
            ```python linenums="1"
            import comet_ml
            import pandas as pd

            # Initialize Comet ML Experiment
            comet_ml.login()
            exp = comet_ml.start()

            # Example: Load a dataset (could be from a file, database, etc.)
            data = pd.read_csv('path/to/your/dataset.csv')

            # Function to create a consistent hash of a pandas DataFrame
            def hash_dataframe(df):
                return pd.util.hash_pandas_object(df).sum()

            # Compute the hash of the dataset
            dataset_hash = hash_dataframe(data)

            # Log the dataset hash to Comet ML
            exp.log_dataset_hash(dataset_hash)

            # End the experiment
            exp.end()
            ```
        """
        try:
            import hashlib

            data_hash = hashlib.md5(str(data).encode("utf-8")).hexdigest()
            self._log_parameter("dataset_hash", data_hash[:12], framework="comet")
        except Exception:
            LOGGER.warning(EXPERIMENT_LOG_DATASET_HASH_WARNING, exc_info=True)

    def log_table(
        self,
        filename: str,
        tabular_data: Any = None,
        headers: Union[str, List[str]] = False,
        **format_kwargs: Any,
    ) -> Optional[Dict[str, str]]:
        """Log tabular data, including data, csv files, tsv files, and Pandas dataframes.

        Args:
            filename (str): A filename ending in ".csv", or ".tsv" (for tabular
                data) or ".json", ".csv", ".md", or ".html" (for Pandas dataframe data).
            tabular_data (Any): Data that can be interpreted as 2D tabular data
                or a Pandas dataframe.
            headers (bool | list): If True, will add column headers automatically
                if tabular_data is given; if False, no headers will be added; if list
                then it will be used as headers. Only useful with tabular data (csv, or tsv).
            format_kwargs (Any): When passed a Pandas dataframe
                these keyword arguments are used in the conversion to "json", "csv",
                "md", or "html". See Pandas Dataframe conversion methods (like `to_json()`)
                for more information.

        See also:

        * [pandas.DataFrame.to_json documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)
        * [pandas.DataFrame.to_csv documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
        * [pandas.DataFrame.to_html documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html)
        * [pandas.DataFrame.to_markdown documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html)

        Example:
            ```python linenums="1"
            import comet_ml
            import pandas as pd

            comet_ml.login()
            exp = comet_ml.start()

            data = {
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'Los Angeles', 'Chicago']
            }
            example_table = pd.DataFrame(data)

            exp.log_table("example_table.csv", example_table)
            ```
        """
        if not self.alive:
            return None

        expanded_filename = expand_user_home_path(filename)
        if os.path.isfile(expanded_filename):
            if headers is not False:
                LOGGER.info(LOG_TABLE_FILENAME_AND_HEADERS)

            return self._log_asset(expanded_filename)

        # Filename is not an existing file - the tabular data expected
        if tabular_data is None:
            LOGGER.warning(LOG_TABLE_NONE_VALUES)
            return None

        # Tabular-data is not None - the file name must be string
        if not isinstance(filename, str):
            raise ValueError(EXPERIMENT_LOG_TABLE_FILENAME_NOT_STRING_EXCEPTION)

        converted = convert_log_table_input_to_io(
            filename=filename,
            tabular_data=tabular_data,
            headers=headers,
            format_kwargs=format_kwargs,
        )

        if not converted:
            self._report_experiment_error(UNEXPECTED_LOGGING_ERROR % "table")
            return None

        fp, asset_type = converted
        return self._log_asset(fp, filename, asset_type=asset_type)

    def _log_embedding_groups(self) -> None:
        """
        Log all embedding groups together in one template config file.

        Example:

        ```python
        >>> experiment.log_embedding(..., group="hidden-layer")
        >>> experiment.log_embedding(..., group="hidden-layer")

        >>> experiment._log_embedding_groups()
        ```
        """
        if not self.alive:
            return

        groups = list(self._embedding_groups.keys())
        for group in groups:
            embedding_list = self._embedding_groups[group]
            self._log_embedding_list(
                embedding_list,
                "template-%s-configs.json" % safe_filename(group),
            )
            del self._embedding_groups[group]

    def _log_embedding_list(
        self, embeddings: List[Embedding], template_filename: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Log a list of Embeddings.

        Args:
            embeddings: list of Embeddings
            template_filename: (optional) name of template JSON file

        Example:
        ```python
        >>> embeddings = [Embedding(...), Emedding(...), ...]
        >>> experiment._log_embedding_list(embeddings)
        ```

        See also: [comet_ml.CometExperiment.log_embedding][] and [comet_ml.Embedding][]
        """
        if not self.alive:
            return None

        # Log the template:
        template = {ASSET_TYPE_EMBEDDINGS: [embed.to_json() for embed in embeddings]}
        if template_filename is None:
            template_filename = make_template_filename()

        return self._log_asset_data(
            template, template_filename, asset_type=ASSET_TYPE_EMBEDDINGS
        )

    def create_embedding_image(
        self,
        image_data: Any,
        image_size: Optional[List[int]],
        image_preprocess_function: Optional[Callable] = None,
        image_transparent_color: Optional[List[int]] = None,
        image_background_color_function: Optional[Callable] = None,
    ) -> Optional[Tuple[Any, str]]:
        """Create an embedding image (a sprite sheet). Returns the image
        and the url to the image.

        Args:
            image_data (Any): List of arrays or Images.
            image_size (list): The size of each image.
            image_preprocess_function (callable): If image_data is an
                array, apply this function to each element first.
            image_transparent_color (list): A (red, green, blue) tuple.
            image_background_color_function (callable): A function that takes an
                index, and returns a (red, green, blue) color tuple.

        Returns:
            tuple: Returns the image and url

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np

            comet_ml.login()
            exp = comet_ml.start()

            num_images = 10
            inputs = np.random.randint(0, 256, (num_images, 28, 28))
            label = np.random.randint(0, 10, num_images)

            def label_to_color(index):
                label_colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (0, 255, 255), (128, 128, 0),
                    (0, 128, 128), (128, 0, 128), (255, 0, 255),
                    (255, 255, 255)
                ]
                return label_colors[label[index] % len(label_colors)]

            image, image_url = exp.create_embedding_image(
                image_data=inputs,
                image_size=(28, 28),
                image_preprocess_function=lambda matrix: np.round(matrix / 255, 0) * 2,
                image_transparent_color=(0, 0, 0),
                image_background_color_function=label_to_color
            )

            print("Generated Image URL:", image_url)

            exp.end()
            ```
        """
        if not self.alive:
            return None

        image, sprite_url, _ = self._create_embedding_image(
            image_data=image_data,
            image_size=image_size,
            image_preprocess_function=image_preprocess_function,
            image_transparent_color=image_transparent_color,
            image_background_color_function=image_background_color_function,
        )
        return image, sprite_url

    def _create_embedding_image(
        self,
        image_data: Any,
        image_size: Optional[List[int]],
        image_preprocess_function: Optional[Callable] = None,
        image_transparent_color: Optional[List[int]] = None,
        image_background_color_function: Optional[Callable] = None,
    ) -> Optional[Tuple[Any, str, str]]:
        try:
            image = dataset_to_sprite_image(
                image_data,
                size=image_size,
                preprocess_function=image_preprocess_function,
                transparent_color=image_transparent_color,
                background_color_function=image_background_color_function,
            )
        except Exception:
            LOGGER.warning(
                EXPERIMENT_CREATE_EMBEDDING_IMAGE_FAILED_CREATING_IMAGE_ERROR,
                exc_info=True,
            )
            self._report_experiment_error(
                EXPERIMENT_CREATE_EMBEDDING_IMAGE_FAILED_CREATING_IMAGE_ERROR
            )
            return None

        # We assume that the error resulting in an empty image is already logged
        if not image:
            return None

        sprite_url = None
        sprite_asset_id = None
        random_id = random.randint(0, 10000000)
        results = self.log_image(image, "embedding-image-%s.png" % random_id)
        if results is not None:
            sprite_url = results["web"]
            sprite_asset_id = results["imageId"]
        return image, sprite_url, sprite_asset_id

    def _create_embedding(
        self,
        vectors: Any,
        labels: Any,
        image_data: Any = None,
        image_size: Optional[List[int]] = None,
        image_preprocess_function: Optional[Callable] = None,
        image_transparent_color: Optional[List[int]] = None,
        image_background_color_function: Optional[Callable] = None,
        title: str = Embedding.DEFAULT_TITLE,
    ) -> Optional[Embedding]:
        """
        Create a multidimensional dataset and metadata for viewing with
        Comet's Embedding Projector.

        Args:
            vectors: the tensors to visualize in 3D
            labels: labels for each tensor
            image_data: (optional) list of arrays or Images, or a URL
            image_size: (optional, required if image_data is given) the size of each image
            image_preprocess_function: (optional) if image_data is an
                array, apply this function to each element first
            image_transparent_color: a (red, green, blue) tuple
            image_background_color_function: a function that takes an
                index, and returns a (red, green, blue) color tuple
            title: (optional) name of tensor

        See also: [comet_ml.CometExperiment.log_embedding][], [comet_ml.CometExperiment._log_embedding_list][], and [comet_ml.Embedding][]
        """

        # Log vector
        random_id = random.randint(0, 10000000)
        vector = self.log_table("vectors-%s.tsv" % random_id, vectors)

        if vector is None:
            LOGGER.error(EXPERIMENT_CREATE_EMBEDDING_EMPTY_VECTOR_ERROR)
            return None

        try:
            vector_asset_id = vector["assetId"]
            vector_url = vector["web"]
            shape_vectors = shape(vectors)
            vector_shape = [
                shape_vectors[0],
                reduce((lambda x, y: x * y), shape_vectors[1:]),
            ]
        except Exception:
            LOGGER.error(
                EXPERIMENT_CREATE_EMBEDDING_VECTOR_SHAPE_ERROR,
                exc_info=True,
            )
            self._report_experiment_error(
                EXPERIMENT_CREATE_EMBEDDING_VECTOR_SHAPE_ERROR
            )
            return None

        # Log metadata
        headers = False
        if (
            labels is not None
            and len(labels) > 0
            and (
                isinstance(labels[0], (list, tuple))
                or (HAS_NUMPY and isinstance(labels[0], np.ndarray))
            )
        ):
            headers, labels = labels[0], labels[1:]

        metadata = self.log_table(
            "metadata-%s.tsv" % random_id, labels, headers=headers
        )
        if metadata is None:
            LOGGER.error(EXPERIMENT_CREATE_EMBEDDING_EMPTY_METADATA_ERROR)
            return None

        # Log image data if any
        sprite_url = None
        sprite_asset_id = None
        if image_data is not None:
            if isinstance(image_data, str):
                sprite_url = image_data
            else:
                if image_size is None:
                    LOGGER.error(EXPERIMENT_CREATE_EMBEDDING_NO_IMAGE_SIZE_ERROR)
                else:
                    results = self._create_embedding_image(
                        image_data=image_data,
                        image_size=image_size,
                        image_preprocess_function=image_preprocess_function,
                        image_transparent_color=image_transparent_color,
                        image_background_color_function=image_background_color_function,
                    )
                    if results is not None and len(results) == 3:
                        sprite_image, sprite_url, sprite_asset_id = results

        # Construct a data structure:
        embedding = Embedding(
            vector_url=vector_url,
            vector_shape=vector_shape,
            vector_asset_id=vector_asset_id,
            metadata_url=metadata["web"],
            metadata_asset_id=metadata["assetId"],
            sprite_url=sprite_url,
            sprite_asset_id=sprite_asset_id,
            image_size=image_size,
            title=title,
            experiment_key=self.id,
        )

        return embedding

    def log_embedding(
        self,
        vectors: Any,
        labels: Union[List[str], List[List[Any]]],
        image_data: Any = None,
        image_size: Optional[List[Tuple[int]]] = None,
        image_preprocess_function: Optional[Callable] = None,
        image_transparent_color: Optional[Tuple[int]] = None,
        image_background_color_function: Optional[Callable] = None,
        title: str = Embedding.DEFAULT_TITLE,
        template_filename: Optional[str] = None,
        group: Optional[str] = None,
    ) -> Union[Optional[Embedding], Optional[Dict[str, str]]]:
        """
        Log a multidimensional dataset and metadata for viewing with
        Comet's Embedding Projector (experimental).

        Args:
            vectors: the tensors to visualize in 3D.
            labels: labels for each tensor, or a table for each tensor
            image_data (optional): list of arrays or Images
            image_size (optional): The size of each image, required if image_data is given.
            image_preprocess_function (optional): If image_data is an
                array, apply this function to each element first
            image_transparent_color: A (red, green, blue) tuple.
            image_background_color_function: a function that takes an
                index, and returns a (red, green, blue) color tuple.
            title: Name of tensor.
            template_filename: Name of template JSON file.
            group: Name of group of embeddings.

        See also: [comet_ml.Embedding][]

        Note:
        `labels` must be a list of strings, or a table where `table`
        is a list of lists of data, and the first row is a header followed
        by a list for each vector. See example below.

        Examples:
            ```python
            from comet_ml import start

            import numpy as np
            from keras.datasets import mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            def label_to_color(index):
                label = y_test[index]
                if label == 0:
                    return (255, 0, 0)
                elif label == 1:
                    return (0, 255, 0)
                elif label == 2:
                    return (0, 0, 255)
                elif label == 3:
                    return (255, 255, 0)
                elif label == 4:
                    return (0, 255, 255)
                elif label == 5:
                    return (128, 128, 0)
                elif label == 6:
                    return (0, 128, 128)
                elif label == 7:
                    return (128, 0, 128)
                elif label == 8:
                    return (255, 0, 255)
                elif label == 9:
                    return (255, 255, 255)

            experiment = start(project_name="projector-embedding")

            experiment.log_embedding(
                vectors=x_test,
                labels=y_test,
                image_data=x_test,
                image_preprocess_function=lambda matrix: np.round(matrix/255,0) * 2,
                image_transparent_color=(0, 0, 0),
                image_size=(28, 28),
                image_background_color_function=label_to_color,
            )
            ```

            ```python
            # With a table:
            experiment.log_embedding(
                vectors=[[3, 6, 2], [6, 1, 3], [9, 1, 1]],
                labels=[
                    ["index", "label"],
                    [      5, "apple"],
                    [     16, "car"],
                    [      2, "person"],
                ]
            )
            ```
        """
        raise NotImplementedError(EXPERIMENT_LOG_EMBEDDING_NOT_IMPLEMENTED_EXCEPTION)

    def log_dataframe_profile(
        self,
        dataframe: Any,
        name: Optional[str] = "dataframe",
        minimal: bool = False,
        log_raw_dataframe: bool = True,
        dataframe_format: str = "json",
        **format_kwargs: Any,
    ) -> Optional[Dict[str, Optional[Dict[str, str]]]]:
        """
        Log a pandas DataFrame profile as an asset. Optionally, can
        also log the dataframe.

        Args:
            dataframe (Any): the dataframe to profile and/or log
            name (str): the basename (without extension) of the dataframe assets.
            minimal (bool): if True, create a
                minimal profile. Useful for large datasets.
            log_raw_dataframe (bool): Log the
                dataframe as an asset (same as calling `log_table()`)
            dataframe_format (str): The format.
                for optionally logging the dataframe.
            format_kwargs (Any): Keyword args for dataframe
                logging as an asset.

        Example:
            ```python linenums="1"
            import comet_ml
            import pandas as pd

            comet_ml.login()
            exp = comet_ml.start()
            df = pd.read_csv("https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv",
                             parse_dates=['year'], encoding='UTF-8')
            exp.log_dataframe_profile(df)
            exp.end()
            ```

        See also: [`CometExperiment.log_table(pandas_dataframe)`][comet_ml.CometExperiment.log_table]
        """
        if not self.alive:
            return None

        if not check_is_pandas_dataframe(dataframe):
            # Check if pandas is the issue
            try:
                import pandas  # noqa
            except ImportError:
                LOGGER.warning(
                    EXPERIMENT_LOG_DATAFRAME_PROFILE_MISSING_PANDAS_LOG_DATAFRAME
                )
            else:
                LOGGER.warning(EXPERIMENT_LOG_DATAFRAME_PROFILE_NOT_PANDAS_DATAFRAME)
            return None

        retval: Dict[str, Optional[Dict[str, str]]] = {
            "profile": None,
            "dataframe": None,
        }

        try:
            profile_html = get_dataframe_profile_html(
                dataframe=dataframe, minimal=minimal
            )
            if profile_html is not None:
                fp = data_to_fp(profile_html)
                results = self._log_asset(
                    fp,
                    "%s-profile.html" % name,
                    asset_type=ASSET_TYPE_DATAFRAME_PROFILE,
                )
                retval["profile"] = results
        except Exception:
            LOGGER.warning(
                EXPERIMENT_LOG_DATAFRAME_PROFILE_DATAFRAME_PROFILE_ERROR, exc_info=True
            )
            self._report_experiment_error(
                EXPERIMENT_LOG_DATAFRAME_PROFILE_DATAFRAME_PROFILE_ERROR
            )

        if log_raw_dataframe:
            results = self.log_table(
                "%s.%s" % (name, dataframe_format),
                tabular_data=dataframe,
                **format_kwargs,
            )
            retval["dataframe"] = results

        return retval

    def log_points_3d(
        self,
        scene_name: str,
        points: Optional[List[Point3D]] = None,
        boxes: Optional[List[Dict[str, Any]]] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log 3d points and bounding boxes as an asset.

        Args:
            scene_name (str): A string identifying the 3d scene to render. A same scene name could be
                logged across different steps.
            points (list): A list of points, each point being a list (or equivalent like Numpy
                array). Each point length should be either 3, if only the position is given: [X, Y, Z].
                The length could also be 6, if color is passed as well: [X, Y, Z, R, G, B]. Red, Green
                and Blue should be a number between 0 and 1. Either points or boxes are required.
            boxes (list): List of box definition Dict.
            step (int): Used to associate the asset to a specific step.
            metadata (dict): Additional custom metadata can be associated with the logged asset.

        Note:
            You can learn more about logging 3d point clouds
            [here](/docs/v2/guides/experiment-management/log-data/3d-point-clouds/).
            You can also visualize these assets with the
            [3D Points](/docs/v2/guides/comet-ui/experiment-management/visualizations/3d-panel/)
            panels.

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np

            comet_ml.login()
            exp = comet_ml.start()

            # Generate some synthetic 3D points
            np.random.seed(42)
            points = np.random.rand(10, 6)  # 10 points, with XYZ and RGB values

            # Normalize the color values to be between 0 and 1
            points[:, 3:] = points[:, 3:] / np.max(points[:, 3:])

            # Define a bounding box as a list of segments (lines)
            # Each segment is a pair of points (x, y, z)
            box_segments = [
                # Base of the box
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [1, 1, 0]],
                [[1, 1, 0], [0, 1, 0]],
                [[0, 1, 0], [0, 0, 0]],
                # Top of the box
                [[0, 0, 1], [1, 0, 1]],
                [[1, 0, 1], [1, 1, 1]],
                [[1, 1, 1], [0, 1, 1]],
                [[0, 1, 1], [0, 0, 1]],
                # Vertical lines
                [[0, 0, 0], [0, 0, 1]],
                [[1, 0, 0], [1, 0, 1]],
                [[1, 1, 0], [1, 1, 1]],
                [[0, 1, 0], [0, 1, 1]],
            ]

            # Create a box dictionary with the segments
            box = {
                "segments": box_segments,
                "color": [255, 0, 0],  # Red color
                "name": "example_box",
            }

            # Log 3D points and the bounding box to Comet
            exp.log_points_3d(
                scene_name="Simple3DExample",
                points=points.tolist(),
                boxes=[box],
            )

            # End the experiment
            exp.end()
            ```
        """
        if not self.alive:
            return

        experiment_loggers.log_3d_points(
            scene_name,
            points,
            boxes,
            metadata,
            step=step,
            points_3d_upload_limits=self.points_3d_upload_limits,
            summary=self._summary,
            enqueue_message_callback=self._enqueue_message,
        )

    def log_code(
        self,
        file_name: Optional[str] = None,
        folder: Optional[str] = None,
        code: Optional[str] = None,
        code_name: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Tuple[str, Dict[str, str]]]]:
        """
        Logs additional source code files. This method allows logging code in different ways:

        * Without parameters, logs the code from the file calling this method.
        * With `file_name`, logs the code from the specified file.
        * Using `folder`, logs all files' code in that folder.
        * Using `code`, logs the content as source code.

        Args:
            file_name (str): File path to log.
            folder (str): Folder path where the code files are stored.
            code (str): Source code, either as text or a file-like object (such as
                StringIO). If passed, `code_name` is mandatory.
            code_name (str): Name of the source code file.
            overwrite (bool): If True, overwrites existing code with the same name.
            metadata (dict): Additional custom metadata can be associated with the logged code.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            # Logs code from the file using this method.
            exp.log_code()

            # Logs code from the file 'dataset.py'.
            exp.log_code(file_name="dataset.py")

            # Logs all files code in the 'training/' folder.
            exp.log_code(folder="training/")

            # Logs any other code
            exp.log_code(code=DYNAMIC_CODE_SOURCE, code_name="user_training_code.py")
            ```
        """
        return self._log_code_asset(
            source_code_type="manual",
            file_name=file_name,
            folder=folder,
            code=code,
            code_name=code_name,
            overwrite=overwrite,
            metadata=metadata,
        )

    def _log_code_asset(
        self,
        source_code_type: str,
        file_name: Optional[str] = None,
        folder: Optional[str] = None,
        code: Optional[str] = None,
        code_name: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Tuple[str, Dict[str, str]]]]:
        """
        Log additional source code files.
        Args:
            file_name: optional, string: the file path of the file to log
            folder: optional, string: the folder path where the code files are stored
        """
        if not self.alive:
            return None

        if metadata is None:
            metadata = {}
        metadata.update({"source_code_type": source_code_type})

        # Check for mutually exclusive params
        non_null_params = [x for x in [code, file_name, folder] if x is not None]
        if len(non_null_params) > 1:
            LOGGER.warning(EXPERIMENT_LOG_CODE_FILE_NAME_FOLDER_MUTUALLY_EXCLUSIVE)
            return None

        if code is None and file_name is None and folder is None:
            if _in_ipython_environment():
                LOGGER.warning(EXPERIMENT_LOG_CODE_CALLER_JUPYTER)
                return None

            caller = get_caller_file_path()
            if caller is None:
                LOGGER.warning(EXPERIMENT_LOG_CODE_CALLER_NOT_FOUND)
                return None

            caller_file_path = caller[1]

            log_result = self._log_asset(
                file_data=caller_file_path,
                file_name=caller_file_path,
                asset_type=ASSET_TYPE_SOURCE_CODE,
                metadata=metadata,
                overwrite=overwrite,
            )

            if log_result is None:
                return None

            return [(caller_file_path, log_result)]

        elif code is not None:
            if code_name is None:
                LOGGER.warning(EXPERIMENT_LOG_CODE_MISSING_CODE_NAME)
                return None

            log_result = self._log_asset_data(
                data=code,
                file_name=code_name,
                asset_type=ASSET_TYPE_SOURCE_CODE,
                metadata=metadata,
                overwrite=overwrite,
            )

            if log_result is None:
                return None

            return [(code_name, log_result)]

        elif file_name is not None:
            if code_name is not None:
                user_file_name = code_name
            else:
                user_file_name = file_name

            log_result = self._log_asset(
                file_data=file_name,
                file_name=user_file_name,
                asset_type=ASSET_TYPE_SOURCE_CODE,
                metadata=metadata,
                overwrite=overwrite,
            )

            if log_result is None:
                return None

            return [(user_file_name, log_result)]

        else:
            return self._log_asset_folder(
                folder,
                log_file_name=True,
                recursive=True,
                asset_type=ASSET_TYPE_SOURCE_CODE,
                extension_filter=[".py", ".yaml", ".yml", ".txt", ".ipynb", ".sh"],
                metadata=metadata,
                overwrite=overwrite,
            )

    def log_artifact(self, artifact: Artifact) -> LoggedArtifact:
        """
        Log an Artifact object, synchronously create a new Artifact Version and upload
        asynchronously all local and remote assets attached to the Artifact object.

        This is only supported for online experiment at the moment.

        Args:
            artifact (Artifact): An Artifact object.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Returns:
            LoggedArtifact: The artifact that was logged
        """
        raise NotImplementedError(EXPERIMENT_LOG_ARTIFACT_NOT_SUPPORTED_EXCEPTION)

    def get_artifact(
        self,
        artifact_name: str,
        workspace: Optional[str] = None,
        version_or_alias: Optional[str] = None,
    ) -> LoggedArtifact:
        """Returns a logged artifact object that can be used to access the artifact version assets and
        download them locally.

        If no version or alias is provided, the latest version for that artifact is returned.

        This is only supported for online experiment at the moment.

        Args:
            artifact_name (str): Retrieve an artifact with that name. This could either be a fully
                qualified artifact name like `workspace/artifact-name:versionOrAlias` or just the name
                of the artifact like `artifact-name`.
            workspace (str): Retrieve an artifact belonging to that workspace
            version_or_alias (str): Retrieve the artifact by the given alias or version.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Returns:
            LoggedArtifact: The artifact requested

        Example:
            ```python
            logged_artifact = experiment.get_artifact(
                "workspace/artifact-name:version_or_alias"
            )
            ```

            Which is equivalent to:

            ```python
            logged_artifact = experiment.get_artifact(
                artifact_name="artifact-name",
                workspace="workspace",
                version_or_alias="version_or_alias")
            ```
        """
        raise NotImplementedError(EXPERIMENT_GET_ARTIFACT_NOT_SUPPORTED_EXCEPTION)

    def get_hf_dataset(
        self,
        name: str,
        workspace: Optional[str] = None,
        version_or_alias: Optional[str] = None,
        download_directory: Optional[str] = None,
        pathname: str = "**",
        recursive: bool = True,
        use_cached_dataset: bool = True,
    ) -> List[str]:
        """
        Fetches Comet artifact files to be used as Hugging Face dataset using specified parameters.

        This method retrieves a dataset artifact from Comet, allowing
        specification of workspace, version or alias, file pathname, and
        recursive fetching. The dataset is downloaded to the specified directory or custom
        directory if no download directory is specified. By default, all files are downloaded
        only once for specified parameters and cached locally. Later calls to this method
        with the same parameters will retrieve the cached dataset.

        Args:
            name: The name of the dataset artifact to retrieve. This could either be a fully
                qualified artifact name like `workspace/artifact-name:versionOrAlias` or just the name
                of the artifact like `artifact-name`.
            workspace: The workspace context where the artifact exists.
            version_or_alias: Specific version or alias to identify the artifact dataset.
            download_directory: The directory to download the dataset to. If not specified, the dataset
                will be downloaded to a directory created using the artifact name and version or alias in the
                current working directory.
            pathname: File path or pattern to match within the dataset. This can be a glob pattern. Defaults to "**".
            recursive: Determines whether to fetch files recursively. Defaults to True.
            use_cached_dataset: Determines whether to use cached dataset. Defaults to True.

        Returns:
            List of strings representing the retrieved dataset files.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Example:
            ```python
            import comet_ml
            from comet_ml import Artifact

            # log a dataset artifact
            exp = comet_ml.start(project_name="hf-datasets")
            dataset_artifact = Artifact(
                artifact_type="dataset",
                name="iris",
            )
            dataset_artifact.add(
                local_path_or_data="./Downloads/iris", # folder with 2 csvs
            )
            exp.log_artifact(dataset_artifact)
            exp.end()

            # retrieve the dataset artifact and load it as HuggingFace dataset
            from datasets import load_dataset

            dataset = load_dataset(
                "csv",
                data_files={
                    "train": exp.get_hf_dataset(
                        name="iris",
                        version_or_alias="1.0.0",
                        pathname="train/*.csv",
                        download_directory="./iris-dataset",
                    ),
                    "test": exp.get_hf_dataset(
                        name="iris",
                        version_or_alias="1.0.0",
                        pathname="test/*.csv",
                        download_directory="./iris-dataset",
                    )
                }
            )
            ```
        """
        raise NotImplementedError(EXPERIMENT_GET_HF_DATASET_NOT_SUPPORTED_EXCEPTION)

    def set_code(
        self,
        code: Optional[str] = None,
        overwrite: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """
        This method is now deprecated, use [comet_ml.Experiment.log_code][] instead.

        Sets the current experiment script's code. Should be called once per experiment.

        Args:
            code (Any): The experiment's source code.
            overwrite (bool): If True, will overwrite previously set code.
            filename (str): name of file to get source code from
        """
        if filename:
            if code is not None:
                LOGGER.warning(EXPERIMENT_SET_CODE_IGNORE_FILENAME_WARNING)
            elif os.path.isfile(filename):
                LOGGER.warning(SET_CODE_FILENAME_DEPRECATED)
                self.log_code(file_name=filename)
            else:
                LOGGER.warning(EXPERIMENT_LOG_CODE_NOT_A_FILE_WARNING, filename)
                return

        self._set_code(code, overwrite)

    def _set_code(
        self, code: str, overwrite: bool = False, framework: Optional[str] = None
    ):
        if self.alive and code is not None:
            if self._code_set and not overwrite:
                if framework:
                    # Called by an auto-logger
                    self._log_once_at_level(
                        logging.DEBUG,
                        "Set code by %r ignored; already called. Future attempts are silently ignored."
                        % framework,
                    )
                else:
                    LOGGER.warning(EXPERIMENT_SET_CODE_ALREADY_CALLED_WARNING)
                return

            self._track_framework_usage(framework)

            self._code_set = True

            LOGGER.warning(SET_CODE_CODE_DEPRECATED)
            self.log_code(code=code, code_name="Default")

    def set_model_graph(self, graph: Any, overwrite: bool = False) -> None:
        """Sets the current experiment's computational graph.

        Args:
            graph (Any): A model's computational graph.
            overwrite (bool): If True, will overwrite a
                previously logged computational graph definition.

        Note:
            The computational graph will be defined in the `Graph definition`
            tab within the Comet UI of the logged experiment.

        Example:
            ```python linenums="1"
            import comet_ml
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            # Initialize Comet.ml experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Define a simple Sequential model
            model = Sequential([
                Dense(64, activation='relu', input_shape=(10,)),
                Dense(10, activation='softmax')
            ])

            # Convert the model to its TensorFlow graph representation
            graph = model.to_json()
            exp.set_model_graph(graph)

            # End the experiment
            exp.end()
            ```
        """
        return self._set_model_graph(graph, overwrite)

    def __internal_api__set_model_graph__(
        self, graph: Any, overwrite: bool = False, framework: Optional[str] = None
    ):
        self._set_model_graph(graph=graph, overwrite=overwrite, framework=framework)

    def _set_model_graph(
        self,
        graph: Any,
        overwrite: Optional[bool] = False,
        framework: Optional[str] = None,
    ) -> None:
        if not self.alive:
            return

        if not graph:
            LOGGER.debug("Empty model graph logged")
            return None

        if self._graph_set and not overwrite:
            if framework:
                # Called by an auto-logger
                self._log_once_at_level(
                    logging.DEBUG,
                    "Set model graph by %r ignored; already called. Future attempts are silently ignored."
                    % framework,
                )
            else:
                LOGGER.warning(EXPERIMENT_SET_MODEL_GRAPH_ALREADY_CALLED_WARNING)
            return

        graph = convert_model_to_string(graph)

        self._graph_set = True

        self._track_framework_usage(framework)

        LOGGER.debug("Set model graph called")

        message = ModelGraphMessage.create(
            context=self.context,
            graph=graph,
        )
        self._enqueue_message(message)

        self._summary.increment_section("uploads", "model graph", framework=framework)

    def set_filename(self, fname: str) -> None:
        """Sets the current experiment filename.

        Args:
            fname (str): The script's filename.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize Comet experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            exp.set_filename('new_experiment.py')

            # End the experiment
            exp.end()
            ```
        """
        self.filename = fname
        if not self.alive:
            return
        message = FileNameMessage.create(
            context=self.context,
            file_name=fname,
        )
        self._enqueue_message(message)
        self._summary.increment_section("uploads", "filename")

    def set_name(self, name: str) -> None:
        """
        Set a name for the experiment. Useful for filtering and searching on Comet.ml.
        Will shown by default under the `Other` tab.

        Args:
            name (str): A name for the experiment.
        """
        self.name = name
        self._log_other("Name", name, include_context=False)

    def set_os_packages(self) -> None:
        """Reads the installed os packages and reports them to server as a message.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize Comet experiment
            comet_ml.init(project_name="os_package_logging")
            exp = comet_ml.start()

            # Log the OS packages installed on the system
            exp.set_os_packages()

            # End the experiment
            exp.end()
            ```

        """
        if not self.alive:
            return

        try:
            os_packages_list = read_unix_packages()
            if os_packages_list is None:
                return
            message = OsPackagesMessage.create(
                context=self.context,
                os_packages=os_packages_list,
            )
            self._enqueue_message(message)
            self._summary.increment_section("uploads", "os packages")

        except Exception:
            LOGGER.warning(EXPERIMENT_SET_OS_PACKAGES_FAILED_WARNING, exc_info=True)

    def set_pip_packages(self) -> None:
        """Get the installed pip packages using pkg resources and reports
        them to server as a message.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize Comet experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Log the Python packages and their versions
            exp.set_pip_packages()

            # End the experiment
            exp.end()
            ```
        """
        if not self.alive:
            return

        try:
            packages = get_pip_packages()
            message = InstalledPackagesMessage.create(
                context=self.context,
                installed_packages=packages,
            )
            self._enqueue_message(message)
            self._summary.increment_section("uploads", "installed packages")
        except Exception:
            LOGGER.warning(EXPERIMENT_SET_PIP_PACKAGES_FAILED_WARNING, exc_info=True)

    def _log_conda_packages(self) -> None:
        """
        Reads the installed pip packages using pip's CLI and reports them to server as a message.

        """
        if not self.alive:
            return

        try:
            if not _in_conda_environment():
                LOGGER.debug("Not in conda environment")
                return

            conda_env = _get_conda_env()

            if conda_env:
                self._log_asset_data(
                    conda_env,
                    file_name=CONDA_ENV_FILE_NAME,
                    asset_type=CONDA_ENV_ASSET_TYPE,
                )

            conda_explicit_packages = _get_conda_explicit_packages()

            if conda_explicit_packages:
                self._log_asset_data(
                    conda_explicit_packages,
                    file_name=CONDA_SPEC_FILE_NAME,
                    asset_type=CONDA_SPEC_ASSET_TYPE,
                )

            conda_info = _get_conda_info()

            if conda_info:
                self._log_asset_data(
                    conda_info,
                    file_name=CONDA_INFO_FILE_NAME,
                    asset_type=ASSET_TYPE_CONDA_INFO,
                )
        except Exception:
            LOGGER.debug("Failing to collect conda information", exc_info=True)

    def set_cmd_args(self) -> None:
        """Logs command-line arguments used to run the script.

        Example:
            ```python linenums="1"
            import comet_ml

            # Initialize Comet experiment
            comet_ml.login()
            exp = comet_ml.start()

            exp.set_cmd_args()

            # Continue on with your experiment
            # ...

            # End the experiment
            exp.end()
            ```
        """
        if self.alive:
            args = parse_command_line_arguments()
            LOGGER.debug("Command line arguments %r", args)
            if args is not None:
                for k, v in args.items():
                    self._log_parameter(
                        k, v, framework="comet", source=ParameterMessage.source_cli
                    )

    # Context context-managers

    @contextmanager
    def context_manager(self, context: str) -> None:
        """
        A context manager to mark the beginning and the end of the training
        phase. This allows you to provide a namespace for metrics/params.
        For example:

        Args:
            context (str): Name of the context manager.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            # Log validation_accuracy using the context manager
            with exp.context_manager("validation"):
                exp.log_metric("accuracy", 0.12)
            ```
        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = context

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def train(self) -> None:
        """A context manager to mark the beginning and the end of the training
        phase. This allows you to provide a namespace for metrics/params.

        Example:
            ```python linenums="1"
            import comet_ml
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # Initialize Comet experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Load a dataset and create a model
            data = load_iris()
            X = data.data
            y = data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.2,
                                                                random_state=42)
            model = LogisticRegression(max_iter=200)

            # Start train context using a with statement
            with exp.train():

                # Train model
                model.fit(X_train, y_train)

                # Perform predictions on the train set
                pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, pred)

                # Log train accuracy
                exp.log_metric("train_accuracy", train_acc)

            # End the experiment
            exp.end()
            ```
        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "train"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def validate(self) -> None:
        """A context manager to mark the beginning and the end of the validating
        phase. This allows you to provide a namespace for metrics/params.

        Example:
            ```python linenums="1"
            import comet_ml
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # Initialize Comet.ml experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Load a dataset and create a model
            data = load_iris()
            X = data.data
            y = data.target
            X_train, X_validation, y_train, y_validation = train_test_split(X, y,
                                                                            test_size=0.2,
                                                                            random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            # Start validation context using a with statement
            with exp.validate():
                # Perform predictions on the validation set
                pred = model.predict(X_validation)
                val_acc = accuracy_score(y_validation, pred)

                # Log validation accuracy
                exp.log_metric("validation_accuracy", val_acc)

            # End the experiment
            exp.end()
            ```
        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "validate"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def test(self) -> None:
        """A context manager to mark the beginning and the end of the testing phase.
        This allows you to provide a namespace for metrics/params.

        Example:
            ```python linenums="1"
            import comet_ml
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # Initialize Comet experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Load a dataset and create a model
            data = load_iris()
            X = data.data
            y = data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.2,
                                                                random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            # Start test context using a with statement
            with exp.test():
                # Perform predictions on the test set
                pred = model.predict(X_test)
                test_acc = accuracy_score(y_test, pred)

                # Log test accuracy
                exp.log_metric("test_accuracy", test_acc)

            # End the experiment
            exp.end()
            ```
        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "test"

        yield self

        # Restore the old one
        self.context = old_context

    def get_keras_callback(self) -> Callable:
        """This method is deprecated. See [Experiment.get_callback("keras")][comet_ml.CometExperiment.get_callback]. Get
        a callback for the Keras framework.
        """
        LOGGER.warning(EXPERIMENT_GET_KERAS_CALLBACK_DEPRECATED_WARNING)
        return self.get_callback("keras")

    def disable_mp(self) -> None:
        """Disabling the auto-collection of metrics and monkey-patching of
        the Machine Learning frameworks.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.disable_mp()

            exp.log_parameter("example_parameter", 123)

            exp.end()
            ```
        """
        self.disabled_monkey_patching = True

    def register_callback(self, function: Callable) -> None:
        """
        Register the function passed as argument to be an RPC.

        This is only supported for online experiment at the moment.

        Args:
            function (callable): Function to register as a callback.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.
        """
        function_name = function.__name__

        if isinstance(function, types.LambdaType) and function_name == "<lambda>":
            raise LambdaUnsupported()

        if function_name in self.rpc_callbacks:
            raise RPCFunctionAlreadyRegistered(function_name)

        self.rpc_callbacks[function_name] = function

    def unregister_callback(self, function: Callable) -> None:
        """
        Unregister the function passed as argument.

        Args:
            function (callable): The function to unregister as the callback.
        """
        function_name = function.__name__

        self.rpc_callbacks.pop(function_name, None)

    def _get_filename(self) -> Optional[str]:
        """
        Get the filename of the executing code, if possible.
        """
        if _in_ipython_environment():
            return DEFAULT_JUPYTER_INTERACTIVE_FILE_NAME
        elif sys.argv:
            pathname = os.path.dirname(sys.argv[0])
            abs_path = os.path.abspath(pathname)
            filename = os.path.basename(sys.argv[0])
            full_path = os.path.join(abs_path, filename)
            return full_path

        return None

    def _log_git_information(self) -> None:
        """
        Try to find a git repository and logs git-patch and git-metadata is the corresponding
        experiment parameter are enabled.

        The directory preference order is:
            2. the COMET_GIT_DIRECTORY
            3. the current working directory
        """
        if not self.alive:
            return

        if not (self.log_git_metadata or self.log_git_patch):
            return

        from .git_logging import (  # Dulwich imports fails when running in sitecustomize.py
            find_git_repo,
        )

        repo_search_path = self.config.get_comet_git_directory() or os.getcwd()

        LOGGER.debug("Looking for git-metadata starting at %r", repo_search_path)

        repo = find_git_repo(repo_search_path)

        if not repo:
            LOGGER.info(GIT_REPO_NOT_FOUND, repo_search_path)
            return None

        try:
            if self.log_git_metadata:
                self._set_git_metadata(repo)
        except Exception:
            LOGGER.error(LOG_GIT_METADATA_ERROR, exc_info=True)

        try:
            if self.log_git_patch:
                self._set_git_patch(repo)
        except Exception:
            LOGGER.error(LOG_GIT_PATCH_ERROR, exc_info=True)
            tb = traceback.format_exc()
            self._report(event_name=GIT_PATCH_GENERATION_FAILED, err_msg=tb)

        # Close the repo to close all opened fds
        repo.close()

    def _set_git_metadata(self, repo):
        # type: (dulwich.repo.Repo) -> None
        """
        Set the git-metadata for this experiment.
        """
        # This shouldn't be important, the only public caller is _log_git_information which should
        # already check for it
        if not self.alive:
            return

        from .git_logging import (  # Dulwich imports fails when running in sitecustomize.py
            get_git_metadata,
        )

        git_metadata = get_git_metadata(repo)

        if git_metadata:
            message = GitMetadataMessage.create(
                context=self.context,
                git_metadata=git_metadata,
            )
            self._enqueue_message(message)
            self._summary.increment_section("uploads", "git metadata")

    def _set_git_patch(self, repo):
        # type: (dulwich.repo.Repo) -> None
        """
        Set the git-patch for this experiment.

        """
        if not self.alive:
            return

        from .git_logging import (  # Dulwich imports fails when running in sitecustomize.py
            get_git_patch,
        )

        git_patch = get_git_patch(repo)
        if not git_patch:
            LOGGER.debug("Git patch is empty, nothing to upload")
            return None

        _, zip_path = compress_git_patch(git_patch)

        # TODO: Previously there was not upload limit check for git-patch
        processor = GitPatchUploadProcessor(
            TemporaryFilePath(zip_path),
            self.asset_upload_limit,
            url_params=None,
            metadata=None,
            copy_to_tmp=False,
            error_message_identifier=None,
            tmp_dir=self.tmpdir,
            critical=False,
        )
        upload_message = processor.process()

        if not upload_message:
            return None

        self._enqueue_message(upload_message)
        self._summary.increment_section(
            "uploads", "git-patch (uncompressed)", size=len(git_patch)
        )

    def _log_env_details(self) -> None:
        if not self.alive:
            return

        message = get_env_details_message()
        self._enqueue_message(message)

        self._summary.increment_section("uploads", "environment details")

    def _log_cloud_details(self) -> None:
        if not self.alive:
            return

        cloud_details = get_env_cloud_details()
        if cloud_details is None:
            return

        message = CloudDetailsMessage.create(
            context=self.context,
            provider=cloud_details["provider"],
            cloud_metadata=cloud_details["metadata"],
        )
        self._enqueue_message(message)

    def _create_gpu_data_logger(self) -> GPUMetricsDataLogger:
        # First sends the static info as a message
        gpu_static_info = get_gpu_static_info()
        message = GpuStaticInfoMessage.create(
            context=self.context,
            gpu_static_info=gpu_static_info,
        )
        self._enqueue_message(message)

        # Second sends the one-time metrics
        one_time_gpu_metrics = get_initial_gpu_metric()
        metrics = convert_gpu_details_to_metrics(
            gpu_details=one_time_gpu_metrics,
            prefix=self.distributed_node_identifier,
        )
        for metric in metrics:
            self._log_metric(metric["name"], metric["value"], framework="comet")

        return GPUMetricsDataLogger(
            initial_interval=self.config["comet.system_gpu_interval"],
            callback=self._log_gpu_details,
        )

    def _is_compute_metric_included(self) -> bool:
        return False

    def _log_system_metrics(self, metrics: NamedSystemMetrics) -> None:
        prefix = self.distributed_node_identifier
        for metric in metrics:
            full_name = metric_name(metric, prefix=prefix)
            self._log_metric(
                full_name, metrics[metric], include_context=False, framework="comet"
            )

    def _log_disk_utilization(self, metrics: NamedSystemMetrics) -> None:
        self._log_system_metrics(metrics)

    def _log_network_rate(self, metrics: NamedSystemMetrics) -> None:
        self._log_system_metrics(metrics)

    def _log_cpu_details(self, metrics: NamedSystemMetrics) -> None:
        self._log_system_metrics(metrics)

    def _log_gpu_details(self, gpu_details: NamedSystemMetricsList) -> None:
        metrics = convert_gpu_details_to_metrics(
            gpu_details, prefix=self.distributed_node_identifier
        )
        for metric in metrics:
            self._log_metric(
                metric["name"],
                metric["value"],
                include_context=False,
                framework="comet",
            )

    def _log_others_from_environment(self) -> None:
        try:
            others_to_log = env_logging.parse_log_other_instructions()
            self._log_others(others_to_log, include_context=False)
        except Exception:
            LOGGER.error(
                EXPERIMENT_LOG_OTHERS_FROM_ENVIRONMENT_FAILED_ERROR,
                exc_info=True,
            )

    def _set_name_from_environment(self) -> None:
        name = self.config["comet.experiment_name"]
        if name is not None:
            self.set_name(name)

    def _get_uploaded_asset_url(self, asset_id: str) -> Dict[str, str]:
        web_url = create_asset_url(
            self.upload_web_asset_url_prefix, assetId=asset_id, experimentKey=self.id
        )
        api_url = create_asset_url(
            self.upload_api_asset_url_prefix, assetId=asset_id, experimentKey=self.id
        )
        return {"web": web_url, "api": api_url, "assetId": asset_id}

    def _get_uploaded_image_url(self, image_id: str) -> Dict[str, str]:
        web_url = create_asset_url(
            self.upload_web_image_url_prefix, imageId=image_id, experimentKey=self.id
        )
        api_url = create_asset_url(
            self.upload_api_image_url_prefix, imageId=image_id, experimentKey=self.id
        )
        return {"web": web_url, "api": api_url, "imageId": image_id}

    def _get_uploaded_figure_url(self, figure_id: str) -> Dict[str, Optional[str]]:
        web_url = create_asset_url(
            self.upload_web_image_url_prefix, imageId=figure_id, experimentKey=self.id
        )
        api_url = create_asset_url(
            self.upload_api_image_url_prefix, imageId=figure_id, experimentKey=self.id
        )
        return {"web": web_url, "api": api_url, "imageId": figure_id}

    def _get_uploaded_audio_url(self, audio_id: str) -> Dict[str, Optional[str]]:
        web_url = create_asset_url(
            self.upload_web_asset_url_prefix, assetId=audio_id, experimentKey=self.id
        )
        api_url = create_asset_url(
            self.upload_api_asset_url_prefix, assetId=audio_id, experimentKey=self.id
        )
        return {"web": web_url, "api": api_url, "assetId": audio_id}

    def _add_pending_call(self, rpc_call: RemoteCall) -> None:
        self._pending_calls.append(rpc_call)

    def _check_rpc_callbacks(self):
        while len(self._pending_calls) > 0:
            call = self._pending_calls.pop()
            if call is not None:
                try:
                    result = self._call_rpc_callback(call)

                    self._send_rpc_callback_result(call.callId, *result)
                except Exception:
                    LOGGER.debug("Failed to call rpc %r", call, exc_info=True)

    def _call_rpc_callback(self, rpc_call: RemoteCall) -> Tuple[Any, int, int]:
        if rpc_call.cometDefined is False:
            function_name = rpc_call.functionName

            start_time = local_timestamp()

            try:
                function = self.rpc_callbacks[function_name]
                remote_call_result = call_remote_function(function, self, rpc_call)
            except KeyError:
                error = "Unregistered remote action %r" % function_name
                remote_call_result = {"success": False, "error": error}

            end_time = local_timestamp()

            return remote_call_result, start_time, end_time

        # Hardcoded internal callbacks
        if rpc_call.functionName == "stop":
            self._log_other("experiment_stopped_by_user", True, include_context=False)
            raise InterruptedExperiment(rpc_call.userName)
        else:
            raise NotImplementedError()

    def _send_rpc_callback_result(
        self,
        call_id: str,
        remote_call_result: Dict[str, Any],
        start_time: int,
        end_time: int,
    ) -> None:
        raise NotImplementedError()

    def add_tag(self, tag: str) -> bool:
        """
        Add a tag to the experiment. Tags will be shown in the dashboard.

        Args:
            tag (str): A tag to add to the experiment.
        """
        try:
            validator = TagValidator(
                tag,
                method_name=self.add_tag.__name__,
                class_name=self.__class__.__name__,
            )
            result = validator.validate()
            if not result:
                validator.print_result(logger=LOGGER)
                raise ValidationError(EXPERIMENT_LOG_TAG_VALIDATION_ERROR)

            self.tags.add(tag)
            return True
        except Exception:
            LOGGER.warning(ADD_TAGS_ERROR, tag, exc_info=True)
            return False

    def add_tags(self, tags: List[str]) -> bool:
        """
        Add several tags to the experiment. Tags will be shown in the
        dashboard.

        Args:
            tags (List[str]): Tags list to add to the experiment.
        """
        try:
            validator = TagsValidator(
                tags,
                method_name=self.add_tags.__name__,
                class_name=self.__class__.__name__,
            )
            result = validator.validate()
            if not result:
                validator.print_result(logger=LOGGER)
                raise ValidationError(EXPERIMENT_LOG_TAG_VALIDATION_ERROR)

            self.tags = self.tags.union(tags)
            return True
        except Exception:
            LOGGER.warning(ADD_TAGS_ERROR, tags, exc_info=True)
            return False

    def get_tags(self) -> List[str]:
        """
        Return the tags of this experiment.

        Returns:
            List[str]: The list of tags.
        """
        return list(self.tags)

    def log_optimization(
        self,
        optimization_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Any = None,
        parameters: Optional[Dict] = None,
        objective: Optional[str] = None,
    ) -> None:
        """Logs an existing optimization result.

        Args:
            optimization_id (str): The id of the optimization result.
            metric_name (str): The name of your metric
            metric_value (Any): The value of the given metric.
            parameters (dict): Additional parameters to be logged.
            objective (str): The objective of the optimization, could be
                 either maximum/minimum.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            # Define a hypothetical scenario for optimization results
            optimization_id = "opt123"
            metric_name = "accuracy"
            metric_value = 0.95
            parameters = {
                'C': 1,
                'kernel': 'linear',
                'gamma': 'scale'
            }
            objective = "maximum"

            # Log the optimization
            exp.log_optimization(
                optimization_id=optimization_id,
                metric_name=metric_name,
                metric_value=metric_value,
                parameters=parameters,
                objective=objective
            )

            # End the experiment
            exp.end()
            ```
        """

        experiment_loggers.log_optimization(
            self, optimization_id, metric_name, metric_value, parameters, objective
        )

    def _set_optimizer(self, optimizer, pid, trial, count):
        """
        Set the optimizer dictionary and logs
        optimizer data.

        Arguments:
            optimizer: the Optimizer object
            pid: the parameter set id
            trial: the trial number
            count: the running count
        """
        self.optimizer = {
            "optimizer": optimizer,
            "pid": pid,
            "trial": trial,
            "count": count,
        }

    def _get_optimizer_data(self) -> Optional[Dict[str, Any]]:
        """
        If this experiment is being run with the Comet Optimizer,
        return the optimizer data.
        """
        if self.optimizer is not None:
            optimizer_data = {
                "id": self.optimizer["optimizer"].id,
                "pid": self.optimizer["pid"],
                "trial": self.optimizer["trial"],
                "count": self.optimizer["count"],
            }
        else:
            optimizer_data = None

        return optimizer_data

    def _set_optimizer_from_data(self, optimizer_data: Dict[str, Any]) -> None:
        """
        Set the Optimizer from optimizer_data

        Args:
            optimizer_data: a dictionary with fields: id, pid, trial, and count

        Used when this experiment has been recreated and needs to
        restore the optimizer.
        """
        from comet_ml import Optimizer

        optimizer = Optimizer(optimizer_data["id"])
        self._set_optimizer(
            optimizer,
            optimizer_data["pid"],
            optimizer_data["trial"],
            optimizer_data["count"],
        )

        self._log_other("optimizer_id", optimizer_data["id"], include_context=False)
        self._log_other("optimizer_pid", optimizer_data["pid"], include_context=False)
        self._log_other(
            "optimizer_trial", optimizer_data["trial"], include_context=False
        )

        if "parameters" in optimizer_data:
            self._log_parameters(
                optimizer_data["parameters"], source=ParameterMessage.source_autologger
            )

    def stop_early(self, epoch: int) -> bool:
        """Used to programmatically stop an experiment early.

        Args:
            epoch: The epoch number at early stop.

        Example:
            ```python linenums="1"
            import comet_ml
            import numpy as np
            import tensorflow as tf
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import OneHotEncoder
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.losses import CategoricalCrossentropy

            # Initialize Comet experiment
            comet_ml.init(project_name="comet-docs")
            exp = comet_ml.start()

            # Set up simple neural network example
            data = load_iris()
            X = data.data
            y = data.target.reshape(-1, 1)

            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(y)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model = Sequential([
                Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(3, activation='softmax')
            ])

            model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

            # Train model with manual early stopping
            for epoch in range(100):
                history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val))
                val_loss = history.history['val_loss'][0]
                exp.log_metric("validation_loss", val_loss, epoch=epoch)

                # Check for early stopping condition
                if val_loss < 0.5:  # Example threshold for early stopping
                    print(f"Stopping early at epoch {epoch} due to reaching loss threshold.")
                    exp.stop_early(epoch=epoch)
                    break
            ```
        """
        return False

    def get_callback(self, framework: str, *args: Any, **kwargs: Any) -> Callable:
        """Get a callback for a particular framework.

        Args:
            framework (str): Specifies the machine learning framework for which you want a callback.

        Returns:
            callable: Callback object that is compatible with the framework specified in the call.

        Note:
            When framework == 'keras' then return an instance of
            Comet.ml's Keras callback.

            When framework == 'tf-keras' then return an instance of
            Comet.ml's TensorflowKeras callback.

            When framework == "tf-estimator-train" then return an instance
            of Comet.ml's Tensorflow Estimator Train callback.

            When framework == "xgboost" then return an instance of
            Comet.ml's xgboost.callback.TrainingCallback implementation

            When framework == "fastai" then return an instance of
            Comet.ml's fastai.learner.Callback implementation.

            The keras callbacks are added to your Keras `model.fit()`
            callbacks list automatically to report model training metrics
            to Comet.ml so you do not need to add them manually.

            The lightgbm callback is added to the `lightgbm.train()`
            callbacks list automatically to report model training metrics
            to Comet.ml so you do not need to add it manually.

        Example:
            ```python linenums="1"
            import comet_ml
            import tensorflow as tf
            from tensorflow import keras

            # Initialize the Comet experiment with your API key and project information
            comet_ml.login()
            exp = comet_ml.start()

            # Load a dataset (using MNIST as an example)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            # Build a simple model
            model = keras.models.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10)
            ])

            # Compile the model
            model.compile(optimizer='adam',
                        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            # Get the Comet ML callback for Keras
            comet_callback = exp.get_callback("keras")

            # Train the model
            model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[comet_callback])

            # End the experiment
            exp.end()
            ```
        """
        if framework in ["keras", "tf-keras", "tensorflow-keras"]:
            if framework == "keras":
                from .callbacks._keras import (
                    CometEmptyKerasCallback,
                    CometKerasCallback,
                )
            elif framework == "tf-keras":
                from .callbacks._tensorflow_keras import (
                    CometEmptyKerasCallback,
                    CometKerasCallback,
                )

            if self.alive:
                return CometKerasCallback(self, **kwargs)
            else:
                return CometEmptyKerasCallback()

        elif framework == "lightgbm":
            from .callbacks._lgbm import CometLGBMCallback

            return CometLGBMCallback(self)

        elif framework in ["tf-estimator-train", "tensorflow-estimator-train"]:
            from .callbacks._tensorflow_estimator import (
                CometTensorflowEstimatorTrainSessionHook,
            )

            return CometTensorflowEstimatorTrainSessionHook(self)

        elif framework == "xgboost":
            from .callbacks._xgboost import XGBoostCometCallback

            return XGBoostCometCallback(self)

        elif framework == "fastai":
            from .callbacks._fastai import CometFastAICallback

            return CometFastAICallback(self)

        else:
            raise NotImplementedError(
                "No such framework for callback: `%s`" % framework
            )

    def send_notification(
        self,
        title: str,
        status: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send yourself a notification through email when an experiment ends.

        This is only supported for online experiment and for newly created Experiments.

        Args:
            title (str): The email subject.
            status (str): The final status of the experiment. Typically,
                something like "finished", "completed" or "aborted".
            additional_data (dict[str, Any]): A dictionary of key/values to notify.

        Note:
            In order to receive the notification, you need to have turned
            on Notifications in your Settings in the Comet user interface.

            If you wish to have the `additional_data` saved with the
            experiment, you should also call `Experiment.log_other()` with
            this data as well.

            This method uses the email address associated with your account.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.send_notification(
                "Experiment %s" % experiment.get_key(),
                "started"
            )

            # Setup additional data to send
            additional_data = {
                "Accuracy": "90%",
                "Loss": "0.5"
            }

            try:
                train(...)
                exp.send_notification(
                    "Experiment %s" % experiment.get_key(),
                    status="completed successfully",
                    additional_data=additional_data
                )
            except Exception:
                exp.send_notification(
                    "Experiment %s" % experiment.get_key(),
                    "failed"
                )
            ```
        """
        pass

    def create_symlink(self, project_name: str) -> None:
        """
        Creates a symlink for this experiment in another project.
        The experiment will now be displayed in the project provided and the original project.
        This is only supported for online experiment at the moment.

        This is only supported for online experiment at the moment.

        Args:
            project_name (str): Represents the project name. Project must exist.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.
        """
        raise NotImplementedError(CREATE_SYMLINK_ONLINE_ONLY_ERROR)

    def set_offline_zip_uploader(self, upload_callback: UploadCallback) -> None:
        """
        This method allows you to specify a callback function that will be
        invoked to upload the offline Experiment archive when it ends or the
        online Experiment archive if it fallbacked to an offline experiment
        due to connectivity issue. The callback function receives the file
        path to the offline ZIP archive and perform the upload before the job
        ends.

        We also provide a callback to upload to S3 directly, see:
        [comet_ml.get_s3_uploader][].

        Args:
            upload_callback (UploadCallback): A user-defined function that
                takes a single argument, `file_path` (str), which is the path
                to the offline ZIP archive. The function should handle the
                upload process to the desired location.


        Example:
            ```python
            def custom_uploader(file_path: str) -> None:
                # Implement your upload logic here
                print(f"Uploading {file_path} to the cloud storage.")
                # Upload logic goes here

            experiment.set_offline_zip_uploader(custom_uploader)
            ```

        Notes:
            - The callback function is invoked at the end of the job if an
              offline fallback occurred after a connectivity issue and the SDK
              couldn't recover from it.
            - The callback function is responsible for handling any errors
              that occur during the upload process.
            - If the callback function fails to upload the ZIP archive, an
              error log message will be printed without retrying the upload.
        """
        pass

    # Internal methods
    def _track_framework_usage(self, framework: Optional[str]) -> None:
        """Track which framework logged data, framework could be internal auto-loggers or external
        integrations.
        """

        if framework is None:
            return

        if framework == "comet":
            return

        self._frameworks.add(framework)


BaseExperiment = CometExperiment


def flush() -> Optional[bool]:
    """
    If there is a running experiment, flush all pending data to the Comet server.
    See [flush][comet_ml.Experiment.flush] for more details. If no experiment
    was running, this does nothing.

    Returns:
        If there is a running experiment, returns a Boolean True or False
            indicating whether the flush was successful or not. If no experiment
            was running, returns None.
    """
    experiment = get_running_experiment()
    if experiment is not None:
        return experiment.flush()

    return None


def end() -> None:
    """
    If there is a running experiment, finalize it and flush all pending data
    to the Comet server. See [end][comet_ml.Experiment.end] for more details. If no experiment
    was running, this does nothing.
    """
    experiment = get_running_experiment()
    if experiment is not None:
        return experiment.end()
