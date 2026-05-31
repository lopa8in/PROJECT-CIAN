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

"""
Author: Boris Feld

This module contains the code related to offline feature

"""
import io
import json
import logging
import os
import os.path
import shutil
import tempfile
import traceback
import zipfile
from os.path import join
from typing import Callable, Union
from zipfile import ZipFile

from jsonschema import ValidationError

from ._reporting import (
    OFFLINE_INVALID_3D_CLOUD_MESSAGE,
    OFFLINE_INVALID_CLOUD_DETAILS_MSG,
    OFFLINE_INVALID_FILE_NAME_MSG,
    OFFLINE_INVALID_GIT_METADATA_MSG,
    OFFLINE_INVALID_GPU_STATIC_INFO_MSG,
    OFFLINE_INVALID_GRAPH_MSG,
    OFFLINE_INVALID_HTML_MSG,
    OFFLINE_INVALID_HTML_OVERRIDE_MSG,
    OFFLINE_INVALID_INSTALLED_PACKAGES_MSG,
    OFFLINE_INVALID_LOG_DEPENDENCY_MESSAGE,
    OFFLINE_INVALID_LOG_OTHER_MSG,
    OFFLINE_INVALID_METRIC_MSG,
    OFFLINE_INVALID_OS_PACKAGES_MSG,
    OFFLINE_INVALID_PARAMETER_MSG,
    OFFLINE_INVALID_REGISTER_MODEL_MESSAGE,
    OFFLINE_INVALID_REMOTE_MODEL_MESSAGE,
    OFFLINE_INVALID_STANDARD_OUTPUT_MESSAGE,
    OFFLINE_INVALID_SYSTEM_DETAILS_MSG,
    OFFLINE_INVALID_SYSTEM_INFO_MSG,
    OFFLINE_INVALID_UPLOAD_MSG,
    OFFLINE_INVALID_WS_MSG,
)
from ._typing import Any, Dict, List, Optional, Tuple
from .api_helpers.experiment_key import get_experiment_key
from .assets import asset_item
from .batch_utils import MessageBatch, MessageBatchItem, ParametersBatch
from .comet_start.start_modes import SUPPORTED_START_MODES
from .config import (
    ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
    DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    OFFLINE_EXPERIMENT_JSON_FILE_NAME,
    OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME,
    get_api_key,
    get_backend_address,
    get_check_tls_certificate,
    get_config,
)
from .connection import RestApiClient, RestServerConnection
from .connection.connection_factory import (
    get_rest_api_client,
    get_rest_server_connection,
)
from .constants import (
    ASSET_TYPE_3D_POINTS,
    ASSET_TYPE_EMBEDDINGS,
    ASSET_TYPE_MODEL_ELEMENT,
    DEPRECATED_OFFLINE_MODE_CREATE,
    DEPRECATED_OFFLINE_MODE_TO_RESUME_STRATEGY_MAP,
    RESUME_STRATEGY_CREATE,
    RESUME_STRATEGY_GET,
    RESUME_STRATEGY_GET_OR_CREATE,
)
from .data_structure import Embedding
from .exceptions import (
    BackendCustomError,
    CometRestApiException,
    ExperimentAlreadyUploaded,
    ExperimentNotFound,
    InvalidAPIKey,
    InvalidExperimentMode,
    InvalidExperimentModeUnsupported,
    OfflineExperimentUploadFailed,
    ProjectConsideredLLM,
)
from .experiment import CometExperiment
from .feature_toggles import FeatureToggles
from .file_upload_manager import FileUploadManager, FileUploadManagerMonitor
from .file_utils import make_template_filename
from .handshake import ExperimentHandshakeResponse
from .logging_messages import (
    CLOUD_DETAILS_MSG_SENDING_ERROR,
    CREATE_SYMLINK_ONLINE_ONLY_ERROR,
    DIRECT_S3_UPLOAD_DISABLED,
    DIRECT_S3_UPLOAD_ENABLED,
    FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
    FILENAME_DETAILS_MSG_SENDING_ERROR,
    GIT_METADATA_MSG_SENDING_ERROR,
    GPU_STATIC_INFO_MSG_SENDING_ERROR,
    HTML_MSG_SENDING_ERROR,
    HTML_OVERRIDE_MSG_SENDING_ERROR,
    INSTALLED_PACKAGES_MSG_SENDING_ERROR,
    LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
    LOG_EMBEDDING_EXPERIMENTAL_WARNING,
    LOG_OTHER_MSG_SENDING_ERROR,
    METRICS_BATCH_MSG_SENDING_ERROR,
    MODEL_GRAPH_MSG_SENDING_ERROR,
    OFFLINE_AT_LEAST_ONE_EXPERIMENT_UPLOAD_FAILED,
    OFFLINE_DISPLAY_METHOD_ERROR,
    OFFLINE_DISPLAY_PROJECT_METHOD_ERROR,
    OFFLINE_EXPERIMENT_ALREADY_EXISTS_CREATE_MODE,
    OFFLINE_EXPERIMENT_ALREADY_UPLOADED,
    OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_CONFIG,
    OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_PARAMETER,
    OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_CONFIG,
    OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_PARAMETER,
    OFFLINE_EXPERIMENT_END,
    OFFLINE_EXPERIMENT_INVALID_3D_CLOUD_MESSAGE,
    OFFLINE_EXPERIMENT_INVALID_CLOUD_DETAILS_MSG,
    OFFLINE_EXPERIMENT_INVALID_FILE_NAME_MSG,
    OFFLINE_EXPERIMENT_INVALID_GIT_METADATA_MSG,
    OFFLINE_EXPERIMENT_INVALID_GPU_STATIC_INFO_MSG,
    OFFLINE_EXPERIMENT_INVALID_GRAPH_MSG,
    OFFLINE_EXPERIMENT_INVALID_HTML_MSG,
    OFFLINE_EXPERIMENT_INVALID_HTML_OVERRIDE_MSG,
    OFFLINE_EXPERIMENT_INVALID_INSTALLED_PACKAGES_MSG,
    OFFLINE_EXPERIMENT_INVALID_LOG_DEPENDENCY_MESSAGE,
    OFFLINE_EXPERIMENT_INVALID_LOG_OTHER_MSG,
    OFFLINE_EXPERIMENT_INVALID_METRIC_MSG,
    OFFLINE_EXPERIMENT_INVALID_OS_PACKAGES_MSG,
    OFFLINE_EXPERIMENT_INVALID_PARAMETER_MSG,
    OFFLINE_EXPERIMENT_INVALID_REGISTER_MODEL_MESSAGE,
    OFFLINE_EXPERIMENT_INVALID_REMOTE_MODEL_MESSAGE,
    OFFLINE_EXPERIMENT_INVALID_STANDARD_OUTPUT_MESSAGE,
    OFFLINE_EXPERIMENT_INVALID_SYSTEM_DETAILS_MSG,
    OFFLINE_EXPERIMENT_INVALID_SYSTEM_INFO_MSG,
    OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG,
    OFFLINE_EXPERIMENT_INVALID_WS_MSG,
    OFFLINE_EXPERIMENT_NAME_ACCESS,
    OFFLINE_EXPERIMENT_NOT_FOUND_GET_MODE,
    OFFLINE_EXPERIMENT_SAVE_ARCHIVE_INFO,
    OFFLINE_EXPERIMENT_UPLOAD_ZIP_CALLBACK_ERROR,
    OFFLINE_EXPERIMENT_UPLOAD_ZIP_COMPLETED_INFO,
    OFFLINE_EXPERIMENT_UPLOAD_ZIP_STARTED_INFO,
    OFFLINE_FAILED_TO_REGISTER_MODEL_NO_MODEL_FILES,
    OFFLINE_FAILED_UPLOADED_EXPERIMENTS,
    OFFLINE_GET_ARTIFACT_METHOD_ERROR,
    OFFLINE_LOG_ARTIFACT_METHOD_ERROR,
    OFFLINE_SENDER_ENDS,
    OFFLINE_SENDER_ENDS_PROCESSING,
    OFFLINE_SENDER_STARTS,
    OFFLINE_SUCCESS_UPLOADED_EXPERIMENTS,
    OFFLINE_UPLOAD_FAILED_UNEXPECTED_ERROR,
    OFFLINE_UPLOADING_EXPERIMENT_FILE_PROMPT,
    OFFLINE_UPLOADS_FAILED_DUE_TIMEOUT,
    OS_PACKAGE_MSG_SENDING_ERROR,
    PARAMETERS_BATCH_MSG_SENDING_ERROR,
    REGISTER_CALLBACK_ONLINE_ONLY_ERROR,
    REGISTER_MODEL_MESSAGE_SENDING_ERROR,
    REMOTE_MODEL_MESSAGE_SENDING_ERROR,
    STANDARD_OUTPUT_SENDING_ERROR,
    SYSTEM_DETAILS_MSG_SENDING_ERROR,
    SYSTEM_INFO_MESSAGE_SENDING_ERROR,
    UNEXPECTED_OFFLINE_PROCESS_MESSAGE_ERROR,
    UPLOADING_DATA_BEFORE_TERMINATION,
    WAITING_FOR_FILE_UPLOADS_COMPLETION,
)
from .messages import (
    CloudDetailsMessage,
    FileNameMessage,
    GitMetadataMessage,
    GpuStaticInfoMessage,
    HtmlMessage,
    HtmlOverrideMessage,
    InstalledPackagesMessage,
    Log3DCloudMessage,
    LogDependencyMessage,
    LogOtherMessage,
    MetricMessage,
    ModelGraphMessage,
    OsPackagesMessage,
    ParameterMessage,
    RegisterModelMessage,
    RemoteAssetMessage,
    RemoteModelMessage,
    StandardOutputMessage,
    SystemDetailsMessage,
    SystemInfoMessage,
    UploadFileMessage,
    WebSocketMessage,
)
from .metrics import MetricsSampler
from .offline_utils import (
    create_experiment_archive,
    create_offline_archive,
    extract_experiment_key_from_offline_file,
    get_offline_data_dir_path,
    write_experiment_meta_file,
)
from .s3.multipart_upload.multipart_upload_options import MultipartUploadOptions
from .schemas import (
    get_cloud_3d_msg_validator,
    get_cloud_details_msg_validator,
    get_experiment_file_validator,
    get_file_name_msg_validator,
    get_git_metadata_msg_validator,
    get_gpu_static_info_msg_validator,
    get_graph_msg_validator,
    get_html_msg_validator,
    get_html_override_msg_validator,
    get_installed_packages_msg_validator,
    get_log_dependency_msg_validator,
    get_log_other_msg_validator,
    get_metric_msg_validator,
    get_os_packages_msg_validator,
    get_parameter_msg_validator,
    get_register_model_msg_validator,
    get_remote_file_msg_validator,
    get_remote_model_msg_validator,
    get_standard_output_msg_validator,
    get_system_details_msg_validator,
    get_system_info_msg_validator,
    get_upload_msg_validator,
    get_ws_msg_validator,
)
from .streamer import OfflineStreamer
from .upload_callback.callback import UploadCallback
from .upload_options import (
    AssetItemUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
    ThumbnailUploadOptions,
)
from .utils import local_timestamp, random_ascii_string, wait_for_done

LOGGER = logging.getLogger(__name__)
UNUSED_INT = 0


class OfflineExperiment(CometExperiment):
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
        offline_directory: Optional[str] = None,
        log_env_gpu: Optional[bool] = True,
        log_env_host: Optional[bool] = True,
        api_key: Optional[str] = None,
        display_summary: Optional[bool] = None,
        log_env_cpu: Optional[bool] = True,
        log_env_network: Optional[bool] = True,
        log_env_disk: Optional[bool] = True,
        display_summary_level: Optional[int] = None,
        auto_weight_logging: Optional[bool] = None,
        auto_log_co2: Optional[bool] = False,
        auto_metric_step_rate: Optional[int] = 10,
        auto_histogram_tensorboard_logging: Optional[bool] = False,
        auto_histogram_epoch_rate: Optional[int] = 1,
        auto_histogram_weight_logging: Optional[bool] = False,
        auto_histogram_gradient_logging: Optional[bool] = False,
        auto_histogram_activation_logging: Optional[bool] = False,
        experiment_key: Optional[str] = None,
        distributed_node_identifier: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Creates a new experiment and serialize it on disk. The experiment file will need to be
        uploaded manually later to appear on the frontend.

        Args:
            project_name (str): Send your experiment to a specific project.
                            Otherwise, will be sent to `Uncategorized Experiments`.
                            If project name does not already exist Comet.ml will create a new project.
            workspace (str): Attach an experiment to a project that belongs to this workspace
            log_code (bool): Allows you to enable/disable code logging
            log_graph (bool): Allows you to enable/disable automatic computation graph logging.
            auto_param_logging (bool): Allows you to enable/disable hyper-parameters logging
            auto_metric_logging (bool): Allows you to enable/disable metrics logging
            auto_metric_step_rate (int): Controls how often batch metrics are logged
            auto_histogram_tensorboard_logging (bool): Allows you to enable/disable automatic histogram logging
            auto_histogram_epoch_rate (int): Controls how often histograms are logged
            auto_histogram_weight_logging (bool): Allows you to enable/disable automatic histogram logging of biases and weights
            auto_histogram_gradient_logging (bool): Allows you to enable/disable automatic histogram logging of gradients
            auto_histogram_activation_logging (bool): Allows you to enable/disable automatic histogram logging of activations
            auto_output_logging (str): Allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"simple"`.
            auto_log_co2 (bool): Automatically tracks the CO2 emission of
                this experiment if `codecarbon` package is installed in the environment
            parse_args (bool): Allows you to enable/disable automatic parsing of CLI arguments
            log_env_details (bool): Log various environment
                information in order to identify where the script is running
            log_env_gpu (bool): Allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu (bool): Allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_network (bool): Allow you to enable/disable the
                automatic collection of network details and metrics (sent, receive rates, etc..).
                `log_env_details` must also be true.
            log_env_disk (bool): Allow you to enable/disable the
                automatic collection of disk utilization metrics (usage, IO rates, etc.).
                `log_env_details` must also be true.
            log_env_host (bool): Allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata (bool): Allow you to enable/disable the
                automatic collection of git details
            display_summary_level (int): Control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled (bool): Allows you to disable all network
                communication with the Comet.ml backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
            offline_directory (str): The directory used to save the offline archive
                for the experiment.
            experiment_key (str): If provided, will be used as the experiment key. If an experiment
                with the same key already exists, it will raise an Exception during upload. Could be set
                through configuration as well. Must be an alphanumeric string whose length is between 32 and 50 characters.
        """
        self.config = get_config()

        self.api_key = get_api_key(
            api_key, self.config
        )  # optional, except for on-line operations

        # Start and ends time
        self.start_time = None
        self.stop_time = None

        self.resume_strategy = kwargs.pop("_resume_strategy", None)
        if self.resume_strategy is None:
            self.resume_strategy = RESUME_STRATEGY_CREATE
        self.user_provided_experiment_key = experiment_key is not None
        self.comet_start_sourced = kwargs.pop("_comet_start_sourced", False)

        self.customer_error_reported = False
        self.customer_error_message: Optional[str] = None

        super().__init__(
            project_name=project_name,
            workspace=workspace,
            log_code=log_code,
            log_graph=log_graph,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            auto_output_logging=auto_output_logging,
            log_env_details=log_env_details,
            log_git_metadata=log_git_metadata,
            log_git_patch=log_git_patch,
            disabled=disabled,
            log_env_gpu=log_env_gpu,
            log_env_host=log_env_host,
            display_summary=display_summary,
            display_summary_level=display_summary_level,
            log_env_cpu=log_env_cpu,
            log_env_network=log_env_network,
            log_env_disk=log_env_disk,
            auto_weight_logging=auto_weight_logging,
            auto_log_co2=auto_log_co2,
            auto_metric_step_rate=auto_metric_step_rate,
            auto_histogram_epoch_rate=auto_histogram_epoch_rate,
            auto_histogram_tensorboard_logging=auto_histogram_tensorboard_logging,
            auto_histogram_weight_logging=auto_histogram_weight_logging,
            auto_histogram_gradient_logging=auto_histogram_gradient_logging,
            auto_histogram_activation_logging=auto_histogram_activation_logging,
            experiment_key=experiment_key,
            distributed_node_identifier=distributed_node_identifier,
        )

        self.offline_archive_name_suffix = None

        self.offline_directory, default_dir_used = self._get_offline_data_dir_path(
            offline_directory
        )

        self.offline_zip_uploader: Optional[UploadCallback] = None

        if self.disabled is False:
            # Check that the offline directory is usable
            # Try to create ZIP file for the experiment
            zip_file, self.offline_directory = self._create_offline_archive(
                fallback_to_temp=default_dir_used
            )
            # Close the file handle, it will be reopened later
            zip_file.close()

            if api_key is not None:
                self._log_once_at_level(
                    logging.WARNING,
                    "api_key was given, but is ignored in offline experiment; remember to set when you upload",
                )
            elif self.api_key is not None:
                self._log_once_at_level(
                    logging.INFO,
                    "COMET_API_KEY was set, but is ignored in offline experiment; remember to set when you upload",
                )

            self._start()

    def set_offline_zip_uploader(self, upload_callback: UploadCallback) -> None:
        """
        This method allows you to specify a callback function that will be
        invoked to upload the offline ZIP archive created if a connectivity
        issue occurs. The callback function receives the file path to the
        offline ZIP archive and perform the upload before the job ends.

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
        self.offline_zip_uploader = upload_callback

    def _create_offline_archive(
        self, fallback_to_temp: bool = True
    ) -> Tuple[ZipFile, str]:
        return create_offline_archive(
            offline_directory=self.offline_directory,
            offline_archive_file_name=self._get_offline_archive_file_name(),
            fallback_to_temp=fallback_to_temp,
            logger=LOGGER,
        )

    def _get_offline_data_dir_path(
        self, offline_directory: Optional[str]
    ) -> Tuple[str, bool]:
        return get_offline_data_dir_path(
            comet_config=self.config,
            offline_directory=offline_directory,
            logger=LOGGER,
        )

    def _get_offline_archive_file_name(self) -> str:
        """Return the offline archive file name, used for creating it on the file-system."""
        if self.resume_strategy == RESUME_STRATEGY_CREATE:
            return "%s.zip" % self.id

        # create unique file name if resume strategy is not to create
        if self.offline_archive_name_suffix is None:
            self.offline_archive_name_suffix = random_ascii_string()

        return "%s-%s.zip" % (self.id, self.offline_archive_name_suffix)

    def display(self, *args: Any, **kwargs: Any) -> None:
        """
        Show the Comet experiment page in an IFrame in a
        Jupyter notebook or Jupyter lab, OR open a browser
        window or tab.

        This is only supported for online experiment at the moment.

        Args:
            clear (bool): To clear the output area, use clear=True. This is only
                used in Notebook environments.
            wait (bool): To wait for the next displayed item, use
                `wait=True` (cuts down on flashing). This is only used in Notebook
                environments.
            new (int): Open a new browser window if `new=1`, otherwise re-use
                existing window/tab. This is only used in non-Notebook
                environments.
            autoraise (bool): Make the browser tab/window active. This is only
                used in non-Notebook environments.
            tab (str): Name of the Tab on Experiment View

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Note:
            The Tab name should be one of:

            * "artifacts"
            * "assets"
            * "audio"
            * "charts"
            * "code"
            * "confusion-matrices"
            * "histograms"
            * "images"
            * "installed-packages"
            * "metrics"
            * "notes"
            * "parameters"
            * "system-metrics"
            * "text"
        """
        raise NotImplementedError(OFFLINE_DISPLAY_METHOD_ERROR)

    def display_project(self, *args: Any, **kwargs: Any) -> None:
        """Show the Comet project page in an IFrame in either (1) a Jupyter
        notebook or Jupyter lab or (2) open a browser window or tab.

        This is only supported for online experiment at the moment.

        Args:
            view_id (str): The id of the view to show.
            clear (bool): To clear the output area, use `clear=True.`
            wait (bool): To wait for the next displayed item, use
                wait=True (cuts down on flashing).
            new (int): Open a new browser window if `new=1`, otherwise
                re-use existing window/tab.
            autoraise (bool): Make the browser tab/window active.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.

        Note:
            For Jupyter environments, you can utilize the `clear` and `wait` parameters.
            For non-Jupyter environments, you can utilize the `new` and `autoraise` parameters.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            # Optionally log some metrics or parameters (not required for displaying the project)
            exp.log_metric("accuracy", 0.95)
            exp.log_parameter("learning_rate", 0.01)

            exp.display_project()

            exp.end()
            ```
        """
        raise NotImplementedError(OFFLINE_DISPLAY_PROJECT_METHOD_ERROR)

    def send_notification(
        self,
        title: str,
        status: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send yourself a notification through email when an experiment ends.

        This is only supported for online experiment.

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
        # Temporary commented because it is being shown automatically
        # on the end of every offline experiment
        # LOGGER.error(OFFLINE_SEND_NOTIFICATION_METHOD_ERROR)
        pass

    def log_embedding(
        self,
        vectors,
        labels,
        image_data=None,
        image_size=None,
        image_preprocess_function=None,
        image_transparent_color=None,
        image_background_color_function=None,
        title=Embedding.DEFAULT_TITLE,
        template_filename=None,
        group=None,
    ):
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
        LOGGER.warning(LOG_EMBEDDING_EXPERIMENTAL_WARNING)

        embedding = self._create_embedding(
            vectors,
            labels,
            image_data,
            image_size,
            image_preprocess_function,
            image_transparent_color,
            image_background_color_function,
            title,
        )

        if embedding is None:
            return None

        if group is not None:
            self._embedding_groups[group].append(embedding)
            return embedding
        else:
            # Log the template:
            template = {ASSET_TYPE_EMBEDDINGS: [embedding.to_json()]}
            if template_filename is None:
                template_filename = make_template_filename()

            return self._log_asset_data(
                template, template_filename, asset_type=ASSET_TYPE_EMBEDDINGS
            )

    def log_artifact(self, artifact):
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
        raise NotImplementedError(OFFLINE_LOG_ARTIFACT_METHOD_ERROR)

    def get_artifact(self, *args, **kwargs):
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
        raise NotImplementedError(OFFLINE_GET_ARTIFACT_METHOD_ERROR)

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

    def register_callback(self, function: Callable) -> None:
        """
        Register the function passed as argument to be an RPC.

        This is only supported for online experiment at the moment.

        Args:
            function (callable): Function to register as a callback.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.
        """
        raise NotImplementedError(REGISTER_CALLBACK_ONLINE_ONLY_ERROR)

    def _start(self, **kwargs):
        self.start_time = local_timestamp()
        super()._start(**kwargs)
        self._log_other("offline_experiment", True, include_context=False)

    def _write_experiment_meta_file(self):
        write_experiment_meta_file(
            tempdir=self.tmpdir,
            experiment_key=self.id,
            workspace=self.workspace,
            project_name=self.project_name,
            start_time=self.start_time,
            stop_time=self.stop_time,
            tags=self.get_tags(),
            resume_strategy=self.resume_strategy,
            customer_error_reported=self.customer_error_reported,
            customer_error_message=self.customer_error_message,
            user_provided_experiment_key=self.user_provided_experiment_key,
            comet_start_sourced=self.comet_start_sourced,
        )

    def _mark_as_ended(self):
        if not self.alive:
            LOGGER.debug("Skipping creating the offline archive as we are not alive")
            return

        LOGGER.info(OFFLINE_EXPERIMENT_SAVE_ARCHIVE_INFO)
        self.stop_time = local_timestamp()

        self._write_experiment_meta_file()

        zip_file_filename, self.offline_directory = create_experiment_archive(
            offline_directory=self.offline_directory,
            offline_archive_file_name=self._get_offline_archive_file_name(),
            data_dir=self.tmpdir,
            logger=LOGGER,
        )

        # Clean the tmpdir to avoid filling up the disk
        try:
            shutil.rmtree(self.tmpdir)
        except OSError:
            # We made our best effort to clean ourselves
            LOGGER.debug(
                "Error cleaning offline experiment's temporary directory: %r",
                self.tmpdir,
                exc_info=True,
            )

        if self.offline_zip_uploader is not None:
            LOGGER.info(OFFLINE_EXPERIMENT_UPLOAD_ZIP_STARTED_INFO, zip_file_filename)
            try:
                self.offline_zip_uploader(zip_file_filename)
                LOGGER.info(OFFLINE_EXPERIMENT_UPLOAD_ZIP_COMPLETED_INFO)
            except Exception:
                LOGGER.error(
                    OFFLINE_EXPERIMENT_UPLOAD_ZIP_CALLBACK_ERROR, exc_info=True
                )

        # Display the full command to upload the offline experiment
        LOGGER.info(OFFLINE_EXPERIMENT_END, zip_file_filename)

    def _report_experiment_error(self, message, has_crashed: bool = False):
        self.customer_error_reported = True
        self.customer_error_message = message

    def _setup_streamer(self) -> bool:
        """
        Initialize the streamer and feature flags.
        """
        # init feature toggles first, thus configuration will be applied before everything else
        self.feature_toggles = FeatureToggles({}, self.config)

        # Initiate the streamer
        self.streamer = OfflineStreamer(
            tmp_dir=self.tmpdir,
            wait_timeout=60,
            on_error_callback=self._report_experiment_error,
            experiment_key=self.get_key(),
        )

        # Start streamer thread.
        self.streamer.start()

        # Mark the experiment as alive
        return True

    def _report(self, *args, **kwrags):
        # TODO WHAT TO DO WITH REPORTING?
        pass

    def _get_experiment_url(self, tab=None) -> str:
        return "[OfflineExperiment will get URL after upload]"

    def get_name(self) -> str:
        """
        Get the name of the experiment, if one.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            exp = comet_ml.start()

            exp.set_name("My Name")

            print(exp.get_name())
            ```
        """
        if self.name is None:
            LOGGER.warning(OFFLINE_EXPERIMENT_NAME_ACCESS)
        return self.name


class ExistingOfflineExperiment(OfflineExperiment):
    def __init__(
        self,
        project_name=None,  # type: Optional[str]
        workspace=None,  # type: Optional[str]
        log_code=True,  # type: Optional[bool]
        log_graph=True,  # type: Optional[bool]
        auto_param_logging=True,  # type: Optional[bool]
        auto_metric_logging=True,  # type: Optional[bool]
        parse_args=True,  # type: Optional[bool]
        auto_output_logging=AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,  # type: Optional[str]
        log_env_details=True,  # type: Optional[bool]
        log_git_metadata=True,  # type: Optional[bool]
        log_git_patch=True,  # type: Optional[bool]
        disabled=False,  # type: Optional[bool]
        offline_directory=None,  # type: Optional[str]
        log_env_gpu=True,  # type: Optional[bool]
        log_env_host=True,  # type: Optional[bool]
        api_key=None,  # type: Optional[str]
        display_summary=None,  # type: Optional[bool]
        log_env_cpu=True,  # type: Optional[bool]
        log_env_disk=False,  # type: Optional[bool]
        display_summary_level=None,  # type: Optional[int]
        auto_weight_logging=False,  # type: Optional[bool]
        previous_experiment=None,  # type: Optional[str]
        experiment_key=None,  # type: Optional[str]
    ):
        # type: (...) -> None
        """
        Continue a previous experiment (identified by previous_experiment) and serialize it on disk.
        The experiment file will need to be uploaded manually later to append new information to the
        previous experiment. The previous experiment need to exist before upload of the
        ExistingOfflineExperiment.

        Args:
            experiment_key (str): Your experiment key from comet.com, could be set through
                configuration as well.
            previous_experiment (str): Deprecated. Use `experiment_key` instead.
            project_name (str): Send your experiment to a specific project.
                            Otherwise, will be sent to `Uncategorized Experiments`.
                            If project name does not already exist Comet.ml will create a new project.
            workspace (str):Attach an experiment to a project that belongs to this workspace
            log_code (bool): Allows you to enable/disable code logging
            log_graph (bool): Allows you to enable/disable automatic computation graph logging.
            auto_param_logging (bool): Allows you to enable/disable hyper-parameters logging
            auto_metric_logging (bool): Allows you to enable/disable metrics logging
            parse_args (bool): Allows you to enable/disable automatic parsing of CLI arguments
            auto_output_logging (str): Allows you to select
                which output logging mode to use. You can pass `"native"`
                which will log all output even when it originated from a C
                native library. You can also pass `"simple"` which will work
                only for output made by Python code. If you want to disable
                automatic output logging, you can pass `False`. The default is
                `"simple"`.
            log_env_details (bool): Log various environment
                information in order to identify where the script is running
            log_env_gpu (bool): Allow you to enable/disable the
                automatic collection of gpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_cpu (bool): Allow you to enable/disable the
                automatic collection of cpu details and metrics (utilization, memory usage etc..).
                `log_env_details` must also be true.
            log_env_disk (bool): Allow you to enable/disable the
                automatic collection of disk utilization metrics (usage, IO rates, etc.).
                `log_env_details` must also be true.
            log_env_host (bool): Allow you to enable/disable the
                automatic collection of host information (ip, hostname, python version, user etc...).
                `log_env_details` must also be true.
            log_git_metadata (bool): Allow you to enable/disable the
                automatic collection of git details
            display_summary_level (int): Control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled (bool): Allows you to disable all network
                communication with the Comet backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
            offline_directory (str): The directory used to save the offline archive
                for the experiment.
        """
        self.config = get_config()

        if previous_experiment is not None and experiment_key is not None:
            # TODO: SHOW LOG MESSAGE?
            pass
        elif previous_experiment is not None:
            experiment_key = previous_experiment

        # Generate once the random string used when creating the offline archive on the file-system
        self.offline_archive_name_suffix = random_ascii_string()

        super().__init__(
            project_name=project_name,
            workspace=workspace,
            log_code=log_code,
            log_graph=log_graph,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            auto_output_logging=auto_output_logging,
            log_env_details=log_env_details,
            log_git_metadata=log_git_metadata,
            log_git_patch=log_git_patch,
            disabled=disabled,
            offline_directory=offline_directory,
            log_env_gpu=log_env_gpu,
            log_env_host=log_env_host,
            api_key=api_key,
            log_env_cpu=log_env_cpu,
            log_env_disk=log_env_disk,
            display_summary=display_summary,
            display_summary_level=display_summary_level,
            auto_weight_logging=auto_weight_logging,
            experiment_key=experiment_key,
            _resume_strategy=RESUME_STRATEGY_GET,
        )


class OfflineSender(object):
    def __init__(
        self,
        api_key: str,
        offline_dir: str,
        force_upload: bool = False,
        display_level: str = "info",
        raise_validation_error_for_tests: bool = False,
        file_upload_waiting_timeout: int = ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
        override_workspace: Optional[str] = None,
        override_project_name: Optional[str] = None,
        message_batch_metric_interval: int = MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
        message_batch_metric_max_size: int = MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
        message_batch_compress: int = MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
        wait_for_finish_sleep_interval: int = DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
        connection: Optional[RestServerConnection] = None,
        rest_api_client: Optional[RestApiClient] = None,
    ) -> None:
        self.config = get_config()
        self.api_key = api_key
        self.offline_dir = offline_dir
        self.force_upload = force_upload
        self.counter = 0
        self.display_level = logging.getLevelName(display_level.upper())

        # Validators
        self._init_message_validators()

        self.server_address = get_backend_address()

        self.override_workspace = override_workspace
        self.override_project_name = override_project_name

        self._read_experiment_file()

        self.check_tls_certificate = get_check_tls_certificate(self.config)

        if connection is None:
            self.connection = get_rest_server_connection(
                api_key=self.api_key,
                experiment_key=self.experiment_id,
                server_address=self.server_address,
                verify_tls=self.check_tls_certificate,
            )
        else:
            self.connection = connection

        self.rest_api_client = rest_api_client
        self.focus_link = None

        self.file_upload_waiting_timeout = file_upload_waiting_timeout
        self.wait_for_finish_sleep_interval = wait_for_finish_sleep_interval

        self.raise_validation_error_for_tests = raise_validation_error_for_tests

        self.message_batch_metrics = MessageBatch(
            base_interval=message_batch_metric_interval,
            max_size=message_batch_metric_max_size,
        )
        self.message_batch_compress = message_batch_compress

        # holds the names of models which has assets scheduled for upload
        self.upload_models_names = set()
        # holds collected register model messages
        self.register_model_messages = list()

        # Self._resuming will be on only if we append to an existing experiment
        self._resuming = False

    def _init_message_validators(self):
        self.experiment_file_validator = get_experiment_file_validator()
        self.ws_msg_validator = get_ws_msg_validator()
        self.parameter_msg_validator = get_parameter_msg_validator()
        self.metric_msg_validator = get_metric_msg_validator()
        self.os_packages_msg_validator = get_os_packages_msg_validator()
        self.graph_msg_validator = get_graph_msg_validator()
        self.system_details_msg_validator = get_system_details_msg_validator()
        self.cloud_details_msg_validator = get_cloud_details_msg_validator()
        self.upload_msg_validator = get_upload_msg_validator()
        self.remote_file_msg_validator = get_remote_file_msg_validator()
        self.log_other_message_validator = get_log_other_msg_validator()
        self.file_name_msg_validator = get_file_name_msg_validator()
        self.html_msg_validator = get_html_msg_validator()
        self.html_override_msg_validator = get_html_override_msg_validator()
        self.installed_packages_validator = get_installed_packages_msg_validator()
        self.gpu_static_info_msg_validator = get_gpu_static_info_msg_validator()
        self.git_metadata_msg_validator = get_git_metadata_msg_validator()
        self.system_info_msg_validator = get_system_info_msg_validator()
        self.standard_output_msg_validator = get_standard_output_msg_validator()
        self.log_dependency_msg_validator = get_log_dependency_msg_validator()
        self.remote_model_msg_validator = get_remote_model_msg_validator()
        self.cloud_3d_msg_validator = get_cloud_3d_msg_validator()
        self.register_model_msg_validator = get_register_model_msg_validator()

    def send(self):
        self._handshake()

        LOGGER.debug(
            "Offline sender started. REST API url: %r, clientlib url: %r, "
            "metric batch size: %d, metrics batch interval: %d seconds",
            self.rest_api_client.base_url,
            self.connection.server_address,
            self.message_batch_metrics.max_size,
            self.message_batch_metrics.base_interval,
        )

        self._status_report_start()

        if self.customer_error_reported:
            self._report_experiment_error(self.customer_error_message)

        LOGGER.log(self.display_level, OFFLINE_SENDER_STARTS)

        self._send_messages()

        self._status_report_end()

        self._send_start_ends_time()

    def _init_file_upload_manager(self):
        # check backend version
        direct_s3_upload_enabled = self.config.has_direct_s3_file_upload_enabled()
        if not direct_s3_upload_enabled:
            LOGGER.debug(DIRECT_S3_UPLOAD_DISABLED)
        else:
            LOGGER.debug(DIRECT_S3_UPLOAD_ENABLED)

        self.file_upload_manager = FileUploadManager(
            worker_cpu_ratio=self.config.get_int(
                None, "comet.internal.file_upload_worker_ratio"
            ),
            worker_count=self.config.get_raw(None, "comet.internal.worker_count"),
            s3_upload_options=MultipartUploadOptions(
                file_size_threshold=self.config.get_int(
                    None, "comet.s3_multipart.size_threshold"
                ),
                upload_expires_in=self.config.get_int(
                    None, "comet.s3_multipart.expires_in"
                ),
                direct_s3_upload_enabled=self.config.has_direct_s3_file_upload_enabled(),
            ),
        )

    def _read_experiment_file(self):
        with io.open(
            join(self.offline_dir, OFFLINE_EXPERIMENT_JSON_FILE_NAME), encoding="utf-8"
        ) as experiment_file:
            self.metadata = json.load(experiment_file)

        self.experiment_file_validator.validate(self.metadata)

        self.experiment_id = self.metadata.get("offline_id")

        # Offline experiments created with old versions of the SDK will be
        # missing this field, so generate a new one if that's the case
        if not self.experiment_id:
            self.experiment_id = get_experiment_key(user_input=None)

        self.start_time = self.metadata["start_time"]
        self.stop_time = self.metadata["stop_time"]
        self.tags: List[str] = self.metadata["tags"]
        self.metadata_project_name = self.metadata[
            "project_name"
        ]  # type: Optional[str]
        self.metadata_workspace = self.metadata["workspace"]  # type: Optional[str]
        self.customer_error_reported = self.metadata.get(
            "customer_error_reported", False
        )
        self.customer_error_message = self.metadata.get("customer_error_message")
        self.user_provided_experiment_key = self.metadata.get(
            "user_provided_experiment_key", False
        )
        self.comet_start_sourced = self.metadata.get("comet_start_sourced", False)

        # Up to Python SDK 3.19.0, we used to have the "mode" metadata
        old_mode = self.metadata.get("mode", DEPRECATED_OFFLINE_MODE_CREATE)
        old_mode_fallback = DEPRECATED_OFFLINE_MODE_TO_RESUME_STRATEGY_MAP[old_mode]

        self.resume_strategy = self.metadata.get("resume_strategy", old_mode_fallback)

    def get_creation_workspace(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the correct workspace to use for experiment creation. The order of priority is:
        * Explicit workspace parameter (either passed in Python when calling main_upload or
          upload_single_offline_experiment, or using the --workspace CLI flag of comet upload)
        * Implicit workspace taken from config
        * The workspace from the offline archive metadata

        Returns a tuple of three items:
        * The workspace to use
        * Optionally, a log message to display when an experiment has been successfully created
        * Optionally, a log message to display when an experiment fails to be created
        """
        workspace_config = self.config.get_string(None, "comet.workspace")
        if self.override_workspace is not None:
            return (
                self.override_workspace,
                OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_PARAMETER,
            )
        elif workspace_config is not None:
            return (
                workspace_config,
                OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_CONFIG,
            )
        else:
            return self.metadata_workspace, None

    def get_creation_project_name(self) -> Tuple[Optional[str], Optional[str]]:
        """Return the correct project_name to use for experiment creation. The order of priority is:
        * Explicit project_name parameter (either passed in Python when calling main_upload or
          upload_single_offline_experiment, or using the --project_name CLI flag of comet upload)
        * Implicit project_name taken from config
        * The project_name from the offline archive metadata
        """
        project_name_config = self.config.get_string(None, "comet.project_name")
        if self.override_project_name is not None:
            return (
                self.override_project_name,
                OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_PARAMETER,
            )
        elif project_name_config is not None:
            return (
                project_name_config,
                OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_CONFIG,
            )
        else:
            return self.metadata_project_name, None

    def _handshake(self) -> None:
        if self.comet_start_sourced:
            handshake_response = self._do_comet_start_handshake()
        else:
            handshake_response = self._do_original_handshake()
        self._post_handshake(handshake_response)

    def _do_comet_start_handshake(self) -> ExperimentHandshakeResponse:
        """The handshake compatible with `comet_ml.start()` way to create offline experiments"""
        (
            creation_workspace,
            workspace_overridden,
        ) = self.get_creation_workspace()

        (
            creation_project_name,
            project_name_overridden,
        ) = self.get_creation_project_name()

        if not self.user_provided_experiment_key:
            LOGGER.debug(
                "No experiment key is provided by user, creating new offline experiment with metadata: %r",
                self.metadata,
            )
            # create new experiment
            self._show_workspace_project_overriden_warnings(
                creation_workspace=creation_workspace,
                creation_project_name=creation_project_name,
                workspace_overridden=workspace_overridden,
                project_name_overridden=project_name_overridden,
            )

            return self.connection.add_run(
                creation_project_name,
                creation_workspace,
                offline=True,
            )

        # Experiment key was provided by user explicitly
        if self.resume_strategy == RESUME_STRATEGY_GET_OR_CREATE:
            # Try to create an experiment and if we get an exception that the experiment_key already
            # exists, try to resume it instead
            try:
                LOGGER.debug(
                    "The experiment key is provided by user, creating new offline experiment with metadata: %r",
                    self.metadata,
                )
                run_id_results = self.connection.add_run(
                    creation_project_name,
                    creation_workspace,
                    offline=True,
                )
                LOGGER.debug("New offline experiment was created")

                # If we successfully create a new experiment, display the log message about
                # overridden workspace and project_name. If we do before, we might display them even
                # if they ends-up not being used because we do fallback to an ExistingExperiment
                self._show_workspace_project_overriden_warnings(
                    creation_workspace=creation_workspace,
                    creation_project_name=creation_project_name,
                    workspace_overridden=workspace_overridden,
                    project_name_overridden=project_name_overridden,
                )
            except ExperimentAlreadyUploaded:
                LOGGER.debug(
                    "The experiment key is provided by user, experiment already exists, "
                    "continue with resuming existing experiment using metadata: %r",
                    self.metadata,
                )
                run_id_results = self.connection.get_run(self.experiment_id)
                self._resuming = True
        elif self.resume_strategy == RESUME_STRATEGY_CREATE:
            try:
                LOGGER.debug(
                    "The experiment key is provided by user, creating new offline experiment with metadata: %r",
                    self.metadata,
                )
                self._show_workspace_project_overriden_warnings(
                    creation_workspace=creation_workspace,
                    creation_project_name=creation_project_name,
                    workspace_overridden=workspace_overridden,
                    project_name_overridden=project_name_overridden,
                )

                run_id_results = self.connection.add_run(
                    creation_project_name,
                    creation_workspace,
                    offline=True,
                )
                LOGGER.debug("New offline experiment was created")
            except ExperimentAlreadyUploaded:
                # If the experiment already exists and the force flag is set, generate a new experiment ID and retry
                if self.force_upload:
                    run_id_results = self._force_experiment_upload(
                        creation_workspace=creation_workspace,
                        creation_project_name=creation_project_name,
                    )
                else:
                    LOGGER.debug(
                        "Can not create new experiment with key: %r and creation mode: %r "
                        "- it is already exists at Comet.",
                        self.experiment_id,
                        self.resume_strategy,
                    )
                    raise InvalidExperimentMode(
                        "Can not create new experiment - it is already exists"
                    )
        elif self.resume_strategy == RESUME_STRATEGY_GET:
            LOGGER.debug(
                "The experiment key is provided by user, resuming existing offline experiment with metadata: %r",
                self.metadata,
            )
            try:
                run_id_results = self.connection.get_run(self.experiment_id)
                self._resuming = True
            except ExperimentNotFound:
                if self.force_upload:
                    run_id_results = self._force_experiment_upload(
                        creation_workspace=creation_workspace,
                        creation_project_name=creation_project_name,
                    )
                else:
                    raise
        else:
            raise InvalidExperimentModeUnsupported(
                mode=self.resume_strategy, supported_modes=SUPPORTED_START_MODES
            )

        return run_id_results

    def _force_experiment_upload(
        self, creation_project_name: str, creation_workspace: str
    ) -> ExperimentHandshakeResponse:
        self.experiment_id = get_experiment_key(user_input=None)

        # Re-create a new RestServerConnection with the new experiment id
        self.connection.close()
        self.connection = get_rest_server_connection(
            api_key=self.api_key,
            experiment_key=self.experiment_id,
            server_address=self.server_address,
            verify_tls=self.check_tls_certificate,
        )

        run_id_results = self.connection.add_run(
            creation_project_name,
            creation_workspace,
            offline=True,
        )
        return run_id_results

    def _do_original_handshake(self) -> ExperimentHandshakeResponse:
        """Backward compatible handshake method for offline experiment created using constructor"""
        (
            creation_workspace,
            workspace_overridden,
        ) = self.get_creation_workspace()

        (
            creation_project_name,
            project_name_overridden,
        ) = self.get_creation_project_name()

        if self.resume_strategy == RESUME_STRATEGY_CREATE:
            try:
                # We know the workspace and project_name are taken into account in all cases,
                # display the log message before the actual creation
                self._show_workspace_project_overriden_warnings(
                    creation_workspace=creation_workspace,
                    creation_project_name=creation_project_name,
                    workspace_overridden=workspace_overridden,
                    project_name_overridden=project_name_overridden,
                )

                run_id_results = self.connection.add_run(
                    creation_project_name,
                    creation_workspace,
                    offline=True,
                )
            except ExperimentAlreadyUploaded:
                # If the experiment already exists and the force flag is set, generate a new experiment ID and retry
                if self.force_upload:
                    run_id_results = self._force_experiment_upload(
                        creation_workspace=creation_workspace,
                        creation_project_name=creation_project_name,
                    )
                else:
                    raise

        elif self.resume_strategy == RESUME_STRATEGY_GET:
            try:
                run_id_results = self.connection.get_run(self.experiment_id)
                self._resuming = True
            except ExperimentNotFound:
                if self.force_upload:
                    run_id_results = self._force_experiment_upload(
                        creation_workspace=creation_workspace,
                        creation_project_name=creation_project_name,
                    )
                else:
                    raise
        elif self.resume_strategy == RESUME_STRATEGY_GET_OR_CREATE:
            # Try to create an experiment and if we get an exception that the experiment_id already
            # exists, try to resume it instead
            try:
                run_id_results = self.connection.add_run(
                    creation_project_name,
                    creation_workspace,
                    offline=True,
                )

                # If we successfully create a new experiment, display the log message about
                # overridden workspace and project_name. If we do before, we might display them even
                # if they ends-up not being used because we do fallback to an ExistingExperiment
                self._show_workspace_project_overriden_warnings(
                    creation_workspace=creation_workspace,
                    creation_project_name=creation_project_name,
                    workspace_overridden=workspace_overridden,
                    project_name_overridden=project_name_overridden,
                )
            except ExperimentAlreadyUploaded:
                run_id_results = self.connection.get_run(self.experiment_id)
                self._resuming = True
        else:
            raise ValueError("Unknown resume strategy value %r" % self.resume_strategy)

        return run_id_results

    def _show_workspace_project_overriden_warnings(
        self,
        creation_workspace: str,
        creation_project_name: str,
        workspace_overridden: Optional[str],
        project_name_overridden: Optional[str],
    ):
        if workspace_overridden is not None:
            LOGGER.info(
                workspace_overridden,
                {
                    "experiment_id": self.experiment_id,
                    "creation_workspace": creation_workspace,
                    "metadata_workspace": self.metadata_workspace,
                },
            )

        if project_name_overridden is not None:
            LOGGER.info(
                project_name_overridden,
                {
                    "experiment_id": self.experiment_id,
                    "creation_project_name": creation_project_name,
                    "metadata_project_name": self.metadata_project_name,
                },
            )

    def _post_handshake(self, handshake_response: ExperimentHandshakeResponse):
        self.run_id = handshake_response.run_id
        self.project_id = handshake_response.project_id
        self.is_github = handshake_response.is_github
        self.focus_link = handshake_response.focus_link

        self.feature_toggles = FeatureToggles(
            handshake_response.feature_toggles, self.config
        )

        # Send tags if present
        if self.tags:
            self.connection.add_tags(self.tags)

        if self.rest_api_client is None:
            self.rest_api_client = get_rest_api_client(
                "v2", api_key=self.api_key, retry_auth_errors=True
            )

        self.config.set_direct_s3_file_upload_enabled(
            handshake_response.s3_direct_access_enabled
        )
        self._init_file_upload_manager()

    def _register_message_handlers(self):
        self._message_handlers = {
            UploadFileMessage.type: self._process_upload_message,
            RemoteAssetMessage.type: self._process_remote_file_message,
            OsPackagesMessage.type: self._process_os_packages_message,
            ModelGraphMessage.type: self._process_graph_message,
            SystemDetailsMessage.type: self._process_system_details_message,
            CloudDetailsMessage.type: self._process_cloud_details_message,
            LogOtherMessage.type: self._process_log_other_message,
            FileNameMessage.type: self._process_file_name_message,
            HtmlMessage.type: self._process_html_message,
            HtmlOverrideMessage.type: self._process_html_override_message,
            InstalledPackagesMessage.type: self._process_installed_packages_message,
            GpuStaticInfoMessage.type: self._process_gpu_static_info_message,
            GitMetadataMessage.type: self._process_git_metadata_message,
            SystemInfoMessage.type: self._process_system_info_message,
            LogDependencyMessage.type: self._process_log_dependency_message,
            RemoteModelMessage.type: self._process_remote_model_message,
            Log3DCloudMessage.type: self._process_3d_cloud_message,
            RegisterModelMessage.type: self._process_register_model_message,
        }

    def _send_messages(self):
        self._register_message_handlers()

        i = 0

        # Samples down the metrics
        sampling_size = self.config["comet.offline_sampling_size"]

        LOGGER.debug("Sampling metrics to %d values per metric name", sampling_size)

        sampler = MetricsSampler(sampling_size)
        parameter_batch = ParametersBatch(0)  # We don't care about the timing here
        stdout_batch = MessageBatch(0, 0)  # All messages will be sent in one batch

        with io.open(
            join(self.offline_dir, OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME),
            encoding="utf-8",
        ) as messages_files:
            for i, line in enumerate(messages_files):
                offset = i + 1
                try:
                    message = json.loads(line)

                    LOGGER.debug("Message %r", message)

                    message_type = message["type"]

                    if (
                        message_type == WebSocketMessage.type
                        or message_type == MetricMessage.type
                    ):
                        message_payload = message["payload"]
                        # Inject the offset now
                        message_payload["offset"] = offset

                        message_metric = message_payload.get(MetricMessage.metric_key)
                        old_message_param = (
                            "param" in message_payload or "params" in message_payload
                        )

                        if message_metric:
                            sampler.sample_metric(message_payload)
                        elif old_message_param:
                            # The new parameter message payload is a subset of the old WS payload
                            # so it should be compatible
                            self._process_parameter_message(
                                message, offset=offset, parameter_batch=parameter_batch
                            )
                        else:
                            self._process_ws_msg(
                                message_payload,
                                offset=offset,
                                stdout_batch=stdout_batch,
                            )
                    elif message_type == ParameterMessage.type:
                        self._process_parameter_message(
                            message, offset=offset, parameter_batch=parameter_batch
                        )
                    elif message_type == StandardOutputMessage.type:
                        self._process_standard_output_message(
                            message, offset=offset, message_batch=stdout_batch
                        )
                    elif message_type in self._message_handlers:
                        self._message_handlers[message_type](message)
                    else:
                        raise ValueError("Unknown message type %r" % message_type)
                except Exception as ex:
                    LOGGER.warning(
                        "Error processing line %d, reason: %r",
                        offset,
                        ex,
                        exc_info=True,
                    )
                    # report experiment error
                    self._report_experiment_error(
                        UNEXPECTED_OFFLINE_PROCESS_MESSAGE_ERROR
                    )

        # Then send the sampled metrics
        samples = sampler.get_samples()
        # send all collected metric samples as batch(-es)
        self._send_sampled_metrics_batch(samples)

        # And the batched hyper-parameters
        if parameter_batch.accept(
            self._send_parameter_messages_batch, unconditional=True
        ):
            LOGGER.debug("Parameters batch was sent")

        if stdout_batch.accept(self._send_stdout_message_batch, unconditional=True):
            LOGGER.debug("Stdout messages batch was sent")

        LOGGER.debug("Done sending %d messages", i)

    def _report_experiment_error(self, message: str) -> None:
        try:
            self.rest_api_client.update_experiment_error_status(
                experiment_key=self.experiment_id, is_alive=True, error_value=message
            )
        except Exception as ex:
            LOGGER.debug(
                "Failed to report offline sender error, reason: %r",
                ex,
                exc_info=True,
            )

    def _send_sampled_metrics_batch(self, samples: List[Dict[str, Any]]) -> None:
        for metric_sample in samples:
            metric = self._parse_metric_message(metric_sample)
            if metric is None:
                return

            self.message_batch_metrics.append(metric)
            # attempt to send batch of collected metrics immediately if batch size limit was hit
            # after appending new metric
            self.message_batch_metrics.accept(self._send_metric_messages_batch)

        # send the last part of messages if appropriate
        self.message_batch_metrics.accept(
            self._send_metric_messages_batch, unconditional=True
        )

    def _send_parameter_messages_batch(
        self, message_items: List[MessageBatchItem]
    ) -> None:
        """Attempts to send batch of parameters"""
        self._process_rest_api_send(
            self.connection.log_parameters_batch,
            rest_fail_prompt=PARAMETERS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending parameters batch (offline experiment)",
            items=message_items,
            compress=self.message_batch_compress,
        )

    def _send_metric_messages_batch(
        self, message_items: List[MessageBatchItem]
    ) -> None:
        """Attempts to send batch of metrics."""
        self._process_rest_api_send(
            self.connection.log_metrics_batch,
            rest_fail_prompt=METRICS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending metrics batch (offline experiment)",
            items=message_items,
            compress=self.message_batch_compress,
        )

    def _process_rest_api_send(
        self,
        sender: Callable,
        rest_fail_prompt: str,
        general_fail_prompt: str,
        **kwargs,
    ) -> None:
        try:
            sender(**kwargs)
        except CometRestApiException as exc:
            LOGGER.error(
                rest_fail_prompt,
                exc.response.status_code,
                exc.response.content,
            )
            # report experiment error
            self._report_experiment_error(
                rest_fail_prompt % (exc.response.status_code, exc.response.content)
            )
        except Exception:
            LOGGER.error(general_fail_prompt, exc_info=True)
            # report experiment error
            self._report_experiment_error(general_fail_prompt)

    def _validate_msg(
        self,
        message: Dict[str, Any],
        msg_validator: Any,
        fail_message: str,
        fail_event_name: str,
        raise_validation_error: bool,
    ) -> bool:
        """Validates message using given JSON schema validator. If validation failed reports experiment error
        and raises exception if appropriate or log corresponding warning and reports failure event to backend
        """
        try:
            msg_validator.validate(message)
            return True
        except ValidationError as ex:
            if raise_validation_error:
                raise

            LOGGER.warning(fail_message, exc_info=True)
            LOGGER.warning("Failure reason: %r", ex)
            self._report_experiment_error(fail_message)

            tb = traceback.format_exc()
            self.connection.report(event_name=fail_event_name, err_msg=tb)

        return False

    def _process_ws_msg(
        self, message: Dict[str, Any], offset: int, stdout_batch: Optional[MessageBatch]
    ) -> None:
        self._validate_msg(
            message=message,
            msg_validator=self.ws_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_WS_MSG,
            fail_event_name=OFFLINE_INVALID_WS_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._convert_and_send_ws_message_via_http(
            message=message, offset=offset, stdout_batch=stdout_batch
        )

    def _parse_metric_message(self, message: Dict[str, Any]) -> Optional[MetricMessage]:
        """Validates and deserialize raw metric message (dictionary)"""
        if not self._validate_msg(
            message=message,
            msg_validator=self.metric_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_METRIC_MSG,
            fail_event_name=OFFLINE_INVALID_METRIC_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        ):
            return None

        return MetricMessage.from_message_dict(message_dict=message)

    def _process_parameter_message(
        self, message: Dict[Any, Any], offset: int, parameter_batch: ParametersBatch
    ) -> None:
        message = message["payload"]

        if not self._validate_msg(
            message=message,
            msg_validator=self.parameter_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_PARAMETER_MSG,
            fail_event_name=OFFLINE_INVALID_PARAMETER_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        ):
            return

        # We know the message is valid, recreate a ParameterMessage from it
        param_message = ParameterMessage.from_message_dict(message)
        if param_message.has_invalid_message_id():
            param_message.message_id = offset

        parameter_batch.append(param_message)

    def _inject_fields(
        self, message: Dict[str, Any], offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhance provided message with relevant meta-data"""

        # Inject CometML specific values
        message["apiKey"] = self.api_key
        message["runId"] = self.run_id
        message["projectId"] = self.project_id
        message["experimentKey"] = self.experiment_id

        if offset:
            message["offset"] = offset

        return message

    def _collect_model_name_if_appropriate(self, message: Dict[str, Any]) -> None:
        additional_params = message.get("additional_params", None)
        if (
            additional_params is None
            or message["upload_type"] != ASSET_TYPE_MODEL_ELEMENT
        ):
            # not a model asset message
            return

        model_name = additional_params.get("groupingName", None)
        if model_name is not None:
            self.upload_models_names.add(model_name)

    def _process_upload_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.upload_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG,
            fail_event_name=OFFLINE_INVALID_UPLOAD_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        # Compute the url from the upload type
        url = self.connection.get_upload_url(message["upload_type"])

        additional_params = message["additional_params"] or {}
        additional_params["runId"] = self.run_id
        # Temporary fix to ensure integers:
        if "step" in additional_params and additional_params["step"] is not None:
            additional_params["step"] = int(additional_params["step"])
        if "epoch" in additional_params and additional_params["epoch"] is not None:
            additional_params["epoch"] = int(additional_params["epoch"])

        file_path = join(self.offline_dir, message["file_path"])
        file_size = os.path.getsize(file_path)

        # try to collect model name if it is an asset for the model
        self._collect_model_name_if_appropriate(message)

        # Mark message to be cleaned after sending, i.e. to delete all extracted files after upload
        message["clean"] = True

        self.file_upload_manager.upload_file_thread(
            options=FileUploadOptions(
                project_id=self.project_id,
                experiment_id=self.experiment_id,
                file_path=file_path,
                metadata=message.get("metadata"),
                upload_endpoint=url,
                api_key=self.api_key,
                additional_params=additional_params,
                clean=True,
                timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                verify_tls=self.check_tls_certificate,
                estimated_size=file_size,
                upload_type=message["upload_type"],
                base_url=self.connection.server_address,
                log_connection_error_as_debug=False,
                critical=False,
            )
        )
        LOGGER.debug("Processing uploading message done")

    def _process_remote_file_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.remote_file_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG,
            fail_event_name=OFFLINE_INVALID_UPLOAD_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        # Compute the url from the upload type
        url = self.connection.get_upload_url(message["upload_type"])

        additional_params = message["additional_params"] or {}
        additional_params["runId"] = self.run_id
        # Temporary fix to ensure integers:
        if "step" in additional_params and additional_params["step"] is not None:
            additional_params["step"] = int(additional_params["step"])
        if "epoch" in additional_params and additional_params["epoch"] is not None:
            additional_params["epoch"] = int(additional_params["epoch"])

        self.file_upload_manager.upload_remote_asset_thread(
            options=RemoteAssetsUploadOptions(
                project_id=self.project_id,
                experiment_id=self.experiment_id,
                remote_uri=message["remote_uri"],
                upload_endpoint=url,
                api_key=self.api_key,
                additional_params=additional_params,
                metadata=message["metadata"],
                timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                verify_tls=self.check_tls_certificate,
                estimated_size=0,
                log_connection_error_as_debug=False,
                upload_type=message["upload_type"],
                critical=False,
            )
        )
        LOGGER.debug("Processing remote uploading message done")

    def _convert_and_send_ws_message_via_http(
        self, message: Dict[str, Any], offset: int, stdout_batch: MessageBatch
    ) -> None:
        http_message = WebSocketMessage.from_message_dict_to_http_message(
            message_dict=message
        )
        http_message.message_id = offset
        message_type = http_message.type
        if message_type == OsPackagesMessage.type:
            self._send_os_packages_message(os_packages=http_message.os_packages)
        elif message_type == ModelGraphMessage.type:
            self._send_graph_message(graph=http_message.graph)
        elif message_type == SystemDetailsMessage.type:
            self._send_system_details_message(
                command=http_message.command,
                env=http_message.env,
                hostname=http_message.hostname,
                ip=http_message.ip,
                machine=http_message.machine,
                os_release=http_message.os_release,
                os_type=http_message.os_type,
                os=http_message.os,
                pid=http_message.pid,
                processor=http_message.processor,
                python_exe=http_message.python_exe,
                python_version_verbose=http_message.python_version_verbose,
                python_version=http_message.python_version,
                user=http_message.user,
            )
        elif message_type == LogOtherMessage.type:
            self._send_log_other(key=http_message.key, value=http_message.value)
        elif message_type == FileNameMessage.type:
            self._send_file_name_message(file_name=http_message.file_name)
        elif message_type == HtmlMessage.type:
            self._send_html_message(html=http_message.html)
        elif message_type == InstalledPackagesMessage.type:
            self._send_installed_packages_message(
                installed_packages=http_message.installed_packages
            )
        elif message_type == HtmlOverrideMessage.type:
            self._send_html_overwrite_message(html=http_message.htmlOverride)
        elif message_type == GpuStaticInfoMessage.type:
            self._send_gpu_static_info_message(
                gpu_static_info=http_message.gpu_static_info
            )
        elif message_type == GitMetadataMessage.type:
            self._send_git_metadata_message(
                user=http_message.git_metadata["user"],
                root=http_message.git_metadata["root"],
                branch=http_message.git_metadata["branch"],
                parent=http_message.git_metadata["parent"],
                origin=http_message.git_metadata["origin"],
            )
        elif message_type == SystemInfoMessage.type:
            self._send_system_info_message(system_info=http_message.system_info)
        elif message_type == LogDependencyMessage.type:
            self._send_log_dependency_message(
                name=http_message.name,
                version=http_message.version,
                timestamp=http_message.local_timestamp,
            )
        elif message_type == StandardOutputMessage.type:
            stdout_batch.append(http_message)
        else:
            raise ValueError(
                "Failed to convert and send WS message via HTTP, unsupported message type: %r"
                % message_type
            )

    def _process_os_packages_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.os_packages_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_OS_PACKAGES_MSG,
            fail_event_name=OFFLINE_INVALID_OS_PACKAGES_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_os_packages_message(os_packages=message["os_packages"])

    def _send_os_packages_message(self, os_packages: List[str]) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_os_packages,
            rest_fail_prompt=OS_PACKAGE_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending os_packages message",
            experiment_key=self.experiment_id,
            os_packages=os_packages,
        )

    def _process_graph_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.graph_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_GRAPH_MSG,
            fail_event_name=OFFLINE_INVALID_GRAPH_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_graph_message(graph=message["graph"])

    def _send_graph_message(self, graph: str) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_model_graph,
            rest_fail_prompt=MODEL_GRAPH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending model graph message",
            experiment_key=self.experiment_id,
            graph_str=graph,
        )

    def _process_system_details_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.system_details_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_SYSTEM_DETAILS_MSG,
            fail_event_name=OFFLINE_INVALID_SYSTEM_DETAILS_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_system_details_message(
            os=message["os"],
            command=message["command"],
            env=message["env"],
            hostname=message["hostname"],
            ip=message["ip"],
            machine=message["machine"],
            os_release=message["os_release"],
            os_type=message["os_type"],
            pid=message["pid"],
            processor=message["processor"],
            python_exe=message["python_exe"],
            python_version_verbose=message["python_version_verbose"],
            python_version=message["python_version"],
            user=message["user"],
        )

    def _send_system_details_message(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]],
        hostname: str,
        ip: str,
        machine: str,
        os_release: str,
        os_type: str,
        os: str,
        pid: int,
        processor: str,
        python_exe: str,
        python_version_verbose: str,
        python_version: str,
        user: str,
    ) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_system_details,
            rest_fail_prompt=SYSTEM_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending system details message",
            experiment_key=self.experiment_id,
            _os=os,
            command=command,
            env=env,
            hostname=hostname,
            ip=ip,
            machine=machine,
            os_release=os_release,
            os_type=os_type,
            pid=pid,
            processor=processor,
            python_exe=python_exe,
            python_version_verbose=python_version_verbose,
            python_version=python_version,
            user=user,
        )

    def _process_cloud_details_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.cloud_details_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_CLOUD_DETAILS_MSG,
            fail_event_name=OFFLINE_INVALID_CLOUD_DETAILS_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._process_rest_api_send(
            self.rest_api_client.set_experiment_cloud_details,
            rest_fail_prompt=CLOUD_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending cloud details message",
            experiment_key=self.experiment_id,
            provider=message["provider"],
            cloud_metadata=message["cloud_metadata"],
        )

    def _process_log_other_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.log_other_message_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_LOG_OTHER_MSG,
            fail_event_name=OFFLINE_INVALID_LOG_OTHER_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_log_other(key=message["key"], value=message["value"])

    def _send_log_other(self, key: str, value: Any) -> None:
        self._process_rest_api_send(
            self.rest_api_client.log_experiment_other,
            rest_fail_prompt=LOG_OTHER_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending log other message",
            experiment_key=self.experiment_id,
            key=key,
            value=value,
        )

    def _process_file_name_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.file_name_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_FILE_NAME_MSG,
            fail_event_name=OFFLINE_INVALID_FILE_NAME_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_file_name_message(file_name=message["file_name"])

    def _send_file_name_message(self, file_name: str) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_filename,
            rest_fail_prompt=FILENAME_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending file name message",
            experiment_key=self.experiment_id,
            filename=file_name,
        )

    def _process_html_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.html_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_HTML_MSG,
            fail_event_name=OFFLINE_INVALID_HTML_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_html_message(html=message["html"])

    def _send_html_message(self, html: str) -> None:
        self._process_rest_api_send(
            self.rest_api_client.log_experiment_html,
            rest_fail_prompt=HTML_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending html message",
            experiment_key=self.experiment_id,
            html=html,
        )

    def _process_installed_packages_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.installed_packages_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_INSTALLED_PACKAGES_MSG,
            fail_event_name=OFFLINE_INVALID_INSTALLED_PACKAGES_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_installed_packages_message(
            installed_packages=message["installed_packages"]
        )

    def _send_installed_packages_message(self, installed_packages: List[str]) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_installed_packages,
            rest_fail_prompt=INSTALLED_PACKAGES_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending installed packages message",
            experiment_key=self.experiment_id,
            installed_packages=installed_packages,
        )

    def _process_html_override_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.html_override_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_HTML_OVERRIDE_MSG,
            fail_event_name=OFFLINE_INVALID_HTML_OVERRIDE_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_html_overwrite_message(html=message["htmlOverride"])

    def _send_html_overwrite_message(self, html: str):
        self._process_rest_api_send(
            self.rest_api_client.log_experiment_html,
            rest_fail_prompt=HTML_OVERRIDE_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending html override message",
            experiment_key=self.experiment_id,
            html=html,
            overwrite=True,
        )

    def _process_gpu_static_info_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.gpu_static_info_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_GPU_STATIC_INFO_MSG,
            fail_event_name=OFFLINE_INVALID_GPU_STATIC_INFO_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_gpu_static_info_message(gpu_static_info=message["gpu_static_info"])

    def _send_gpu_static_info_message(
        self, gpu_static_info: List[Dict[str, Any]]
    ) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_gpu_static_info,
            rest_fail_prompt=GPU_STATIC_INFO_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending gpu static info message",
            experiment_key=self.experiment_id,
            gpu_static_info=gpu_static_info,
        )

    def _process_git_metadata_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.git_metadata_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_GIT_METADATA_MSG,
            fail_event_name=OFFLINE_INVALID_GIT_METADATA_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        git_metadata = message["git_metadata"]
        self._send_git_metadata_message(
            user=git_metadata["user"],
            root=git_metadata["root"],
            branch=git_metadata["branch"],
            parent=git_metadata["parent"],
            origin=git_metadata["origin"],
        )

    def _send_git_metadata_message(
        self, user: str, root: str, branch: str, parent: str, origin: str
    ) -> None:
        self._process_rest_api_send(
            self.rest_api_client.set_experiment_git_metadata,
            rest_fail_prompt=GIT_METADATA_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending git metadata message",
            experiment_key=self.experiment_id,
            user=user,
            root=root,
            branch=branch,
            parent=parent,
            origin=origin,
        )

    def _process_system_info_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.system_info_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_SYSTEM_INFO_MSG,
            fail_event_name=OFFLINE_INVALID_SYSTEM_INFO_MSG,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_system_info_message(system_info=message["system_info"])

    def _send_system_info_message(self, system_info: Dict[str, Any]) -> None:
        self._process_rest_api_send(
            self.rest_api_client.log_experiment_system_info,
            rest_fail_prompt=SYSTEM_INFO_MESSAGE_SENDING_ERROR,
            general_fail_prompt="Error sending system info message",
            experiment_key=self.experiment_id,
            system_info=[system_info],
        )

    def _process_standard_output_message(
        self, message: Dict[str, Any], offset: int, message_batch: MessageBatch
    ) -> None:
        message = message["payload"]

        if not self._validate_msg(
            message=message,
            msg_validator=self.standard_output_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_STANDARD_OUTPUT_MESSAGE,
            fail_event_name=OFFLINE_INVALID_STANDARD_OUTPUT_MESSAGE,
            raise_validation_error=self.raise_validation_error_for_tests,
        ):
            return

        # We know the message is valid, recreate a StandardOutputMessage from it
        stdout_message = StandardOutputMessage.from_message_dict(message)
        if stdout_message.has_invalid_message_id():
            stdout_message.message_id = offset

        message_batch.append(stdout_message)

    def _send_stdout_message_batch(self, batch_items: List[MessageBatchItem]) -> None:
        self._process_rest_api_send(
            sender=self.rest_api_client.send_stdout_batch,
            rest_fail_prompt=STANDARD_OUTPUT_SENDING_ERROR,
            general_fail_prompt="Error sending stdout/stderr batch (offline experiment)",
            batch_items=batch_items,
            compress=self.message_batch_compress,
            experiment_key=self.experiment_id,
        )

    def _process_log_dependency_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.log_dependency_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_LOG_DEPENDENCY_MESSAGE,
            fail_event_name=OFFLINE_INVALID_LOG_DEPENDENCY_MESSAGE,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_log_dependency_message(
            name=message["name"],
            version=message["version"],
            timestamp=message["local_timestamp"],
        )

    def _send_log_dependency_message(self, name: Any, version: Any, timestamp: int):
        self._process_rest_api_send(
            sender=self.rest_api_client.log_experiment_dependency,
            rest_fail_prompt=LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
            general_fail_prompt="Error sending log dependency message",
            experiment_key=self.experiment_id,
            name=name,
            version=version,
            timestamp=timestamp,
        )

    def _process_remote_model_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.remote_model_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_REMOTE_MODEL_MESSAGE,
            fail_event_name=OFFLINE_INVALID_REMOTE_MODEL_MESSAGE,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        self._send_remote_model_message(
            model_name=message["model_name"],
            remote_assets=message["remote_assets"],
        )

    def _send_remote_model_message(
        self, model_name: str, remote_assets: List[Dict[str, Any]]
    ) -> None:
        self._process_rest_api_send(
            sender=self.rest_api_client.log_experiment_remote_model,
            rest_fail_prompt=REMOTE_MODEL_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
            experiment_key=self.experiment_id,
            model_name=model_name,
            remote_assets=remote_assets,
            on_model_upload=None,
            on_failed_model_upload=None,
        )

    def _process_3d_cloud_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.cloud_3d_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_3D_CLOUD_MESSAGE,
            fail_event_name=OFFLINE_INVALID_3D_CLOUD_MESSAGE,
            raise_validation_error=self.raise_validation_error_for_tests,
        )

        asset_id = self.connection.create_asset(
            experiment_key=self.experiment_id,
            asset_name=message["name"],
            asset_type=ASSET_TYPE_3D_POINTS,
            metadata=message["metadata"],
            step=None,
        )

        thumbnail_upload_url = "%sasset/%s/thumbnail" % (
            self.connection.server_address,
            asset_id,
        )
        thumbnail_path = message["thumbnail_path"]

        if thumbnail_path:
            self.file_upload_manager.upload_thumbnail_thread(
                options=ThumbnailUploadOptions(
                    api_key=self.api_key,
                    experiment_id=self.experiment_id,
                    project_id=self.project_id,
                    timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                    verify_tls=self.check_tls_certificate,
                    upload_endpoint=thumbnail_upload_url,
                    estimated_size=UNUSED_INT,
                    thumbnail_path=thumbnail_path,
                    log_connection_error_as_debug=False,
                    critical=False,
                ),
            )

        asset_item_url = self.connection.get_upload_url(message["upload_type"])
        items = asset_item.deserialize_items(message["items"])
        for item in items:
            self.file_upload_manager.upload_asset_item_thread(
                options=AssetItemUploadOptions(
                    api_key=self.api_key,
                    experiment_id=self.experiment_id,
                    project_id=self.project_id,
                    timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                    verify_tls=self.check_tls_certificate,
                    upload_endpoint=asset_item_url,
                    estimated_size=UNUSED_INT,
                    asset_id=asset_id,
                    asset_item=item,
                    all_items=items,
                    log_connection_error_as_debug=False,
                    critical=False,
                    upload_type=message["upload_type"],
                    asset_name=message["name"],
                ),
            )

    def _process_register_model_message(self, message: Dict[str, Any]) -> None:
        message = message["payload"]

        self._validate_msg(
            message=message,
            msg_validator=self.register_model_msg_validator,
            fail_message=OFFLINE_EXPERIMENT_INVALID_REGISTER_MODEL_MESSAGE,
            fail_event_name=OFFLINE_INVALID_REGISTER_MODEL_MESSAGE,
            raise_validation_error=self.raise_validation_error_for_tests,
        )
        # save message for later processing
        self.register_model_messages.append(message)

    def _send_register_model_message(
        self,
        model_name: str,
        version: str,
        registry_name: str,
        public: bool,
        description: str,
        comment: str,
        tags: List[str],
        status: str,
        stages: List[str],
    ) -> None:
        self._process_rest_api_send(
            sender=self.rest_api_client.register_model_v2,
            rest_fail_prompt=REGISTER_MODEL_MESSAGE_SENDING_ERROR,
            general_fail_prompt="Failed to send register model message",
            experiment_id=self.experiment_id,
            model_name=model_name,
            version=version,
            workspace=self.get_creation_workspace(),
            registry_name=registry_name,
            public=public,
            description=description,
            comment=comment,
            tags=tags,
            status=status,
            stages=stages,
        )

    def _flush_register_model_messages(self) -> None:
        for m in self.register_model_messages:
            model_name = m["model_name"]
            if model_name in self.upload_models_names:
                self._send_register_model_message(
                    model_name=model_name,
                    version=m["version"],
                    registry_name=m["registry_name"],
                    public=m["public"],
                    description=m["description"],
                    comment=m["comment"],
                    tags=m["tags"],
                    status=m["status"],
                    stages=m["stages"],
                )
            else:
                LOGGER.warning(
                    OFFLINE_FAILED_TO_REGISTER_MODEL_NO_MODEL_FILES, model_name
                )

    def _status_report_start(self):
        self.connection.update_experiment_status(
            self.run_id, self.project_id, True, offline=True
        )

    def _status_report_end(self):
        self.connection.update_experiment_status(
            self.run_id, self.project_id, False, offline=True
        )

    def _send_start_ends_time(self):
        # We created a new experiment, update the start time and stop time
        if self._resuming is False:
            self.connection.offline_experiment_start_end_time(
                self.run_id, self.start_time, self.stop_time
            )
        else:
            self.connection.offline_experiment_start_end_time(
                self.run_id, None, self.stop_time
            )

    def _get_experiment_url(self):
        if self.focus_link:
            return self.focus_link + self.experiment_id

        return ""

    def close(self):
        LOGGER.info(UPLOADING_DATA_BEFORE_TERMINATION)

        self.file_upload_manager.close()
        # Finish remained uploads and display upload progress
        if not self.file_upload_manager.all_done():
            monitor = FileUploadManagerMonitor(self.file_upload_manager)
            LOGGER.info(WAITING_FOR_FILE_UPLOADS_COMPLETION)

            wait_for_done(
                monitor.all_done,
                self.file_upload_waiting_timeout,
                progress_callback=monitor.log_remaining_uploads,
                sleep_time=self.wait_for_finish_sleep_interval,
            )

        if not self.file_upload_manager.all_done():
            remaining_upload = self.file_upload_manager.remaining_uploads()
            LOGGER.error(OFFLINE_UPLOADS_FAILED_DUE_TIMEOUT, remaining_upload)
            self._report_experiment_error(
                OFFLINE_UPLOADS_FAILED_DUE_TIMEOUT % remaining_upload
            )

        self.file_upload_manager.join()
        LOGGER.debug("Upload threads %r", self.file_upload_manager)

        # send register model messages collected
        self._flush_register_model_messages()

        # close the REST API client last to make sure that experiment error reported if any
        if self.rest_api_client is not None:
            self.rest_api_client.close()

        LOGGER.log(self.display_level, OFFLINE_SENDER_ENDS, self._get_experiment_url())
        LOGGER.log(self.display_level, OFFLINE_SENDER_ENDS_PROCESSING)


def unzip_offline_archive(offline_archive_path):
    temp_dir = tempfile.mkdtemp()

    zip_file = zipfile.ZipFile(offline_archive_path, mode="r", allowZip64=True)

    # Extract the archive
    zip_file.extractall(temp_dir)

    return temp_dir


def upload_single_offline_experiment(
    offline_archive_path: str,
    api_key: str,
    force_upload: bool,
    display_level: str = "info",
    override_workspace: Optional[str] = None,
    override_project_name: Optional[str] = None,
) -> bool:
    unzipped_directory = unzip_offline_archive(offline_archive_path)
    settings = get_config()
    sender = OfflineSender(
        api_key=api_key,
        offline_dir=unzipped_directory,
        force_upload=force_upload,
        display_level=display_level,
        override_workspace=override_workspace,
        override_project_name=override_project_name,
        message_batch_compress=settings.get_bool(
            None, "comet.message_batch.use_compression"
        ),
        message_batch_metric_interval=settings.get_int(
            None, "comet.message_batch.metric_interval"
        ),
        message_batch_metric_max_size=settings.get_int(
            None, "comet.message_batch.metric_max_size"
        ),
    )
    try:
        sender.send()
        sender.close()

        return True
    except ExperimentAlreadyUploaded:
        # original upload flow
        LOGGER.error(OFFLINE_EXPERIMENT_ALREADY_UPLOADED, offline_archive_path)
    except InvalidExperimentModeUnsupported as ex:
        # comet_ml.start() upload flow
        LOGGER.error(ex)
    except InvalidExperimentMode:
        # comet_ml.start() upload flow
        experiment_key = extract_experiment_key_from_offline_file(offline_archive_path)
        LOGGER.error(
            OFFLINE_EXPERIMENT_ALREADY_EXISTS_CREATE_MODE,
            offline_archive_path,
            experiment_key,
        )
    except ExperimentNotFound:
        # comet_ml.start() upload flow
        LOGGER.error(OFFLINE_EXPERIMENT_NOT_FOUND_GET_MODE, offline_archive_path)
    except BackendCustomError as e:
        # comet_ml.start() upload flow
        LOGGER.error(str(e), exc_info=True)
    except ProjectConsideredLLM as e:
        # comet_ml.start() upload flow
        LOGGER.error(str(e), exc_info=True)
    finally:
        try:
            shutil.rmtree(unzipped_directory)
        except OSError:
            # We made our best effort to clean after ourselves
            msg = "Failed to clean the Offline sender tmpdir %r"
            LOGGER.debug(msg, unzipped_directory, exc_info=True)

    return False


def main_upload(
    archives: List[str],
    force_upload: bool,
    override_workspace: Optional[str] = None,
    override_project_name: Optional[str] = None,
) -> None:
    upload_count = 0
    fail_count = 0

    # Common code
    config = get_config()
    api_key = get_api_key(None, config)

    for filename in archives:
        LOGGER.info(OFFLINE_UPLOADING_EXPERIMENT_FILE_PROMPT, filename)
        try:
            success = upload_single_offline_experiment(
                offline_archive_path=filename,
                api_key=api_key,
                force_upload=force_upload,
                override_workspace=override_workspace,
                override_project_name=override_project_name,
            )
            if success:
                upload_count += 1
                LOGGER.info("    Done!")
            else:
                fail_count += 1

        except InvalidAPIKey:
            # raise an exception - no need to continue with other archives as it will fail for them as well
            raise
        except Exception:
            # log exception and continue with other experiments
            LOGGER.error(
                OFFLINE_UPLOAD_FAILED_UNEXPECTED_ERROR,
                filename,
                exc_info=True,
                extra={"show_traceback": True},
            )
            fail_count += 1

    LOGGER.info(OFFLINE_SUCCESS_UPLOADED_EXPERIMENTS, upload_count)
    if fail_count > 0:
        LOGGER.info(OFFLINE_FAILED_UPLOADED_EXPERIMENTS, fail_count)
        raise OfflineExperimentUploadFailed(
            OFFLINE_AT_LEAST_ONE_EXPERIMENT_UPLOAD_FAILED
        )
