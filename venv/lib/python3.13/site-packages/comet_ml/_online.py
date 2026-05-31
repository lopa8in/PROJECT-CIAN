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

"""comet-ml"""
from __future__ import print_function

import logging
import traceback
from typing import Callable

from comet_ml.artifacts import get_artifact, log_artifact, parse_artifact_name

import requests

from . import announcements, reporter
from ._reporting import EXPERIMENT_CREATION_FAILED
from ._typing import (
    Any,
    Dict,
    ExperimentThrottledStatus,
    HeartBeatResponse,
    List,
    Optional,
    Tuple,
)
from .api_helpers.experiment_key import get_experiment_key
from .artifacts import Artifact, LoggedArtifact, hf_dataset
from .backend_response_helper import (
    get_latest_metric_value,
    get_other_value,
    get_param_value,
    is_backend_timestamp_newer,
)
from .config import (
    AUTO_OUTPUT_LOGGING_DEFAULT_VALUE,
    INITIAL_HEARTBEAT_INTERVAL,
    MAXIMAL_KEY_LENGTH,
    discard_api_key,
    get_api_key,
    get_check_tls_certificate,
    get_config,
    get_previous_experiment,
)
from .connection import RestApiClient, RestServerConnection
from .connection.connection_factory import (
    get_rest_api_client,
    get_rest_server_connection,
)
from .connection.connection_monitor import ServerConnectionMonitor
from .constants import ASSET_TYPE_EMBEDDINGS
from .convert_utils import convert_to_string_key
from .data_structure import Embedding
from .exceptions import (
    BackendCustomError,
    BadCallbackArguments,
    CometRestApiException,
    ExperimentCleaningException,
    ExperimentDisabledException,
    ExperimentNotAlive,
    InvalidAPIKey,
)
from .experiment import CometExperiment
from .feature_toggles import SDK_ANNOUNCEMENT, FeatureToggles
from .file_utils import make_template_filename
from .handshake import ExperimentHandshakeResponse
from .heartbeat import HeartbeatThread
from .json_encoder import NestedEncoder
from .logging_messages import (
    ADD_SYMLINK_ERROR,
    ADD_TAGS_ERROR,
    DIRECT_S3_UPLOAD_DISABLED,
    DIRECT_S3_UPLOAD_ENABLED,
    EXPERIMENT_LIVE,
    EXPERIMENT_MARK_AS_ENDED_FAILED,
    EXPERIMENT_MARK_AS_STARTED_FAILED,
    FAILED_TO_CREATE_EXPERIMENT_COMET_ERROR,
    GET_ARTIFACT_VERSION_OR_ALIAS_GIVEN_TWICE,
    GET_ARTIFACT_WORKSPACE_GIVEN_TWICE,
    INTERNET_CONNECTION_ERROR,
    LOG_EMBEDDING_EXPERIMENTAL_WARNING,
    ONLINE_EXISTING_INIT_INVALID_KEY_EXCEPTION,
    ONLINE_GET_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION,
    ONLINE_GET_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION,
    ONLINE_GET_EXPERIMENT_URL_TAB_INVALID_INFO,
    ONLINE_INIT_API_KEY_MISSING_EXCEPTION,
    ONLINE_LOG_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION,
    ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION,
    ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ARTIFACT_EXCEPTION,
    ONLINE_ON_END_FAILED_TO_LOG_RUN_WARNING,
    ONLINE_ON_END_STREAMER_UPLOAD_FAILED_EXCEPTION,
    REGISTER_RPC_FAILED,
    SEND_NOTIFICATION_FAILED,
)
from .rpc import create_remote_call, get_remote_action_definition
from .semantic_version import SemanticVersion
from .streamer import FallbackStreamer, OnlineStreamer
from .upload_callback.callback import UploadCallback
from .utils import (
    get_time_monotonic,
    is_valid_experiment_key,
    local_timestamp as get_local_timestamp,
    merge_url,
    valid_ui_tabs,
)

LOGGER = logging.getLogger(__name__)


class Experiment(CometExperiment):
    """
    Experiment is a unit of measurable research that defines a single run with some data/parameters/code/results.

    Creating an Experiment object in your code will report a new experiment to your Comet.ml project. Your Experiment
    will automatically track and collect many things and will also allow you to manually report anything.

    You can create multiple objects in one script (such as when looping over multiple hyper parameters).

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
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
        **kwargs,
    ) -> None:
        """
        Creates a new experiment on Comet.

        Args:
            api_key (str): Your API key obtained from comet.com
            project_name (str): Send your experiment to a specific project. Otherwise will be sent to
                `Uncategorized Experiments`. If project name does not already exists Comet will create a new project.
            workspace (str): Attach an experiment to a project that belongs to this workspace
            log_code (bool): Allows you to enable/disable code logging
            log_graph (bool): Allows you to enable/disable automatic computation graph logging.
            auto_param_logging (bool): Allows you to enable/disable hyper parameters logging
            auto_metric_logging (bool): Allows you to enable/disable metrics logging
            auto_metric_step_rate (int): Controls how often batch metrics are logged
            auto_histogram_tensorboard_logging (bool): Allows you to enable/disable automatic tensorboard histogram logging
            auto_histogram_epoch_rate (int): Controls how often histograms are logged
            auto_histogram_weight_logging (bool): Allows you to enable/disable histogram logging for biases and weights
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
            log_git_patch (bool): Allow you to enable/disable the
                automatic collection of git patch
            display_summary_level (int): Control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled (bool): Allows you to disable all network
                communication with the Comet backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.
            experiment_key (str): If provided, will be used as the experiment key. If an experiment
                with the same key already exists, it will raises an Exception. Could be set through
                configuration as well. Must be an alphanumeric string whose length is between 32 and 50 characters.
        """
        self._startup_duration_start = get_time_monotonic()
        self.config = get_config()

        self.disabled = self.config.get_bool(
            disabled, "comet.auto_log.disable", False, not_set_value=False
        )

        self.api_key = get_api_key(api_key, self.config)

        self._raise_for_missing_api_key()

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
            display_summary=display_summary,  # deprecated
            display_summary_level=display_summary_level,
            log_env_cpu=log_env_cpu,
            log_env_network=log_env_network,
            log_env_disk=log_env_disk,
            optimizer_data=optimizer_data,
            auto_weight_logging=auto_weight_logging,  # deprecated
            auto_log_co2=auto_log_co2,
            auto_metric_step_rate=auto_metric_step_rate,
            auto_histogram_tensorboard_logging=auto_histogram_tensorboard_logging,
            auto_histogram_epoch_rate=auto_histogram_epoch_rate,
            auto_histogram_weight_logging=auto_histogram_weight_logging,
            auto_histogram_gradient_logging=auto_histogram_gradient_logging,
            auto_histogram_activation_logging=auto_histogram_activation_logging,
            experiment_key=experiment_key,
            distributed_node_identifier=distributed_node_identifier,
        )

        self.connection: Optional[RestServerConnection] = None
        self.rest_api_client: Optional[RestApiClient] = None
        self._heartbeat_thread: Optional[HeartbeatThread] = None

        self._check_tls_certificate = get_check_tls_certificate(self.config)

        self._sentry_client = None

        self._start_time = get_time_monotonic()

        self._use_get_or_create_mode = kwargs.get("enable_get_or_create_mode", False)

        self._offline_directory = kwargs.get("offline_directory", None)

        if self.disabled is not True:
            self._start()

            if self.alive is True:
                LOGGER.info(EXPERIMENT_LIVE, self._get_experiment_url())

    def _raise_for_missing_api_key(self):
        if not self.disabled and self.api_key is None:
            raise ValueError(ONLINE_INIT_API_KEY_MISSING_EXCEPTION)

    def _setup_streamer(self) -> bool:
        """
        Do the necessary work to create mandatory objects, like the streamer
        and feature flags
        """
        # make sure that we have an API key
        self._raise_for_missing_api_key()

        self.connection = get_rest_server_connection(
            api_key=self.api_key,
            experiment_key=self.id,
        )
        self.rest_api_client = get_rest_api_client(
            "v2",
            api_key=self.api_key,
            use_cache=False,
            headers={"X-COMET-SDK-SOURCE": "Experiment"},
            retry_auth_errors=True,
        )

        authenticated = False
        initial_offset = 0
        try:
            authenticated, initial_offset = self._authenticate()
        except (ConnectionError, requests.ConnectionError):
            LOGGER.error(INTERNET_CONNECTION_ERROR, exc_info=True)
        except InvalidAPIKey as e:
            # discard invalid API key so it would be displayed in error log without masking
            discard_api_key(e.api_key)
            raise e
        except CometRestApiException as e:
            tb = traceback.format_exc()
            LOGGER.error(
                FAILED_TO_CREATE_EXPERIMENT_COMET_ERROR,
                e,
                exc_info=True,
            )
            self._report(event_name=EXPERIMENT_CREATION_FAILED, err_msg=tb)
        except BackendCustomError as e:
            tb = traceback.format_exc()
            LOGGER.error(str(e), exc_info=True)
            self._report(event_name=EXPERIMENT_CREATION_FAILED, err_msg=tb)

        # Authentication failed somehow
        if not authenticated:
            LOGGER.debug(
                "Experiment creation at backend failed, skipping streamer creation"
            )
            return False

        # Initiate the streamer
        self._initialize_streamer(initial_offset=initial_offset)

        # Initiate the heartbeat thread
        self._heartbeat_thread = HeartbeatThread(
            beat_duration=INITIAL_HEARTBEAT_INTERVAL,
            status_update_callback=self._send_status_update,
            pending_rpcs_callback=self._on_pending_rpcs_callback,
        )

        # setup parameters update callback
        self._heartbeat_thread.on_parameters_update_interval_callback = (
            self.streamer.parameters_update_interval_callback
        )

        self._heartbeat_thread.start()
        return True

    def _authenticate(self) -> Tuple[bool, int]:
        """
        Do the handshake with the Backend to authenticate the api key and get
        various parameters and settings
        """
        add_run_response = self._add_or_get_run()
        points_3d_upload_limits = self.connection.get_points_3d_upload_limits()

        self.run_id = add_run_response.run_id
        self.project_id = add_run_response.project_id
        self.is_github = add_run_response.is_github
        self.focus_link = add_run_response.focus_link
        self.upload_limit = add_run_response.upload_limit
        self.video_upload_limit = add_run_response.video_upload_limit
        self.asset_upload_limit = add_run_response.asset_upload_limit
        self.upload_web_asset_url_prefix = add_run_response.web_asset_url
        self.upload_web_image_url_prefix = add_run_response.web_image_url
        self.upload_api_asset_url_prefix = add_run_response.api_asset_url
        self.upload_api_image_url_prefix = add_run_response.api_image_url
        self.name = add_run_response.experiment_name

        if points_3d_upload_limits is not None:
            self.points_3d_upload_limits = points_3d_upload_limits

        self.feature_toggles = FeatureToggles(
            add_run_response.feature_toggles, self.config
        )

        self.config.set_direct_s3_file_upload_enabled(
            add_run_response.s3_direct_access_enabled
        )

        authenticated = add_run_response.run_id is not None

        return authenticated, add_run_response.last_offset

    def _add_or_get_run(self) -> ExperimentHandshakeResponse:
        return self.connection.add_run(
            project_name=self.project_name,
            workspace=self.workspace,
            get_or_create_mode=self._use_get_or_create_mode,
        )

    def _initialize_streamer(self, initial_offset: int) -> None:
        backend_version = self.rest_api_client.get_api_backend_version()

        direct_s3_upload_enabled = self.config.has_direct_s3_file_upload_enabled()
        if not direct_s3_upload_enabled:
            LOGGER.debug(DIRECT_S3_UPLOAD_DISABLED)
        else:
            LOGGER.debug(DIRECT_S3_UPLOAD_ENABLED)

        online_streamer = OnlineStreamer(
            beat_duration=self.config.get_int(
                None, "comet.internal.streamer_beat_duration"
            ),
            connection=self.connection,
            initial_offset=initial_offset,
            experiment_key=self.id,
            api_key=self.api_key,
            run_id=self.run_id,
            project_id=self.project_id,
            rest_api_client=self.rest_api_client,
            worker_cpu_ratio=self.config.get_int(
                None, "comet.internal.file_upload_worker_ratio"
            ),
            worker_count=self.config.get_raw(None, "comet.internal.worker_count"),
            verify_tls=self._check_tls_certificate,
            msg_waiting_timeout=self.config["comet.timeout.cleaning"],
            wait_for_finish_sleep_interval=self.config.get_int(
                None, "comet.internal.streamer_wait_sleep_interval"
            ),
            file_upload_waiting_timeout=self.config["comet.timeout.upload"],
            file_upload_read_timeout=self.config.get_int(
                None, "comet.timeout.file_upload"
            ),
            message_batch_compress=self.config.get_bool(
                None, "comet.message_batch.use_compression"
            ),
            message_batch_metric_interval=self.config.get_int(
                None, "comet.message_batch.metric_interval"
            ),
            message_batch_metric_max_size=self.config.get_int(
                None, "comet.message_batch.metric_max_size"
            ),
            parameters_batch_base_interval=self.config.get_int(
                None, "comet.message_batch.parameters_interval"
            ),
            message_batch_stdout_interval=self.config.get_int(
                None, "comet.message_batch.stdout_interval"
            ),
            message_batch_stdout_max_size=self.config.get_int(
                None, "comet.message_batch.stdout_max_size"
            ),
            s3_multipart_threshold=self.config.get_int(
                None, "comet.s3_multipart.size_threshold"
            ),
            s3_multipart_expires_in=self.config.get_int(
                None, "comet.s3_multipart.expires_in"
            ),
            s3_multipart_upload_enabled=direct_s3_upload_enabled,
            artifact_remote_assets_batch_metric_interval=self.config.get_int(
                None, "comet.message_batch.artifact_remote_assets_interval"
            ),
            artifact_remote_assets_batch_metric_max_size=self.config.get_int(
                None, "comet.message_batch.artifact_remote_assets_max_size"
            ),
            artifact_remote_assets_batch_enabled=self.config.has_artifact_remote_assets_batching_enabled(
                backend_version
            ).feature_supported,
            use_raw_throttling_messages=self.config.should_use_raw_throttling_messages(
                backend_version
            ).feature_supported,
        )
        connection_monitor = ServerConnectionMonitor(
            ping_interval=self.config.get_int(
                None, "comet.fallback_streamer.connection_check_interval"
            ),
        )
        self.streamer = FallbackStreamer(
            online_streamer=online_streamer,
            initial_offset=initial_offset,
            server_connection_monitor=connection_monitor,
            rest_server_connection=self.connection,
            enable_fallback_to_offline=self._has_fallback_to_offline_enabled(
                backend_version
            ),
            keep_offline_zip=self.config.get_bool(
                None, "comet.fallback_streamer.keep_offline_zip"
            ),
            offline_directory=self._offline_directory,
        )

        # Start streamer thread.
        self.streamer.start()

    def _has_fallback_to_offline_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> bool:
        check_result = self.config.has_fallback_to_offline_enabled(
            current_backend_version=current_backend_version,
        )

        if not check_result.feature_supported:
            LOGGER.debug(
                "Fallback to offline disabled. Backend version: %s, required minimal version: %r"
                % (current_backend_version, check_result.min_backend_version_supported)
            )
            return False

        return True

    def _mark_as_started(self) -> None:
        try:
            self.connection.update_experiment_status(
                self.run_id, self.project_id, self.alive
            )
        except Exception:
            LOGGER.error(EXPERIMENT_MARK_AS_STARTED_FAILED, exc_info=True)
            self._report_experiment_error(EXPERIMENT_MARK_AS_STARTED_FAILED)

    def _is_compute_metric_included(self) -> bool:
        backend_version = self.rest_api_client.backend_version
        if backend_version is not None and backend_version >= (3, 3, 53):
            return True

        return False

    def _mark_as_ended(self) -> None:
        if not self.alive:
            return

        if not self._streamer_has_connection():
            LOGGER.debug(
                "Failed to send experiment ended status. No server connection."
            )
            return

        LOGGER.debug("Update experiment status as ENDED, experiment id: %s", self.id)
        try:
            self.connection.update_experiment_status(
                self.run_id, self.project_id, False
            )
        except Exception:
            LOGGER.error(EXPERIMENT_MARK_AS_ENDED_FAILED, exc_info=True)
            # report error
            self._report_experiment_error(EXPERIMENT_MARK_AS_ENDED_FAILED)

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        if not self.alive:
            return
        if self.streamer is None:
            return
        if not self._streamer_has_connection():
            LOGGER.debug("Failed to report experiment error. No server connection.")
            return

        self.streamer._report_experiment_error(message, has_crashed=has_crashed)

    def _report(self, *args, **kwargs) -> None:
        if self.alive or kwargs["event_name"] == EXPERIMENT_CREATION_FAILED:
            self.connection.report(*args, **kwargs)

    def _send_status_update(self) -> Optional[HeartBeatResponse]:
        if self._streamer_has_connection():
            return self.connection.heartbeat()
        else:
            return None

    def __internal_api__announce__(self) -> None:
        if not self.feature_toggles[SDK_ANNOUNCEMENT]:
            return

        try:
            announcements.announce(self.id)
        except Exception:
            LOGGER.debug("Announcement not reported", exc_info=True)

    def _streamer_wait_for_finish(self) -> bool:
        """Called to wait for experiment streamer's cleanup procedures"""
        return self.streamer.wait_for_finish(
            experiment_key=self.id,
            workspace=self.workspace,
            project_name=self.project_name,
            tags=self.get_tags(),
            comet_config=self.config,
        )

    def _report_accumulated_bi_events(self) -> None:
        try:
            reporter.report(self)
        except Exception:
            LOGGER.debug("An error occurred during bi events reporting", exc_info=True)

    def _on_end(self, wait=True) -> bool:
        """Called when the Experiment is replaced by another one or at the
        end of the script
        """
        if self._experiment_fully_ended() is True:
            # already ended, no need to process it twice
            return True

        self._report_accumulated_bi_events()

        successful_clean = super()._on_end(wait=wait)

        if not successful_clean:
            LOGGER.warning(ONLINE_ON_END_FAILED_TO_LOG_RUN_WARNING)
        elif self.alive:
            LOGGER.info(EXPERIMENT_LIVE, self._get_experiment_url())

        # make sure to close heartbeat thread
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.close()
            if wait is True:
                self._heartbeat_thread.join(10)

        if self.connection is not None:
            self.connection.close()

        if self.streamer is not None and self.streamer.has_upload_failed():
            raise ExperimentCleaningException(
                ONLINE_ON_END_STREAMER_UPLOAD_FAILED_EXCEPTION
            )

        elapsed = get_time_monotonic() - self._start_time
        LOGGER.debug(
            "Full experiment's (%s) elapsed time: %r seconds, cleaned successfully: %s, waited: %s\n\n",
            self.id,
            elapsed,
            successful_clean,
            wait,
        )

        return successful_clean

    def _check_experiment_throttled(
        self,
    ) -> ExperimentThrottledStatus:
        experiment_metadata = self.rest_api_client.get_experiment_metadata(self.id)
        throttled = experiment_metadata.get("throttle", False)
        if throttled:
            message = experiment_metadata.get("throttleMessage")
            reasons = experiment_metadata.get("throttleReasons")
            return ExperimentThrottledStatus(True, message, reasons)

        return ExperimentThrottledStatus(False, None, None)

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
        return self._get_experiment_url()

    def _get_experiment_url(self, tab: Optional[str] = None) -> str:
        if self.focus_link:
            if tab:
                if tab in valid_ui_tabs():
                    return merge_url(
                        self.focus_link + self.id,
                        {"experiment-tab": valid_ui_tabs(tab)},
                    )
                else:
                    LOGGER.info(
                        ONLINE_GET_EXPERIMENT_URL_TAB_INVALID_INFO,
                        valid_ui_tabs(preferred=True),
                    )
            return self.focus_link + self.id

        return ""

    def _on_pending_rpcs_callback(self) -> None:
        """Called by heartbeat thread when we have pending RPCs"""
        LOGGER.debug("Checking pending RPCs")
        calls = self.connection.get_pending_rpcs()["remoteProcedureCalls"]

        LOGGER.debug("Got pending RPCs: %r", calls)
        for raw_call in calls:
            call = create_remote_call(raw_call)
            if call is None:
                continue
            self._add_pending_call(call)

    def _send_rpc_callback_result(
        self,
        call_id: str,
        remote_call_result: Dict[str, Any],
        start_time: int,
        end_time: int,
    ) -> None:
        # Send the result to the backend
        self.connection.send_rpc_result(
            call_id, remote_call_result, start_time, end_time
        )

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
        try:
            if self.alive:
                self.connection.send_new_symlink(project_name)
        except Exception:
            msg = ADD_SYMLINK_ERROR % project_name
            LOGGER.warning(msg, exc_info=True)
            # report error
            self._report_experiment_error(msg)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the experiment. Tags will be shown in the dashboard.

        Args:
            tag (str): A tag to add to the experiment.
        """
        try:
            result = super().add_tag(tag)
            if result is False:
                return

            if self.alive:
                self.connection.add_tags([tag])
        except Exception:
            msg = ADD_TAGS_ERROR % tag
            LOGGER.warning(msg, exc_info=True)
            # report error
            self._report_experiment_error(msg)

    def add_tags(self, tags: List[str]) -> None:
        """
        Add several tags to the experiment. Tags will be shown in the
        dashboard.

        Args:
            tags (List[str]): Tags list to add to the experiment.
        """
        try:
            result = super().add_tags(tags)
            if result is False:
                return None

            if self.alive:
                self.connection.add_tags(tags)
        except Exception:
            msg = ADD_TAGS_ERROR % tags
            LOGGER.warning(msg, exc_info=True)
            # report error
            self._report_experiment_error(msg)

    def register_callback(self, function: Callable) -> None:
        """
        Register the function passed as argument to be an RPC.

        This is only supported for online experiment at the moment.

        Args:
            function (callable): Function to register as a callback.

        Raises:
            NotImplementedError: This is not yet supported for offline experiments.
        """
        super(Experiment, self).register_callback(function)

        try:
            remote_action_definition = get_remote_action_definition(function)
        except BadCallbackArguments as exc:
            # Don't keep bad callbacks registered
            self.unregister_callback(function)
            LOGGER.warning(str(exc), exc_info=True)
            return

        try:
            self._register_callback_remotely(remote_action_definition)
        except Exception:
            # Don't keep bad callbacks registered
            self.unregister_callback(function)
            msg = REGISTER_RPC_FAILED % remote_action_definition["functionName"]
            LOGGER.warning(msg)
            # report error
            self._report_experiment_error(msg)

    def _register_callback_remotely(self, remote_action_definition):
        self.connection.register_rpc(remote_action_definition)

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
            experiment, you should also call [comet_ml.CometExperiment.log_other][] with
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
        if not self._streamer_has_connection():
            LOGGER.debug(
                "Failed to send notification. No server connection. Title %s with data: %r",
                title,
                additional_data,
            )
            return

        try:
            name = self.others.get("Name")

            if additional_data is None:
                additional_data = {}

            self.connection.send_notification(
                title,
                status,
                name,
                self._get_experiment_url(),
                additional_data,
                custom_encoder=NestedEncoder,
            )

        except Exception as ex:
            LOGGER.debug(
                "Failed to send notification. Title %s with data: %r",
                title,
                additional_data,
            )
            msg = SEND_NOTIFICATION_FAILED % ex
            LOGGER.error(msg, exc_info=True)
            # report error
            self._report_experiment_error(msg)

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
        if not self.alive:
            return None

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
        if self.disabled:
            raise ExperimentDisabledException(
                ONLINE_LOG_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION % self
            )
        elif not self.alive:
            raise ExperimentNotAlive(
                ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION % self
            )

        if not isinstance(artifact, Artifact):
            raise ValueError(
                ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ARTIFACT_EXCEPTION % artifact
            )

        return log_artifact(
            artifact, experiment=self, rest_api_client=self.rest_api_client
        )

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
        if self.disabled:
            raise ExperimentDisabledException(
                ONLINE_GET_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION % self
            )
        elif not self.alive:
            raise ExperimentNotAlive(
                ONLINE_GET_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION % self
            )

        # Parse the artifact_name
        parsed_name = parse_artifact_name(artifact_name)

        params = {}  # type: Dict[str, Optional[str]]

        if parsed_name.workspace is None and workspace is None:
            # In that case, the backend will use the experiment id to get the workspace
            param_workspace = None
        elif parsed_name.workspace is not None and workspace is not None:
            if parsed_name.workspace != workspace:
                LOGGER.warning(
                    GET_ARTIFACT_WORKSPACE_GIVEN_TWICE
                    % (parsed_name.workspace, parsed_name.name)
                )
            param_workspace = workspace
        elif workspace is None:
            param_workspace = parsed_name.workspace
        else:
            param_workspace = workspace

        if parsed_name.version_or_alias is not None and version_or_alias is not None:
            if parsed_name.version_or_alias != version_or_alias:
                LOGGER.warning(
                    GET_ARTIFACT_VERSION_OR_ALIAS_GIVEN_TWICE
                    % (parsed_name.version_or_alias, parsed_name.name)
                )
            param_version_or_alias = version_or_alias
        elif parsed_name.version_or_alias is not None:
            param_version_or_alias = parsed_name.version_or_alias
        else:
            param_version_or_alias = version_or_alias

        params = {
            "consumer_experiment_key": self.id,
            "experiment_key": self.id,
            "name": parsed_name.name,
            "version_or_alias": param_version_or_alias,
            "workspace": param_workspace,
        }

        logged_artifact = get_artifact(
            self.rest_api_client, params, self.id, self._summary, self.config
        )

        self._summary.increment_section("downloads", "artifacts")

        return logged_artifact

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
        logged_artifact = self.get_artifact(
            artifact_name=name,
            workspace=workspace,
            version_or_alias=version_or_alias,
        )

        return hf_dataset.get_hf_dataset(
            logged_artifact,
            download_directory=download_directory,
            pathname=pathname,
            recursive=recursive,
            use_cached_dataset=use_cached_dataset,
        )

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
        if self.streamer is not None:
            self.streamer.set_offline_zip_uploader(upload_callback)


class ExistingExperiment(Experiment):
    """Existing Experiment allows you to report information to an
    experiment that already exists on comet.com and is not currently
    running. This is useful when your training and testing happen on
    different scripts.

    For example:

    train.py:
    ```
    import comet_ml
    exp = comet_ml.start()
    score = train_model()
    exp.log_metric("train accuracy", score)
    ```

    Now obtain the experiment key from comet.com. If it's not visible
    on your experiment table you can click `Customize` and add it as a
    column.


    test.py:
    ```
    import comet_ml
    exp = comet_ml.start(experiment_key="your experiment key from comet.com")
    score = test_model()
    exp.log_metric("test accuracy", score)
    ```

    Alternatively, you can pass the api_key via an environment
    variable named `COMET_API_KEY` and the previous experiment id via
    an environment variable named `COMET_EXPERIMENT_KEY` and omit them
    from the ExistingExperiment constructor:

    ```python linenums="1"
    import comet_ml

    comet_ml.login()

    exp = comet_ml.start()
    exp.log_metric("test accuracy", 0.3)
    ```

    """

    def __init__(self, api_key=None, previous_experiment=None, **kwargs):
        """
        Append to an existing experiment on the Comet.ml frontend.

        Args:
            api_key (str): Your API key obtained from comet.com
            experiment_key (str): Your experiment key from comet.com, could be set through configuration as well.
            previous_experiment: Deprecated. Use `experiment_key` instead.
            project_name (str): Send your experiment to a specific project. Otherwise will be sent to
                `Uncategorized Experiments`. If project name does not already exists Comet will create a new project.
            workspace (str): Attach an experiment to a project that belongs to this workspace
            log_code (bool): Allows you to enable/disable code logging
            log_graph (bool): Allows you to enable/disable automatic computation graph logging.
            auto_param_logging (bool): Allows you to enable/disable hyper parameters logging
            auto_metric_logging (bool): Allows you to enable/disable metrics logging
            auto_metric_step_rate (int): Controls how often batch metrics are logged
            auto_histogram_tensorboard_logging (bool): Allows you to enable/disable automatic tensorboard histogram logging
            auto_histogram_epoch_rate (int): Controls how often histograms are logged
            auto_histogram_weight_logging(bool): Allows you to enable/disable histogram logging for biases and weights
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
            log_git_patch (bool): Allow you to enable/disable the
                automatic collection of git patch
            display_summary_level (int): Control the summary detail that is
                displayed on the console at end of experiment. If 0, the summary
                notification is still sent. Valid values are 0 to 2.
            disabled (bool): Allows you to disable all network
                communication with the Comet backend. It is useful when you
                want to test to make sure everything is working, without actually
                logging anything.

        Note:
            ExistingExperiment does not alter nor destroy previously
            logged information. To override or add to previous information
            you will have to set the appropriate following parameters to True:

            * log_code
            * log_graph
            * parse_args
            * log_env_details
            * log_git_metadata
            * log_git_patch
            * log_env_gpu
            * log_env_cpu
            * log_env_host

        Example:
            For example, to continue to collect GPU information in an
            `ExistingExperiment` you will need to override these parameters:

            ```python
            >>> experiment = ExistingExperiment(
            ...                 log_env_details=True,
            ...                 log_env_gpu=True)
            ```
        """
        # Validate the previous experiment id
        self.config = get_config()

        if previous_experiment is not None and "experiment_key" in kwargs:
            # TODO: SHOW LOG MESSAGE?
            pass

        # TODO: Document the parameter
        self.step_copy = kwargs.pop("step_copy", None)

        self.previous_experiment = None
        if "experiment_key" in kwargs:
            self.previous_experiment = kwargs["experiment_key"]
        elif previous_experiment is not None:
            self.previous_experiment = previous_experiment
            kwargs["experiment_key"] = previous_experiment

        self.previous_experiment = get_previous_experiment(
            self.previous_experiment, self.config
        )

        if not is_valid_experiment_key(self.previous_experiment):
            raise ValueError(
                ONLINE_EXISTING_INIT_INVALID_KEY_EXCEPTION % self.previous_experiment
            )

        kwargs["experiment_key"] = self.previous_experiment

        # Defaults for ExistingExperiment:
        # For now, don't destroy previous Experiment information by default:

        for key, config_name, default in [
            ("log_code", "comet.auto_log.code", False),
            ("log_graph", "comet.auto_log.graph", False),
            ("parse_args", "comet.auto_log.cli_arguments", False),
            ("log_env_details", "comet.auto_log.env_details", False),
            ("log_git_metadata", "comet.auto_log.git_metadata", False),
            ("log_git_patch", "comet.auto_log.git_patch", False),
            ("log_env_gpu", "comet.auto_log.env_gpu", False),
            ("log_env_cpu", "comet.auto_log.env_cpu", False),
            ("log_env_host", "comet.auto_log.env_host", False),
            ("log_env_network", "comet.auto_log.env_network", False),
            ("log_env_disk", "comet.auto_log.env_disk", False),
        ]:
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = self.config.get_bool(
                    None, config_name, default, not_set_value=None
                )

        self._metrics_with_timestamp: Dict[str, Tuple[Any, int]] = dict()
        self._params_with_timestamp: Dict[str, Tuple[Any, int]] = dict()
        self._others_with_timestamp: Dict[str, Tuple[Any, int]] = dict()

        super().__init__(api_key, **kwargs)

    def _get_experiment_key(self, user_experiment_key: Optional[str]) -> str:
        # In this case we know user_experiment_key is not None as we validated it in __init__
        assert user_experiment_key is not None

        # If we don't copy the experiment, uses the same experiment ID
        if self.step_copy is None:
            return user_experiment_key
        else:
            # When copying, generate a new one
            return get_experiment_key(user_input=None)

    def _add_or_get_run(self):
        if self.step_copy is None:
            return self.connection.get_run(self.previous_experiment)
        else:
            return self.connection.copy_run(self.previous_experiment, self.step_copy)

    def send_notification(
        self,
        title: str,
        status: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        With an `Experiment`, this method will send you a notification
        through email when an experiment ends. However, with an
        `ExistingExperiment` this method does nothing.
        """
        pass

    def get_tags(self) -> List[str]:
        """
        Return the tags of this experiment.

        Returns:
            List[str]: The list of tags.
        """
        if self.alive:
            tags = self.rest_api_client.get_experiment_tags(self.previous_experiment)
            self.tags = self.tags.union(tags.get("tags", []))

        return super().get_tags()

    def _log_metric(
        self,
        name: Any,
        value: Any,
        step: Optional[Any] = None,
        epoch: Optional[Any] = None,
        include_context: bool = True,
        framework: Optional[str] = None,
    ) -> None:

        timestamp_before_sending = get_local_timestamp()

        super()._log_metric(name, value, step, epoch, include_context, framework)

        result = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH)
        self._metrics_with_timestamp[result.value] = (value, timestamp_before_sending)

    def get_metric(self, name: str) -> Any:
        """
        Get the last value for a metric from those logged.

        Args:
            name (str): The name of the metric to get. This method will return the
                last value logged to this metric.
        """
        name = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH).value

        if not self.alive:
            return super().get_metric(name)

        saved_metrics = self.rest_api_client.get_experiment_metric(
            experiment_key=self.previous_experiment, metric=name
        )
        latest_metric = get_latest_metric_value(saved_metrics)

        if latest_metric is None:
            return super().get_metric(name)

        latest_backend_value, latest_backend_timestamp = latest_metric
        local_value, local_timestamp = self._metrics_with_timestamp.get(
            name, (None, None)
        )
        backend_timestamp_is_newer = is_backend_timestamp_newer(
            local_timestamp, latest_backend_timestamp
        )

        if backend_timestamp_is_newer:
            self.metrics[name] = latest_backend_value
            self._metrics_with_timestamp[name] = (
                latest_backend_value,
                latest_backend_timestamp,
            )

        return super().get_metric(name)

    def _log_parameter(
        self,
        name: Any,
        value: Any,
        step: Optional[Any] = None,
        framework: Optional[str] = None,
        source: Optional[str] = None,
        include_context: bool = True,
    ) -> None:

        timestamp_before_sending = get_local_timestamp()

        super()._log_parameter(name, value, step, framework, source, include_context)

        result = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH)
        if include_context:
            param_full_name = self._fullname(result.value)
        else:
            param_full_name = result.value
        self._params_with_timestamp[param_full_name] = (value, timestamp_before_sending)

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
            exp = comet_ml.start()

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

        if not self.alive:
            return super().get_parameter(name)

        saved_values = self.rest_api_client.get_experiment_parameters_summaries(
            experiment_key=self.previous_experiment
        )

        full_name = self._fullname(name)

        for param_name in (full_name, name):
            param_value_with_timestamp = get_param_value(param_name, saved_values)
            if param_value_with_timestamp is None:
                continue

            backend_value, backend_timestamp = param_value_with_timestamp
            local_value, local_timestamp = self._params_with_timestamp.get(
                param_name, (None, None)
            )
            backend_timestamp_is_newer = is_backend_timestamp_newer(
                local_timestamp, backend_timestamp
            )

            if backend_timestamp_is_newer:
                self.params[param_name] = backend_value
                self._params_with_timestamp[param_name] = (
                    backend_value,
                    backend_timestamp,
                )
                # always use shorten "name" for deprecated params
                self.deprecated_params[name] = backend_value

                break

        return super().get_parameter(name)

    def _log_other(
        self,
        key: Any,
        value: Any,
        framework: Optional[str] = None,
        include_context: bool = True,
    ) -> None:

        timestamp_before_sending = get_local_timestamp()

        super()._log_other(key, value, framework, include_context)

        # this condition is from the base class method
        # "others" will not be logged if experiment is not alive
        if not self.alive:
            return

        result = convert_to_string_key(key, max_length=MAXIMAL_KEY_LENGTH)
        self._others_with_timestamp[result.value] = (value, timestamp_before_sending)

    def get_other(self, name: str) -> Any:
        """
        Get the latest value logged for an `other` parameter from those previously logged.

        Args:
            name (str): The name of the other parameter to get.
        """
        name = convert_to_string_key(name, max_length=MAXIMAL_KEY_LENGTH).value

        if not self.alive:
            return super().get_other(name)

        saved_values = self.rest_api_client.get_experiment_others_summaries(
            experiment_key=self.previous_experiment
        )

        other_value_with_timestamp = get_other_value(name, saved_values)
        if other_value_with_timestamp is None:
            return super().get_other(name)

        backend_value, backend_timestamp = other_value_with_timestamp
        local_value, local_timestamp = self._params_with_timestamp.get(
            name, (None, None)
        )
        backend_timestamp_is_newer = is_backend_timestamp_newer(
            local_timestamp, backend_timestamp
        )

        if backend_timestamp_is_newer:
            self.others[name] = backend_value
            self._others_with_timestamp[name] = (
                backend_value,
                backend_timestamp,
            )

        return super().get_other(name)
