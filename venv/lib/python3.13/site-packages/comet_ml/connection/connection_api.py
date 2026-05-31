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

import gzip
import itertools
import json
import logging
import os
from http import HTTPStatus
from typing import (
    IO,
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urlparse

import requests
import requests.utils

from .._typing import HeartBeatResponse, PreparedRequest
from ..batch_utils import MessageBatchItem
from ..config import (
    MAXIMAL_LENGTH_OF_OUTPUT_LINE,
    Config,
    get_backend_address,
    get_check_tls_certificate,
    get_comet_timeout_http,
    get_config,
)
from ..constants import (
    ASSET_TYPE_AUDIO,
    ASSET_TYPE_VIDEO,
    PAYLOAD_ADDITIONAL_SYSTEM_INFO_LIST,
    PAYLOAD_API_KEY,
    PAYLOAD_ASSET_NAME,
    PAYLOAD_COMMAND,
    PAYLOAD_DEPENDENCIES,
    PAYLOAD_DEPENDENCY_NAME,
    PAYLOAD_DEPENDENCY_VERSION,
    PAYLOAD_ENV,
    PAYLOAD_EXECUTABLE,
    PAYLOAD_EXPERIMENT_KEY,
    PAYLOAD_FILE_PATH,
    PAYLOAD_GPU_STATIC_INFO_LIST,
    PAYLOAD_HOSTNAME,
    PAYLOAD_HTML,
    PAYLOAD_INSTALLED_PACKAGES,
    PAYLOAD_IP,
    PAYLOAD_LOCAL_TIMESTAMP,
    PAYLOAD_MACHINE,
    PAYLOAD_METADATA,
    PAYLOAD_MODEL_GRAPH,
    PAYLOAD_MODEL_NAME,
    PAYLOAD_OFFSET,
    PAYLOAD_OS,
    PAYLOAD_OS_PACKAGES,
    PAYLOAD_OS_RELEASE,
    PAYLOAD_OS_TYPE,
    PAYLOAD_OUTPUT,
    PAYLOAD_OUTPUT_LINES,
    PAYLOAD_OVERRIDE,
    PAYLOAD_PARAMETER_NAME,
    PAYLOAD_PID,
    PAYLOAD_PROCESSOR,
    PAYLOAD_PROVIDER,
    PAYLOAD_PYTHON_VERSION,
    PAYLOAD_PYTHON_VERSION_VERBOSE,
    PAYLOAD_REMOTE_ASSETS,
    PAYLOAD_RUN_CONTEXT,
    PAYLOAD_STDERR,
    PAYLOAD_TIMESTAMP,
    PAYLOAD_TOTAL_RAM,
    PAYLOAD_USED_RAM,
    PAYLOAD_USER,
    STATUS_RESPONSE_CPU_MONITOR_INTERVAL_MILLIS,
    STATUS_RESPONSE_GPU_MONITOR_INTERVAL_MILLIS,
    STATUS_RESPONSE_IS_ALIVE_BEAT_DURATION_MILLIS,
    STATUS_RESPONSE_PARAMETER_UPDATE_INTERVAL_MILLIS,
    STATUS_RESPONSE_PENDING_RPCS,
    SUPPORTED_VIDEO_FORMATS,
)
from ..exceptions import (
    API_KEY_NOT_REGISTERED,
    BACKEND_CUSTOM_ERROR,
    EXPERIMENT_ALREADY_EXISTS,
    MAX_EXPERIMENTS_NUMBER_REACHED,
    NO_PROJECT_NAME_SPECIFIED,
    NO_WRITE_ACCESS_TO_EXPERIMENT,
    NON_EXISTING_TEAM,
    OBSOLETE_SDK_VERSION,
    PROJECT_CONSIDERED_LLM,
    PROJECT_NAME_TOO_LONG,
    SERVICE_ACCOUNT_WORKSPACE_RESTRICTED,
    VIEW_ONLY_PERMISSIONS,
    BackendCustomError,
    BackendVersionTooOld,
    CometException,
    CometRestApiException,
    CometRestApiValueError,
    ExperimentAlreadyUploaded,
    ExperimentNotFound,
    InvalidAPIKey,
    InvalidWorkspace,
    MaxExperimentNumberReachedException,
    NotFound,
    PaymentRequired,
    ProjectConsideredLLM,
    ProjectNameEmpty,
    ProjectNameIsTooLong,
    SDKVersionIsTooOldException,
    ServiceAccountWorkspaceRestricted,
    Unauthorized,
    ViewOnlyAccessException,
)
from ..file_uploader import AssetDataUploadProcessor, AssetUploadProcessor
from ..file_utils import get_file_extension
from ..handshake import ExperimentHandshakeResponse, parse_experiment_handshake_response
from ..heartbeat import (
    HEARTBEAT_CPU_MONITOR_INTERVAL,
    HEARTBEAT_GPU_MONITOR_INTERVAL,
    HEARTBEAT_PARAMETERS_BATCH_UPDATE_INTERVAL,
)
from ..json_encoder import NestedEncoder
from ..logging_messages import (
    BACKEND_VERSION_CHECK_ERROR,
    CONNECTION_DOWNLOAD_REGISTRY_MODEL_VERSION_OR_STAGE_EXCEPTION,
    CONNECTION_LOG_EXPERIMENT_ASSET_NO_NAME_WARNING,
    CONNECTION_LOG_EXPERIMENT_SYSTEM_INFO_EXCEPTION,
    CONNECTION_MISSING_CPU_MONITOR_INTERVAL_EXCEPTION,
    CONNECTION_MISSING_GPU_MONITOR_INTERVAL_EXCEPTION,
    CONNECTION_MISSING_HEARTBEAT_DURATION_EXCEPTION,
    CONNECTION_MISSING_PARAMETERS_UPDATE_INTERVAL_EXCEPTION,
    CONNECTION_NAME_TOO_LONG_DEFAULT_ERROR,
    CONNECTION_REGISTER_MODEL_DESCRIPTION_WARNING,
    CONNECTION_REGISTER_MODEL_INVALID_MODEL_NAME_EXCEPTION,
    CONNECTION_REGISTER_MODEL_INVALID_STAGES_LIST_EXCEPTION,
    CONNECTION_REGISTER_MODEL_NO_MODEL_EXCEPTION,
    CONNECTION_REGISTER_MODEL_PUBLIC_WARNING,
    CONNECTION_REGISTER_MODEL_STAGES_IGNORED_WARNING,
    CONNECTION_REGISTER_MODEL_STATUS_IGNORED_WARNING,
    CONNECTION_REGISTER_MODEL_SUCCESS_INFO,
    CONNECTION_REGISTER_MODEL_TAGS_IGNORED_WARNING,
    CONNECTION_SET_EXPERIMENT_STATE_UNSUPPORTED_EXCEPTION,
    CONNECTION_UPDATE_PROJECT_BY_ID_MISSING_REQUIRED_EXCEPTION,
    CONNECTION_UPDATE_PROJECT_MISSING_REQUIRED_EXCEPTION,
    CONNECTION_VIEW_ONLY_CREATE_EXPERIMENT_EXCEPTION,
    EXPERIMENT_LOG_LLM_PROJECT_ERROR_MSG,
    EXTENSION_NOT_FOUND,
    EXTENSION_NOT_SUPPORTED,
    INVALID_CONFIG_MINIMAL_BACKEND_VERSION,
    REPORTING_ERROR,
)
from ..messages import UploadFileMessage
from ..semantic_version import SemanticVersion
from ..utils import (
    encode_metadata,
    get_comet_version,
    get_time_monotonic,
    local_timestamp,
    masked_api_key,
    optional_update,
    proper_registry_model_name,
    timestamp_milliseconds,
    url_join,
)
from .connection_helpers import (
    API_KEY_HEADER,
    format_message_batch_items,
    format_remote_assets_batch_items,
    format_stdout_message_batch_items,
    raise_for_status_code,
    split_output_by_breaks_and_length,
)
from .connection_upload import send_file, send_file_like
from .connection_url_helpers import (
    SYSTEM_DETAILS_ENDPOINT,
    SYSTEM_DETAILS_WRITE_ENDPOINT,
    add_run_url,
    add_tags_url,
    copy_experiment_url,
    create_asset_url,
    get_backend_version_url,
    get_ping_backend_url,
    get_points_3d_upload_limits_url,
    get_run_url,
    get_upload_url,
    metrics_batch_url,
    new_symlink_url,
    notification_url,
    notify_url,
    offline_experiment_times_url,
    parameters_batch_url,
    pending_rpcs_url,
    register_rpc_url,
    rpc_result_url,
    status_report_update_url,
)
from .http_session import (
    STATUS_FORCELIST_FULL,
    STATUS_FORCELIST_NO_AUTH_ERRORS,
    get_comet_http_session,
    get_retry_strategy,
    get_retry_strategy_for_get_or_add_run,
)

LOGGER = logging.getLogger("comet_ml.connection")


def _should_report(config: Config, server_address: str) -> bool:
    backend_host = urlparse(server_address).hostname

    if backend_host.endswith("comet.com"):
        default = True
    else:
        default = False

    return config.get_bool(None, "comet.internal.reporting", default=default)


def _json_post(url, session, headers, body, timeout):
    response = session.post(
        url=url, data=json.dumps(body), headers=headers, timeout=timeout
    )

    raise_for_status_code(response)
    return response


class LowLevelHTTPClient(object):
    """A low-level HTTP client that centralize common code and behavior between
    the two backends clients, the optimizer"""

    def __init__(
        self,
        api_key: str,
        server_address: str,
        default_timeout: float,
        verify_tls: bool,
        config: Config,
        headers: Optional[Dict[str, Optional[str]]] = None,
        default_retry: bool = True,
        retry_auth_errors: bool = False,
    ) -> None:
        self.server_address = server_address
        self.verify_tls = verify_tls

        self.session = get_comet_http_session(
            verify_tls=verify_tls, api_key=api_key, config=config
        )

        status_forcelist = (
            STATUS_FORCELIST_FULL
            if retry_auth_errors
            else STATUS_FORCELIST_NO_AUTH_ERRORS
        )
        self.retry_session = get_comet_http_session(
            retry_strategy=get_retry_strategy(status_forcelist, config=config),
            verify_tls=self.verify_tls,
            api_key=api_key,
            config=config,
        )

        if headers is None:
            headers = {}

        self.headers = headers

        self.default_retry = default_retry
        self.default_timeout = default_timeout

    def close(self):
        self.session.close()
        self.retry_session.close()

    def get(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry: Optional[bool] = None,
        timeout: Optional[float] = None,
        check_status_code: bool = False,
        stream: bool = False,
    ) -> requests.Response:
        if retry is None:
            retry = self.default_retry

        if timeout is None:
            timeout = self.default_timeout

        final_headers = self.headers.copy()
        if headers:
            final_headers.update(headers)

        # Do not log the headers as they may contain the authentication keys
        LOGGER.debug(
            "GET HTTP Call, url %r, params %r, retry %r, timeout %r",
            url,
            params,
            retry,
            timeout,
        )

        if retry:
            session = self.retry_session
        else:
            session = self.session

        response: requests.Response = session.get(
            url,
            params=params,
            headers=final_headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=True,  # NB: we need this to allow direct assets download (made explicit)
        )

        if check_status_code:
            raise_for_status_code(response=response)

        return response

    def post(
        self,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        retry: Optional[bool] = None,
        timeout: Optional[float] = None,
        params: Optional[Any] = None,
        files: Optional[Any] = None,
        check_status_code: bool = False,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
        session: Optional[requests.Session] = None,
    ) -> requests.Response:
        return self.do(
            method="POST",
            url=url,
            payload=payload,
            headers=headers,
            retry=retry,
            timeout=timeout,
            params=params,
            files=files,
            check_status_code=check_status_code,
            custom_encoder=custom_encoder,
            compress=compress,
            session=session,
        )

    def put(
        self,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        retry: Optional[bool] = None,
        timeout: Optional[float] = None,
        params: Optional[Any] = None,
        files: Optional[Any] = None,
        check_status_code: bool = False,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
    ) -> requests.Response:
        return self.do(
            method="PUT",
            url=url,
            payload=payload,
            headers=headers,
            retry=retry,
            timeout=timeout,
            params=params,
            files=files,
            check_status_code=check_status_code,
            custom_encoder=custom_encoder,
            compress=compress,
        )

    def do(
        self,
        method: str,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        retry: Optional[bool] = None,
        timeout: Optional[float] = None,
        params: Optional[Any] = None,
        files: Optional[Any] = None,
        check_status_code: bool = False,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
        session: Optional[requests.Session] = None,
    ) -> requests.Response:

        if retry is None:
            retry = self.default_retry

        if timeout is None:
            timeout = self.default_timeout

        final_headers = self.headers.copy()
        if headers:
            final_headers.update(headers)

        # Do not log the headers as they might contain the authentication keys
        LOGGER.debug(
            "%s HTTP Call, url %r, payload %r, retry %r, timeout %r",
            method,
            url,
            payload,
            retry,
            timeout,
        )

        if session is None:
            if retry:
                session = self.retry_session
            else:
                session = self.session

        if files:
            # File upload, multipart request
            response = session.request(
                method=method,
                url=url,
                data=payload,
                headers=final_headers,
                timeout=timeout,
                files=files,
                params=params,
            )
        else:
            # JSON request

            # Format the payload with potentially some custom encoder
            data = json.dumps(payload, cls=custom_encoder).encode("utf-8")
            final_headers["Content-Type"] = "application/json;charset=utf-8"
            if compress is True:
                data = gzip.compress(data)
                final_headers["Content-Encoding"] = "gzip"

            response = session.request(
                method=method,
                url=url,
                data=data,
                headers=final_headers,
                timeout=timeout,
                params=params,
            )

        LOGGER.debug(
            "%s HTTP Response, url %r, status_code %d, response %r",
            method,
            url,
            response.status_code,
            response.content,
        )
        if response.status_code != 200:
            LOGGER.debug(
                "Not OK %s HTTP Response headers: %s", method, response.headers
            )

        if check_status_code:
            raise_for_status_code(response=response)

        return response


class RestServerConnection(object):
    """
    A static class that handles the connection with the server endpoints.
    """

    def __init__(
        self,
        api_key: str,
        experiment_id: str,
        server_address: str,
        default_timeout: float,
        verify_tls: bool,
        config: Optional[Config] = None,
    ) -> None:
        self.api_key = api_key
        self.experiment_id = experiment_id

        # Set once get_run_id is called
        self.run_id = None
        self.project_id = None

        self._server_address = server_address

        if config is None:
            config = get_config()
        self.config = config

        self.default_timeout = default_timeout
        self._low_level_http_client = LowLevelHTTPClient(
            api_key=api_key,
            server_address=server_address,
            default_retry=False,
            default_timeout=default_timeout,
            headers={"X-COMET-DEBUG-EXPERIMENT-KEY": experiment_id},
            verify_tls=verify_tls,
            config=self.config,
        )

    @property
    def server_address(self) -> str:
        return self._server_address

    def server_hostname(self) -> AnyStr:
        parsed = urlparse(self.server_address)
        return parsed.netloc

    def close(self):
        self._low_level_http_client.close()

    def heartbeat(self) -> HeartBeatResponse:
        """Inform the backend that we are still alive"""
        LOGGER.debug("Doing an heartbeat")
        return self.update_experiment_status(self.run_id, self.project_id, True)

    def ping_backend(self) -> None:
        ping_url = get_ping_backend_url(self.server_address)
        self._low_level_http_client.get(ping_url, check_status_code=True, retry=False)

    def update_experiment_status(
        self, run_id: str, project_id: str, is_alive: bool, offline=False
    ) -> Optional[HeartBeatResponse]:
        endpoint_url = status_report_update_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            "runId": run_id,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "projectId": project_id,
            "is_alive": is_alive,
            "local_timestamp": local_timestamp(),
            "offline": offline,
        }

        if run_id is None:
            LOGGER.debug(
                "Unable to send experiment status update, run_id is None. Payload: %r, url: %r",
                payload,
                endpoint_url,
            )
            return None

        LOGGER.debug(
            "Sending experiment status update with payload: %r, url: %r",
            payload,
            endpoint_url,
        )

        r = self._low_level_http_client.post(
            url=endpoint_url, payload=payload, retry=True
        )

        if r.status_code != 200:
            raise ValueError(r.content)

        data = r.json()
        LOGGER.debug("Update experiment status response payload: %r", data)
        beat_duration = data.get(STATUS_RESPONSE_IS_ALIVE_BEAT_DURATION_MILLIS)

        if beat_duration is None:
            raise ValueError(CONNECTION_MISSING_HEARTBEAT_DURATION_EXCEPTION)

        gpu_monitor_interval = data.get(STATUS_RESPONSE_GPU_MONITOR_INTERVAL_MILLIS)

        if gpu_monitor_interval is None:
            raise ValueError(CONNECTION_MISSING_GPU_MONITOR_INTERVAL_EXCEPTION)

        # Default the backend response until it responds:
        cpu_monitor_interval = data.get(
            STATUS_RESPONSE_CPU_MONITOR_INTERVAL_MILLIS, 68 * 1000
        )

        if cpu_monitor_interval is None:
            raise ValueError(CONNECTION_MISSING_CPU_MONITOR_INTERVAL_EXCEPTION)

        # Default the backend response until actual data received
        parameters_update_interval = data.get(
            STATUS_RESPONSE_PARAMETER_UPDATE_INTERVAL_MILLIS,
            self.config.get_int(None, "comet.message_batch.parameters_interval") * 1000,
        )
        if parameters_update_interval is None:
            raise ValueError(CONNECTION_MISSING_PARAMETERS_UPDATE_INTERVAL_EXCEPTION)

        pending_rpcs = data.get(STATUS_RESPONSE_PENDING_RPCS, False)

        return_data = {
            HEARTBEAT_GPU_MONITOR_INTERVAL: gpu_monitor_interval,
            HEARTBEAT_CPU_MONITOR_INTERVAL: cpu_monitor_interval,
            HEARTBEAT_PARAMETERS_BATCH_UPDATE_INTERVAL: parameters_update_interval,
        }

        return beat_duration, return_data, pending_rpcs

    def add_run(
        self,
        project_name: Optional[str],
        workspace: Optional[str],
        offline: bool = False,
        get_or_create_mode: bool = False,
    ) -> ExperimentHandshakeResponse:
        """
        Adds new experiment record to the backend and return assigned experiment options from the server.
        :param project_name: project name for the new experiment, can be None
        :param workspace: workspace name for the new experiment, can be None
        :param offline: should the new experiment be marked as offline
        :param get_or_create_mode: the flag to indicate to backend if to raise an error if experiment already exists (False)
            or to skip error raising and just return experiment handshake data (True)
        :return: ExperimentHandshakeResponse
        """
        endpoint_url = add_run_url(self.server_address)

        # We used to pass the team name as second parameter then we migrated
        # to workspaces. We keep using the same payload field, so the compatibility
        # is ensured by the backend and old SDK version still use it anyway
        payload = {
            "apiKey": self.api_key,
            "local_timestamp": local_timestamp(),
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "offline": offline,
            "projectName": project_name,
            "teamName": workspace,
            "libVersion": get_comet_version(),
            "getOrCreateEnabled": get_or_create_mode,
        }

        config = get_config()
        retry_strategy = get_retry_strategy_for_get_or_add_run(
            status_forcelist=STATUS_FORCELIST_NO_AUTH_ERRORS,
            config=config,
        )
        session = get_comet_http_session(
            retry_strategy=retry_strategy,
            verify_tls=self._low_level_http_client.verify_tls,
            api_key=self.api_key,
            config=config,
        )

        LOGGER.debug("Get run id URL: %s", endpoint_url)
        with session:
            r = self._low_level_http_client.post(
                url=endpoint_url, payload=payload, session=session
            )

            if r.status_code != 200:
                if r.status_code in [400, 403]:
                    # Check if the api key was invalid
                    data = r.json()  # Raise a ValueError if failing
                    code = data.get("sdk_error_code")
                    if code == API_KEY_NOT_REGISTERED:
                        raise InvalidAPIKey(self.api_key, self.server_hostname())

                    elif code == NON_EXISTING_TEAM:
                        raise InvalidWorkspace(workspace)

                    elif code == SERVICE_ACCOUNT_WORKSPACE_RESTRICTED:
                        raise ServiceAccountWorkspaceRestricted(workspace)

                    elif code == NO_PROJECT_NAME_SPECIFIED:
                        raise ProjectNameEmpty()

                    elif code == EXPERIMENT_ALREADY_EXISTS:
                        raise ExperimentAlreadyUploaded(self.experiment_id)

                    elif code == PROJECT_NAME_TOO_LONG:
                        # Add fallback if the backend stop sending the msg
                        err_msg = data.get(
                            "msg",
                            CONNECTION_NAME_TOO_LONG_DEFAULT_ERROR,
                        )
                        raise ProjectNameIsTooLong(err_msg)

                    elif code == VIEW_ONLY_PERMISSIONS:
                        raise ViewOnlyAccessException(
                            CONNECTION_VIEW_ONLY_CREATE_EXPERIMENT_EXCEPTION
                        )

                    elif code == OBSOLETE_SDK_VERSION:
                        raise SDKVersionIsTooOldException(data.get("msg", None))

                    elif code == MAX_EXPERIMENTS_NUMBER_REACHED:
                        raise MaxExperimentNumberReachedException(
                            project_name=project_name, workspace=workspace
                        )

                    elif code == PROJECT_CONSIDERED_LLM:
                        raise ProjectConsideredLLM(
                            EXPERIMENT_LOG_LLM_PROJECT_ERROR_MSG.format(project_name)
                        )

                # raise Comet specific error which can be meaningfully processed at later stages
                raise CometRestApiException("POST", response=r)

            res_body = json.loads(r.content.decode("utf-8"))

        LOGGER.debug("New run response body: %s", res_body)

        return self._parse_run_id_res_body(res_body)

    def get_run(self, previous_experiment: str) -> ExperimentHandshakeResponse:
        """
        Gets from the backend options assigned to an existing experiment.
        :param previous_experiment: the experiment id to continue logging to
        :return: ExperimentHandshakeResponse
        """
        endpoint_url = get_run_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            "local_timestamp": local_timestamp(),
            "previousExperiment": previous_experiment,
            "libVersion": get_comet_version(),
        }
        config = get_config()
        retry_strategy = get_retry_strategy_for_get_or_add_run(
            status_forcelist=STATUS_FORCELIST_NO_AUTH_ERRORS,
            config=config,
        )
        session = get_comet_http_session(
            retry_strategy=retry_strategy,
            verify_tls=self._low_level_http_client.verify_tls,
            api_key=self.api_key,
            config=config,
        )

        LOGGER.debug("Get old run id URL: %s", endpoint_url)
        with session:
            r = self._low_level_http_client.post(
                url=endpoint_url, payload=payload, session=session
            )

            if r.status_code != 200:
                if r.status_code == 400:

                    data = r.json()  # Raise a ValueError if failing
                    code = data.get("sdk_error_code")
                    if code == API_KEY_NOT_REGISTERED:
                        raise InvalidAPIKey(self.api_key, self.server_hostname())
                    elif code == NO_WRITE_ACCESS_TO_EXPERIMENT:
                        raise ExperimentNotFound(data.get("msg"))
                    elif code == BACKEND_CUSTOM_ERROR:
                        raise BackendCustomError(data.get("msg"))

                # raise Comet specific error which can be meaningfully processed at later stages
                raise CometRestApiException("POST", response=r)

            res_body = json.loads(r.content.decode("utf-8"))

        LOGGER.debug("Old run response body: %s", res_body)

        return self._parse_run_id_res_body(res_body)

    def get_points_3d_upload_limits(self) -> Optional[Dict[str, int]]:
        endpoint_url = get_points_3d_upload_limits_url(self.server_address)

        payload = {"apiKey": self.api_key, "experimentKey": self.experiment_id}

        try:
            response = self._low_level_http_client.post(
                url=endpoint_url, payload=payload
            )

            return json.loads(response.text)

        except Exception:
            return None

    def copy_run(
        self, previous_experiment: str, copy_step: bool
    ) -> ExperimentHandshakeResponse:
        """
        Gets a run id from an existing experiment.
        :param previous_experiment: the experiment id to copy
        :param copy_step: copy up the step passed
        :return: ExperimentHandshakeResponse
        """
        endpoint_url = copy_experiment_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            "copiedExperimentKey": previous_experiment,
            "newExperimentKey": self.experiment_id,
            "stepToCopyTo": copy_step,
            "localTimestamp": local_timestamp(),
            "libVersion": get_comet_version(),
        }
        LOGGER.debug("Copy run URL: %s", endpoint_url)

        r = self._low_level_http_client.post(
            url=endpoint_url, payload=payload, retry=True
        )

        if r.status_code != 200:
            if r.status_code == 400:
                # Check if the api key was invalid
                data = r.json()  # Raise a ValueError if failing
                if data.get("sdk_error_code") == API_KEY_NOT_REGISTERED:
                    raise InvalidAPIKey(self.api_key, self.server_hostname())

            raise ValueError(r.content)

        res_body = json.loads(r.content.decode("utf-8"))

        LOGGER.debug("Copy run response body: %s", res_body)

        return self._parse_run_id_res_body(res_body)

    def _parse_run_id_res_body(
        self, res_body: Dict[str, Any]
    ) -> ExperimentHandshakeResponse:
        response = parse_experiment_handshake_response(res_body)

        # Save run_id and project_id around
        self.run_id = response.run_id
        self.project_id = response.project_id

        return response

    def report(self, event_name: str = None, err_msg: str = None) -> None:
        try:
            if event_name is not None:

                if not _should_report(self.config, self.server_address):
                    return None

                endpoint_url = notify_url(self.server_address)
                # Automatically add the sdk_ prefix to the event name
                real_event_name = "sdk_{}".format(event_name)

                payload = {
                    "event_name": real_event_name,
                    "api_key": self.api_key,
                    "run_id": self.run_id,
                    "experiment_key": self.experiment_id,
                    "project_id": self.project_id,
                    "err_msg": err_msg,
                    "timestamp": local_timestamp(),
                }

                LOGGER.debug("Report notify URL: %s", endpoint_url)

                # We use half of the timeout as the call might happen in the
                # main thread that we don't want to block and report data is
                # usually not critical
                timeout = int(self.default_timeout / 2)

                self._low_level_http_client.post(
                    endpoint_url,
                    payload,
                    timeout=timeout,
                    check_status_code=True,
                )

        except Exception:
            LOGGER.debug("Error reporting %s", event_name, exc_info=True)
            pass

    def offline_experiment_start_end_time(
        self, run_id: str, start_time: Optional[int], stop_time: int
    ) -> None:
        endpoint_url = offline_experiment_times_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            "runId": run_id,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "startTimestamp": start_time,
            "endTimestamp": stop_time,
        }

        LOGGER.debug(
            "Offline experiment start time and end time update with payload: %s",
            payload,
        )

        r = self._low_level_http_client.post(endpoint_url, payload)

        if r.status_code != 200:
            raise ValueError(r.content)

        return None

    def add_tags(self, added_tags: List[str]) -> None:
        endpoint_url = add_tags_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "addedTags": added_tags,
        }

        LOGGER.debug("Add tags with payload: %s", payload)

        r = self._low_level_http_client.post(
            url=endpoint_url, payload=payload, retry=True
        )

        if r.status_code != 200:
            raise ValueError(r.content)

        return None

    def get_upload_url(self, upload_type: str) -> str:
        return get_upload_url(self.server_address, upload_type)

    def get_pending_rpcs(self) -> Dict[str, Any]:
        endpoint_url = pending_rpcs_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
        }

        LOGGER.debug("Get pending RPCS with payload: %s", payload)

        r = self._low_level_http_client.get(url=endpoint_url, params=payload)

        if r.status_code != 200:
            raise ValueError(r.content)

        res_body = json.loads(r.content.decode("utf-8"))

        LOGGER.debug("Pending RPC response: %r", res_body)

        return res_body

    def register_rpc(self, function_definition: Dict[str, Any]) -> None:
        endpoint_url = register_rpc_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
        }
        # We might replace this payload.update by defining the exact
        # parameters once the payload body has been stabilized
        payload.update(function_definition)

        LOGGER.debug("Register RPC with payload: %s", payload)

        r = self._low_level_http_client.post(endpoint_url, payload=payload)

        if r.status_code != 200:
            raise ValueError(r.content)

        return None

    def send_rpc_result(
        self, callId: str, result: Dict[str, Any], start_time: int, end_time: int
    ) -> None:
        endpoint_url = rpc_result_url(self.server_address)

        error = result.get("error", "")
        error_stacktrace = result.get("traceback", "")
        result = result.get("result", "")

        payload = {
            "apiKey": self.api_key,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "callId": callId,
            "result": result,
            "errorMessage": error,
            "errorStackTrace": error_stacktrace,
            "startTimeMs": start_time,
            "endTimeMs": end_time,
        }

        LOGGER.debug("Sending RPC result with payload: %s", payload)

        r = self._low_level_http_client.post(endpoint_url, payload=payload)

        if r.status_code != 200:
            raise ValueError(r.content)

        return None

    def send_new_symlink(self, project_name: str) -> None:
        endpoint_url = new_symlink_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            PAYLOAD_EXPERIMENT_KEY: self.experiment_id,
            "projectName": project_name,
        }

        LOGGER.debug("new symlink: %s", payload)

        r = self._low_level_http_client.get(url=endpoint_url, params=payload)

        if r.status_code != 200:
            raise ValueError(r.content)

    def send_notification(
        self,
        title: str,
        status: Optional[str],
        experiment_name: str,
        experiment_link: str,
        notification_map: Optional[Dict[str, Any]],
        custom_encoder: Optional[Type[json.JSONEncoder]],
    ) -> None:
        endpoint_url = notification_url(self.server_address)

        payload = {
            "api_key": self.api_key,
            "title": title,
            "status": status,
            "experiment_key": self.experiment_id,
            "experiment_name": experiment_name,
            "experiment_link": experiment_link,
            "additional_data": notification_map,
        }

        LOGGER.debug("Notification: %s", payload)
        r = self._low_level_http_client.post(
            endpoint_url,
            payload=payload,
            check_status_code=False,
            custom_encoder=custom_encoder,
        )
        if r.status_code != 200 and r.status_code != 204:
            if r.status_code == 404:
                # for backwards and forwards compatibility, this endpoint was introduced backend v1.1.103
                return

            # raise Comet specific error which can be meaningfully processed at later stages
            raise CometRestApiException("POST", response=r)

    def log_parameters_batch(
        self, items: List[MessageBatchItem], compress: bool = True
    ) -> None:
        endpoint_url = parameters_batch_url(self.server_address)

        LOGGER.debug(
            "Sending parameters batch, length: %d, compression enabled: %s, endpoint: %s",
            len(items),
            compress,
            endpoint_url,
        )

        self._send_batch(
            batch_items=items, endpoint_url=endpoint_url, compress=compress
        )

    def log_metrics_batch(
        self, items: List[MessageBatchItem], compress: bool = True
    ) -> None:
        endpoint_url = metrics_batch_url(self.server_address)

        LOGGER.debug(
            "Sending metrics batch, length: %d, compression enabled: %s, endpoint: %s",
            len(items),
            compress,
            endpoint_url,
        )

        self._send_batch(
            batch_items=items, endpoint_url=endpoint_url, compress=compress
        )

    def create_asset(
        self,
        experiment_key: str,
        asset_name: str,
        asset_type: str,
        metadata: Dict,
        step: Optional[int],
    ) -> str:
        endpoint_url = create_asset_url(self.server_address)

        payload = {
            "apiKey": self.api_key,
            "experimentKey": experiment_key,
            "userFileName": asset_name,
            "assetType": asset_type,
            "step": step,
        }

        if metadata is not None:
            encoded_metadata = encode_metadata(metadata)
            if encoded_metadata:
                payload["metadata"] = encoded_metadata

        response = self._low_level_http_client.post(
            url=endpoint_url,
            payload=payload,
            check_status_code=True,
        )

        asset_id = json.loads(response.text)["assetId"]

        return asset_id

    def _send_batch(
        self,
        batch_items: List[MessageBatchItem],
        endpoint_url: str,
        compress: bool = True,
    ) -> None:
        payload = format_message_batch_items(
            batch_items=batch_items, experiment_key=self.experiment_id
        )
        LOGGER.debug("ENCODED BATCH BODY DATA %r", payload)

        headers = {API_KEY_HEADER: self.api_key}
        r = self._low_level_http_client.post(
            url=endpoint_url,
            payload=payload,
            check_status_code=False,
            custom_encoder=NestedEncoder,
            headers=headers,
            compress=compress,
            retry=True,
        )

        raise_for_status_code(response=r)

        LOGGER.debug(
            "Batch of %d messages has been sent to %r", len(batch_items), endpoint_url
        )


class Reporting(object):
    @staticmethod
    def report(
        config: Config,
        event_name: Optional[str] = None,
        api_key: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_key: Optional[str] = None,
        project_id: Optional[str] = None,
        err_msg: Optional[str] = None,
        is_alive: Optional[bool] = None,
    ) -> None:
        try:
            if event_name is not None:
                server_address = get_backend_address(config)

                if not _should_report(config, server_address):
                    return None

                verify_tls = get_check_tls_certificate(config)

                endpoint_url = notify_url(server_address)
                headers = {"Content-Type": "application/json;charset=utf-8"}
                # Automatically add the sdk_ prefix to the event name
                real_event_name = "sdk_{}".format(event_name)

                payload = {
                    "event_name": real_event_name,
                    "api_key": api_key,
                    "run_id": run_id,
                    "experiment_key": experiment_key,
                    "project_id": project_id,
                    "err_msg": err_msg,
                    "timestamp": local_timestamp(),
                }

                session = get_comet_http_session(
                    retry_strategy=get_retry_strategy(
                        status_forcelist=STATUS_FORCELIST_NO_AUTH_ERRORS, config=config
                    ),
                    verify_tls=verify_tls,
                    api_key=api_key,
                )

                if experiment_key:
                    headers["X-COMET-DEBUG-EXPERIMENT-KEY"] = experiment_key

                # We use half of the timeout as the call might happens in the
                # main thread that we don't want to block and report data is
                # usually not critical
                timeout = int(get_comet_timeout_http(config) / 2)

                with session:
                    _json_post(endpoint_url, session, headers, payload, timeout)

        except Exception:
            LOGGER.debug(REPORTING_ERROR, event_name, exc_info=True)


def _debug_proxy_for_http(target_url: str) -> Optional[str]:
    """Return the proxy (or None) used by requests for the target url"""
    proxies = requests.utils.get_environ_proxies(target_url)
    return requests.utils.select_proxy(target_url, proxies)


class OptimizerAPI(object):
    """
    API for talking to Optimizer Server.
    """

    def __init__(
        self,
        api_key: str,
        low_level_api_client: LowLevelHTTPClient,
        server_address: str,
    ) -> None:
        """ """
        self.DEFAULT_VERSION = "v1"
        self.URLS = {"v1": {"SERVER": server_address}}
        self._version = self.DEFAULT_VERSION
        self._api_key = api_key
        self.low_level_api_client = low_level_api_client

    def get_url(self, version: Optional[str] = None) -> str:
        """
        Returns the URL for this version of the API.
        """
        version = version if version is not None else self._version
        return self.URLS[version]["SERVER"]

    def get_url_server(self, version: Optional[str] = None) -> str:
        """
        Returns the URL server for this version of the API.
        """
        version = version if version is not None else self._version
        return self.URLS[version]["SERVER"]

    def get_url_end_point(self, end_point: str, version: Optional[str] = None) -> str:
        """
        Return the URL + end point.
        """
        return url_join(self.get_url(version), end_point)

    def get_request(
        self, end_point: str, params: Dict[str, str], return_type: str = "json"
    ) -> Any:
        """
        Given an end point and a dictionary of params,
        return the results.
        """
        from .. import __version__

        url = self.get_url_end_point(end_point)
        LOGGER.debug("API.get_request: url = %s, params = %s", url, params)
        headers = {"X-API-KEY": self._api_key, "PYTHON-SDK-VERSION": __version__}

        response = self.low_level_api_client.get(
            url, params=params, headers=headers, retry=False
        )

        raise_exception = None
        try:
            response.raise_for_status()
        except requests.HTTPError as exception:
            if exception.response.status_code == 401:
                raise_exception = ValueError("Invalid COMET_API_KEY")
            else:
                raise
        if raise_exception:
            raise raise_exception
        ### Return data based on return_type:
        if return_type == "json":
            return response.json()
        elif return_type == "binary":
            return response.content
        elif return_type == "text":
            return response.text
        elif return_type == "response":
            return response

    def post_request(
        self, end_point: str, json: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Given an end point and a dictionary of json,
        post the json, and return the results.
        """
        from .. import __version__

        url = self.get_url_end_point(end_point)
        if json is None:
            json = {}
        LOGGER.debug("API.post_request: url = %s, json = %s", url, json)
        headers = {
            "PYTHON-SDK-VERSION": __version__,
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        if "files" in kwargs:
            del headers["Content-Type"]

        response = self.low_level_api_client.post(
            url, headers=headers, payload=json, retry=False
        )

        raise_exception = None
        try:
            response.raise_for_status()
        except requests.HTTPError as exception:
            if exception.response.status_code == 401:
                raise_exception = ValueError("Invalid COMET_API_KEY")
            else:
                raise
        if raise_exception:
            raise raise_exception
        return response.json()

    def get_version(self) -> str:
        """
        Return the default version of the API.
        """
        return self._version

    def optimizer_next(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get the next set of parameters to evaluate.
        """
        ## /next?id=ID
        params = {"id": id}
        results = self.get_request("next", params=params)
        if results["code"] == 200:
            result_params = results["parameters"]  # type: Dict[str, Any]
            return result_params
        else:
            return None

    def optimizer_spec(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get the full spec for an optimizer by name.
        """
        if algorithm_name is not None:
            params = {"algorithmName": algorithm_name}
        else:
            raise ValueError("Optimizer must have an algorithm name")

        results = self.get_request("spec", params=params)
        if "code" in results:
            raise ValueError(results["message"])
        return results

    def optimizer_status(self, id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the status of optimizer instance or
        search.
        """
        ## /status
        ## /status?id=ID
        if id is not None:
            params = {"id": id}
        else:
            params = {}
        results = self.get_request("status", params=params)
        return results

    def optimizer_algorithms(self) -> Dict[str, Any]:
        """
        Get a list of algorithms.
        """
        ## /algorithms
        results = self.get_request("algorithms", {})
        return results

    def optimizer_update(
        self, id, pid, trial, status=None, score=None, epoch=None
    ) -> Dict[str, Any]:
        """
        Post the status of a search.
        """
        ## /update {"pid": PID, "trial": TRIAL,
        ##          "status": STATUS, "score": SCORE, "epoch": EPOCH}
        json = {
            "id": id,
            "pid": pid,
            "trial": trial,
            "status": status,
            "score": score,
            "epoch": epoch,
        }
        results = self.post_request("update", json=json)
        return results

    def optimizer_insert(self, id: str, p: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a completed parameter package.
        """
        ## /insert {"pid": PID, "trial": TRIAL,
        ##          "status": STATUS, "score": SCORE, "epoch": EPOCH}
        json = {
            "id": id,
            "pid": p["pid"],
            "trial": p["trial"],
            "status": p["status"],
            "score": p["score"],
            "epoch": p["epoch"],
            "parameters": p["parameters"],
            "tries": p["tries"],
            "startTime": p["startTime"],
            "endTime": p["endTime"],
            "lastUpdateTime": p["lastUpdateTime"],
            "count": p["count"],
        }
        results = self.post_request("insert", json=json)
        return results


def _check_response_status(response: requests.Response) -> requests.Response:
    if response.status_code < HTTPStatus.BAD_REQUEST:
        return response

    if response.status_code == HTTPStatus.PAYMENT_REQUIRED:
        raise PaymentRequired(response.request.method, response)
    else:
        raise CometRestApiException(response.request.method, response)


class BaseApiClient(object):
    """A base api client to centralize how we build urls and treat exceptions"""

    def __init__(
        self,
        server_url: str,
        base_url: List[str],
        low_level_api_client: LowLevelHTTPClient,
        api_key: Optional[str],
        config: Config,
    ) -> None:
        self.server_url = server_url
        self._base_url = url_join(server_url, *base_url)
        self.low_level_api_client = low_level_api_client
        self.api_key = api_key
        self.config = config

    @property
    def base_url(self) -> str:
        return self._base_url

    def get(
        self,
        url: str,
        params: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> requests.Response:
        response = self.low_level_api_client.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            stream=stream,
        )

        if response.status_code != 200:
            if response.status_code == 401:
                LOGGER.debug("Got 401 response, reason: %r", response.content)
                raise Unauthorized(masked_api_key(self.api_key), response=response)
            elif response.status_code == 404:
                raise NotFound("GET", response)
            else:
                raise CometRestApiException("GET", response)

        return response

    def get_from_endpoint(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]],
        return_type: str = "json",
        timeout: Optional[float] = None,
        stream: bool = False,
        alternate_base_url: Optional[str] = None,
    ) -> Any:
        url = self._endpoint_url(
            endpoint=endpoint, alternate_base_url=alternate_base_url
        )
        response = self.get(
            url=url,
            params=params,
            timeout=timeout,
            stream=stream,
        )

        # Return data based on return_type:
        if return_type == "json":
            content_type = response.headers["content-type"]
            # octet-stream is a GZIPPED stream:
            if content_type in ["application/json", "application/octet-stream"]:
                retval = response.json()
            else:
                raise CometRestApiValueError("GET", "data is not json", response)
        elif return_type == "binary":
            retval = response.content
        elif return_type == "text":
            retval = response.text
        elif return_type == "response":
            retval = response
        else:
            raise CometRestApiValueError(
                "GET",
                "invalid return_type %r: should be 'json', 'binary', or 'text'"
                % return_type,
                response,
            )

        return retval

    def post(
        self,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        files: Optional[Any] = None,
        params: Optional[Any] = None,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
        retry: Optional[bool] = None,
    ) -> requests.Response:
        response = self.low_level_api_client.post(
            url,
            payload=payload,
            headers=headers,
            files=files,
            params=params,
            custom_encoder=custom_encoder,
            compress=compress,
            retry=retry,
        )

        return _check_response_status(response)

    def put(
        self,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        files: Optional[Any] = None,
        params: Optional[Any] = None,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
    ) -> requests.Response:
        response = self.low_level_api_client.put(
            url,
            payload=payload,
            headers=headers,
            files=files,
            params=params,
            custom_encoder=custom_encoder,
            compress=compress,
        )

        return _check_response_status(response)

    def post_from_endpoint(
        self,
        endpoint: str,
        payload: Any,
        alternate_base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> requests.Response:
        return self._result_from_http_method(
            method=self.post,
            endpoint=endpoint,
            payload=payload,
            alternate_base_url=alternate_base_url,
            **kwargs,
        )

    def put_from_endpoint(
        self,
        endpoint: str,
        payload: Any,
        alternate_base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> requests.Response:
        return self._result_from_http_method(
            method=self.put,
            endpoint=endpoint,
            payload=payload,
            alternate_base_url=alternate_base_url,
            **kwargs,
        )

    def _result_from_http_method(
        self,
        method: Callable,
        endpoint: str,
        payload: Any,
        alternate_base_url: Optional[str],
        **kwargs: Any,
    ) -> requests.Response:
        url = self._endpoint_url(
            endpoint=endpoint, alternate_base_url=alternate_base_url
        )
        return method(url=url, payload=payload, **kwargs)

    def _endpoint_url(
        self,
        endpoint: str,
        alternate_base_url: Optional[str] = None,
    ) -> str:
        if alternate_base_url is not None:
            return url_join(alternate_base_url, endpoint)
        else:
            return url_join(self.base_url, endpoint)


class RestApiClient(BaseApiClient):
    """This API Client is meant to discuss to the REST API and handle, params and payload formatting,
    input validation if necessary, creating the url and parsing the output. All the HTTP
    communication is handled by the low-level HTTP client.

    Inputs must be JSON-encodable, any conversion must be done by the caller.

    One method equals one endpoint and one call"""

    def __init__(
        self,
        server_url: str,
        version: str,
        low_level_api_client: LowLevelHTTPClient,
        api_key: str,
        config: Config,
        check_version: bool,
    ) -> None:
        super().__init__(
            server_url,
            ["api/rest/", version + "/"],
            low_level_api_client,
            api_key,
            config,
        )
        self._version = version
        # this is going to be used for some endpoints
        self.alternate_base_url = url_join(server_url, "clientlib/rest/", version + "/")

        self.use_cache = False
        self.backend_version = None

        if check_version:
            self._check_version()

    def _check_version(self) -> None:
        config_minimal_backend_version = self.config[
            "comet.rest_v2_minimal_backend_version"
        ]
        minimal_backend_version = None
        try:
            # Invalid version will raise exception:
            minimal_backend_version = SemanticVersion.parse(
                config_minimal_backend_version
            )
        except Exception:
            LOGGER.warning(
                INVALID_CONFIG_MINIMAL_BACKEND_VERSION, config_minimal_backend_version
            )

        if minimal_backend_version:
            self._check_api_backend_version(minimal_backend_version)

    def get(
        self,
        url: str,
        params: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> requests.Response:
        headers = {API_KEY_HEADER: self.api_key}

        return super().get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            stream=stream,
        )

    def post(
        self,
        url: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
        retry: Optional[bool] = None,
    ) -> requests.Response:
        headers = {API_KEY_HEADER: self.api_key}

        return super().post(
            url,
            payload=payload,
            headers=headers,
            files=files,
            params=params,
            custom_encoder=custom_encoder,
            compress=compress,
            retry=retry,
        )

    def put(
        self,
        url: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
    ) -> requests.Response:
        headers = {API_KEY_HEADER: self.api_key}

        return super().put(
            url,
            payload=payload,
            headers=headers,
            files=files,
            params=params,
            custom_encoder=custom_encoder,
            compress=compress,
        )

    def reset(self):
        pass

    # Read Experiment methods:

    def get_account_details(self) -> Dict[str, Any]:
        """
        Example: {'userName': 'USERNAME', 'defaultWorkspaceName': 'WORKSPACE'}
        """
        payload = None
        response = self.get_from_endpoint("account-details", payload)
        return response

    def _get_experiment_system_details_single_field(
        self, experiment_key: str, field: str
    ) -> Optional[Any]:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        response = self.get_from_endpoint(
            SYSTEM_DETAILS_ENDPOINT,
            payload,
        )
        if response:
            return response[field]
        else:
            return None

    def get_experiment_os_packages(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_OS_PACKAGES
        )

    def get_experiment_user(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_USER
        )

    def get_experiment_installed_packages(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_INSTALLED_PACKAGES
        )

    def get_experiment_command(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_COMMAND
        )

    def get_experiment_executable(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_EXECUTABLE
        )

    def get_experiment_hostname(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_HOSTNAME
        )

    def get_experiment_gpu_static_info(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_GPU_STATIC_INFO_LIST
        )

    def get_experiment_additional_system_info(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_ADDITIONAL_SYSTEM_INFO_LIST
        )

    def get_experiment_ip(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_IP
        )

    def get_experiment_max_memory(self, experiment_key):
        # FIXME: always None
        return self._get_experiment_system_details_single_field(
            experiment_key, "maxTotalMemory"
        )

    def get_experiment_network_interface_ips(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, "networkInterfaceIps"
        )

    def get_experiment_os(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_OS
        )

    def get_experiment_os_type(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_OS_TYPE
        )

    def get_experiment_os_release(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_OS_RELEASE
        )

    def get_experiment_pid(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_PID
        )

    def get_experiment_python_version(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_PYTHON_VERSION
        )

    def get_experiment_python_version_verbose(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_PYTHON_VERSION_VERBOSE
        )

    def get_experiment_total_memory(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_TOTAL_RAM
        )

    def get_experiment_machine(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_MACHINE
        )

    def get_experiment_processor(self, experiment_key):
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_PROCESSOR
        )

    def get_experiment_system_info(self, experiment_key):
        """
        Deprecated.
        """
        return self._get_experiment_system_details_single_field(
            experiment_key, PAYLOAD_ADDITIONAL_SYSTEM_INFO_LIST
        )

    def get_experiment_system_metric_names(self, experiment_key):
        """ """
        return self._get_experiment_system_details_single_field(
            experiment_key, "systemMetricNames"
        )

    def get_experiment_model_graph(self, experiment_key):
        """
        Get the associated graph/model description for this
        experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/graph", params)

    def get_experiment_tags(self, experiment_key):
        """
        Get the associated tags for this experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/tags", params)

    def get_experiment_parameters_summaries(self, experiment_key):
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/parameters", params)

    def get_experiment_metrics_summaries(self, experiment_key):
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/metrics/summary", params)

    def get_experiment_metric(self, experiment_key, metric):
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "metricName": metric}
        return self.get_from_endpoint("experiment/metrics/get-metric", params)

    def get_all_experiment_metrics(self, experiment_key):
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/metrics/get-all", params)

    def get_experiment_multi_metrics(
        self, experiment_keys, metrics, parameters=None, independent=True, full=True
    ):
        payload = {
            "targetedExperiments": experiment_keys,
            "metrics": metrics,
            "params": parameters,
            "independentMetrics": independent,
            "fetchFull": full,
        }
        return self.post_from_endpoint("experiments/multi-metric-chart", payload)

    def get_experiment_asset_list(
        self,
        experiment_key: str,
        asset_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get a list of assets associated with the experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        if asset_type is not None:
            params["type"] = asset_type
        results = self.get_from_endpoint(
            "experiment/asset/list", params=params, timeout=timeout
        )
        if results is not None:
            return results["assets"]

        return None

    def get_experiment_assets_list_by_name(
        self,
        experiment_key: str,
        asset_name: str,
        asset_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch a list of experiment assets by their name and optional type.

        This method interacts with an endpoint to retrieve a list of assets for a given
        experiment, filtered by the provided asset name and optionally by asset type.

        Args:
            experiment_key:
                The unique key identifying the experiment.
            asset_name:
                The name of the asset to fetch.
            asset_type:
                The type of the asset to filter the results. Default is None.
            timeout: Optional[float]
                The maximum time in seconds to wait for the endpoint response. Default is None.

        Returns:
            A list of assets matching the provided criteria, or None if no assets are found.
        """
        params = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_ASSET_NAME: asset_name,
        }
        if asset_type is not None:
            params["type"] = asset_type
        results = self.get_from_endpoint(
            "experiment/asset/get-by-name", params=params, timeout=timeout
        )
        if results is not None:
            return results["assets"]

        return None

    def _prepare_experiment_asset_request(
        self,
        asset_id: str,
        experiment_key: Optional[str] = None,
        artifact_version_id: Optional[str] = None,
        allow301: bool = False,
    ) -> PreparedRequest:
        params = {"assetId": asset_id, "allow301": allow301}

        if experiment_key is not None:
            params[PAYLOAD_EXPERIMENT_KEY] = experiment_key

        if artifact_version_id is not None:
            params["artifactVersionId"] = artifact_version_id

        url = url_join(self.base_url, "experiment/asset/get-asset")

        return PreparedRequest(
            api_key=self.api_key,
            url=url,
            json=params,
            headers=self.low_level_api_client.headers,
        )

    def get_experiment_asset(
        self,
        asset_id: str,
        experiment_key: Optional[str] = None,
        artifact_version_id: Optional[str] = None,
        return_type: str = "binary",
        stream: bool = False,
        allow301: bool = False,
    ) -> Union[bytes, Dict[str, Any], requests.Response]:
        request = self._prepare_experiment_asset_request(
            asset_id=asset_id,
            experiment_key=experiment_key,
            artifact_version_id=artifact_version_id,
            allow301=allow301,
        )

        response = self.get_from_endpoint(
            "experiment/asset/get-asset",
            params=request.json,
            return_type=return_type,
            timeout=self.config["comet.timeout.file_download"],
            stream=stream,
        )
        return response

    def get_experiment_others_summaries(self, experiment_key):
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/log-other", params)

    def get_experiment_system_details(self, experiment_key):
        """
        Return the dictionary of system details.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint(
            "experiment/system-details",
            params,
        )

    def get_experiment_html(self, experiment_key):
        """
        Get the HTML associated with the experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/html", params=params)

    def get_experiment_code(self, experiment_key):
        """
        Get the associated source code for this experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/code", params)

    def get_experiment_output(self, experiment_key):
        """
        Get the associated standard output for this experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/output", params)

    def get_experiment_metadata(self, experiment_key):
        """
        Returns the JSON metadata for an experiment

        Returns:

        ```python
        {
            'archived': False,
            'durationMillis': 7,
            'endTimeMillis': 1586174765277,
            'experimentKey': 'EXPERIMENT-KEY',
            'experimentName': None,
            'fileName': None,
            'filePath': None,
            'optimizationId': None,
            'projectId': 'PROJECT-ID',
            'projectName': 'PROJECT-NAME',
            'running': False,
            'startTimeMillis': 1586174757596,
            'throttle': False,
            'workspaceName': 'WORKSPACE-NAME',
        }
        ```
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        results = self.get_from_endpoint("experiment/metadata", params)
        return results

    def get_experiment_git_patch(self, experiment_key):
        """
        Get the git-patch associated with this experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        results = self.get_from_endpoint("experiment/git/patch", params, "binary")
        # NOTE: returns either a binary JSON message or a binary git patch
        if results.startswith(b'{"msg"'):
            return None  # JSON indicates no patch
        else:
            return results

    def get_experiment_git_metadata(self, experiment_key):
        """
        Get the git-metadata associated with this experiment.
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("experiment/git/metadata", params)

    def get_experiments_metadata(self, workspace, project_name):
        """
        Return the names of the projects in a workspace.
        """
        payload = {"workspaceName": workspace, "projectName": project_name}
        results = self.get_from_endpoint("projects", payload)
        return results

    # Other methods:

    def get_workspaces(self) -> Dict:
        """
        Get workspace names.
        """
        return self.get_from_endpoint("workspaces", {})

    def get_project_experiments(
        self,
        workspace: str,
        project_name: str,
        archived: bool = False,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """
        Return the metadata JSONs of the experiments in a project.

        Args:
            workspace: The workspace name containing the project
            project_name: The name of the project
            archived: Whether to return archived experiments (default: False)
            page: Page number for pagination (1-indexed). If provided, page_size is required.
            page_size: Number of experiments per page. Required when page is specified.
            sort_by: Field to sort by. Must be "startTime" or "endTime" if provided.
            sort_order: Sort direction. Must be "asc" or "desc" if provided.
                Required when page, page_size, and sort_by are all specified.

        Returns:
            List of experiment metadata dictionaries

        Raises:
            ValueError: If pagination or sorting parameters are invalid
        """
        # Validate pagination and sorting parameters
        if page is not None:
            if page_size is None:
                raise ValueError("page_size is required when page is specified")

        # Validate sort_by parameter
        if sort_by is not None:
            if sort_by not in ["startTime", "endTime"]:
                raise ValueError("sort_by must be 'startTime' or 'endTime'")

        # Validate sort_order parameter
        if sort_order is not None:
            if sort_order not in ["asc", "desc"]:
                raise ValueError("sort_order must be 'asc' or 'desc'")

        # Validate that sort_order is required when page, page_size, and sort_by are all specified
        if page is not None and page_size is not None and sort_by is not None:
            if sort_order is None:
                raise ValueError(
                    "sort_order is required when page, page_size, and sort_by are all specified"
                )

        payload = {
            "workspaceName": workspace,
            "projectName": project_name,
            "archived": archived,
        }
        if page is not None:
            payload["page"] = page
        if page_size is not None:
            payload["size"] = page_size
        if sort_by is not None:
            payload["sortBy"] = sort_by
        if sort_order is not None:
            payload["sortOrder"] = sort_order
        results = self.get_from_endpoint("experiments", payload)
        # Returns list of experiment dicts
        return results

    def get_project_notes_by_id(self, project_id):
        """
        Return the notes of a project.
        """
        payload = {"projectId": project_id}
        results = self.get_from_endpoint("project/notes", payload)
        if results and "notes" in results:
            return results["notes"]

    def get_project_jsons(self, workspace):
        """
        Return the JSONs of the projects in a workspace.
        """
        payload = {"workspaceName": workspace}
        return self.get_from_endpoint("projects", payload)

    def get_projects(self, workspace):
        """
        Get project names in a workspace.
        """
        payload = {"workspaceName": workspace}
        results = self.get_from_endpoint("projects", payload)
        if results and "projects" in results:
            projects = [
                project_json["projectName"] for project_json in results["projects"]
            ]
            return projects

    def get_project(self, workspace=None, project_name=None, project_id=None):
        """
        Get project details.
        """
        if project_id is not None:
            payload = {"projectId": project_id}
        else:
            payload = {"workspaceName": workspace, "projectName": project_name}
        results = self.get_from_endpoint("project", payload)
        return results

    def get_project_by_id(self, project_id):
        """
        Get project details.
        """
        payload = {"projectId": project_id}
        results = self.get_from_endpoint("project", payload)
        return results

    def get_project_json(self, workspace, project_name):
        """
        Get Project metadata JSON.
        """
        payload = {"workspaceName": workspace}
        results = self.get_from_endpoint("projects", payload)
        if results and "projects" in results:
            projects = [
                project_json
                for project_json in results["projects"]
                if project_json["projectName"] == project_name
            ]
            if len(projects) > 0:
                # Get the first if more than one
                return projects[0]
            # else, return None

    def query_project(self, workspace, project_name, predicates, archived=False):
        """
        Given a workspace, project_name, and predicates, return matching experiments.
        """
        payload = {
            "workspaceName": workspace,
            "projectName": project_name,
            "predicates": predicates,
            "archived": archived,
        }
        return self.post_from_endpoint("project/query", payload)

    def get_project_columns(self, workspace, project_name):
        """
        Given a workspace and project_name return the column names, types, etc.
        """
        payload = {"workspaceName": workspace, "projectName": project_name}
        return self.get_from_endpoint("project/column-names", payload)

    # Write methods:

    def update_project(
        self,
        workspace,
        project_name,
        new_project_name=None,
        description=None,
        public=None,
    ):
        """
        Update the metadata of a project by project_name
        and workspace.

        Args:
            workspace: name of workspace
            project_name: name of project
            new_project_name: new name of project (optional)
            description: new description of project (optional)
            public: new setting of visibility (optional)
        """
        payload = {}
        if project_name is None or workspace is None:
            raise ValueError(CONNECTION_UPDATE_PROJECT_MISSING_REQUIRED_EXCEPTION)
        payload["projectName"] = project_name
        payload["workspaceName"] = workspace
        if new_project_name is not None:
            payload["newProjectName"] = new_project_name
        if description is not None:
            payload["newProjectDescription"] = description
        if public is not None:
            payload["isPublic"] = public
        response = self.post_from_endpoint("write/project/update", payload)
        return response

    def update_project_by_id(
        self, project_id, new_project_name=None, description=None, public=None
    ):
        """
        Update the metadata of a project by project_id.

        Args:
            project_id: project id
            new_project_name: new name of project (optional)
            description: new description of project (optional)
            public: new setting of visibility (optional)
        """
        if project_id is None:
            raise ValueError(CONNECTION_UPDATE_PROJECT_BY_ID_MISSING_REQUIRED_EXCEPTION)
        payload = {}
        payload["projectId"] = project_id
        if new_project_name is not None:
            payload["newProjectName"] = new_project_name
        if description is not None:
            payload["newProjectDescription"] = description
        if public is not None:
            payload["isPublic"] = public
        response = self.post_from_endpoint("write/project/update", payload)
        return response

    def create_project_share_key(self, project_id):
        """
        Create a sharable key for a private project.

        Args:
            project_id: project id

        Example:
        ```python
        >>> api = API()
        >>> SHARE_KEY = api.create_project_share_key(PROJECT_ID)
        ```
        You can now share the private project with:
        https://comet.com/workspace/project?shareable=SHARE_KEY

        See also: get_project_share_keys(), and delete_project_share_key().
        """
        payload = {"projectId": project_id}
        response = self.get_from_endpoint("write/project/add-share-link", payload)
        return response

    def get_project_share_keys(self, project_id):
        """
        Get all sharable keys for a private project.

        Args:
            project_id: project id

        Example:
        ```python
        >>> api = API()
        >>> SHARE_KEYS = api.get_project_share_keys(PROJECT_ID)
        ```

        See also: create_project_share_key(), and delete_project_share_key().
        """
        payload = {"projectId": project_id}
        response = self.get_from_endpoint("project/get-project-share-links", payload)
        return response

    def delete_project_share_key(self, project_id, share_key):
        """
        Delete a sharable key for a private project.

        Args:
            project_id: project id
            share_key: the share key

        Example:
        ```python
        >>> api = API()
        >>> SHARE_KEYS = api.get_project_share_keys(PROJECT_ID)
        >>> api.delete_project_share_key(PROJECT_ID, SHARE_KEYS[0])
        ```

        See also: create_project_share_key(), and get_project_share_keys().
        """
        payload = {
            "projectId": project_id,
            "shareCode": share_key,
        }
        response = self.get_from_endpoint(
            "write/project/delete-project-share-link", payload
        )
        return response

    def add_experiment_gpu_metrics(self, experiment_key, gpu_metrics):
        """
        Add an instance of GPU metrics.

        Args:
            experiment_key: an experiment id
            gpu_metrics: a list of dicts with keys:
                * gpuId: required, Int identifier
                * freeMemory: required, Long
                * usedMemory: required, Long
                * gpuUtilization: required, Int percentage utilization
                * totalMemory: required, Long
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "gpus": gpu_metrics}
        response = self.post_from_endpoint("write/experiment/gpu-metrics", payload)
        return response

    def add_experiment_cpu_metrics(
        self,
        experiment_key,
        cpu_metrics,
        context=None,
        step=None,
        epoch=None,
        timestamp=None,
    ):
        """
        Add an instance of CPU metrics.

        Args:
            experiment_key: an experiment id
            cpu_metrics: a list of integer percentages, ordered by cpu
            context: optional, a run context
            step: optional, the current step
            epoch: optional, the current epoch
            timestamp: optional": current time, in milliseconds since the Epoch
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "cpuPercentUtilization": cpu_metrics,
        }
        if context is not None:
            payload["context"] = context
        if step is not None:
            payload["step"] = step
        if epoch is not None:
            payload["epoch"] = epoch
        if timestamp is not None:
            payload["timestamp"] = timestamp
        response = self.post_from_endpoint("write/experiment/cpu-metrics", payload)
        return response

    def add_experiment_ram_metrics(
        self,
        experiment_key,
        total_ram,
        used_ram,
        context=None,
        step=None,
        epoch=None,
        timestamp=None,
    ):
        """
        Add an instance of RAM metrics.

        Args:
            experiment_key: an experiment id
            total_ram: required, total RAM available
            used_ram: required,  RAM used
            context: optional, the run context
            step: optional, the current step
            epoch: optional, the current epoch
            timestamp: optional, the current timestamp
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_TOTAL_RAM: total_ram,
            PAYLOAD_USED_RAM: used_ram,
        }
        if context is not None:
            payload["context"] = context
        if step is not None:
            payload["step"] = step
        if epoch is not None:
            payload["epoch"] = epoch
        if timestamp is not None:
            payload["timestamp"] = timestamp
        response = self.post_from_endpoint("write/experiment/ram-metrics", payload)
        return response

    def add_experiment_load_metrics(
        self,
        experiment_key,
        load_avg,
        context=None,
        step=None,
        epoch=None,
        timestamp=None,
    ):
        """
        Add an instance of system load metrics.

        Args:
            experiment_key: an experiment id
            load_avg: required, the load average
            context: optional, the run context
            step: optional, the current step
            epoch: optional, the current epoch
            timestamp: optional, the current timestamp
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "loadAverage": load_avg}
        if context is not None:
            payload["context"] = context
        if step is not None:
            payload["step"] = step
        if epoch is not None:
            payload["epoch"] = epoch
        if timestamp is not None:
            payload["timestamp"] = timestamp
        response = self.post_from_endpoint("write/experiment/load-metrics", payload)
        return response

    def set_experiment_git_metadata(
        self, experiment_key, user, root, branch, parent, origin
    ):
        """ """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "user": user,
            "root": root,
            "branch": branch,
            "parent": parent,
            "origin": origin,
        }
        response = self.post_from_endpoint("write/experiment/git/metadata", payload)
        return response

    def set_experiment_git_patch(self, experiment_key: str, git_patch: IO):
        """ """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        files = {"file": ("filename", git_patch)}
        response = self.post_from_endpoint(
            "write/experiment/git/patch", {}, files=files, params=params
        )
        return response

    def set_experiment_code(self, experiment_key: str, code: str) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "code": code}
        response = self.post_from_endpoint("write/experiment/code", payload)
        return response

    def set_experiment_model_graph(
        self, experiment_key: str, graph_str: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_MODEL_GRAPH: graph_str,
        }
        response = self.post_from_endpoint("write/experiment/graph", payload)
        return response

    def set_experiment_os_packages(
        self, experiment_key: str, os_packages: List[str]
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_OS_PACKAGES: os_packages,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_user(self, experiment_key: str, user: str) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_USER: user}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_installed_packages(
        self, experiment_key: str, installed_packages: List[str]
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_INSTALLED_PACKAGES: installed_packages,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_command(
        self, experiment_key: str, command: Union[str, List[str]]
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_COMMAND: command}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_gpu_static_info(
        self, experiment_key: str, gpu_static_info: List[Dict[str, Any]]
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_GPU_STATIC_INFO_LIST: gpu_static_info,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_executable(
        self, experiment_key: str, executable: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_EXECUTABLE: executable,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_filename(
        self, experiment_key: str, filename: str
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_FILE_PATH: filename}
        response = self.post_from_endpoint("write/experiment/file-path", payload)
        return response

    def set_experiment_hostname(
        self, experiment_key: str, hostname: str
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_HOSTNAME: hostname}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_ip(self, experiment_key: str, ip: str) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_IP: ip}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def log_experiment_system_info(
        self, experiment_key: str, system_info: List[Dict[str, str]]
    ) -> requests.Response:
        if not isinstance(system_info, list):
            raise ValueError(CONNECTION_LOG_EXPERIMENT_SYSTEM_INFO_EXCEPTION)
        for si in system_info:
            if "key" not in si or "value" not in si:
                raise ValueError(CONNECTION_LOG_EXPERIMENT_SYSTEM_INFO_EXCEPTION)

        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_ADDITIONAL_SYSTEM_INFO_LIST: system_info,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_network_interface_ips(
        self, experiment_key: str, network_interface_ips: List[str]
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "networkInterfaceIps": network_interface_ips,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_os(self, experiment_key: str, os: str) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_OS: os}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_os_type(
        self, experiment_key: str, os_type: str
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_OS_TYPE: os_type}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_os_release(
        self, experiment_key: str, os_release: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_OS_RELEASE: os_release,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_pid(self, experiment_key: str, pid: int) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_PID: pid}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_python_version(
        self, experiment_key: str, python_version: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_PYTHON_VERSION: python_version,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_python_version_verbose(
        self, experiment_key: str, python_version_verbose: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_PYTHON_VERSION_VERBOSE: python_version_verbose,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_machine(
        self, experiment_key: str, machine: str
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_MACHINE: machine}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_processor(
        self, experiment_key: str, processor: str
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, PAYLOAD_PROCESSOR: processor}
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def set_experiment_system_details(
        self,
        _os: str,
        command: Union[str, List[str]],
        env: Dict[str, str],
        experiment_key: str,
        hostname: str,
        ip: str,
        machine: str,
        os_release: str,
        os_type: str,
        pid: int,
        processor: str,
        python_exe: str,
        python_version_verbose: str,
        python_version: str,
        user: str,
    ) -> requests.Response:
        payload = {
            PAYLOAD_COMMAND: command,
            PAYLOAD_ENV: env,
            PAYLOAD_EXECUTABLE: python_exe,
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_HOSTNAME: hostname,
            PAYLOAD_IP: ip,
            PAYLOAD_MACHINE: machine,
            PAYLOAD_OS: _os,
            PAYLOAD_OS_RELEASE: os_release,
            PAYLOAD_OS_TYPE: os_type,
            PAYLOAD_PID: pid,
            PAYLOAD_PROCESSOR: processor,
            PAYLOAD_PYTHON_VERSION: python_version,
            PAYLOAD_PYTHON_VERSION_VERBOSE: python_version_verbose,
            PAYLOAD_USER: user,
        }
        response = self.post_from_endpoint(
            SYSTEM_DETAILS_WRITE_ENDPOINT,
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def update_experiment_status(self, experiment_key: str) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        response = self.get_from_endpoint("write/experiment/set-status", payload)
        return response

    def set_experiment_start_end(
        self, experiment_key: str, start_time: int, end_time: int
    ) -> requests.Response:
        """
        Set the start/end time of an experiment.

        Note: times are in milliseconds.
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "startTimeMillis": start_time,
            "endTimeMillis": end_time,
        }
        response = self.post_from_endpoint(
            "write/experiment/set-start-end-time", payload
        )
        return response

    def set_project_notes_by_id(self, project_id, notes):
        """
        Set the notes of a project.
        """
        payload = {"projectId": project_id, "notes": notes}
        results = self.post_from_endpoint("write/project/notes", payload)
        if results:
            return results.json()

    def set_experiment_cloud_details(
        self, experiment_key: str, provider: str, cloud_metadata: Dict[Any, Any]
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_PROVIDER: provider,
            PAYLOAD_METADATA: cloud_metadata,
        }
        response = self.post_from_endpoint("write/experiment/cloud-details", payload)
        return response

    def log_experiment_output(
        self,
        experiment_key: str,
        output: str,
        context: Optional[str] = None,
        stderr: bool = False,
        timestamp: Optional[float] = None,
        max_line_length: int = MAXIMAL_LENGTH_OF_OUTPUT_LINE,
    ) -> requests.Response:
        """
        Logs the output of an experiment.

        This method sends the experiment output data to a specific endpoint. It allows for
        splitting output into lines and adding metadata such as timestamps and offsets.

        Args:
            experiment_key: The unique identifier for the experiment.
            output: The output string to be logged.
            context: An optional context string related to the experiment run.
            stderr: A flag indicating whether the output is from a standard error. Defaults to False.
            timestamp: Optional timestamp for the output. If not provided, a monotonic time value
                       will be used.
            max_line_length: The maximum allowable length for a line in the output. Defaults to
                             5_000_000 characters.

        Returns:
            requests.Response: The response object resulting from the POST request to the endpoint.
        """
        if timestamp is None:
            timestamp = get_time_monotonic()

        stdout_lines = []

        for offset, line in enumerate(
            split_output_by_breaks_and_length(output, max_line_length=max_line_length)
        ):
            stdout_lines.append(
                {
                    PAYLOAD_STDERR: stderr,
                    PAYLOAD_OUTPUT: line,
                    PAYLOAD_LOCAL_TIMESTAMP: timestamp_milliseconds(timestamp),
                    PAYLOAD_OFFSET: offset,
                }
            )

        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_RUN_CONTEXT: context,
            PAYLOAD_OUTPUT_LINES: stdout_lines,
        }
        response = self.post_from_endpoint(
            "write/experiment/output",
            payload,
            alternate_base_url=self.alternate_base_url,
        )
        return response

    def send_artifact_remote_assets_batch(
        self,
        batch_items: List[MessageBatchItem],
        experiment_key: str,
        compress: bool = True,
    ) -> None:
        endpoint_url = "write/artifact/assets/remote/batch"

        LOGGER.debug(
            "Sending artifact remote assets batch, length: %d, compression enabled: %s, endpoint: %s",
            len(batch_items),
            compress,
            endpoint_url,
        )

        payload = format_remote_assets_batch_items(batch_items)
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        self.post_from_endpoint(
            endpoint_url,
            params=params,
            payload=payload,
            compress=compress,
            alternate_base_url=self.alternate_base_url,
        )

        LOGGER.debug(
            "Batch of %d artifact remote assets messages has been sent to %r",
            len(batch_items),
            endpoint_url,
        )

    def send_stdout_batch(
        self,
        batch_items: List[MessageBatchItem],
        experiment_key: str,
        compress: bool = True,
        timestamp: Optional[int] = None,
        max_line_length: int = MAXIMAL_LENGTH_OF_OUTPUT_LINE,
    ) -> None:

        endpoint_url = "write/experiment/output"

        LOGGER.debug(
            "Sending stdout messages batch, length: %d, compression enabled: %s, endpoint: %s",
            len(batch_items),
            compress,
            endpoint_url,
        )

        if timestamp is None:
            timestamp = get_time_monotonic()

        stderr_flags = [False, True]
        for stderr in stderr_flags:
            payload = format_stdout_message_batch_items(
                batch_items=batch_items,
                timestamp=timestamp,
                experiment_key=experiment_key,
                stderr=stderr,
                max_line_length=max_line_length,
            )
            if payload is not None:
                self.post_from_endpoint(
                    endpoint_url,
                    payload,
                    compress=compress,
                    alternate_base_url=self.alternate_base_url,
                )

    def log_experiment_other(
        self,
        experiment_key: str,
        key: str,
        value: Any,
        timestamp: Optional[float] = None,
    ) -> requests.Response:
        """
        Set an other key/value pair for an experiment.

        Args:
            experiment_key: str, the experiment id
            key: str, the name of the other value
            value: any, the value of the other key
            timestamp: int, time in seconds, since epoch
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "key": key,
            "value": value,
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp_milliseconds(timestamp)
        response = self.post_from_endpoint("write/experiment/log-other", payload)
        return response

    def log_experiment_parameter(
        self,
        experiment_key: str,
        parameter: str,
        value: Any,
        step: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> requests.Response:
        """
        Set a parameter name/value pair for an experiment.

        Args:
            experiment_key: str, the experiment id
            parameter: str, the name of the parameter
            value: any, the value of the parameter
            step: int, the step number at time of logging
            timestamp: int, time in seconds, since epoch
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "parameterName": parameter,
            "parameterValue": value,
        }
        if step is not None:
            payload["step"] = step
        if timestamp is not None:
            payload["timestamp"] = timestamp_milliseconds(timestamp)
        response = self.post_from_endpoint("write/experiment/parameter", payload)
        return response

    def delete_experiment_parameter(
        self, experiment_key: str, parameter: str
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_PARAMETER_NAME: parameter,
        }
        response = self.post_from_endpoint("write/experiment/parameter/delete", payload)
        return response

    def delete_experiment_parameters(
        self, experiment_key: str, parameters: List[str]
    ) -> requests.Response:
        payload: List[Dict[str, str]] = [
            {
                PAYLOAD_EXPERIMENT_KEY: experiment_key,
                PAYLOAD_PARAMETER_NAME: param,
            }
            for param in parameters
        ]
        response = self.post_from_endpoint(
            "write/experiments/parameters/delete", payload
        )
        return response

    def log_experiment_metric(
        self,
        experiment_key: str,
        metric: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        timestamp: Optional[float] = None,
        context: Optional[str] = None,
    ) -> requests.Response:
        """
        Set a metric name/value pair for an experiment.
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "metricName": metric,
            "metricValue": value,
        }
        if epoch is not None:
            payload["epoch"] = epoch
        if context is not None:
            payload["context"] = context
        if step is not None:
            payload["step"] = step
        if timestamp is not None:
            payload["timestamp"] = timestamp_milliseconds(timestamp)
        response = self.post_from_endpoint("write/experiment/metric", payload)
        return response

    def log_experiment_html(
        self,
        experiment_key: str,
        html: str,
        overwrite: bool = False,
        timestamp: Optional[float] = None,
    ) -> requests.Response:
        """
        Set, or append onto, an experiment's HTML.

        Args:
            experiment_key: str, the experiment id
            html: str, the html string to log
            overwrite: bool, if, true overwrite previously-logged html
            timestamp: int, time in seconds, since epoch
        """
        if timestamp is None:
            timestamp = local_timestamp()
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_HTML: html,
            PAYLOAD_OVERRIDE: overwrite,
            PAYLOAD_TIMESTAMP: timestamp_milliseconds(timestamp),
        }
        response = self.post_from_endpoint("write/experiment/html", payload)
        return response

    def log_experiment_dependency(
        self,
        experiment_key: str,
        name: str,
        version: str,
        timestamp: Optional[float] = None,
    ) -> requests.Response:
        if timestamp is None:
            timestamp = local_timestamp()
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_TIMESTAMP: timestamp_milliseconds(timestamp),
            PAYLOAD_DEPENDENCIES: [
                {
                    PAYLOAD_DEPENDENCY_NAME: name,
                    PAYLOAD_DEPENDENCY_VERSION: version,
                }
            ],
        }
        return self.put_from_endpoint(
            endpoint="write/experiment/dependencies",
            payload=payload,
        )

    def add_experiment_tags(
        self, experiment_key: str, tags: List[str]
    ) -> requests.Response:
        """
        Append onto an experiment's list of tags.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "addedTags": tags}
        response = self.post_from_endpoint("write/experiment/tags", payload)
        return response

    def delete_experiment_tags(
        self, experiment_key: str, tags: List[str]
    ) -> requests.Response:
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "tags": tags}
        response = self.post_from_endpoint("write/experiment/tags/delete", payload)
        return response

    def log_experiment_asset(
        self,
        experiment_key: str,
        file_data: Any,
        step: Optional[int] = None,
        overwrite: Optional[bool] = None,
        context: Optional[str] = None,
        ftype: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extension: Optional[str] = None,
        file_content: Optional[Any] = None,
        file_name: Optional[str] = None,
    ) -> requests.Response:
        """
        Upload an asset to an experiment.
        """
        if file_name is None:
            if not isinstance(file_data, str):
                LOGGER.warning(CONNECTION_LOG_EXPERIMENT_ASSET_NO_NAME_WARNING)
                file_name = "unnamed"
            else:
                file_name = file_data

        params = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "fileName": file_name}
        if step is not None:
            params["step"] = step
        if overwrite is not None:
            params["overwrite"] = overwrite
        if context is not None:
            params["context"] = context
        if ftype is not None:
            params["type"] = ftype
        if metadata is not None:
            params["metadata"] = metadata
        if extension is not None:
            params["extension"] = extension
        else:
            ext = os.path.splitext(file_name)[1]
            if ext:
                params["extension"] = ext

        headers = {API_KEY_HEADER: self.api_key}
        headers.update(self.low_level_api_client.headers)

        url = url_join(self.base_url, "write/experiment/upload-asset")

        if file_content:
            processor = AssetDataUploadProcessor(
                file_content,
                ftype,
                params,
                upload_limit=float("+inf"),
                copy_to_tmp=False,
                error_message_identifier=None,
                metadata=metadata,
                tmp_dir=None,
                critical=False,
            )
            message = processor.process()
        else:
            processor = AssetUploadProcessor(
                file_data,
                ftype,
                params,
                upload_limit=float("+inf"),
                copy_to_tmp=False,
                error_message_identifier=None,
                metadata=metadata,
                tmp_dir=None,
                critical=False,
            )
            message = processor.process()

        # We could get a file-like upload message in case filename is not a file-path or an invalid one
        if isinstance(message, UploadFileMessage):
            response = send_file(
                url,
                message.file_path,
                params=message.additional_params,
                headers=headers,
                timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                metadata=message.metadata,
                session=self.low_level_api_client.session,
            )
        else:
            response = send_file_like(
                url,
                message.file_like,
                params=message.additional_params,
                headers=headers,
                timeout=self.config.get_int(None, "comet.timeout.file_upload"),
                metadata=message.metadata,
                session=self.low_level_api_client.session,
            )

        return _check_response_status(response)

    def log_experiment_image(
        self,
        experiment_key: str,
        filename: str,
        image_name: Optional[str] = None,
        step: Optional[int] = None,
        overwrite: Optional[bool] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Upload an image asset to an experiment.
        """
        _, filename_extension = os.path.splitext(filename)
        with open(filename, "rb") as fp:
            params = {
                PAYLOAD_EXPERIMENT_KEY: experiment_key,
                "type": "image",
                "extension": filename_extension,
            }
            files = {"file": (filename, fp)}
            if image_name is not None:
                params["fileName"] = image_name
            if step is not None:
                params["step"] = step
            if overwrite is not None:
                params["overwrite"] = overwrite
            if context is not None:
                params["context"] = context
            if metadata is not None:
                params["metadata"] = metadata
            response = self.post_from_endpoint(
                "write/experiment/upload-asset", {}, params=params, files=files
            )
            return response

    def log_experiment_video(
        self,
        experiment_key: str,
        filename: str,
        video_name: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        overwrite: Optional[bool] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        extension = get_file_extension(filename)
        if extension is None:
            raise CometException(EXTENSION_NOT_FOUND)

        if extension.upper() not in SUPPORTED_VIDEO_FORMATS:
            raise CometException(
                EXTENSION_NOT_SUPPORTED
                % (extension, ", ".join(SUPPORTED_VIDEO_FORMATS))
            )

        with open(filename, "rb") as fp:
            params = {
                PAYLOAD_EXPERIMENT_KEY: experiment_key,
                "type": ASSET_TYPE_VIDEO,
                "extension": extension,
            }
            files = {"file": (filename, fp)}
            if video_name is not None:
                params["fileName"] = video_name
            if step is not None:
                params["step"] = step
            if epoch is not None:
                params["epoch"] = epoch
            if overwrite is not None:
                params["overwrite"] = overwrite
            if context is not None:
                params["context"] = context
            if metadata is not None:
                files["metadata"] = encode_metadata(metadata)

            response = self.post_from_endpoint(
                "write/experiment/upload-asset", {}, params=params, files=files
            )
            return response

    def log_experiment_remote_model(
        self,
        experiment_key: str,
        model_name: str,
        remote_assets: List[dict],
        on_model_upload: Optional[Callable],
        on_failed_model_upload: Optional[Callable],
    ) -> requests.Response:
        payload = {
            PAYLOAD_API_KEY: self.api_key,
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            PAYLOAD_MODEL_NAME: model_name,
            PAYLOAD_REMOTE_ASSETS: remote_assets,
        }
        try:
            response = self.post_from_endpoint(
                endpoint="asset/log-remote-model",
                payload=payload,
                alternate_base_url=self.server_url + "/clientlib/",
            )

            if on_model_upload is not None:
                try:
                    on_model_upload(response)
                except Exception:
                    LOGGER.warning("Failed to call on_asset_upload", exc_info=True)

            return response
        except Exception as exception:
            if on_failed_model_upload is not None:
                try:
                    on_failed_model_upload(exception)
                except Exception:
                    LOGGER.warning(
                        "Failed to call on_failed_asset_upload", exc_info=True
                    )
            raise exception

    def stop_experiment(self, experiment_key: str) -> Any:
        """
        Stop a running experiment.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("write/experiment/stop", payload)

    def set_experiment_not_throttled(self, experiment_key: str) -> Any:
        """
        Unthrottles an experiment, removing a throttling flag from the experiment metadata.

        Args:
            experiment_key: The unique identifier for the experiment.
        Returns:
            The result of the unthrottling operation, depending on the implementation.
        """
        try:
            self._check_api_backend_version(SemanticVersion.parse("4.7.416"))

            url = f"write/experiment/{experiment_key}/mark-not-throttled"
            response = self.post_from_endpoint(url, {})
            return response
        except BackendVersionTooOld as ex:
            LOGGER.info(
                "Failed to set experiment as not throttled, backend version '%s' is too old, expected version '%s' or higher",
                ex.backend_version,
                ex.minimal_backend_version,
            )

    def get_artifact_lineage(
        self, experiment_key: str, direction: str
    ) -> Dict[str, Any]:
        parameters = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "filter": direction}
        return self.get_from_endpoint("get/artifacts", params=parameters)

    def get_artifact_list(
        self,
        workspace: str,
        artifact_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {"workspace": workspace}

        if artifact_type is not None:
            params["type"] = artifact_type

        return self.get_from_endpoint("artifacts/get-all", params)

    def get_artifact_details(
        self,
        artifact_id: Optional[str] = None,
        workspace: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "artifact_id": artifact_id,
            "workspace": workspace,
            "artifactName": name,
        }

        return self.get_from_endpoint("artifacts/get", params)

    def get_artifact_version_details(
        self,
        workspace: Optional[str] = None,
        name: Optional[str] = None,
        artifact_id: Optional[str] = None,
        version: Optional[str] = None,
        alias: Optional[str] = None,
        artifact_version_id: Optional[str] = None,
        version_or_alias: Optional[str] = None,
        experiment_key: Optional[str] = None,
        consumer_experiment_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "alias": alias,
            "artifactId": artifact_id,
            "artifactName": name,
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "consumerExperimentKey": consumer_experiment_key,
            "version": version,
            "versionId": artifact_version_id,
            "versionOrAlias": version_or_alias,
            "workspace": workspace,
        }

        return self.get_from_endpoint("artifacts/version", params)

    def get_artifact_files(
        self,
        artifact_id: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "artifact_id": artifact_id,
            "workspace": workspace,
            "artifactName": name,
            "version": version,
            "alias": alias,
        }

        return self.get_from_endpoint("artifacts/version/files", params)

    def upsert_artifact(
        self,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
        description: Optional[str] = None,
        experiment_key: Optional[str] = None,
        is_public: Optional[str] = None,
        metadata: Optional[Dict[Any, Any]] = None,
        version: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        version_tags: Optional[List[str]] = None,
    ) -> requests.Response:
        version_metadata = encode_metadata(metadata)

        payload = {
            "artifactName": artifact_name,
            "artifactType": artifact_type,
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "versionMetadata": version_metadata,
            "version": version,
            "alias": aliases,
            "versionTags": version_tags,
        }

        return self.post_from_endpoint("write/artifacts/upsert", payload)

    def update_artifact(
        self,
        artifact_id: str,
        artifact_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> requests.Response:
        artifact_metadata = encode_metadata(metadata)

        payload = {
            "artifactId": artifact_id,
            "artifactType": artifact_type,
            "versionMetadata": artifact_metadata,
            "version": version,
            "tags": tags,
        }

        return self.post_from_endpoint("write/artifacts/details", payload)

    def update_artifact_version(
        self,
        artifact_version_id: str,
        version_aliases: Optional[List[str]] = None,
        version_metadata: Optional[Dict[str, Any]] = None,
        version_tags: Optional[Sequence[str]] = None,
    ) -> requests.Response:
        artifact_version_metadata = encode_metadata(version_metadata)

        payload = {
            "alias": version_aliases,
            "artifactVersionId": artifact_version_id,
            "versionMetadata": artifact_version_metadata,
            "versionTags": version_tags,
        }

        return self.post_from_endpoint("write/artifacts/version/labels", payload)

    def set_experiment_state(
        self, experiment_key: str, state: str
    ) -> requests.Response:
        if state == "running":
            is_alive = True
            has_crashed = False
        elif state == "finished":
            is_alive = False
            has_crashed = False
        elif state == "crashed":
            is_alive = False
            has_crashed = True
        else:
            raise ValueError(
                CONNECTION_SET_EXPERIMENT_STATE_UNSUPPORTED_EXCEPTION,
                state,
            )

        return self.update_experiment_error_status(
            experiment_key=experiment_key,
            is_alive=is_alive,
            error_value=None,
            has_crashed=has_crashed,
        )

    def update_experiment_error_status(
        self,
        experiment_key: str,
        is_alive: bool,
        error_value: Optional[str],
        has_crashed: bool = False,
    ) -> requests.Response:
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "isAlive": is_alive,
            "error": error_value,
            "hasCrashed": has_crashed,
        }

        return self.post_from_endpoint(
            "write/experiment/update-status", payload=payload
        )

    def _prepare_update_artifact_version_state(
        self,
        artifact_version_id: str,
        experiment_key: str,
        state: str,
    ) -> PreparedRequest:
        params = {
            "artifactVersionId": artifact_version_id,
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "state": state,
        }

        url = url_join(self.base_url, "write/artifacts/state")

        return PreparedRequest(
            api_key=self.api_key,
            url=url,
            json=params,
            headers=self.low_level_api_client.headers,
        )

    # Create, Delete, Archive and Move methods:

    def move_experiments(
        self,
        experiment_keys: List[str],
        target_workspace: str,
        target_project_name: str,
        symlink: bool = False,
    ) -> requests.Response:
        """
        Move/symlink list of experiments to another workspace/project_name
        """
        payload = {
            "targetWorkspaceName": target_workspace,
            "targetProjectName": target_project_name,
            "experimentKeys": experiment_keys,
            "symlink": symlink,
        }
        return self.post_from_endpoint("write/experiment/move", payload)

    def delete_experiment(self, experiment_key: str) -> Any:
        """
        Delete one experiment.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("write/experiment/delete", payload)

    def delete_experiment_asset(self, experiment_key: str, asset_id: str) -> Any:
        """
        Delete an experiment's asset.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key, "assetId": asset_id}
        return self.get_from_endpoint("write/experiment/asset/delete", payload)

    def delete_project(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        delete_experiments: bool = False,
    ) -> requests.Response:
        """
        Delete a project.
        """
        if project_id is not None:
            payload = {
                "projectId": project_id,
                "deleteAllExperiments": delete_experiments,
            }
        else:
            payload = {
                "workspaceName": workspace,
                "projectName": project_name,
                "deleteAllExperiments": delete_experiments,
            }
        return self.post_from_endpoint("write/project/delete", payload)

    def restore_experiment(self, experiment_key: str) -> Dict[str, Any]:
        """
        Restore one experiment.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("write/experiment/restore", payload)

    def delete_experiments(self, experiment_keys: List[str]) -> List[Any]:
        """
        Delete list of experiments.
        """
        return [self.delete_experiment(key) for key in experiment_keys]

    def archive_experiment(self, experiment_key: str) -> Any:
        """
        Archive one experiment.
        """
        payload = {PAYLOAD_EXPERIMENT_KEY: experiment_key}
        return self.get_from_endpoint("write/experiment/archive", payload)

    def archive_experiments(self, experiment_keys: List[str]) -> List[Any]:
        """
        Archive list of experiments.
        """
        return [self.archive_experiment(key) for key in experiment_keys]

    def create_project(
        self,
        workspace: str,
        project_name: str,
        project_description: Optional[str] = None,
        public: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a project.
        """
        payload = {
            "workspaceName": workspace,
            "projectName": project_name,
            "projectDescription": project_description,
            "isPublic": public,
        }
        response = self.post_from_endpoint("write/project/create", payload)
        return response.json()

    def create_experiment(
        self, workspace: str, project_name: str, experiment_name: Optional[str] = None
    ) -> requests.Response:
        """
        Create an experiment and return its associated APIExperiment.
        """
        payload = {"workspaceName": workspace, "projectName": project_name}
        if experiment_name is not None:
            payload["experimentName"] = experiment_name
        return self.post_from_endpoint("write/experiment/create", payload)

    def create_experiment_symlink(
        self, experiment_key: str, project_name: str
    ) -> Dict[str, Any]:
        """
        Create a copy of this experiment in another project
        in the workspace.
        """
        payload = {
            PAYLOAD_EXPERIMENT_KEY: experiment_key,
            "projectName": project_name,
        }
        return self.get_from_endpoint(
            "write/project/symlink", payload, return_type="json"
        )

    # Experiment model methods:

    def get_experiment_models(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Given an experiment id, return a list of model data associated
        with an experiment.

        Args:
            experiment_id: the experiment's key

        Returns [{'experimentModelId': 'MODEL-ID'
                  'experimentKey': 'EXPERIMENT-KEY',
                  'modelName': 'MODEL-NAME'}, ...]
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_id}
        response = self.get_from_endpoint("experiment/model", params)
        if response:
            return response["models"]

    def get_experiment_model_asset_list(
        self, experiment_id: str, model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get an experiment model's asset list by model name.

        Args:
            experiment_id: the experiment's key
            model_name: str, the name of the model

        Returns: a list of asset dictionaries with these fields:
            * fileName
            * fileSize
            * runContext
            * step
            * link
            * createdAt
            * dir
            * canView
            * audio
            * histogram
            * image
            * type
            * metadata
            * assetId
        """
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_id, "modelName": model_name}
        response = self.get_from_endpoint("experiment/model/asset/list", params)
        if response:
            return response["assets"]

    def get_experiment_model_zipfile(
        self, experiment_id: str, model_name: str
    ) -> bytes:
        params = {PAYLOAD_EXPERIMENT_KEY: experiment_id, "modelName": model_name}
        return self.get_from_endpoint(
            "experiment/model/download",
            params,
            return_type="binary",
            timeout=self.config.get_int(None, "comet.timeout.file_download"),
        )

    # Registry model methods:

    def get_registry_models(self, workspace: str) -> List[Dict[str, Any]]:
        """
        Return a list of registered models in workspace.

        Args:
            workspace: the name of workspace
        """
        params = {"workspaceName": workspace}
        response = self.get_from_endpoint("registry-model", params)
        if response:
            return response["registryModels"]
        else:
            return []

    def get_registry_model_count(self, workspace: str) -> int:
        """
        Return a count of registered models in workspace.

        Args:
            workspace: the name of workspace
        """
        params = {"workspaceName": workspace}
        response = self.get_from_endpoint("registry-model/count", params)
        if response:
            return response["registryModelCount"]
        else:
            return 0

    def get_registry_model_versions(
        self, workspace: str, registry_name: str
    ) -> List[str]:
        """
        Return a list of versions of the registered model in the given
        workspace.

        Args:
            workspace: the name of workspace
            registry_name: the name of the registered model
        """
        return [
            m["version"]
            for m in self.get_registry_model_details(workspace, registry_name)[
                "versions"
            ]
        ]

    def get_registry_model_details(
        self, workspace: str, registry_name: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Return a dictionary of details of the model in given
        workspace.

        Args:
            workspace: the name of workspace
            registry_name: the name of the registered model
            version: optional, the string version number
        """
        params = {"workspaceName": workspace, "modelName": registry_name}
        response = self.get_from_endpoint("registry-model/details", params)
        if response:
            if version is None:
                return response
            else:
                select = [
                    model
                    for model in response["versions"]
                    if model["version"] == version
                ]
                if len(select) > 0:
                    return select[0]
                else:
                    return None

    def get_latest_registry_model_details(
        self,
        workspace: str,
        registry_name: str,
        stage: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = {
            "workspaceName": workspace,
            "modelName": registry_name,
        }

        optional_update(
            params,
            {
                "versionMajor": version_major,
                "versionMinor": version_minor,
                "stage": stage,
            },
        )

        return self.get_from_endpoint(
            "registry-model/latest_version",
            params,
            return_type="json",
        )

    def get_registry_model_notes(self, workspace: str, registry_name: str) -> str:
        """
        Return the notes for a registered model.
        """
        params = {"workspaceName": workspace, "modelName": registry_name}
        response = self.get_from_endpoint("registry-model/notes", params)
        if response:
            return response["notes"]
        else:
            return ""

    def get_registry_model_zipfile(
        self,
        workspace: str,
        registry_name: str,
        version: Optional[str],
        stage: Optional[str],
    ) -> bytes:
        params = {
            "workspaceName": workspace,
            "modelName": registry_name,
        }

        if version and stage:
            raise ValueError(
                CONNECTION_DOWNLOAD_REGISTRY_MODEL_VERSION_OR_STAGE_EXCEPTION
            )

        if version is not None:
            params["version"] = version

        if stage is not None:
            params["stage"] = stage

        return self.get_from_endpoint(
            "registry-model/item/download",
            params,
            return_type="binary",
            timeout=self.config["comet.timeout.file_download"],
        )

    def get_registry_model_items_download_links(
        self,
        workspace: str,
        registry_name: str,
        version: Optional[str],
        stage: Optional[str],
    ) -> Dict[str, Any]:
        params = {
            "workspaceName": workspace,
            "modelName": registry_name,
        }

        if version and stage:
            raise ValueError(
                CONNECTION_DOWNLOAD_REGISTRY_MODEL_VERSION_OR_STAGE_EXCEPTION
            )

        if version is not None:
            params["version"] = version

        if stage is not None:
            params["stage"] = stage

        return self.get_from_endpoint(
            "registry-model/item/download-instructions",
            params,
            return_type="json",
        )

    # Write registry methods:

    def register_model_v2(
        self,
        experiment_id: str,
        model_name: str,
        version: str,
        workspace: str,
        registry_name: str,
        public: bool,
        description: str,
        comment: str,
        tags: List[str],
        status: str,
        stages: List[str],
    ) -> Optional[requests.Response]:
        # we need to import here to avoid circular imports
        import comet_ml.api_objects.model

        if not comet_ml.api_objects.model.Model.__internal_api_compatible_backend__(
            client=self
        ):
            if tags is not None:
                LOGGER.warning(
                    CONNECTION_REGISTER_MODEL_TAGS_IGNORED_WARNING.format(tags)
                )
            if status is not None:
                LOGGER.warning(
                    CONNECTION_REGISTER_MODEL_STATUS_IGNORED_WARNING.format(status)
                )
            return self.register_model(
                experiment_id,
                model_name,
                version,
                workspace,
                registry_name,
                public,
                description,
                comment,
                stages,
            )
        if stages is not None:
            LOGGER.warning(
                CONNECTION_REGISTER_MODEL_STAGES_IGNORED_WARNING.format(stages)
            )
        comet_ml.api_objects.model.Model.__internal_api__register__(
            experiment_id,
            model_name,
            version,
            workspace,
            registry_name,
            public,
            description,
            comment,
            tags,
            status,
            api_key=self.api_key,
        )

        LOGGER.info(
            CONNECTION_REGISTER_MODEL_SUCCESS_INFO,
            registry_name,
            version,
            workspace,
        )
        return None

    def register_model(
        self,
        experiment_id: str,
        model_name: str,
        version: str,
        workspace: str,
        registry_name: str,
        public: bool,
        description: str,
        comment: str,
        stages: List[str],
    ) -> requests.Response:
        """
        Register an experiment model in the workspace registry.

        Args:
            experiment_id: the experiment key
            model_name: the name of the experiment model
            workspace: the name of workspace
            version: a version string
            registry_name: the name of the registered workspace model
            public: if True, then the model will be publicly viewable
            description: optional, a textual description of the model
            comment: optional, a textual comment about the model
            stages: optional, a list of textual tags such as ["production", "staging"] etc.

        Returns 200 Response if successful
        """
        models = self.get_experiment_models(experiment_id)
        if len(models) == 0:
            raise ValueError(
                CONNECTION_REGISTER_MODEL_NO_MODEL_EXCEPTION % experiment_id
            )
        # Look up the model name:
        details = [model for model in models if model["modelName"] == model_name]
        # If model name found:
        if len(details) == 1:
            registry_name = proper_registry_model_name(
                registry_name
            ) or proper_registry_model_name(model_name)
            registry_models = [model for model in self.get_registry_models(workspace)]
            model_id = details[0]["experimentModelId"]
            payload = {
                "experimentModelId": model_id,
                "registryModelName": registry_name,
                "version": version,
            }
            if public is not None:
                payload["isPublic"] = public
            if description is not None:
                payload["description"] = description
            if comment is not None:
                payload["comment"] = comment
            if stages is not None:
                if not isinstance(stages, (list, tuple)) or any(
                    not isinstance(s, str) for s in stages
                ):
                    raise ValueError(
                        CONNECTION_REGISTER_MODEL_INVALID_STAGES_LIST_EXCEPTION
                    )
                payload["stages"] = stages

            # Now we create or add a new version:
            if payload["registryModelName"] in [
                model["modelName"] for model in registry_models
            ]:
                # Adding a new version of existing registry model:
                if "description" in payload:
                    del payload["description"]
                    LOGGER.warning(CONNECTION_REGISTER_MODEL_DESCRIPTION_WARNING)
                if "isPublic" in payload:
                    del payload["isPublic"]
                    LOGGER.warning(CONNECTION_REGISTER_MODEL_PUBLIC_WARNING)
                # Update:
                response = self.post_from_endpoint(
                    "write/registry-model/item", payload=payload
                )
            else:
                # Create:
                response = self.post_from_endpoint(
                    "write/registry-model", payload=payload
                )

            LOGGER.info(
                CONNECTION_REGISTER_MODEL_SUCCESS_INFO,
                registry_name,
                version,
                workspace,
            )
            return response
        else:
            # Model name not found
            model_names = [model["modelName"] for model in models]
            raise ValueError(
                CONNECTION_REGISTER_MODEL_INVALID_MODEL_NAME_EXCEPTION
                % (model_name, model_names)
            )

    def update_registry_model_version(
        self,
        workspace: str,
        registry_name: str,
        version: str,
        comment: Optional[str] = None,
        stages: Optional[List[str]] = None,
    ) -> requests.Response:
        """
        Updates a registered model version's comments and/or stages.
        """
        details = self.get_registry_model_details(workspace, registry_name, version)
        payload = {"registryModelItemId": details["registryModelItemId"]}
        # update the registry model version: comment and stages
        if comment is not None:
            payload["comment"] = comment
        if stages is not None:
            if not isinstance(stages, (list, tuple)) or any(
                not isinstance(s, str) for s in stages
            ):
                raise ValueError("Invalid stages list: should be a list of strings")
            payload["stages"] = stages
        return self.post_from_endpoint("write/registry-model/item/update", payload)

    def update_registry_model(
        self,
        workspace: str,
        registry_name: str,
        new_name: Optional[str] = None,
        description: Optional[str] = None,
        public: Optional[bool] = None,
    ) -> requests.Response:
        """
        Updates a registered model's name, description, and/or visibility.
        """
        details = self.get_registry_model_details(workspace, registry_name)
        payload = {"registryModelId": details["registryModelId"]}
        # update the registry model top level: name, description, and public
        if new_name is not None:
            payload["registryModelName"] = new_name
        if description is not None:
            payload["description"] = description
        if public is not None:
            payload["isPublic"] = public
        return self.post_from_endpoint("write/registry-model/update", payload)

    def delete_registry_model_version(
        self, workspace: str, registry_name: str, version: str
    ) -> Any:
        """
        Delete a registered model version
        """
        details = self.get_registry_model_details(workspace, registry_name, version)
        payload = {"modelItemId": details["registryModelItemId"]}
        response = self.get_from_endpoint(
            "write/registry-model/item/delete", payload, return_type="response"
        )
        if response:
            return response

    def delete_registry_model(self, workspace: str, registry_name: str) -> Any:
        """
        Delete a registered model
        """
        params = {"workspaceName": workspace, "modelName": registry_name}
        response = self.get_from_endpoint(
            "write/registry-model/delete", params, return_type="response"
        )
        if response:
            return response

    def update_registry_model_notes(
        self, workspace: str, registry_name: str, notes: str
    ) -> requests.Response:
        """
        Update the notes of a registry model.
        """
        payload = {
            "workspaceName": workspace,
            "registryModelName": registry_name,
            "notes": notes,
        }
        return self.post_from_endpoint("write/registry-model/notes", payload)

    def add_registry_model_version_stage(
        self, workspace, registry_name, version, stage
    ):
        details = self.get_registry_model_details(workspace, registry_name, version)
        if details is None:
            raise CometException(
                "could not find details for model {} version {} in workspace {}".format(
                    registry_name, version, workspace
                )
            )
        params = {"modelItemId": details["registryModelItemId"], "stage": stage}
        response = self.get_from_endpoint(
            "write/registry-model/item/stage", params, return_type="response"
        )
        if response:
            return response

    def delete_registry_model_version_stage(
        self, workspace: str, registry_name: str, version: str, stage: str
    ) -> Any:
        details = self.get_registry_model_details(workspace, registry_name, version)
        if details is None:
            raise CometException(
                "could not find details for model {} version {} in workspace {}".format(
                    registry_name, version, workspace
                )
            )
        params = {"modelItemId": details["registryModelItemId"], "stage": stage}
        response = self.get_from_endpoint(
            "write/registry-model/item/stage/delete", params, return_type="response"
        )
        if response:
            return response

    # Other helpers
    def _check_api_backend_version(
        self, minimal_backend_version: SemanticVersion
    ) -> None:
        version_url = get_backend_version_url(self.server_url)
        if self.backend_version is None:
            self.backend_version = self._get_api_backend_version(version_url)

        if self.backend_version is None:
            return

        # Compare versions
        if self.backend_version < minimal_backend_version:
            raise BackendVersionTooOld(
                version_url, self.backend_version, minimal_backend_version
            )

    def _get_api_backend_version(self, version_url: str) -> Optional[SemanticVersion]:
        # Get the backend version
        try:
            response = self.low_level_api_client.get(
                version_url, check_status_code=True
            )
            # Invalid version will raise exception:
            response_body = response.json()
            if "version" not in response_body:
                raise ValueError("No version field in the backend response.")
            return SemanticVersion.parse(response.json()["version"])
        except Exception as e:
            LOGGER.warning(BACKEND_VERSION_CHECK_ERROR, version_url, e, exc_info=True)
            return None

    def get_api_backend_version(self) -> Optional[SemanticVersion]:
        version_url = get_backend_version_url(self.server_url)
        if self.backend_version is None:
            self.backend_version = self._get_api_backend_version(version_url)

        return self.backend_version

    # General methods:

    def close(self) -> None:
        self.low_level_api_client.close()

    def do_not_cache(self, *items) -> None:
        """
        Add these items from the do-not-cache list. Ignored
        as this class does not have cache.
        """
        pass

    def do_cache(self, *items) -> None:
        """
        Remove these items from the do-not-cache list. Raises
        Exception as this class does not have cache.
        """
        raise Exception("this implementation does not have cache")


class RestApiClientWithCache(RestApiClient):
    """
    Same as RestApiClient, except with optional cache.

    When you post_from_endpoint(write_endpoint) you clear the
    associated read_endpoints.

    When you get_from_endpoint(read_endpoint) you attempt to
    read from cache and save to cache unless in the NOCACHE.

    If you read from a read_endpoint that is not listed, then
    a debug message is shown.
    """

    # map of write endpoints to read endpoints
    # POST-ENDPOINT: [(GET-ENDPOINT, [GET-ARGS]), ...]
    ENDPOINTS = {
        "write/project/symlink": [],  # Nothing to do
        "write/experiment/set-status": [],  # Nothing to do
        "write/experiment/set-start-end-time": [
            ("experiment/metadata", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/file-path": [
            ("experiment/metadata", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/": [("experiment/metadata", [PAYLOAD_EXPERIMENT_KEY])],
        SYSTEM_DETAILS_WRITE_ENDPOINT: [
            ("experiment/system-details", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/code": [("experiment/code", [PAYLOAD_EXPERIMENT_KEY])],
        "write/experiment/graph": [("experiment/graph", [PAYLOAD_EXPERIMENT_KEY])],
        "write/experiment/output": [("experiment/output", [PAYLOAD_EXPERIMENT_KEY])],
        "write/experiment/log-other": [
            ("experiment/log-other", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/parameter": [
            ("experiment/parameters", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/metric": [
            ("experiment/metrics/get-metric", [PAYLOAD_EXPERIMENT_KEY]),
            ("experiment/metrics/summary", [PAYLOAD_EXPERIMENT_KEY]),
        ],
        "write/experiment/html": [("experiment/html", [PAYLOAD_EXPERIMENT_KEY])],
        "write/experiment/tags": [("experiment/tags", [PAYLOAD_EXPERIMENT_KEY])],
        "write/experiment/upload-asset": [
            (
                "experiment/asset/list",
                [PAYLOAD_EXPERIMENT_KEY],
            ),  # not usually cached
            (
                "experiment/asset/get-asset",
                [PAYLOAD_EXPERIMENT_KEY, "assetId"],
            ),  # not usually cached
        ],
        "write/experiment/git/metadata": [
            ("experiment/git/metadata", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/git/patch": [
            ("experiment/git/patch", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/cpu-metrics": [
            ("experiment/system-details", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/gpu-metrics": [
            ("experiment/system-details", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/load-metrics": [
            ("experiment/system-details", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/experiment/ram-metrics": [
            ("experiment/system-details", [PAYLOAD_EXPERIMENT_KEY])
        ],
        "write/project/notes": [("project/notes", ["projectId"])],
        # Only read that is really a POST:
        "experiments/multi-metric-chart": [],  # Nothing to do
    }  # type: Dict[str, List[Tuple[str, List[str]]]]

    # Don't use cache on these GET endpoints:
    NOCACHE_ENDPOINTS = {
        "write/experiment/delete",
        "write/experiment/archive",
        "write/experiment/restore",
        "write/experiment/set-status",
        "write/project/symlink",
        "projects",
        "project/column-names",
        "experiments",
        "experiment/metadata",
        "experiment/asset/get-asset",
        "experiment/asset/list",
        "experiment/model",
        "experiment/model/asset/list",
        "experiment/model/download",
        "registry-model",
        "registry-model/count",
        "registry-model/details",
        "registry-model/notes",
        "write/registry-model/delete",
        "write/registry-model/item/delete",
        "write/registry-model/item/stage",
        "write/registry-model/item/stage/delete",
        "write/project/delete-project-share-link",
        "project/get-project-share-links",
        "write/project/add-share-link",
    }

    # Some read endpoints have additional payload key/values to clear:
    EXTRA_PAYLOAD = {
        "experiment/asset/list": {
            "type": [
                "all",
                "histogram_combined_3d",
                "image",
                ASSET_TYPE_AUDIO,
                ASSET_TYPE_VIDEO,
            ]
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = True
        self.cache = {}
        self.READ_ENDPOINTS = list(
            itertools.chain.from_iterable(
                [[ep for (ep, items) in self.ENDPOINTS[key]] for key in self.ENDPOINTS]
            )
        )

    # Cache methods:

    def do_not_cache(self, *items):
        """
        Add these items from the do-not-cache list.
        """
        for item in items:
            self.NOCACHE_ENDPOINTS.add(item)

    def do_cache(self, *items):
        """
        Remove these items from the do-not-cache list.
        """
        for item in items:
            if item in self.NOCACHE_ENDPOINTS:
                self.NOCACHE_ENDPOINTS.remove(item)

    def cacheable(self, read_endpoint):
        return read_endpoint not in self.NOCACHE_ENDPOINTS

    def get_hash(self, kwargs):
        key = hash(json.dumps(kwargs, sort_keys=True, default=str))
        return key

    def cache_get(self, **kwargs):
        # Look up in cache
        key = self.get_hash(kwargs)
        if key in self.cache:
            hit = True
            value = self.cache[key]
        else:
            hit = False
            value = None
        return hit, value

    def cache_put(self, value, **kwargs):
        # Put in cache
        if not self.check_read_endpoint(kwargs["endpoint"]):
            LOGGER.debug(
                "this endpoint cannot be cleared from cache: %r", kwargs["endpoint"]
            )
        key = self.get_hash(kwargs)
        LOGGER.debug("cache_put: %s, key: %s", kwargs, key)
        self.cache[key] = value

    def cache_clear_return_types(self, **kwargs):
        # Remove all return_types from cache for this endpoint/params:
        for return_type in ["json", "binary", "text"]:
            kwargs["return_type"] = return_type
            key = self.get_hash(kwargs)
            LOGGER.debug("attempting cache_clear: %s, key: %s", kwargs, key)
            if key in self.cache:
                LOGGER.debug("cache_clear: CLEARED!")
                del self.cache[key]

    def cache_clear(self, **kwargs):
        extra = self.EXTRA_PAYLOAD.get(kwargs["endpoint"])
        if extra:
            # First, without extras:
            self.cache_clear_return_types(**kwargs)
            # If more than one extra, we have to do all combinations:
            for key in extra:
                for value in extra[key]:
                    kwargs["payload"][key] = value
                    self.cache_clear_return_types(**kwargs)
        else:
            self.cache_clear_return_types(**kwargs)

    def get_read_endpoints(self, write_endpoint):
        """
        Return the mapping from a write endpoint to a list
        of tuples of (read-endpoint, [payload keys]) to
        clear the associated read endpoint caches.
        """
        return self.ENDPOINTS.get(write_endpoint, None)

    def check_read_endpoint(self, read_endpoint):
        """
        Check to see if the read_endpoint is in the
        list of known ones, or if it is not cached.
        If it is neither, then there is no way to
        clear it.
        """
        return (read_endpoint in self.READ_ENDPOINTS) or not self.cacheable(
            read_endpoint
        )

    # Overridden methods:

    def reset(self):
        self.cache.clear()

    def cache_clear_read_endpoints(self, endpoint, payload):
        # Clear read cache:
        endpoints = self.get_read_endpoints(endpoint)
        if endpoints:
            for read_endpoint, keys in endpoints:
                # Build read payload:
                read_payload = {}
                for key in keys:
                    if key in payload:
                        read_payload[key] = payload[key]
                self.cache_clear(endpoint=read_endpoint, payload=read_payload)

    def get_from_endpoint(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]],
        return_type: str = "json",
        alternate_base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> Any:
        """
        Wrapper around RestApiClient.get_from_endpoint() that adds cache.

        """
        if self.use_cache and self.cacheable(endpoint) and not stream:
            hit, result = self.cache_get(
                endpoint=endpoint, payload=params, return_type=return_type
            )
            if hit:
                # LOGGER.debug(
                #     "RestApiClientWithCache, hit: endpoint = %s, params = %s, return_type = %s",
                #     endpoint,
                #     params,
                #     return_type,
                # )
                return result

        # LOGGER.debug(
        #     "RestApiClientWithCache, miss: endpoint = %s, params = %s, return_type = %s",
        #     endpoint,
        #     params,
        #     return_type,
        # )
        retval = super().get_from_endpoint(
            endpoint,
            params,
            return_type,
            alternate_base_url=alternate_base_url,
            timeout=timeout,
            stream=stream,
        )

        if (
            self.use_cache
            and self.cacheable(endpoint)
            and retval is not None
            and not stream
        ):
            self.cache_put(
                retval, endpoint=endpoint, payload=params, return_type=return_type
            )

        return retval

    def post_from_endpoint(
        self,
        endpoint: str,
        payload: Any,
        alternate_base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Wrapper that clears the cache after posting.
        """
        response = super().post_from_endpoint(
            endpoint=endpoint,
            payload=payload,
            alternate_base_url=alternate_base_url,
            **kwargs,
        )
        self.cache_clear_read_endpoints(endpoint, payload)
        if "params" in kwargs:
            self.cache_clear_read_endpoints(endpoint, kwargs["params"])
        return response


class CometApiClient(BaseApiClient):
    """
    Inputs must be JSON-encodable, any conversion must be done by the caller.

    One method equals one endpoint and one call
    """

    def __init__(
        self, server_url: str, low_level_api_client: LowLevelHTTPClient, config: Config
    ) -> None:
        super().__init__(server_url, ["api/auth/"], low_level_api_client, None, config)

    def get(
        self,
        url: str,
        params: Dict[str, str],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> requests.Response:
        # Overrides to suppress exceptions on errors
        response = self.low_level_api_client.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            stream=stream,
        )
        return response

    def post(
        self,
        url: str,
        payload: Any,
        headers: Optional[Mapping[str, Optional[str]]] = None,
        files: Optional[Any] = None,
        params: Optional[Any] = None,
        custom_encoder: Optional[Type[json.JSONEncoder]] = None,
        compress: bool = False,
        retry: Optional[bool] = None,
    ) -> requests.Response:
        return super().post(
            url,
            payload=payload,
            headers=headers,
            files=files,
            params=params,
            custom_encoder=custom_encoder,
            compress=compress,
            retry=retry,
        )

    def check_email(self, email: str, reason: str) -> int:
        """
        Check if the given email is associated with a user.

        Args:
            email: str, the email of the user
            reason: str, the reason for the check

        Returns: a status code

        * 200: ok, existing user
        * 204: unknown user
        """
        payload = {"email": email, "reason": reason}
        response = self.get_from_endpoint("users", payload, return_type="response")
        return response.status_code

    def create_user(
        self, email: str, username: str, signup_source: str, send_email: bool = True
    ) -> Dict[str, Any]:
        """
        Creates a temporary user token for the email/username.

        Args:
            email: str, an email address
            username: str, a proper Comet username
            signup_source: str, description of signup source
            send_email: bool, if True (the default), the new user will receive the welcome emails

        Returns: dict (if successful), a JSON response as follows.
            Otherwise, a CometRestApiException with reason for
            failure.

        POST /api/auth/users?sendEmail=true
        ```json
        {
         'cometUserName': 'new-usernane',
         'token': 'MXZRU1WXsEJAzeFn0I235423549',
         'apiKey': 'X6Wr0uKXZwOLnPFpNvV39874857'
        }
        ```
        """
        payload = {
            "email": email,
            "userName": username,
            "signupSource": signup_source,
        }
        # The backend expect boolean to be formatted like in JSON
        params = {"sendEmail": json.dumps(send_email)}
        response = self.post_from_endpoint("users", payload, params=params)
        return response.json()
