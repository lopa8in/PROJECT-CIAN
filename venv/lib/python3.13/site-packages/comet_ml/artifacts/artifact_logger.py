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

import logging
import threading
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Set, Tuple

import requests

from .._typing import PreparedRequest
from ..config import get_check_tls_certificate, get_comet_timeout_http
from ..connection.connection_api import RestApiClient
from ..connection.http_session import get_cached_comet_http_session
from ..debug import debug_helpers
from ..exceptions import (
    CometRestApiException,
    FileUploadThrottledException,
    LogArtifactException,
)
from ..file_uploader import is_asset_remote
from ..logging_messages import (
    ARTIFACT_ASSET_UPLOAD_FAILED,
    ARTIFACT_FAILED_TO_MARK_VERSION_AS_CLOSED_ERROR,
    ARTIFACT_FAILED_TO_MARK_VERSION_AS_FAILED_ERROR,
    ARTIFACT_LOG_ASSETS_SCHEDULING_INFO,
    ARTIFACT_LOG_EMPTY_ASSETS_WARNING,
    ARTIFACT_NAME_TRUNCATED_WARNING,
    ARTIFACT_UPLOAD_FINISHED,
    ARTIFACT_UPLOAD_STARTED,
    ARTIFACT_VERSION_CREATED_WITH_PREVIOUS,
    ARTIFACT_VERSION_CREATED_WITHOUT_PREVIOUS,
    LOG_ARTIFACT_IN_PROGRESS_MESSAGE,
)
from ..utils import (
    IterationProgressCallback,
    format_bytes,
    generate_guid,
    truncate_to_length_or_raise,
)
from .artifact import Artifact
from .artifact_getter import get_artifact
from .logged_artifact import LoggedArtifact

if TYPE_CHECKING:
    # noinspection PyUnusedImports
    from ..experiment import CometExperiment


LOGGER = logging.getLogger(__name__)


def log_artifact(
    artifact: Artifact, experiment: "CometExperiment", rest_api_client: RestApiClient
) -> LoggedArtifact:
    """
    Logs an artifact to a given experiment using a REST API client.

    This function handles the process of logging an artifact by first truncating
    its name if it surpasses the maximum allowed length, upserting the artifact
    details, and updating its version state. The function manages the artifact's
    assets upload and ensures proper handling of empty artifacts while also logging
    relevant warnings or errors during the process.

    Args:
        artifact:
            The artifact object to be logged, containing its details and associated assets.
        experiment:
            The experiment object where the artifact will be logged. Includes configuration
            and other relevant metadata for handling the artifact.
        rest_api_client:
            The REST API client used to communicate with the backend and perform operations
            related to the artifact.

    Returns:
        The artifact that has been successfully logged. Includes metadata such as
        workspace, name, and version.

    Raises:
        ValueError: If the artifact name cannot be truncated to the accepted length.
        Other errors may be raised depending on the status of API interactions or the
        configured debug helpers.
    """
    # truncate Artifact name if appropriate
    name, truncated = truncate_to_length_or_raise(
        artifact.name,
        length=experiment.artifact_name_max_length,
        raise_on_truncate=debug_helpers.has_enabled_debug_exception_raising(),
    )
    if truncated:
        artifact.name = name
        LOGGER.warning(
            ARTIFACT_NAME_TRUNCATED_WARNING, experiment.artifact_name_max_length
        )

    artifact_id, artifact_version_id = _upsert_artifact(
        artifact, rest_api_client, experiment.id
    )

    success_prepared_request = _prepare_update_artifact_version_state(
        rest_api_client, artifact_version_id, experiment.id, state="CLOSED"
    )
    timeout = get_comet_timeout_http(experiment.config)
    verify_tls = get_check_tls_certificate(experiment.config)

    logged_artifact = get_artifact(
        rest_api_client=rest_api_client,
        get_artifact_params={
            "artifact_id": artifact_id,
            "artifact_version_id": artifact_version_id,
        },
        experiment_id=experiment.id,
        summary=experiment._summary,
        config=experiment.config,
        raise_on_warning=debug_helpers.has_enabled_debug_exception_raising(),
    )

    if len(artifact._assets) == 0:
        debug_helpers.log_warning_or_raise(
            ARTIFACT_LOG_EMPTY_ASSETS_WARNING, logged_artifact, logger=LOGGER
        )

        _call_post_prepared_request(
            success_prepared_request, timeout, verify_tls=verify_tls
        )
    else:
        failed_prepared_request = _prepare_update_artifact_version_state(
            rest_api_client,
            artifact_version_id,
            experiment.id,
            state="ERROR",
        )

        _log_artifact_assets(
            artifact=artifact,
            experiment=experiment,
            artifact_version_id=artifact_version_id,
            logged_artifact_workspace=logged_artifact.workspace,
            logged_artifact_name=logged_artifact.name,
            logged_artifact_version=str(logged_artifact.version),
            success_prepared_request=success_prepared_request,
            failed_prepared_request=failed_prepared_request,
            timeout=timeout,
            verify_tls=verify_tls,
        )

        LOGGER.info(
            ARTIFACT_UPLOAD_STARTED,
            logged_artifact.workspace,
            logged_artifact.name,
            logged_artifact.version,
        )

    experiment._summary.increment_section("uploads", "artifacts")

    return logged_artifact


def _on_artifact_asset_upload(
    lock: threading.Lock,
    all_asset_ids: Set[str],
    asset_id: str,
    prepared_request: PreparedRequest,
    timeout: float,
    success_log_message: str,
    success_log_message_args: Tuple,
    verify_tls: bool,  # -> partial arguments end
    response: Optional[Any] = None,
    batched_asset_ids: Optional[Set[str]] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    with lock:
        if batched_asset_ids is not None:
            # remove ids from batch upload
            all_asset_ids.difference_update(batched_asset_ids)
        else:
            all_asset_ids.remove(asset_id)

        if len(all_asset_ids) == 0:
            try:
                _call_post_prepared_request(
                    prepared_request, timeout, verify_tls=verify_tls
                )
                LOGGER.info(success_log_message, *success_log_message_args)
            except Exception:
                LOGGER.error(
                    ARTIFACT_FAILED_TO_MARK_VERSION_AS_CLOSED_ERROR, exc_info=True
                )


def _on_artifact_failed_asset_upload(
    asset_id: str,
    prepared_request: PreparedRequest,
    timeout: float,
    error_log_message: str,
    error_log_message_args: Tuple,
    verify_tls: bool,  # -> partial arguments end
    response: Optional[Any] = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    if isinstance(response, FileUploadThrottledException):
        LOGGER.debug(
            f"Artifact asset ({asset_id}) upload has been throttled. It will be retried later."
        )
        return

    LOGGER.error(error_log_message, *error_log_message_args)

    try:
        _call_post_prepared_request(prepared_request, timeout, verify_tls=verify_tls)
    except Exception:
        LOGGER.error(ARTIFACT_FAILED_TO_MARK_VERSION_AS_FAILED_ERROR, exc_info=True)


def _log_artifact_assets(
    artifact: Artifact,
    experiment: "CometExperiment",
    artifact_version_id: str,
    logged_artifact_workspace: str,
    logged_artifact_name: str,
    logged_artifact_version: str,
    success_prepared_request: PreparedRequest,
    failed_prepared_request: PreparedRequest,
    timeout: float,
    verify_tls: bool,
) -> None:
    artifact_assets = artifact._assets.values()

    all_asset_ids = {artifact_asset.asset_id for artifact_asset in artifact_assets}

    lock = threading.Lock()

    # At the starts, it's the total numbers, but then it's the remaining numbers
    total_num_assets = len(artifact_assets)
    total_size = sum(asset.size for asset in artifact_assets)

    remote_assets_num = sum(is_asset_remote(asset) for asset in artifact_assets)
    local_assets_num = total_num_assets - remote_assets_num

    remote_assets_size = sum(
        asset.size for asset in artifact_assets if is_asset_remote(asset)
    )
    local_assets_size = total_size - remote_assets_size

    LOGGER.info(
        ARTIFACT_LOG_ASSETS_SCHEDULING_INFO.format(
            total_num_assets,
            local_assets_num,
            format_bytes(local_assets_size),
            remote_assets_num,
        ),
    )

    def progress_callback():
        LOGGER.info(
            LOG_ARTIFACT_IN_PROGRESS_MESSAGE.format(
                local_assets_num,
                format_bytes(local_assets_size),
                remote_assets_num,
            )
        )

    frequency = 5

    success_log_message = ARTIFACT_UPLOAD_FINISHED
    success_log_message_args = (
        logged_artifact_workspace,
        logged_artifact_name,
        logged_artifact_version,
    )

    error_log_message = ARTIFACT_ASSET_UPLOAD_FAILED
    error_log_message_args = (
        logged_artifact_workspace,
        logged_artifact_name,
        logged_artifact_version,
    )

    for artifact_file in IterationProgressCallback(
        artifact_assets, progress_callback, frequency
    ):
        asset_id = artifact_file.asset_id

        # If the asset id is from a downloaded artifact version asset, generate a new ID here.
        # TODO: Need to find a way to not re-upload it
        if artifact_file.asset_id in artifact._downloaded_asset_ids:
            artifact_file = artifact_file._replace(asset_id=generate_guid())

        if is_asset_remote(artifact_file):
            experiment._log_preprocessed_remote_asset(
                artifact_file,
                artifact_version_id=artifact_version_id,
                critical=True,
                on_asset_upload=partial(
                    _on_artifact_asset_upload,
                    lock,
                    all_asset_ids,
                    asset_id,
                    success_prepared_request,
                    timeout,
                    success_log_message,
                    success_log_message_args,
                    verify_tls,
                ),
                on_failed_asset_upload=partial(
                    _on_artifact_failed_asset_upload,
                    asset_id,
                    failed_prepared_request,
                    timeout,
                    error_log_message,
                    (asset_id,) + error_log_message_args,
                    verify_tls,
                ),
                return_url=False,
            )
            remote_assets_num -= 1
            remote_assets_size -= artifact_file.size

        else:
            experiment._log_preprocessed_asset(
                artifact_file,
                artifact_version_id=artifact_version_id,
                critical=True,
                on_asset_upload=partial(
                    _on_artifact_asset_upload,
                    lock,
                    all_asset_ids,
                    asset_id,
                    success_prepared_request,
                    timeout,
                    success_log_message,
                    success_log_message_args,
                    verify_tls,
                ),
                on_failed_asset_upload=partial(
                    _on_artifact_failed_asset_upload,
                    asset_id,
                    failed_prepared_request,
                    timeout,
                    error_log_message,
                    (asset_id,) + error_log_message_args,
                    verify_tls,
                ),
                return_url=False,
            )
            local_assets_num -= 1
            local_assets_size -= artifact_file.size

        total_num_assets -= 1
        total_size -= artifact_file.size


def _upsert_artifact(
    artifact: Artifact, rest_api_client: RestApiClient, experiment_key: str
) -> Tuple[str, str]:
    try:
        artifact_version = artifact.version
        if artifact_version is not None:
            artifact_version = str(artifact_version)

        response = rest_api_client.upsert_artifact(
            artifact_name=artifact.name,
            artifact_type=artifact.artifact_type,
            experiment_key=experiment_key,
            metadata=artifact.metadata,
            version=artifact_version,
            aliases=list(artifact.aliases),
            version_tags=list(artifact.version_tags),
        )
    except CometRestApiException as e:
        raise LogArtifactException(e.safe_msg, e.sdk_error_code) from e
    except requests.RequestException as e:
        raise LogArtifactException() from e

    result = response.json()

    artifact_id = result["artifactId"]
    artifact_version_id = result["artifactVersionId"]

    version = result["currentVersion"]
    _previous_version = result["previousVersion"]

    if _previous_version is None:
        LOGGER.info(ARTIFACT_VERSION_CREATED_WITHOUT_PREVIOUS, artifact.name, version)
    else:
        LOGGER.info(
            ARTIFACT_VERSION_CREATED_WITH_PREVIOUS,
            artifact.name,
            version,
            _previous_version,
        )

    return artifact_id, artifact_version_id


def _prepare_update_artifact_version_state(
    rest_api_client: RestApiClient,
    artifact_version_id: str,
    experiment_key: str,
    state: str,
) -> PreparedRequest:
    # Extracted to ease the monkey-patching of Experiment.log_artifact
    return rest_api_client._prepare_update_artifact_version_state(
        artifact_version_id, experiment_key, state
    )


def _call_post_prepared_request(
    prepared_request: PreparedRequest, timeout: float, verify_tls: bool
) -> requests.Response:
    session = get_cached_comet_http_session(
        retry=True,
        verify_tls=verify_tls,
        api_key=prepared_request.api_key,
    )

    LOGGER.debug(
        "POST HTTP Call, url %r, json_body %r, timeout %r",
        prepared_request.url,
        prepared_request.json,
        timeout,
    )

    response = session.post(
        prepared_request.url,
        json=prepared_request.json,
        headers=prepared_request.headers,
        timeout=timeout,
    )
    status = response.status_code
    if status != 200:
        if status == 400:
            data = response.json()
            sdk_error_code = data.get("sdk_error_code", None)
            sdk_error_msg = data.get("msg", None)
            raise LogArtifactException(
                backend_err_msg=sdk_error_msg, sdk_error_code=sdk_error_code
            )

        response.raise_for_status()
    return response
