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
import json
import logging
import os
import time
from http import HTTPStatus
from typing import IO, Any, Callable, Dict, Optional, Tuple, Union

import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

from .._reporting import FILE_UPLOADED_FAILED
from ..config import get_config
from ..constants import (
    ASSET_TYPE_CURVE,
    ASSET_TYPE_EMBEDDINGS,
    ASSET_TYPE_HISTOGRAM_3D,
    ASSET_TYPE_MODEL_ELEMENT,
    ASSET_TYPE_NOTEBOOK,
    ASSET_TYPE_TEXT_SAMPLE,
    ASSET_TYPE_VIDEO,
    ASSET_TYPE_VISUALIZATION,
)
from ..exceptions import (
    ASSET_STATE_IMMUTABLE,
    CURVE_LIMIT_REACHED,
    EMBEDDINGS_LIMIT_REACHED,
    HISTOGRAM_LIMIT_REACHED,
    IMAGE_STATE_IMMUTABLE,
    IMAGE_UPLOAD_LIMIT_REACHED,
    MODEL_ELEMENT_LIMIT_REACHED,
    NOTEBOOK_LIMIT_REACHED,
    TEXT_SAMPLES_LIMIT_REACHED,
    VIDEO_LIMIT_REACHED,
    BackendCustomError,
    FileUploadThrottledException,
    ImagesUploadLimitReachedException,
    UploadLimitReachedException,
)
from ..file_upload_limits_guard import FileUploadLimitsGuard
from ..file_upload_size_monitor import UploadSizeMonitor
from ..logging_extensions.filters import ASSETS_UPLOAD_THROTTLING_MSG_EXTRA
from ..logging_messages import (
    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING,
    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
    CONNECTION_FILE_LIKE_UPLOAD_FAILED_ERROR,
    CONNECTION_FILE_LIKE_UPLOAD_FAILED_EXCEPTION,
    CONNECTION_FILE_LIKE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION,
    CONNECTION_FILE_UPLOAD_FAILED_ERROR,
    CONNECTION_FILE_UPLOAD_FAILED_EXCEPTION,
    CONNECTION_FILE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION,
    CONNECTION_ONLY_S3_DIRECT_UPLOAD_TYPES_SUPPORTED_EXCEPTION,
    CONNECTION_S3_DIRECT_FILE_LIKE_UPLOAD_FAILED_ERROR,
    CONNECTION_S3_DIRECT_FILE_UPLOAD_FAILED_ERROR,
    CONNECTION_UPLOAD_ASSET_ITEM_FAILED_ERROR,
    CONNECTION_UPLOAD_ASSET_ITEM_FAILED_WITH_STATUS_EXCEPTION,
    CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_ERROR,
    CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_WITH_STATUS_EXCEPTION,
    CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_ERROR,
    CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_WITH_STATUS_EXCEPTION,
    UPLOAD_LIMIT_REACHED,
)
from ..s3.multipart_upload.assets_helper import S3MultipartAssetsHelper
from ..s3.multipart_upload.base_helper import S3MultipartBaseHelper
from ..s3.multipart_upload.file_parts_strategy import (
    FileLikePartsStrategy,
    FilePartsStrategy,
)
from ..s3.multipart_upload.multipart_upload_options import MultipartUploadOptions
from ..s3.multipart_upload.retry_strategy import UploadRetryStrategyOp
from ..s3.multipart_upload.upload_error import S3UploadFileError
from ..s3.multipart_upload.upload_types import (
    ASSET_S3_UPLOAD_TYPE,
    DIRECT_S3_UPLOAD_TYPES,
    MODEL_ELEMENT_S3_UPLOAD_TYPE,
    VISUALIZATION_S3_UPLOAD_TYPE,
)
from ..s3.multipart_upload.uploader import S3MultipartUploader
from ..s3.multipart_upload.visualizations_helper import S3MultipartVisualizationsHelper
from ..upload_options import (
    AssetItemUploadOptions,
    FileLikeUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
    ThumbnailUploadOptions,
    UploadOptions,
)
from ..utils import encode_metadata
from .connection_helpers import (
    calculate_backoff_time,
    get_api_key_header,
    get_backend_custom_error_msg,
    get_clientlib_headers,
    get_clientlib_params,
)
from .http_session import get_cached_comet_http_session

LOGGER = logging.getLogger(__name__)


def _report_error(
    event_name: Optional[str],
    api_key: Optional[str],
    experiment_key: Optional[str],
    project_id: Optional[str],
    err_msg: Optional[str],
) -> None:
    from .connection_api import Reporting

    Reporting.report(
        event_name=event_name,
        experiment_key=experiment_key,
        project_id=project_id,
        api_key=api_key,
        err_msg=err_msg,
        config=get_config(),
    )


def _send_file(
    url: str,
    data: Any,
    params: Dict[str, str],
    headers: Dict[str, Any],
    timeout: float,
    session: requests.Session,
) -> requests.Response:
    LOGGER.debug("Uploading files %r to %s with params %s", data, url, params)

    r = session.post(
        url,
        params=params,
        data=data,
        timeout=timeout,
        headers=headers,
    )

    LOGGER.debug("Uploading file to %s done", url)

    return r


def send_thumbnail(
    post_endpoint: str,
    thumbnail_path: str,
    payload: Dict[str, str],
    timeout: float,
    session: requests.Session,
) -> requests.Response:
    with open(thumbnail_path, "rb") as _file:
        files = {
            "requestParams": (None, json.dumps(payload), "application/json"),
            "thumbnail": ("file", _file, "application/octet-stream"),
        }

        return session.put(
            post_endpoint,
            files=files,
            timeout=timeout,
        )


def send_asset_item(
    post_endpoint: str,
    file_path: str,
    payload: Dict[str, Any],
    timeout: float,
    session: requests.Session,
) -> requests.Response:
    with open(file_path, "rb") as _file:
        files = {
            "requestParams": (None, json.dumps(payload), "application/json"),
            "assetItemContent": ("file", _file, "application/octet-stream"),
        }

        return session.post(
            post_endpoint,
            files=files,
            timeout=timeout,
        )


def send_file(
    post_endpoint: str,
    file_path: str,
    params: Dict[str, str],
    headers: Dict[str, str],
    timeout: float,
    session: requests.Session,
    monitor: Optional[UploadSizeMonitor] = None,
    metadata: Optional[Dict[Any, Any]] = None,
) -> requests.Response:
    with open(file_path, "rb") as _file:
        fields: Dict[str, Any] = {"file": ("file", _file)}

        if metadata is not None:
            encoded_metadata = encode_metadata(metadata)
            if encoded_metadata:
                fields["metadata"] = encoded_metadata

        encoder = MultipartEncoder(fields=fields)

        if monitor is not None:
            monitor.total_size = encoder.len
            data = MultipartEncoderMonitor(encoder, monitor.monitor_callback)
        else:
            data = encoder

        headers.update({"Content-Type": encoder.content_type})

        return _send_file(
            post_endpoint,
            params=params,
            data=data,
            timeout=timeout,
            headers=headers,
            session=session,
        )


def send_file_like(
    post_endpoint: str,
    file_like: Any,
    params: Dict[str, str],
    headers: Dict[str, str],
    timeout: float,
    session: requests.Session,
    monitor: Optional[UploadSizeMonitor] = None,
    metadata: Optional[Dict[Any, Any]] = None,
) -> requests.Response:
    fields: Dict[str, Any] = {"file": ("file", file_like)}

    if metadata is not None:
        encoded_metadata = encode_metadata(metadata)
        if encoded_metadata:
            fields["metadata"] = encoded_metadata

    encoder = MultipartEncoder(fields=fields)

    if monitor is not None:
        monitor.total_size = encoder.len
        data = MultipartEncoderMonitor(encoder, monitor.monitor_callback)
    else:
        data = encoder

    headers.update({"Content-Type": encoder.content_type})

    return _send_file(
        post_endpoint,
        params=params,
        data=data,
        timeout=timeout,
        headers=headers,
        session=session,
    )


def send_remote_asset(
    post_endpoint: str,
    remote_uri: str,
    params: Dict[str, str],
    headers: Dict[str, str],
    timeout: float,
    session: requests.Session,
    monitor: Optional[UploadSizeMonitor] = None,
    metadata: Optional[Dict[Any, Any]] = None,
) -> requests.Response:
    fields: Dict[str, Any] = {"link": ("link", remote_uri)}

    if metadata is not None:
        encoded_metadata = encode_metadata(metadata)
        if encoded_metadata:
            fields["metadata"] = encoded_metadata

    encoder = MultipartEncoder(fields=fields)

    if monitor is not None:
        monitor.total_size = encoder.len
        data = MultipartEncoderMonitor(encoder, monitor.monitor_callback)
    else:
        data = encoder

    headers.update({"Content-Type": encoder.content_type})

    return _send_file(
        post_endpoint,
        params=params,
        data=data,
        timeout=timeout,
        headers=headers,
        session=session,
    )


def _process_upload_with_retries(
    upload_func: Callable, max_retries: int, backoff_factor: float, **kwargs: Any
) -> Optional[requests.Response]:
    retry_attempt = 0
    response: Optional[requests.Response] = None
    monitor = kwargs.get("monitor")
    connection_error = None
    while retry_attempt < max_retries:
        failed = False
        try:
            response = upload_func(**kwargs)
            connection_error = None
        except (ConnectionError, requests.ConnectionError) as ex:
            LOGGER.debug(
                "ConnectionError when uploading file. Attempt: %d of %d.",
                retry_attempt,
                max_retries,
                exc_info=True,
            )
            connection_error = ex

        if response is not None:
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                # stop retries - either hard request limit or rate limit is reached at backend
                return response
            elif response.status_code != HTTPStatus.OK:
                failed = True

        elif response is None or connection_error is not None:
            failed = True

        if not failed:
            return response

        # do backoff if appropriate
        retry_attempt += 1
        if failed and retry_attempt < max_retries:
            if monitor is not None and hasattr(monitor, "reset"):
                # reset upload monitor
                monitor.reset()
            backoff_time = calculate_backoff_time(
                backoff_factor=backoff_factor, retry_attempt=retry_attempt
            )
            LOGGER.debug(
                "Failed to upload file. Attempt: %d of %d. Retrying after: %r seconds",
                retry_attempt,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)

    if connection_error is not None:
        # raise ConnectionError thus it would be properly intercepted by automatic recovery mechanism
        raise connection_error

    return response


def _prepare_upload_params_and_headers(
    options: UploadOptions,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    params = get_clientlib_params(
        options.experiment_id, options.project_id, options.api_key
    )

    if options.additional_params is not None:
        params.update(options.additional_params)

    headers = get_clientlib_headers(options.experiment_id)
    headers.update(get_api_key_header(options.api_key))
    return params, headers


def send_direct_s3_multipart_file(
    file_path: str,
    file_size: int,
    upload_helper: S3MultipartBaseHelper,
    session: requests.Session,
    _monitor: Optional[UploadSizeMonitor] = None,
) -> Tuple[int, requests.Response]:
    LOGGER.debug(
        "Started S3 direct file upload of %r with size: %d", file_path, file_size
    )

    file_parts_strategy = FilePartsStrategy(
        file_path=file_path,
        file_size=file_size,
    )
    uploader = S3MultipartUploader(
        file_parts_strategy=file_parts_strategy, s3_helper=upload_helper
    )
    return uploader.upload(
        session=session,
        monitor=_monitor,
    )


def send_direct_s3_multipart_file_like(
    file_name: str,
    file_like: IO,
    file_size: int,
    upload_helper: S3MultipartBaseHelper,
    session: requests.Session,
    _monitor: Optional[UploadSizeMonitor] = None,
) -> Tuple[int, requests.Response]:
    LOGGER.debug(
        "Started S3 direct file-like upload of %r with size: %d", file_name, file_size
    )

    file_parts_strategy = FileLikePartsStrategy(
        file_name=file_name,
        file_like=file_like,
        file_size=file_size,
    )
    uploader = S3MultipartUploader(
        file_parts_strategy=file_parts_strategy, s3_helper=upload_helper
    )
    return uploader.upload(
        session=session,
        monitor=_monitor,
    )


def _create_strategy_and_helper_for_s3_direct_upload(
    options: Union[FileUploadOptions, FileLikeUploadOptions],
    upload_expires_in: int,
) -> Tuple[UploadRetryStrategyOp, S3MultipartBaseHelper]:
    params, headers = _prepare_upload_params_and_headers(options)
    retry_strategy = UploadRetryStrategyOp(
        max_upload_start_attempts=options.max_retries,
        max_upload_complete_attempts=options.max_retries,
        max_s3_file_part_upload_attempts=options.max_retries,
        retry_backoff_factor=options.retry_backoff_factor,
    )
    if (
        options.upload_type == ASSET_S3_UPLOAD_TYPE
        or options.upload_type == MODEL_ELEMENT_S3_UPLOAD_TYPE
    ):
        upload_helper = S3MultipartAssetsHelper(
            base_url=options.base_url,
            parameters=params,
            headers=headers,
            asset_metadata=options.metadata,
            expires_in=upload_expires_in,
            upload_retry_strategy=retry_strategy,
        )
    elif options.upload_type == VISUALIZATION_S3_UPLOAD_TYPE:
        upload_helper = S3MultipartVisualizationsHelper(
            base_url=options.base_url,
            parameters=params,
            headers=headers,
            asset_metadata=options.metadata,
            expires_in=upload_expires_in,
            upload_retry_strategy=retry_strategy,
        )
    else:
        raise ValueError(
            CONNECTION_ONLY_S3_DIRECT_UPLOAD_TYPES_SUPPORTED_EXCEPTION
            % DIRECT_S3_UPLOAD_TYPES
        )
    return retry_strategy, upload_helper


def upload_s3_multipart_file(
    options: FileUploadOptions,
    multipart_options: MultipartUploadOptions,
    _monitor: Optional[UploadSizeMonitor] = None,
) -> requests.Response:
    retry_strategy, upload_helper = _create_strategy_and_helper_for_s3_direct_upload(
        options=options, upload_expires_in=multipart_options.upload_expires_in
    )
    try:
        # we need session without retries because it would be done by uploader
        session = get_cached_comet_http_session(
            retry=False,
            verify_tls=options.verify_tls,
            api_key=options.api_key,
        )
        bytes_sent, response = send_direct_s3_multipart_file(
            file_path=options.file_path,
            file_size=options.estimated_size,
            upload_helper=upload_helper,
            session=session,
            _monitor=_monitor,
        )

        if options.clean is True:
            # Cleanup file in case of success
            try:
                os.remove(options.file_path)
                LOGGER.debug(
                    "Removed file after direct upload to S3: %r", options.file_path
                )
            except OSError:
                LOGGER.debug(
                    "Failed to remove file after direct upload to S3: %r",
                    options.file_path,
                    exc_info=True,
                )
                pass

        LOGGER.debug(
            "File successfully uploaded to S3: %r, bytes sent: %d, upload type: %r",
            options.file_path,
            bytes_sent,
            options.upload_type,
        )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(None)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING, exc_info=True
                )

        return response
    except Exception as e:
        if (
            isinstance(e, S3UploadFileError)
            and e.due_connection_error
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_S3_DIRECT_FILE_UPLOAD_FAILED_ERROR,
                e,
                options.file_path,
                options.upload_type,
                exc_info=True,
            )
        else:
            LOGGER.error(
                CONNECTION_S3_DIRECT_FILE_UPLOAD_FAILED_ERROR,
                e,
                options.file_path,
                options.upload_type,
                exc_info=True,
            )

        _report_error(
            event_name=FILE_UPLOADED_FAILED,
            experiment_key=options.experiment_id,
            project_id=options.project_id,
            api_key=options.api_key,
            err_msg=str(e),
        )

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )
        raise


def upload_s3_multipart_file_like(
    options: FileLikeUploadOptions,
    multipart_options: MultipartUploadOptions,
    _monitor: Optional[UploadSizeMonitor] = None,
) -> requests.Response:
    retry_strategy, upload_helper = _create_strategy_and_helper_for_s3_direct_upload(
        options=options, upload_expires_in=multipart_options.upload_expires_in
    )
    file_name = options.additional_params.get("fileName", "file-like")
    try:
        # we need session without retries because it would be done by uploader
        session = get_cached_comet_http_session(
            retry=False,
            verify_tls=options.verify_tls,
            api_key=options.api_key,
        )
        bytes_sent, response = send_direct_s3_multipart_file_like(
            file_name=file_name,
            file_like=options.file_like,
            file_size=options.estimated_size,
            upload_helper=upload_helper,
            session=session,
            _monitor=_monitor,
        )

        LOGGER.debug(
            "File-like successfully uploaded to S3: %r, bytes sent: %d, upload type: %r",
            file_name,
            bytes_sent,
            options.upload_type,
        )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(response)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING, exc_info=True
                )

        return response
    except Exception as e:
        if (
            isinstance(e, S3UploadFileError)
            and e.due_connection_error
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_S3_DIRECT_FILE_LIKE_UPLOAD_FAILED_ERROR,
                e,
                file_name,
                options.upload_type,
                exc_info=True,
            )
        else:
            LOGGER.error(
                CONNECTION_S3_DIRECT_FILE_LIKE_UPLOAD_FAILED_ERROR,
                e,
                file_name,
                options.upload_type,
                exc_info=True,
            )

        _report_error(
            event_name=FILE_UPLOADED_FAILED,
            experiment_key=options.experiment_id,
            project_id=options.project_id,
            api_key=options.api_key,
            err_msg=str(e),
        )

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )
        raise


def raise_on_upload_from_backend_error_response(
    response: requests.Response, upload_limits_guard: FileUploadLimitsGuard
) -> None:
    custom_error_msg = get_backend_custom_error_msg(response)
    if custom_error_msg is not None:
        raise BackendCustomError(custom_error_msg)

    # check if file upload limits reached
    try:
        data = response.json()
        code = data.get("sdk_error_code", None)
        if code is None:
            return
    except Exception:
        LOGGER.debug(
            msg="Failed to get sdk_error_code from server response after file upload operation",
            exc_info=True,
        )
        return

    if code == IMAGE_UPLOAD_LIMIT_REACHED:
        upload_limits_guard.image_upload_limit_exceeded()
        # We need specific exception here because visualization type can be for both: images and figures
        raise ImagesUploadLimitReachedException()
    elif code == EMBEDDINGS_LIMIT_REACHED:
        upload_limits_guard.embedding_upload_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_EMBEDDINGS)
    elif code == HISTOGRAM_LIMIT_REACHED:
        upload_limits_guard.histogram_upload_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_HISTOGRAM_3D)
    elif code == TEXT_SAMPLES_LIMIT_REACHED:
        upload_limits_guard.text_samples_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_TEXT_SAMPLE)
    elif code == MODEL_ELEMENT_LIMIT_REACHED:
        upload_limits_guard.model_element_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_MODEL_ELEMENT)
    elif code == CURVE_LIMIT_REACHED:
        upload_limits_guard.curve_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_CURVE)
    elif code == NOTEBOOK_LIMIT_REACHED:
        upload_limits_guard.notebook_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_NOTEBOOK)
    elif code == VIDEO_LIMIT_REACHED:
        upload_limits_guard.video_limit_exceeded()
        raise UploadLimitReachedException(asset_type=ASSET_TYPE_VIDEO)


def check_upload_limit_not_reached(
    options: Union[FileUploadOptions, FileLikeUploadOptions],
    upload_limits_guard: FileUploadLimitsGuard,
):
    if upload_limits_guard.has_upload_limit_exceeded(upload_type=options.upload_type):
        LOGGER.debug(
            "Already exceed limit of number of %r assets to be uploaded",
            options.upload_type,
        )
        if options.upload_type == ASSET_TYPE_VISUALIZATION:
            # We need specific exception here because visualization type can be for both: images and figures
            raise ImagesUploadLimitReachedException()
        else:
            raise UploadLimitReachedException(asset_type=options.upload_type)


def check_asset_was_already_uploaded(response: requests.Response) -> bool:
    try:
        data = response.json()
        code = data.get("sdk_error_code")

        if code is None:
            return False

        if code in [
            ASSET_STATE_IMMUTABLE,
            IMAGE_STATE_IMMUTABLE,
        ]:
            return True
    except Exception:
        LOGGER.debug("Failed to check if asset was already uploaded")

    return False


def upload_file(
    options: FileUploadOptions,
    upload_limits_guard: FileUploadLimitsGuard,
    _monitor: Optional[UploadSizeMonitor] = None,
) -> None:
    check_upload_limit_not_reached(options, upload_limits_guard)

    params, headers = _prepare_upload_params_and_headers(options)
    try:
        response = _process_upload_with_retries(
            upload_func=send_file,
            max_retries=options.max_retries,
            backoff_factor=options.retry_backoff_factor,
            post_endpoint=options.upload_endpoint,
            file_path=options.file_path,
            params=params,
            headers=headers,
            metadata=options.metadata,
            timeout=options.timeout,
            session=get_cached_comet_http_session(
                retry=False,
                verify_tls=options.verify_tls,
                api_key=options.api_key,
            ),
            monitor=_monitor,
        )

        if response is None:
            raise ValueError(
                CONNECTION_FILE_UPLOAD_FAILED_EXCEPTION
                % (options.max_retries, options.upload_endpoint)
            )

        if response.status_code != HTTPStatus.OK:
            # raise for supported backend errors - this should go first to handle hard upload limits
            raise_on_upload_from_backend_error_response(response, upload_limits_guard)

            # Handle soft upload limits - HTTP 429 (Too Many Requests) specifically for retry integration
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise FileUploadThrottledException(response, options)

            if check_asset_was_already_uploaded(response):
                LOGGER.debug(
                    "File was rejected by backend since it was already uploaded before, exception NOT raised: %r",
                    response.content,
                )
            else:
                raise ValueError(
                    CONNECTION_FILE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION
                    % (
                        response.status_code,
                        options.max_retries,
                        options.upload_endpoint,
                        response.content,
                    )
                )

        if options.clean is True:
            # Cleanup file in case of success
            try:
                os.remove(options.file_path)
                LOGGER.debug("Removed file after sending: %r", options.file_path)
            except OSError:
                LOGGER.debug(
                    "Failed to remove file after sending: %r",
                    options.file_path,
                    exc_info=True,
                )

        LOGGER.debug(
            "File successfully uploaded to: %s",
            options.upload_endpoint,
        )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(response)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING, exc_info=True
                )
    except Exception as e:
        if isinstance(e, UploadLimitReachedException):
            LOGGER.error(
                msg=UPLOAD_LIMIT_REACHED.format(asset_type=e.asset_type),
                extra=ASSETS_UPLOAD_THROTTLING_MSG_EXTRA,
            )

        elif (
            isinstance(e, (ConnectionError, requests.ConnectionError))
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_FILE_UPLOAD_FAILED_ERROR,
                e,
                options.file_path,
                options.upload_endpoint,
                options.max_retries,
                exc_info=True,
            )
        elif not isinstance(e, FileUploadThrottledException):
            LOGGER.error(
                CONNECTION_FILE_UPLOAD_FAILED_ERROR,
                e,
                options.file_path,
                options.upload_endpoint,
                options.max_retries,
                exc_info=True,
            )

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )

        if not isinstance(e, FileUploadThrottledException):
            _report_error(
                event_name=FILE_UPLOADED_FAILED,
                experiment_key=options.experiment_id,
                project_id=options.project_id,
                api_key=options.api_key,
                err_msg=str(e),
            )
            raise


def upload_file_like(
    options: FileLikeUploadOptions,
    upload_limits_guard: FileUploadLimitsGuard,
    _monitor=None,
) -> None:
    check_upload_limit_not_reached(options, upload_limits_guard)

    params, headers = _prepare_upload_params_and_headers(options)
    try:
        response = _process_upload_with_retries(
            upload_func=send_file_like,
            max_retries=options.max_retries,
            backoff_factor=options.retry_backoff_factor,
            post_endpoint=options.upload_endpoint,
            file_like=options.file_like,
            params=params,
            headers=headers,
            metadata=options.metadata,
            timeout=options.timeout,
            session=get_cached_comet_http_session(
                retry=False,
                verify_tls=options.verify_tls,
                api_key=options.api_key,
            ),
            monitor=_monitor,
        )

        if response is None:
            raise ValueError(
                CONNECTION_FILE_LIKE_UPLOAD_FAILED_EXCEPTION
                % (options.max_retries, options.upload_endpoint)
            )

        if response.status_code != HTTPStatus.OK:
            # raise for supported backend errors - this should go first to handle hard upload limits
            raise_on_upload_from_backend_error_response(response, upload_limits_guard)

            # Handle soft upload limits - HTTP 429 (Too Many Requests) specifically for retry integration
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise FileUploadThrottledException(response, options)

            if check_asset_was_already_uploaded(response):
                LOGGER.debug(
                    "File-like was rejected by backend since it was already uploaded before: %r",
                    response.content,
                )
            else:
                raise ValueError(
                    CONNECTION_FILE_LIKE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION
                    % (
                        response.status_code,
                        options.max_retries,
                        options.upload_endpoint,
                        response.content,
                    )
                )

        LOGGER.debug(
            "File-like successfully uploaded to: %s",
            options.upload_endpoint,
        )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(response)
            except Exception:
                LOGGER.warning("Failed to call on_asset_upload", exc_info=True)
    except Exception as e:
        if isinstance(e, UploadLimitReachedException):
            LOGGER.error(
                msg=UPLOAD_LIMIT_REACHED.format(asset_type=e.asset_type),
                extra=ASSETS_UPLOAD_THROTTLING_MSG_EXTRA,
            )
        elif (
            isinstance(e, (ConnectionError, requests.ConnectionError))
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_FILE_LIKE_UPLOAD_FAILED_ERROR,
                e,
                options.upload_endpoint,
                options.max_retries,
                exc_info=True,
            )
        elif not isinstance(e, FileUploadThrottledException):
            LOGGER.error(
                CONNECTION_FILE_LIKE_UPLOAD_FAILED_ERROR,
                e,
                options.upload_endpoint,
                options.max_retries,
                exc_info=True,
            )

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )

        if not isinstance(e, FileUploadThrottledException):
            _report_error(
                event_name=FILE_UPLOADED_FAILED,
                experiment_key=options.experiment_id,
                project_id=options.project_id,
                api_key=options.api_key,
                err_msg=str(e),
            )
            raise


def upload_remote_asset(
    options: RemoteAssetsUploadOptions,
    _monitor=None,
) -> None:
    params, headers = _prepare_upload_params_and_headers(options)

    try:
        response = send_remote_asset(
            post_endpoint=options.upload_endpoint,
            remote_uri=options.remote_uri,
            params=params,
            headers=headers,
            metadata=options.metadata,
            timeout=options.timeout,
            session=get_cached_comet_http_session(
                True,
                verify_tls=options.verify_tls,
                api_key=options.api_key,
            ),
            monitor=_monitor,
        )

        if response.status_code != HTTPStatus.OK:
            # Handle HTTP 429 (Too Many Requests) specifically for retry integration
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise FileUploadThrottledException(response, options)

            raise ValueError(
                CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_WITH_STATUS_EXCEPTION
                % (response.status_code, options.upload_endpoint, response.content)
            )

        LOGGER.debug(
            "Remote Asset successfully uploaded to (%s): %s",
            response.status_code,
            options.upload_endpoint,
        )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(response)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING, exc_info=True
                )
    except Exception as e:
        if (
            isinstance(e, (ConnectionError, requests.ConnectionError))
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_ERROR, e, exc_info=True)
        elif not isinstance(e, FileUploadThrottledException):
            LOGGER.error(CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_ERROR, e, exc_info=True)

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )

        if not isinstance(e, FileUploadThrottledException):
            _report_error(
                event_name=FILE_UPLOADED_FAILED,
                experiment_key=options.experiment_id,
                project_id=options.project_id,
                api_key=options.api_key,
                err_msg=str(e),
            )
            raise


def upload_asset_item(
    options: AssetItemUploadOptions,
    _monitor=None,
) -> None:
    payload = {
        "apiKey": options.api_key,
        "assetId": options.asset_id,
        "experimentKey": options.experiment_id,
        "userFileName": options.asset_item.file_name,
        "assetItemType": options.asset_item.type,
    }

    try:
        response = _process_upload_with_retries(
            upload_func=send_asset_item,
            max_retries=options.max_retries,
            backoff_factor=options.retry_backoff_factor,
            post_endpoint=options.upload_endpoint,
            file_path=options.asset_item.file_path,
            payload=payload,
            timeout=options.timeout,
            session=get_cached_comet_http_session(
                retry=False,
                verify_tls=options.verify_tls,
                api_key=options.api_key,
            ),
        )

        if response.status_code != HTTPStatus.OK:
            # Handle soft upload limits - HTTP 429 (Too Many Requests) specifically for retry integration
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise FileUploadThrottledException(response, options)

            raise ValueError(
                CONNECTION_UPLOAD_ASSET_ITEM_FAILED_WITH_STATUS_EXCEPTION
                % (response.status_code, options.upload_endpoint, response.content)
            )

        if options.on_asset_upload is not None:
            try:
                options.on_asset_upload(response)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING, exc_info=True
                )

    except Exception as e:
        if (
            isinstance(e, (ConnectionError, requests.ConnectionError))
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_UPLOAD_ASSET_ITEM_FAILED_ERROR,
                options.asset_item.file_name,
                options.upload_endpoint,
                e,
                exc_info=True,
            )
        elif not isinstance(e, FileUploadThrottledException):
            LOGGER.error(
                CONNECTION_UPLOAD_ASSET_ITEM_FAILED_ERROR,
                options.asset_item.file_name,
                options.upload_endpoint,
                e,
                exc_info=True,
            )

        if options.on_failed_asset_upload is not None:
            try:
                options.on_failed_asset_upload(e)
            except Exception:
                LOGGER.warning(
                    CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING,
                    exc_info=True,
                )

        if not isinstance(e, FileUploadThrottledException):
            _report_error(
                event_name=FILE_UPLOADED_FAILED,
                experiment_key=options.experiment_id,
                project_id=options.project_id,
                api_key=options.api_key,
                err_msg=str(e),
            )

            raise


def upload_thumbnail(
    options: ThumbnailUploadOptions,
    _monitor=None,
) -> None:

    payload = {
        "apiKey": options.api_key,
        "experimentKey": options.experiment_id,
        "thumbnailExtension": ".png",
    }

    try:
        response = _process_upload_with_retries(
            upload_func=send_thumbnail,
            max_retries=options.max_retries,
            backoff_factor=options.retry_backoff_factor,
            post_endpoint=options.upload_endpoint,
            thumbnail_path=options.thumbnail_path,
            payload=payload,
            timeout=options.timeout,
            session=get_cached_comet_http_session(
                retry=False,
                verify_tls=options.verify_tls,
                api_key=options.api_key,
            ),
        )

        if response.status_code != HTTPStatus.OK:
            raise ValueError(
                CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_WITH_STATUS_EXCEPTION
                % (response.status_code, options.upload_endpoint, response.content)
            )

        LOGGER.debug(
            "Thumbnail successfully uploaded to (%s): %s",
            response.status_code,
            options.upload_endpoint,
        )

    except Exception as e:
        if (
            isinstance(e, (ConnectionError, requests.ConnectionError))
            and options.log_connection_error_as_debug
        ):
            # to avoid spamming user with errors
            LOGGER.debug(
                CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_ERROR,
                options.thumbnail_path,
                options.upload_endpoint,
                e,
                exc_info=True,
            )
        else:
            LOGGER.error(
                CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_ERROR,
                options.thumbnail_path,
                options.upload_endpoint,
                e,
                exc_info=True,
            )

        _report_error(
            event_name=FILE_UPLOADED_FAILED,
            experiment_key=options.experiment_id,
            project_id=options.project_id,
            api_key=options.api_key,
            err_msg=str(e),
        )

        raise
