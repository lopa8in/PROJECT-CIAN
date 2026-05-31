# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import functools
import logging
import os
import time
from http import HTTPStatus
from typing import Callable, Dict, List, NamedTuple, Optional, Type

import requests
from requests import RequestException

from ...batch_utils import MessageBatch, MessageBatchItem, ParametersBatch
from ...connection import RestApiClient, RestServerConnection
from ...connection.connection_url_helpers import upload_thumbnail_url
from ...exceptions import CometRestApiException, FileUploadThrottledException
from ...file_upload_manager import FileUploadManager
from ...logging_messages import (
    ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_REST_ERROR,
    ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_UNKNOWN_ERROR,
    CLOUD_DETAILS_MSG_SENDING_ERROR,
    FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH,
    FAILED_TO_FLUSH_METRICS_BATCH,
    FAILED_TO_FLUSH_PARAMETERS_BATCH,
    FAILED_TO_FLUSH_STDOUT_BATCH,
    FAILED_TO_REGISTER_MODEL,
    FAILED_TO_SEND_CLOUD_DETAILS_MESSAGE_ERROR,
    FAILED_TO_SEND_FILE_NAME_MESSAGE_ERROR,
    FAILED_TO_SEND_GIT_METADATA_MESSAGE_ERROR,
    FAILED_TO_SEND_GPU_STATIC_MESSAGE_ERROR,
    FAILED_TO_SEND_HTML_MESSAGE_ERROR,
    FAILED_TO_SEND_HTML_OVERRIDE_MESSAGE_ERROR,
    FAILED_TO_SEND_INSTALLED_PACKAGES_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_DEPENDENCY_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_OTHER_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
    FAILED_TO_SEND_MODEL_GRAPH_MESSAGE_ERROR,
    FAILED_TO_SEND_OS_PACKAGES_MESSAGE_ERROR,
    FAILED_TO_SEND_SYSTEM_DETAILS_MESSAGE_ERROR,
    FAILED_TO_SEND_SYSTEM_INFO_MESSAGE_ERROR,
    FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH,
    FILENAME_DETAILS_MSG_SENDING_ERROR,
    GIT_METADATA_MSG_SENDING_ERROR,
    GPU_STATIC_INFO_MSG_SENDING_ERROR,
    HTML_MSG_SENDING_ERROR,
    HTML_OVERRIDE_MSG_SENDING_ERROR,
    INSTALLED_PACKAGES_MSG_SENDING_ERROR,
    LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
    LOG_OTHER_MSG_SENDING_ERROR,
    METRICS_BATCH_MSG_SENDING_ERROR,
    MODEL_GRAPH_MSG_SENDING_ERROR,
    OS_PACKAGE_MSG_SENDING_ERROR,
    PARAMETERS_BATCH_MSG_SENDING_ERROR,
    REGISTER_FAILED_DUE_TO_UPLOADS_FAILED,
    REMOTE_MODEL_MESSAGE_SENDING_ERROR,
    STANDARD_OUTPUT_SENDING_ERROR,
    STREAMER_FAILED_TO_REMOVE_TMP_FILE_COPY_WARNING,
    SYSTEM_DETAILS_MSG_SENDING_ERROR,
    SYSTEM_INFO_MESSAGE_SENDING_ERROR,
)
from ...messages import (
    BaseMessage,
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
    UploadInMemoryMessage,
)
from ...messages_utils import create_upload_message_from_options
from ...s3.multipart_upload.upload_error import S3UploadError
from ...upload_options import (
    AssetItemUploadOptions,
    FileLikeUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
    ThumbnailUploadOptions,
)
from ...utils import seconds_to_datetime_str
from ..retry.retry_helpers import calculate_reset_at_from_response
from ..retry.retry_incidents_manager import RetryIncidentsManager
from .base_handler import BaseMessageHandler, MessageHandlerType
from .handler_context import ErrorReport, HandlerContext

LOGGER = logging.getLogger(__name__)


class RestApiSendResult(NamedTuple):
    success: bool
    connection_error: bool
    throttled: bool


class OnlineMessageHandler(BaseMessageHandler):
    """
    Handles online messages and orchestrates their processing.

    This class is used to handle, batch, and process online messages systematically. It manages
    the sending of batched messages to a server while ensuring proper error handling and cleanup of
    temporary files. It supports the processing of various message types, including metrics, standard
    output, and remote asset uploads. The class works with dependencies such as connection managers,
    rest clients, and file upload managers to perform its tasks.

    Attributes:
        experiment_key: str
            Identifier of the experiment the handler is associated with.
        api_key: str
            API key used for authenticating requests.
        run_id: str
            Unique ID of the run being processed.
        project_id: str
            Unique ID of the project associated with.
        on_message_sent_callback: Optional[Callable]
            A callback function to be executed after a message is sent.
        on_messages_batch_sent_callback: Optional[Callable]
            A callback function to be executed after a batch of messages is sent.
        use_raw_throttling_messages: bool
            Specifies whether to use raw throttling messages.
    """

    def __init__(
        self,
        connection: RestServerConnection,
        experiment_key: str,
        api_key: str,
        run_id: str,
        project_id: str,
        rest_api_client: RestApiClient,
        verify_tls: bool,
        file_upload_read_timeout: float,
        parameters_batch_base_interval: float,
        message_batch_compress: bool,
        message_batch_metric_interval: float,
        message_batch_metric_max_size: int,
        message_batch_stdout_interval: float,
        message_batch_stdout_max_size: int,
        artifact_remote_assets_batch_enabled: bool,
        artifact_remote_assets_batch_metric_interval: float,
        artifact_remote_assets_batch_metric_max_size: int,
        use_raw_throttling_messages: bool,
        file_upload_manager: FileUploadManager,
        retry_incidents_manager: RetryIncidentsManager,
    ):
        super().__init__()

        self._connection = connection
        self._rest_api_client = rest_api_client

        self._file_upload_read_timeout = file_upload_read_timeout

        self._file_upload_manager = file_upload_manager
        self._file_uploads_to_clean = list()

        self._retry_incidents_manager = retry_incidents_manager

        self.experiment_key = experiment_key
        self.api_key = api_key
        self.run_id = run_id
        self.project_id = project_id

        self.on_message_sent_callback = None
        self.on_messages_batch_sent_callback = None

        self._verify_tls = verify_tls

        self._parameters_batch = ParametersBatch(parameters_batch_base_interval)

        self._message_batch_compress = message_batch_compress
        self._message_batch_metrics = MessageBatch(
            base_interval=message_batch_metric_interval,
            max_size=message_batch_metric_max_size,
        )

        self._message_batch_stdout = MessageBatch(
            base_interval=message_batch_stdout_interval,
            max_size=message_batch_stdout_max_size,
        )

        self._artifact_remote_assets_batch_enabled = (
            artifact_remote_assets_batch_enabled
        )
        self._artifact_remote_assets_batch = MessageBatch(
            base_interval=artifact_remote_assets_batch_metric_interval,
            max_size=artifact_remote_assets_batch_metric_max_size,
        )
        self._has_artifact_remote_assets_batch_send_failed = False

        self.use_raw_throttling_messages = use_raw_throttling_messages

    def has_batches_unsent(self) -> bool:
        return (
            not self._parameters_batch.empty()
            or not self._message_batch_metrics.empty()
            or not self._message_batch_stdout.empty()
            or (
                self._artifact_remote_assets_batch_enabled
                and not self._artifact_remote_assets_batch.empty()
            )
        )

    def send_batches(
        self,
        context: HandlerContext,
        flush: bool = False,
    ) -> bool:
        if not self.has_batches_unsent():
            return True

        batch_start_time = time.time()
        LOGGER.debug("OnlineHandler: start send batches, flush: %s", flush)

        success = True
        if not self._parameters_batch.empty():
            if not self._parameters_batch.accept(
                functools.partial(self._send_parameter_messages_batch, context=context),
                unconditional=flush,
            ):
                if flush:
                    # the batch was not sent while a full batch flush is requested
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_PARAMETERS_BATCH)
                    context.report_error_callback(
                        ErrorReport(FAILED_TO_FLUSH_PARAMETERS_BATCH)
                    )
            else:
                LOGGER.debug("Parameters batch has been sent")

        if not self._message_batch_metrics.empty():
            if not self._message_batch_metrics.accept(
                functools.partial(self._send_metric_messages_batch, context=context),
                unconditional=flush,
            ):
                if flush:
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_METRICS_BATCH)
                    context.report_error_callback(
                        ErrorReport(FAILED_TO_FLUSH_METRICS_BATCH)
                    )
            else:
                LOGGER.debug("Metrics batch has been sent")

        if not self._message_batch_stdout.empty():
            if not self._message_batch_stdout.accept(
                functools.partial(self._send_stdout_messages_batch, context=context),
                unconditional=flush,
            ):
                if flush:
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_STDOUT_BATCH)
                    context.report_error_callback(
                        ErrorReport(FAILED_TO_FLUSH_STDOUT_BATCH)
                    )
            else:
                LOGGER.debug("stdout/stderr batch has been sent")

        if self._artifact_remote_assets_batch_enabled:
            if not self._artifact_remote_assets_batch.empty():
                if not self._artifact_remote_assets_batch.accept(
                    functools.partial(
                        self._send_artifact_remote_assets_batch, context=context
                    ),
                    unconditional=flush,
                ):
                    if flush:
                        success = False
                        LOGGER.error(FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH)
                        context.report_error_callback(
                            ErrorReport(FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH)
                        )
                else:
                    LOGGER.debug("Artifact remote assets batch has been sent")

        elapsed = time.time() - batch_start_time
        LOGGER.debug(
            "OnlineHandler: send batches completed, successful: %s, elapsed: %r, flush: %s",
            success,
            elapsed,
            flush,
        )

        return success

    def clean_file_uploads(self):
        for file_path in self._file_uploads_to_clean:
            LOGGER.debug("Removing temporary copy of the uploaded file: %r", file_path)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                LOGGER.warning(
                    STREAMER_FAILED_TO_REMOVE_TMP_FILE_COPY_WARNING,
                    file_path,
                    exc_info=True,
                )

    def has_upload_failed(self) -> bool:
        return (
            self._file_upload_manager.has_failed()
            or self._has_artifact_remote_assets_batch_send_failed
        )

    def has_artifact_remote_assets_batch_send_failed(self) -> bool:
        return self._has_artifact_remote_assets_batch_send_failed

    def _on_upload_failed_callback(
        self, message_id: int, message_callback: Optional[Callable] = None
    ):
        def _callback(response):
            if (
                isinstance(response, ConnectionError)
                or isinstance(response, requests.ConnectionError)
                or isinstance(response, S3UploadError)
            ):
                self._on_messages_sent_completed(
                    [message_id],
                    success=False,
                    connection_error=True,
                    failure_reason=str(response),
                )
            elif isinstance(response, FileUploadThrottledException):
                # Handle HTTP 429 (Too Many Requests) for file uploads
                reset_at = calculate_reset_at_from_response(response.response, LOGGER)

                # Create a retry upload message using the utility function
                retry_message = create_upload_message_from_options(
                    upload_options=response.upload_options, message_id=message_id
                )
                if retry_message is None:
                    LOGGER.error(
                        "Failed to recreate upload message from options: %s during throttling retry handling",
                        response.upload_options,
                    )
                    return

                self._retry_incidents_manager.add_or_update_incident(
                    retry_message.type, reset_at=reset_at, messages=[retry_message]
                )

                LOGGER.info(
                    "File upload ('%s') throttled by backend for type '%s', retry after: %s (%d seconds)",
                    retry_message.get_user_friendly_identifier(),
                    retry_message.type,
                    seconds_to_datetime_str(reset_at),
                    int(reset_at - time.time()),
                )

                # Don't call _on_messages_sent_completed since this will be retried
                return

        if message_callback is None:
            callback = lambda response: _callback(response)  # noqa: E731
        else:
            callback = lambda response: (  # noqa: E731
                _callback(response),
                message_callback(response),
            )

        return callback

    def _on_upload_success_callback(
        self, message_id: int, message_callback: Optional[Callable] = None
    ):
        if message_callback is None:
            callback = lambda response: self._on_messages_sent_completed(  # noqa: E731
                [message_id]
            )
        else:
            callback = lambda response: (  # noqa: E731
                self._on_messages_sent_completed([message_id]),
                message_callback(response),
            )

        return callback

    def _process_upload_message(
        self, message: UploadFileMessage, _: HandlerContext
    ) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_file_thread(
            options=FileUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                clean=False,  # do not clean immediately after upload - this would be handled later
                experiment_id=self.experiment_key,
                file_path=message.file_path,
                metadata=message.metadata,
                project_id=self.project_id,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
                upload_type=message.upload_type,
                base_url=self._connection.server_address,
                critical=message._critical,
            ),
        )
        if message.clean is True:
            self._file_uploads_to_clean.append(message.file_path)

    def _process_upload_in_memory_message(
        self, message: UploadInMemoryMessage, _: HandlerContext
    ) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_file_like_thread(
            options=FileLikeUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                experiment_id=self.experiment_key,
                file_like=message.file_like,
                metadata=message.metadata,
                project_id=self.project_id,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
                upload_type=message.upload_type,
                base_url=self._connection.server_address,
                critical=message._critical,
            ),
        )
        LOGGER.debug("Processing in-memory uploading message done")

    def _process_upload_remote_asset_message(
        self, message: RemoteAssetMessage, context: HandlerContext
    ) -> None:
        if self._artifact_remote_assets_batch_enabled and message.is_artifact_asset():
            # add a message to the batch
            self._artifact_remote_assets_batch.append(message)
        else:
            self._upload_remote_asset_message(message, context)

    def _upload_remote_asset_message(
        self, message: RemoteAssetMessage, _: HandlerContext
    ) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_remote_asset_thread(
            options=RemoteAssetsUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                experiment_id=self.experiment_key,
                metadata=message.metadata,
                project_id=self.project_id,
                remote_uri=message.remote_uri,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                upload_type=message.upload_type,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
                critical=message._critical,
            ),
        )
        LOGGER.debug(
            "Remote asset has been scheduled for upload, asset URI: %r",
            message.remote_uri,
        )

    def _send_artifact_remote_assets_batch(
        self, message_items: List[MessageBatchItem], context: HandlerContext
    ) -> None:
        if len(message_items) == 0:
            LOGGER.debug("Empty artifact remote assets batch, send request ignored")
            return

        # check that we have only artifact remote assets in the batch
        is_artifact_assets = all(
            [
                isinstance(m.message, RemoteAssetMessage)
                and m.message.is_artifact_asset()
                for m in message_items
            ]
        )
        if not is_artifact_assets:
            LOGGER.debug("Only artifact remote assets should be in the batch")
            raise ValueError("Only artifact remote assets should be in the batch")

        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_artifact_remote_assets_batch started")

        messages = [m.message for m in message_items]
        send_result = self._process_rest_api_send(
            sender=self._rest_api_client.send_artifact_remote_assets_batch,
            message_type=RemoteAssetMessage.type,
            context=context,
            rest_fail_prompt=ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_REST_ERROR,
            general_fail_prompt=ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_UNKNOWN_ERROR,
            messages=messages,
            batch_items=message_items,
            compress=self._message_batch_compress,
            experiment_key=self.experiment_key,
        )

        # notify callbacks
        if not send_result.throttled:
            for m in message_items:
                callbacks = m.message.get_message_callbacks()
                if send_result.success and callbacks.on_asset_upload is not None:
                    callbacks.on_asset_upload()
                elif (
                    not send_result.connection_error
                    and callbacks.on_failed_asset_upload is not None
                ):
                    # signal failure if not a connection error
                    callbacks.on_failed_asset_upload()
                    self._has_artifact_remote_assets_batch_send_failed = True

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_artifact_remote_assets_batch completed, elapsed: %r, throttled: %r",
            elapsed,
            send_result.throttled,
        )

    def _send_stdout_messages_batch(
        self, message_items: List[MessageBatchItem], context: HandlerContext
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_stdout_messages_batch started")

        messages = [m.message for m in message_items]
        result = self._process_rest_api_send(
            sender=self._rest_api_client.send_stdout_batch,
            message_type=StandardOutputMessage.type,
            context=context,
            rest_fail_prompt=STANDARD_OUTPUT_SENDING_ERROR,
            general_fail_prompt="Error sending stdout/stderr batch (online experiment)",
            messages=messages,
            batch_items=message_items,
            compress=self._message_batch_compress,
            experiment_key=self.experiment_key,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_stdout_messages_batch completed, elapsed: %r, throttled: %r",
            elapsed,
            result.throttled,
        )

    def _send_metric_messages_batch(
        self, message_items: List[MessageBatchItem], context: HandlerContext
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_metric_messages_batch started")

        messages = [m.message for m in message_items]
        result = self._process_rest_api_send(
            sender=self._connection.log_metrics_batch,
            message_type=MetricMessage.type,
            context=context,
            rest_fail_prompt=METRICS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending metrics batch (online experiment)",
            messages=messages,
            items=message_items,
            compress=self._message_batch_compress,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_metric_messages_batch completed, elapsed: %r, throttled: %r",
            elapsed,
            result.throttled,
        )

    def _send_parameter_messages_batch(
        self, message_items: List[MessageBatchItem], context: HandlerContext
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_parameter_messages_batch started")

        messages = [m.message for m in message_items]
        result = self._process_rest_api_send(
            sender=self._connection.log_parameters_batch,
            message_type=ParameterMessage.type,
            context=context,
            rest_fail_prompt=PARAMETERS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending parameters batch (online experiment)",
            messages=messages,
            items=message_items,
            compress=self._message_batch_compress,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_parameter_messages_batch completed, elapsed: %r, throttled: %r",
            elapsed,
            result.throttled,
        )

    def _process_parameter_message(
        self, message: ParameterMessage, context: HandlerContext
    ) -> None:
        # add a message to the parameters batch
        if not self._parameters_batch.append(message):
            LOGGER.warning(FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH, message)
            # report experiment error
            context.report_error_callback(
                ErrorReport(FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH % message)
            )

    def _process_metric_message(
        self, message: MetricMessage, _: HandlerContext
    ) -> None:
        self._message_batch_metrics.append(message)

    def _process_rest_api_send(
        self,
        sender: Callable,
        message_type: str,
        context: HandlerContext,
        rest_fail_prompt: str,
        general_fail_prompt: str,
        messages: List[BaseMessage],
        **kwargs,
    ) -> RestApiSendResult:
        if not self._should_send_messages(message_type, messages=messages):
            return RestApiSendResult(
                success=False, connection_error=False, throttled=True
            )

        # do send a message if it doesn't hit rate limits yet
        messages_ids = [message.message_id for message in messages]
        try:
            sender(**kwargs)

            # notify successful delivery
            self._on_messages_sent_completed(messages_ids)
            return RestApiSendResult(
                success=True, connection_error=False, throttled=False
            )

        except (ConnectionError, requests.ConnectionError) as conn_err:
            # just log and do not report because there is no connection
            LOGGER.debug(general_fail_prompt, exc_info=True)
            self._on_messages_sent_completed(
                messages_ids,
                success=False,
                connection_error=True,
                failure_reason=str(conn_err),
            )
            return RestApiSendResult(
                success=False, connection_error=True, throttled=False
            )

        except (CometRestApiException, RequestException) as exc:
            if exc.response is not None:
                if (
                    isinstance(exc, CometRestApiException)
                    and exc.response.status_code == HTTPStatus.TOO_MANY_REQUESTS  # 429
                ):
                    # save retry incident with messages to be replayed later
                    reset_at = calculate_reset_at_from_response(
                        exc.response, logger=LOGGER
                    )
                    self._retry_incidents_manager.add_or_update_incident(
                        message_type, reset_at=reset_at, messages=messages
                    )

                    LOGGER.info(
                        "Message(s) of type '%s' were throttled due to backend limits and will be retried after: %s (in %s seconds)",
                        message_type,
                        seconds_to_datetime_str(reset_at),
                        int(reset_at - time.time()),
                    )

                    # stop here - it would be processed
                    return RestApiSendResult(
                        success=False, connection_error=False, throttled=True
                    )
                else:
                    msg = rest_fail_prompt % (
                        exc.response.status_code,
                        exc.response.content,
                    )
            else:
                msg = rest_fail_prompt % (-1, str(exc))

            LOGGER.error(msg, exc_info=True)
            # report experiment error
            context.report_error_callback(ErrorReport(msg))

        except Exception as exception:
            error_message = "%s. %s: %s" % (
                general_fail_prompt,
                exception.__class__.__name__,
                str(exception),
            )
            LOGGER.error(error_message, exc_info=True)
            # report experiment error
            context.report_error_callback(ErrorReport(error_message))

        return RestApiSendResult(success=False, connection_error=False, throttled=False)

    def _on_messages_sent_completed(
        self,
        messages_ids: List[int],
        success: bool = True,
        connection_error: bool = False,
        failure_reason: Optional[str] = None,
    ) -> None:
        size = len(messages_ids)
        if size == 1 and self.on_message_sent_callback is not None:
            self.on_message_sent_callback(
                messages_ids[0], success, connection_error, failure_reason
            )
        elif size > 1 and self.on_messages_batch_sent_callback is not None:
            batch_start = time.time()
            LOGGER.debug("Streamer: batch - start on_messages_batch_sent_callback")

            self.on_messages_batch_sent_callback(
                messages_ids, success, connection_error, failure_reason
            )

            elapsed = time.time() - batch_start
            LOGGER.debug(
                "Streamer: batch - on_messages_batch_sent_callback completed, size: %d, elapsed: %r",
                len(messages_ids),
                elapsed,
            )

    def _process_os_package_message(
        self, message: OsPackagesMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_os_packages,
            message_type=message.type,
            context=context,
            rest_fail_prompt=OS_PACKAGE_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_OS_PACKAGES_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            os_packages=message.os_packages,
        )

    def _process_model_graph_message(
        self, message: ModelGraphMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_model_graph,
            message_type=message.type,
            context=context,
            rest_fail_prompt=MODEL_GRAPH_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_MODEL_GRAPH_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            graph_str=message.graph,
        )

    def _process_system_details_message(
        self, message: SystemDetailsMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_system_details,
            message_type=message.type,
            context=context,
            rest_fail_prompt=SYSTEM_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_SYSTEM_DETAILS_MESSAGE_ERROR,
            messages=[message],
            _os=message.os,
            command=message.command,
            env=message.env,
            experiment_key=self.experiment_key,
            hostname=message.hostname,
            ip=message.ip,
            machine=message.machine,
            os_release=message.os_release,
            os_type=message.os_type,
            pid=message.pid,
            processor=message.processor,
            python_exe=message.python_exe,
            python_version_verbose=message.python_version_verbose,
            python_version=message.python_version,
            user=message.user,
        )

    def _process_log_other_message(
        self, message: LogOtherMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_other,
            message_type=message.type,
            context=context,
            rest_fail_prompt=LOG_OTHER_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_OTHER_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            key=message.key,
            value=message.value,
        )

    def _process_cloud_details_message(
        self, message: CloudDetailsMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_cloud_details,
            message_type=message.type,
            context=context,
            rest_fail_prompt=CLOUD_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_CLOUD_DETAILS_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            provider=message.provider,
            cloud_metadata=message.cloud_metadata,
        )

    def _process_file_name_message(
        self, message: FileNameMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_filename,
            message_type=message.type,
            context=context,
            rest_fail_prompt=FILENAME_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_FILE_NAME_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            filename=message.file_name,
        )

    def _process_html_message(
        self, message: HtmlMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_html,
            message_type=message.type,
            context=context,
            rest_fail_prompt=HTML_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_HTML_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            html=message.html,
        )

    def _process_installed_packages_message(
        self, message: InstalledPackagesMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_installed_packages,
            message_type=message.type,
            context=context,
            rest_fail_prompt=INSTALLED_PACKAGES_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_INSTALLED_PACKAGES_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            installed_packages=message.installed_packages,
        )

    def _process_html_override_message(
        self, message: HtmlOverrideMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_html,
            message_type=message.type,
            context=context,
            rest_fail_prompt=HTML_OVERRIDE_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_HTML_OVERRIDE_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            html=message.htmlOverride,
            overwrite=True,
        )

    def _process_gpu_static_info_message(
        self, message: GpuStaticInfoMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_gpu_static_info,
            message_type=message.type,
            context=context,
            rest_fail_prompt=GPU_STATIC_INFO_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_GPU_STATIC_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            gpu_static_info=message.gpu_static_info,
        )

    def _process_git_metadata_message(
        self, message: GitMetadataMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_git_metadata,
            message_type=message.type,
            context=context,
            rest_fail_prompt=GIT_METADATA_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_GIT_METADATA_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            user=message.git_metadata.get("user"),
            root=message.git_metadata.get("root"),
            branch=message.git_metadata.get("branch"),
            parent=message.git_metadata.get("parent"),
            origin=message.git_metadata.get("origin"),
        )

    def _process_system_info_message(
        self, message: SystemInfoMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_system_info,
            message_type=message.type,
            context=context,
            rest_fail_prompt=SYSTEM_INFO_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_SYSTEM_INFO_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            system_info=[message.system_info],
        )

    def _process_log_dependency_message(
        self, message: LogDependencyMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_dependency,
            message_type=message.type,
            context=context,
            rest_fail_prompt=LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_DEPENDENCY_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            name=message.name,
            version=message.version,
            timestamp=message.local_timestamp,
        )

    def _process_log_3d_cloud_message(
        self, message: Log3DCloudMessage, _: HandlerContext
    ) -> None:
        if message.asset_id is not None:
            # The asset is already created - most probably retrying to send its items
            asset_id = message.asset_id
        else:
            asset_id = self._connection.create_asset(
                experiment_key=self.experiment_key,
                asset_name=message.name,
                asset_type=message.type,
                metadata=message.metadata,
                step=message.step,
            )

        if message.thumbnail_path is not None:
            thumbnail_upload_url = upload_thumbnail_url(
                self._connection.server_address,
                asset_id=asset_id,
            )

            self._file_upload_manager.upload_thumbnail_thread(
                options=ThumbnailUploadOptions(
                    api_key=self.api_key,
                    experiment_id=self.experiment_key,
                    project_id=self.project_id,
                    timeout=self._file_upload_read_timeout,
                    verify_tls=self._verify_tls,
                    upload_endpoint=thumbnail_upload_url,
                    estimated_size=0,
                    thumbnail_path=message.thumbnail_path,
                    critical=False,
                ),
            )

        asset_item_url = self._connection.get_upload_url(message.upload_type)
        for item in message.items:
            self._file_upload_manager.upload_asset_item_thread(
                options=AssetItemUploadOptions(
                    api_key=self.api_key,
                    experiment_id=self.experiment_key,
                    project_id=self.project_id,
                    timeout=self._file_upload_read_timeout,
                    verify_tls=self._verify_tls,
                    upload_endpoint=asset_item_url,
                    estimated_size=0,
                    asset_id=asset_id,
                    asset_item=item,
                    all_items=message.items,  # we need this to restore an original message if throttled
                    upload_type=message.upload_type,
                    asset_name=message.name,
                    critical=False,
                    on_asset_upload=self._on_upload_success_callback(
                        message_id=message.message_id,
                    ),
                    on_failed_asset_upload=self._on_upload_failed_callback(
                        message_id=message.message_id,
                    ),
                ),
            )

    def _process_standard_output_message(
        self, message: StandardOutputMessage, _: HandlerContext
    ) -> None:
        self._message_batch_stdout.append(message)

    def _process_register_model_message(
        self, message: RegisterModelMessage, context: HandlerContext
    ) -> None:
        try:
            status = message.upload_status_observer_callback()
            if status == "IN_PROGRESS":
                if context.message_loop_active:
                    context.push_back_callback(message)
                    return

                # message loop is not active - force to wait for upload complete
                LOGGER.debug(
                    "Message loop is not active! Force wait for model %r upload to complete.",
                    message.model_name,
                )
                while status == "IN_PROGRESS":
                    time.sleep(0.5)
                    status = message.upload_status_observer_callback()

            LOGGER.debug(
                "Model %r upload complete with status: %r", message.model_name, status
            )
            if status == "FAILED":
                LOGGER.error(REGISTER_FAILED_DUE_TO_UPLOADS_FAILED, message.model_name)
                return

            workspace = (
                message.workspace
                or self._rest_api_client.get_experiment_metadata(message.experiment_id)[
                    "workspaceName"
                ]
            )

            LOGGER.debug(
                "Trying to register model %r with registry name %r and version %r",
                message.model_name,
                message.registry_name,
                message.version,
            )
            self._rest_api_client.register_model_v2(
                message.experiment_id,
                message.model_name,
                message.version,
                workspace,
                message.registry_name,
                message.public,
                message.description,
                message.comment,
                message.tags,
                message.status,
                message.stages,
            )
            message.on_model_register()
            # notify message tracker
            self.on_message_sent_callback(message.message_id, True, False, None)
        except CometRestApiException as exception:
            error_message = "{} {}".format(FAILED_TO_REGISTER_MODEL, exception.safe_msg)
            LOGGER.error(error_message)
            context.report_error_callback(ErrorReport(error_message))
            message.on_failed_model_register()
        except (ConnectionError, requests.ConnectionError) as conn_err:
            LOGGER.debug("Failed to register model - connection failure", exc_info=True)
            self.on_message_sent_callback(
                message.message_id, False, True, str(conn_err)
            )
        except Exception as exception:
            error_message = "{} {}".format(FAILED_TO_REGISTER_MODEL, str(exception))
            LOGGER.error(error_message)
            context.report_error_callback(ErrorReport(error_message))
            message.on_failed_model_register()

    def _process_remote_model_message(
        self, message: RemoteModelMessage, context: HandlerContext
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_remote_model,
            message_type=message.type,
            context=context,
            rest_fail_prompt=REMOTE_MODEL_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
            messages=[message],
            experiment_key=self.experiment_key,
            model_name=message.model_name,
            remote_assets=message.remote_assets,
            on_model_upload=message.on_model_upload,
            on_failed_model_upload=message.on_failed_model_upload,
        )

    def _should_send_messages(
        self, message_type: str, messages: List[BaseMessage]
    ) -> bool:
        incident = self._retry_incidents_manager.get_incident(message_type)
        if incident is not None:
            # add messages to be replayed later
            incident.add_messages(messages)

            LOGGER.debug(
                "%d message(s) '%s' were throttled due to active retry incident and will be retried after: %s",
                len(messages),
                message_type,
                seconds_to_datetime_str(incident.reset_at),
            )
            return False
        else:
            return True

    def create_message_handlers(self) -> Dict[Type[BaseMessage], MessageHandlerType]:
        return {
            UploadFileMessage: self._process_upload_message,
            UploadInMemoryMessage: self._process_upload_in_memory_message,
            RemoteAssetMessage: self._process_upload_remote_asset_message,
            MetricMessage: self._process_metric_message,
            ParameterMessage: self._process_parameter_message,
            OsPackagesMessage: self._process_os_package_message,
            ModelGraphMessage: self._process_model_graph_message,
            SystemDetailsMessage: self._process_system_details_message,
            CloudDetailsMessage: self._process_cloud_details_message,
            LogOtherMessage: self._process_log_other_message,
            FileNameMessage: self._process_file_name_message,
            HtmlMessage: self._process_html_message,
            HtmlOverrideMessage: self._process_html_override_message,
            InstalledPackagesMessage: self._process_installed_packages_message,
            GpuStaticInfoMessage: self._process_gpu_static_info_message,
            GitMetadataMessage: self._process_git_metadata_message,
            SystemInfoMessage: self._process_system_info_message,
            LogDependencyMessage: self._process_log_dependency_message,
            StandardOutputMessage: self._process_standard_output_message,
            RegisterModelMessage: self._process_register_model_message,
            RemoteModelMessage: self._process_remote_model_message,
            Log3DCloudMessage: self._process_log_3d_cloud_message,
        }

    @property
    def file_upload_read_timeout(self) -> float:
        return self._file_upload_read_timeout

    @property
    def message_batch_compress(self) -> bool:
        return self._message_batch_compress

    @property
    def parameters_batch(self) -> ParametersBatch:
        return self._parameters_batch
