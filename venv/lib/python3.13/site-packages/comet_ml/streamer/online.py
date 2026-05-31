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

"""
Author: Gideon Mendels

This module contains the main components of comet client side

"""
import functools
import logging
import time
from typing import List, NamedTuple, Optional

from requests import RequestException

from .._reporting import ON_EXIT_DIDNT_FINISH_UPLOAD_SDK
from .._typing import OnMessagesBatchSentCallback, OnMessageSentCallback
from ..config import (
    ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
    DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
    DEFAULT_STREAMER_MSG_TIMEOUT,
    DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
    MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    PROGRESS_CALLBACK_INTERVAL,
    S3_MULTIPART_EXPIRES_IN,
    S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
)
from ..connection import RestApiClient, RestServerConnection
from ..file_upload_manager import FileUploadManager, FileUploadManagerMonitor
from ..logging_messages import (
    EXPERIMENT_MESSAGE_QUEUE_FLUSH_PROMPT,
    FILE_UPLOADS_PROMPT,
    STREAMER_FAILED_TO_PROCESS_ALL_MESSAGES,
    STREAMER_FLUSH_DIDNT_COMPLETE_SUCCESSFULLY_INFO,
    STREAMER_PROGRESS_MESSAGE_INFO,
    STREAMER_WAIT_FOR_FINISH_FAILED,
    UNEXPECTED_STREAMING_ERROR,
)
from ..messages import BaseMessage, CloseMessage
from ..s3.multipart_upload.multipart_upload_options import MultipartUploadOptions
from ..utils import log_once_at_level, seconds_to_datetime_str, wait_for_done
from .base import BaseStreamer
from .handlers.handler_context import HandlerContext
from .handlers.online_handler import OnlineMessageHandler
from .progress_helpers import FixedIntervalProgressTracker
from .retry.retry_incidents_manager import ActiveRetryIncident, RetryIncidentsManager

LOGGER = logging.getLogger(__name__)


class RestApiSendResult(NamedTuple):
    success: bool
    connection_error: bool


class OnlineStreamer(BaseStreamer):
    """
    Manages the streaming of messages and artifacts for an experiment.

    The Streamer class handles the processing of various types of messages and artifact
    uploads associated with an experiment run. It is responsible for managing communication
    between the client and a remote server, batching messages for efficient transmission,
    and ensuring reliability in data transmission through retries and error handling. It
    uses a message-handling loop to process incoming messages, batch data for upload, and
    interact with a REST API server. This class is critical for the proper functioning of
    experiment logging and result reporting.
    """

    def __init__(
        self,
        beat_duration: float,
        connection: RestServerConnection,
        initial_offset: int,
        experiment_key: str,
        api_key: str,
        run_id: str,
        project_id: str,
        rest_api_client: RestApiClient,
        worker_cpu_ratio: int,
        worker_count: Optional[int],
        verify_tls: bool,
        msg_waiting_timeout: float = DEFAULT_STREAMER_MSG_TIMEOUT,
        file_upload_waiting_timeout: float = ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
        file_upload_read_timeout: float = DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
        wait_for_finish_sleep_interval: float = DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
        parameters_batch_base_interval: float = MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
        message_batch_compress: bool = MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
        message_batch_metric_interval: float = MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
        message_batch_metric_max_size: int = MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
        message_batch_stdout_interval: float = MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
        message_batch_stdout_max_size: int = MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
        s3_multipart_threshold: int = S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
        s3_multipart_expires_in: int = S3_MULTIPART_EXPIRES_IN,
        s3_multipart_upload_enabled: bool = False,
        artifact_remote_assets_batch_enabled: bool = False,
        artifact_remote_assets_batch_metric_interval: float = ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
        artifact_remote_assets_batch_metric_max_size: int = ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
        use_raw_throttling_messages: bool = True,
        progress_callback_interval: float = PROGRESS_CALLBACK_INTERVAL,
    ) -> None:
        super().__init__(initial_offset=initial_offset, queue_timeout=beat_duration)

        self.daemon = True
        self.name = "OnlineStreamer(experiment=%r)" % experiment_key

        self.experiment_key = experiment_key
        self._connection = connection
        self._rest_api_client = rest_api_client
        self._stop_message_queue_processing = False
        self._msg_waiting_timeout = msg_waiting_timeout
        self._wait_for_finish_sleep_interval = wait_for_finish_sleep_interval
        self._file_upload_waiting_timeout = file_upload_waiting_timeout
        self._progress_callback_interval = progress_callback_interval

        self._file_upload_manager = FileUploadManager(
            worker_cpu_ratio=worker_cpu_ratio,
            worker_count=worker_count,
            s3_upload_options=MultipartUploadOptions(
                file_size_threshold=s3_multipart_threshold,
                upload_expires_in=s3_multipart_expires_in,
                direct_s3_upload_enabled=s3_multipart_upload_enabled,
            ),
        )

        self._retry_incidents_manager = RetryIncidentsManager()

        self.message_handler = OnlineMessageHandler(
            connection=connection,
            file_upload_manager=self._file_upload_manager,
            retry_incidents_manager=self._retry_incidents_manager,
            experiment_key=experiment_key,
            api_key=api_key,
            run_id=run_id,
            project_id=project_id,
            rest_api_client=rest_api_client,
            verify_tls=verify_tls,
            file_upload_read_timeout=file_upload_read_timeout,
            parameters_batch_base_interval=parameters_batch_base_interval,
            message_batch_compress=message_batch_compress,
            message_batch_metric_interval=message_batch_metric_interval,
            message_batch_metric_max_size=message_batch_metric_max_size,
            message_batch_stdout_interval=message_batch_stdout_interval,
            message_batch_stdout_max_size=message_batch_stdout_max_size,
            artifact_remote_assets_batch_enabled=artifact_remote_assets_batch_enabled,
            artifact_remote_assets_batch_metric_interval=artifact_remote_assets_batch_metric_interval,
            artifact_remote_assets_batch_metric_max_size=artifact_remote_assets_batch_metric_max_size,
            use_raw_throttling_messages=use_raw_throttling_messages,
        )

        LOGGER.debug(
            "Streamer created with REST API url: %r, clientlib url: %r",
            self._rest_api_client.base_url,
            self._connection.server_address,
        )
        LOGGER.debug(
            "---------------------------------------------------------------------"
        )
        LOGGER.debug(
            "Parameters batch base interval: %s seconds", parameters_batch_base_interval
        )
        LOGGER.debug(
            "Metrics max batch size: %d, metrics batch interval: %s seconds",
            message_batch_metric_max_size,
            message_batch_metric_interval,
        )
        LOGGER.debug(
            "Stdout batch max size: %d, batch interval: %s seconds",
            message_batch_stdout_max_size,
            message_batch_stdout_interval,
        )
        LOGGER.debug(
            "Artifact remote assets batching enabled: %s, max batch size: %d, batch interval: %s seconds",
            artifact_remote_assets_batch_enabled,
            artifact_remote_assets_batch_metric_max_size,
            artifact_remote_assets_batch_metric_interval,
        )
        LOGGER.debug("Batches compression enabled: %s", message_batch_compress)
        LOGGER.debug(
            "---------------------------------------------------------------------"
        )

    def register_message_sent_callback(
        self, message_sent_callback: OnMessageSentCallback
    ) -> None:
        self.message_handler.on_message_sent_callback = message_sent_callback

    def register_messages_batch_sent_callback(
        self, messages_batch_sent_callback: OnMessagesBatchSentCallback
    ) -> None:
        self.message_handler.on_messages_batch_sent_callback = (
            messages_batch_sent_callback
        )

    def _current_handler_context(self) -> HandlerContext:
        return HandlerContext(
            message_loop_active=self.is_message_loop_active(),
            push_back_callback=self.put_message_in_q,
            report_error_callback=lambda report: self._report_experiment_error(
                report.message, report.has_crashed
            ),
        )

    def _process_incidents(self, incidents: List[ActiveRetryIncident]) -> None:
        for incident in incidents:
            LOGGER.debug(
                "OnlineStreamer: processing retry incident for messages type '%s', holding %d message(s), at: %s",
                incident.message_type,
                len(incident.messages),
                seconds_to_datetime_str(time.time()),
            )
            for m in incident.messages:
                self.messages.put(m)

    def _loop(self) -> Optional[BaseMessage]:
        """
        A single loop of running
        """
        try:
            # If we should stop processing the queue, abort early
            if self._stop_message_queue_processing:
                LOGGER.debug("OnlineStreamer: force close event loop")
                return CloseMessage()

            # check if any retry incident should be released and its messages retried
            if self._retry_incidents_manager.has_active_incidents():
                incidents = self._retry_incidents_manager.release_outdated_incidents(
                    now=time.time()
                )
                self._process_incidents(incidents)

            messages = self.getn(1)
            context = self._current_handler_context()

            if messages is not None:
                for message in messages:
                    if isinstance(message, CloseMessage):
                        LOGGER.debug(
                            "OnlineStreamer: closing event loop - CloseMessage found in the messages\n"
                        )
                        # force batches to flush
                        self._flush_batches()

                        LOGGER.debug(
                            "OnlineStreamer: closing event loop - no more messages to process"
                        )
                        return message
                    else:
                        self.message_handler.handle(message, context=context)

            # send collected batch data
            self.message_handler.send_batches(flush=False, context=context)

        except Exception as ex:
            LOGGER.debug(UNEXPECTED_STREAMING_ERROR, ex, exc_info=True)
            # report experiment error
            self._report_experiment_error(UNEXPECTED_STREAMING_ERROR % ex)

    def has_connection_to_server(self) -> bool:
        return True

    def is_message_loop_active(self) -> bool:
        with self.__lock_closed__:
            return not self.closed and not self._stop_message_queue_processing

    def _flush_message_queue(self, show_all_prompts, timeout: Optional[int] = None):
        mq_flush_start = time.time()
        LOGGER.debug("OnlineStreamer: start flushing messages queue")

        success = True
        if not self._is_msg_queue_empty():
            timeout = timeout if timeout is not None else self._msg_waiting_timeout

            if show_all_prompts:
                log_once_at_level(
                    logging.INFO,
                    EXPERIMENT_MESSAGE_QUEUE_FLUSH_PROMPT,
                    timeout,
                )

            progress_callback = FixedIntervalProgressTracker(
                interval=self._progress_callback_interval,
                progress_callback=self._show_remaining_messages,
            )
            wait_for_done(
                self._is_msg_queue_empty,
                timeout=timeout,
                progress_callback=progress_callback,
                sleep_time=self._wait_for_finish_sleep_interval,
            )
            success = self._is_msg_queue_empty()

        elapsed = time.time() - mq_flush_start
        LOGGER.debug(
            "OnlineStreamer: flushing messages queue completed, successful: %s, elapsed: %r",
            success,
            elapsed,
        )

        if not success:
            LOGGER.warning(STREAMER_FAILED_TO_PROCESS_ALL_MESSAGES)

        return success

    def _send_batches(self, flush: bool = False) -> bool:
        return self.message_handler.send_batches(
            flush=flush, context=self._current_handler_context()
        )

    def _flush_file_upload_manager(
        self, show_all_prompts: bool, timeout: Optional[int] = None
    ) -> bool:
        fu_start_time = time.time()
        LOGGER.debug("OnlineStreamer: start flushing file upload manager")

        timeout = timeout if timeout is not None else self._file_upload_waiting_timeout
        success = True
        if not self._file_upload_manager.all_done():
            monitor = FileUploadManagerMonitor(self._file_upload_manager)
            if show_all_prompts:
                LOGGER.info(FILE_UPLOADS_PROMPT, timeout)

            progress_callback = FixedIntervalProgressTracker(
                interval=self._progress_callback_interval,
                progress_callback=monitor.log_remaining_uploads,
            )
            wait_for_done(
                monitor.all_done,
                timeout=timeout,
                progress_callback=progress_callback,
                sleep_time=self._wait_for_finish_sleep_interval,
            )

            success = self._file_upload_manager.all_done()

        elapsed = time.time() - fu_start_time
        LOGGER.debug(
            "OnlineStreamer: flushing file upload manager completed, successful: %s, elapsed: %r",
            success,
            elapsed,
        )

        return success

    def _send_batches_if_has_messages(self, flush: bool) -> bool:
        if self._has_messages_to_process():
            self._send_batches(flush=flush)

        return not self._has_messages_to_process()

    def _flush_batches(self, timeout: Optional[int] = None) -> bool:
        # repeat until all batched data is sent
        timeout = timeout if timeout is not None else self._msg_waiting_timeout
        progress_callback = FixedIntervalProgressTracker(
            interval=self._progress_callback_interval,
            progress_callback=lambda: LOGGER.info(
                "Waiting for remaining data to be uploaded to Comet. %d messages remaining.",
                self._get_remaining_messages(),
            ),
        )
        wait_for_done(
            functools.partial(self._send_batches_if_has_messages, flush=True),
            timeout=timeout,
            sleep_time=self._wait_for_finish_sleep_interval,
            progress_callback=progress_callback,
        )
        return not self._has_messages_to_process()

    def flush(self, timeout: Optional[int] = None) -> bool:
        """Flushes all pending data but do not close any threads.
        This method can be invoked multiple times during the experiment lifetime."""

        LOGGER.debug("Start flushing all pending data to Comet")

        message_queue_flushed = self._flush_message_queue(
            show_all_prompts=False, timeout=timeout
        )

        batches_flushed = self._flush_batches(timeout=timeout)

        uploads_flushed = self._flush_file_upload_manager(
            show_all_prompts=False, timeout=timeout
        )

        if not (message_queue_flushed and batches_flushed and uploads_flushed):
            LOGGER.info(
                STREAMER_FLUSH_DIDNT_COMPLETE_SUCCESSFULLY_INFO
                % (message_queue_flushed, batches_flushed, uploads_flushed)
            )

        return message_queue_flushed & batches_flushed & uploads_flushed

    def wait_for_finish(self, **kwargs) -> bool:
        """Blocks the experiment from exiting until all data was sent to server
        OR the configured timeouts have expired."""
        # We need to wait for online streamer to be closed first.
        # The streamer closed in an asynchronous manner to allow all pending messages to be logged before closing.
        wf_start_time = time.time()
        LOGGER.debug("OnlineStreamer: started wait_for_finish")

        start = time.monotonic()

        wait_for_done(
            lambda: self.closed, timeout=self._msg_waiting_timeout, sleep_time=0.5
        )

        LOGGER.debug(
            "OnlineStreamer: Wait for closed elapsed: %f", (time.monotonic() - start)
        )

        start = time.monotonic()

        message_queue_flushed = self._flush_message_queue(show_all_prompts=True)

        LOGGER.debug(
            "OnlineStreamer: Wait for messages queue flushed elapsed: %f",
            (time.monotonic() - start),
        )

        start = time.monotonic()

        batches_flushed = self._flush_batches()

        LOGGER.debug(
            "OnlineStreamer: Wait for batches flushed elapsed: %f",
            (time.monotonic() - start),
        )

        # stop message processing only after message queue flushed to give thread loop a chance to go through
        # all accumulated messages in _flush_message_queue() - loop process one message at a time
        self._stop_message_queue_processing = True

        self._file_upload_manager.close()
        uploads_flushed = self._flush_file_upload_manager(show_all_prompts=True)

        LOGGER.debug(
            ""
            "OnlineStreamer: waiting for finish. Message queue flushed: %s, batches flushed: %s, uploads flushed: %s, throttling incidents detected: %d",
            message_queue_flushed,
            batches_flushed,
            uploads_flushed,
            self._retry_incidents_manager.registered_incidents_count,
        )

        if not self._is_msg_queue_empty() or not self._file_upload_manager.all_done():
            remaining = self.messages.qsize()
            remaining_upload = self._file_upload_manager.remaining_uploads()
            error_message = STREAMER_WAIT_FOR_FINISH_FAILED % (
                remaining,
                remaining_upload,
                self.experiment_key,
            )
            LOGGER.error(error_message)
            # report experiment error
            self._report_experiment_error(error_message)

            self._connection.report(
                event_name=ON_EXIT_DIDNT_FINISH_UPLOAD_SDK, err_msg=error_message
            )

            return False

        self._file_upload_manager.join()

        self._clean_throttled_experiment_mark()

        elapsed = time.time() - wf_start_time
        LOGGER.debug(
            "OnlineStreamer: wait_for_finish completed successfully, elapsed: %r",
            elapsed,
        )

        return True

    def _clean_throttled_experiment_mark(self):
        if (
            self._retry_incidents_manager.registered_incidents_count > 0
            and not self._retry_incidents_manager.has_active_incidents()
        ):
            LOGGER.debug("Cleaning throttled experiment mark")
            try:
                self._rest_api_client.set_experiment_not_throttled(
                    experiment_key=self.experiment_key
                )
            except Exception as ex:
                LOGGER.warning(
                    "Failed to clean throttled experiment mark (%s). Reason: %s",
                    self.experiment_key,
                    ex,
                    exc_info=True,
                )

    def _clean_file_uploads(self) -> None:
        self.message_handler.clean_file_uploads()

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        try:
            self._rest_api_client.update_experiment_error_status(
                experiment_key=self.experiment_key,
                is_alive=True,
                error_value=message,
                has_crashed=has_crashed,
            )
        except (ConnectionError, RequestException):
            LOGGER.debug("Failed to report experiment error", exc_info=True)
        except Exception as ex:
            LOGGER.debug("Failed to report experiment error, %r", ex, exc_info=True)

    def _has_messages_to_process(self) -> bool:
        return (
            not self._is_msg_queue_empty() or self.message_handler.has_batches_unsent()
        )

    def _is_msg_queue_empty(self) -> bool:
        empty = (
            self.messages.empty()
            and not self._retry_incidents_manager.has_active_incidents()
        )

        if not empty:
            LOGGER.debug(
                "OnlineStreamer: Messages queue is not empty, %d messages remaining, streamer closed: %s, has retry incidents: %s",
                self._get_remaining_messages(),
                self.closed,
                self._retry_incidents_manager.has_active_incidents(),
            )

        return empty

    def _get_remaining_messages(self) -> int:
        return self.messages.qsize() + self._retry_incidents_manager.messages_to_retry()

    def _show_remaining_messages(self) -> None:
        LOGGER.info(STREAMER_PROGRESS_MESSAGE_INFO, self._get_remaining_messages())

    def has_upload_failed(self) -> bool:
        return self.message_handler.has_upload_failed()

    @property
    def has_artifact_remote_assets_batch_send_failed(self) -> bool:
        return self.message_handler.has_artifact_remote_assets_batch_send_failed()

    @property
    def msg_waiting_timeout(self) -> float:
        return self._msg_waiting_timeout

    @property
    def file_upload_waiting_timeout(self) -> float:
        return self._file_upload_waiting_timeout

    @property
    def retry_manager(self) -> RetryIncidentsManager:
        return self._retry_incidents_manager
