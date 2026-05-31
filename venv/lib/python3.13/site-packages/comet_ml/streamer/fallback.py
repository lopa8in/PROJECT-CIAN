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
import logging
import random
import shutil
import string
import tempfile
import threading
import time
from queue import Empty
from typing import Any, List, Optional

from .. import offline_utils
from .._reporting import ON_RECONNECTION_EVENT
from ..config import Config
from ..connection import RestServerConnection
from ..connection.connection_monitor import ConnectionStatus, ServerConnectionMonitor
from ..constants import RESUME_STRATEGY_CREATE
from ..convert_utils import convert_dict_to_string
from ..logging_messages import (
    FALLBACK_STREAMER_ARCHIVE_UPLOAD_MESSAGE_KEEP_ENABLED,
    FALLBACK_STREAMER_FAILED_NO_CONNECTION_NO_OFFLINE,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE_ERROR,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_STREAMER_WARNING,
    FALLBACK_STREAMER_FAILED_TO_WRITE_OFFLINE_DATA_ERROR,
    FALLBACK_STREAMER_ONLINE_FAILED_ARCHIVE_UPLOAD_MESSAGE,
)
from ..messages import BaseMessage, CloseMessage
from ..offline_utils import get_offline_data_dir_path
from ..replay_manager import MessageStatus, ReplayManager
from ..upload_callback.callback import UploadCallback
from ..utils import local_timestamp
from .base import BaseStreamer
from .offline import OfflineStreamer
from .online import OnlineStreamer
from .retry.retry_incidents_manager import RetryIncidentsManager

LOGGER = logging.getLogger(__name__)


class FallbackStreamer(BaseStreamer):
    """
    Manages streaming of messages with functionality for fallback to offline mode.

    The FallbackStreamer handles the streaming of messages primarily via an online
    streamer. If a server connection issue arises, it can fall back to offline
    streaming mode, optionally storing messages locally in an offline archive file.
    The class also supports replaying failed messages upon reconnection, tracking
    the status of sent messages, and enabling a seamless transition between online
    and offline operation modes.

    Attributes:
        online_streamer: The primary online message streamer.
        server_connection_monitor: Monitors the connection
            status with the server.
        rest_server_connection: Handles backend server
            communication, including connection checks.
        terminate_timeout: The timeout value in seconds for termination of
            the streamer. Default is 10.
        offline_directory: Directory path to store offline message
            files, if enabled. Default is None.
        enabled_fallback_to_offline: Whether fallback to offline mode is
            enabled. Default is True.
        keep_offline_zip: Whether to retain offline zip files regardless of
            connection status. Default is False.
    """

    def __init__(
        self,
        online_streamer: OnlineStreamer,
        server_connection_monitor: ServerConnectionMonitor,
        rest_server_connection: RestServerConnection,
        initial_offset: int,
        terminate_timeout: int = 10,
        offline_directory: Optional[str] = None,
        enable_fallback_to_offline: bool = True,
        keep_offline_zip: bool = False,  # allows to retain offline zip even without connection interruption
    ) -> None:
        super().__init__(initial_offset=initial_offset, queue_timeout=0.5)

        self.daemon = True
        self.name = "FallbackStreamer(experiment=%r)" % online_streamer.experiment_key

        self.online_streamer = online_streamer
        self.online_streamer.register_message_sent_callback(self._track_message)
        self.online_streamer.register_messages_batch_sent_callback(
            self._track_messages_batch
        )

        self.server_connection_monitor = server_connection_monitor
        self.rest_server_connection = rest_server_connection
        self.terminate_timeout = terminate_timeout
        self.offline_directory = offline_directory

        self.start_time = None
        self.stop_time = None

        self.offline_archive_file: Optional[str] = None
        self.offline_streamer = None
        self.temp_dir = None

        self.offline_zip_uploader: Optional[UploadCallback] = None

        self.enabled_fallback_to_offline = enable_fallback_to_offline
        self.keep_offline_zip = keep_offline_zip

        self.replay_manager = ReplayManager()

        self.queue_empty = threading.Condition()

        if self.enabled_fallback_to_offline:
            try:
                self._create_offline_streamer()
                self.offline_streamer_disabled_or_failed = False
            except Exception:
                LOGGER.warning(
                    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_STREAMER_WARNING,
                    exc_info=True,
                )
                self.offline_streamer_disabled_or_failed = True
        else:
            self.offline_streamer_disabled_or_failed = True
            LOGGER.debug(
                "Skip creation of fallback offline streamer - disabled by configuration"
            )

    def parameters_update_interval_callback(self, parameters_update_interval):
        self.online_streamer.message_handler.parameters_batch.update_interval(
            parameters_update_interval
        )

    def _create_offline_streamer(self):
        self.temp_dir = tempfile.mkdtemp()
        self.offline_streamer = OfflineStreamer(
            tmp_dir=self.temp_dir,
            wait_timeout=60,
            fallback_mode=True,
            experiment_key=self.online_streamer.experiment_key,
        )
        self.offline_streamer.on_error = self._offline_streamer_on_error_callback
        LOGGER.debug(
            "Offline streamer created with temporary data dir: %r", self.temp_dir
        )

    def _offline_streamer_on_error_callback(self, message: str) -> None:
        LOGGER.debug("Offline streamer failed with error: %s", message)
        self.offline_streamer_disabled_or_failed = True

    def _before_run(self) -> None:
        self.start_time = local_timestamp()
        LOGGER.debug("FallbackStreamer._before_run(), start time: %r", self.start_time)

    def _after_run(self) -> None:
        self.stop_time = local_timestamp()
        LOGGER.debug("FallbackStreamer._after_run(), stop time: %r", self.stop_time)

    def _loop(self) -> Optional[BaseMessage]:
        status = self.server_connection_monitor.tick(
            connection_probe=self._check_server_connection,
        )
        if status == ConnectionStatus.connection_restored:
            self._on_connection_restored()

        # process buffered messages if any
        self._process_buffered_messages()

        # notify waiting thread that queue already processed
        with self.queue_empty:
            self.queue_empty.notify()

        with self.__lock_closed__:
            if self.closed is True:
                # one more try to process pending messages before closing streamers
                self._process_buffered_messages()
                self._close_streamers()
                # force stop the thread run()
                return CloseMessage()

        return None

    def _process_buffered_messages(self):
        messages = []
        try:
            while True:
                message = self.messages.get(timeout=0.1)
                self._counter += 1
                message.message_id = self._counter
                messages.append(message)
        except Empty:
            pass
        if len(messages) > 0:
            self._process_queue_messages(messages)

    def _close_streamers(self):
        self.online_streamer.close()
        if self.offline_streamer is not None:
            self.offline_streamer.close()

    def _process_queue_messages(self, messages: List[BaseMessage]) -> None:
        connected = self.has_connection_to_server()
        status = MessageStatus.registered if connected else MessageStatus.failed
        # register message with replay manager
        self.replay_manager.register_messages(messages, status=status)

        for message in messages:
            if connected:
                self.online_streamer.put_message_in_q(message)

            if (
                self.offline_streamer is not None
                and self.offline_streamer_disabled_or_failed is False
            ):
                self.offline_streamer.put_message_in_q(message)

    def _check_server_connection(self) -> (bool, Optional[str]):
        try:
            self.rest_server_connection.ping_backend()
            return True, None
        except Exception as ex:
            LOGGER.debug("Backend ping failed, reason: %r", ex)
            return False, str(ex)

    def _replay_message(self, message: BaseMessage) -> None:
        self.online_streamer.put_message_in_q(message)

    def _on_connection_restored(self):
        # send BI event if appropriate
        if self.server_connection_monitor.disconnect_reason is not None:
            err_msg_dict = {
                "disconnect_reason": self.server_connection_monitor.disconnect_reason,
                "disconnect_time": self.server_connection_monitor.disconnect_time,
            }
            self.online_streamer._connection.report(
                event_name=ON_RECONNECTION_EVENT,
                err_msg=convert_dict_to_string(err_msg_dict),
            )
        self.server_connection_monitor.reset()

        # replay all failed messages
        self.replay_manager.replay_failed_messages(self._replay_message)

    def _track_message(
        self,
        message_id: int,
        delivered: bool,
        connection_error: bool,
        failure_reason: Optional[str] = None,
    ):
        if connection_error:
            self.server_connection_monitor.connection_failed(failure_reason)

        status = MessageStatus.delivered if delivered else MessageStatus.failed
        self.replay_manager.update_message(message_id=message_id, status=status)

    def _track_messages_batch(
        self,
        message_ids: List[int],
        delivered: bool,
        connection_error: bool,
        failure_reason: Optional[str] = None,
    ):
        if connection_error:
            self.server_connection_monitor.connection_failed(failure_reason)

        status = MessageStatus.delivered if delivered else MessageStatus.failed
        self.replay_manager.update_messages_batch(
            message_ids=message_ids, status=status
        )

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        if self.has_connection_to_server():
            self.online_streamer._report_experiment_error(
                message, has_crashed=has_crashed
            )

    def put_message_in_q(self, message) -> None:
        with self.__lock_closed__:
            if self.closed:
                return

            LOGGER.debug("Put message in queue: %r", message)
            self.messages.put(message, block=False)

    def has_upload_failed(self) -> bool:
        return self.online_streamer.has_upload_failed()

    def start(self) -> None:
        super().start()
        self.online_streamer.start()
        if self.offline_streamer is not None:
            self.offline_streamer.start()

    def has_connection_to_server(self) -> bool:
        return self.server_connection_monitor.has_server_connection

    def flush(self, timeout: Optional[int] = None) -> bool:
        """Flushes all pending data in online streamer.
        This method can be invoked multiple times during the experiment lifetime."""
        with self.queue_empty:
            self.queue_empty.wait()
        return self.online_streamer.flush(timeout=timeout)

    def set_offline_zip_uploader(self, upload_callback: UploadCallback) -> None:
        self.offline_zip_uploader = upload_callback

    def wait_for_finish(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool = True,  # this is for testing purposes only
    ) -> bool:
        fb_wait_start = time.time()
        LOGGER.debug(
            "FallbackStreamer: wait_for_finish started, experiment ID: %s",
            self.online_streamer.experiment_key,
        )

        online_finished_successfully = False
        if self.has_connection_to_server():
            online_finished_successfully = self.online_streamer.wait_for_finish()

        # Also wait for the current thread to finish
        if _join is True:
            self.join(self.terminate_timeout)

        # Close replay manager after online and this streamer complete its run
        self.replay_manager.close()

        if online_finished_successfully:
            create_offline_zip = self.keep_offline_zip
        elif not self.offline_streamer_disabled_or_failed:
            create_offline_zip = True
        else:
            # failed both modes - clean temporary directory
            self._clean_offline_data()
            LOGGER.error(FALLBACK_STREAMER_FAILED_NO_CONNECTION_NO_OFFLINE)
            return False

        finished_successfully = online_finished_successfully
        archive_created = False
        if create_offline_zip:
            LOGGER.debug(
                "FallbackStreamer: trying to create offline ZIP, experiment ID: %s",
                self.online_streamer.experiment_key,
            )
            # try to create offline archive as fallback
            archive_created = self._create_offline_archive_and_clean(
                experiment_key=experiment_key,
                workspace=workspace,
                project_name=project_name,
                tags=tags,
                comet_config=comet_config,
                _join=_join,
            )
            LOGGER.debug(
                "FallbackStreamer: offline ZIP created: %r, "
                "online streamer finished successfully: %r, experiment ID: %s",
                archive_created,
                online_finished_successfully,
                self.online_streamer.experiment_key,
            )

            if not online_finished_successfully:
                # offline mode - return flag indicating whether archive was created
                if archive_created:
                    # Display the full command to upload the offline experiment if archive was created
                    LOGGER.warning(
                        FALLBACK_STREAMER_ONLINE_FAILED_ARCHIVE_UPLOAD_MESSAGE,
                        self.offline_archive_file,
                    )
                    self._try_to_upload_offline_zip_file(self.offline_archive_file)
                finished_successfully = archive_created
            elif archive_created:
                # online finished successfully and archive retained - prompt user about having it
                LOGGER.info(
                    FALLBACK_STREAMER_ARCHIVE_UPLOAD_MESSAGE_KEEP_ENABLED,
                    self.offline_archive_file,
                )
        else:
            self._abort_offline_and_clean_offline(_join)

        elapsed = time.time() - fb_wait_start
        LOGGER.debug(
            "FallbackStreamer: wait_for_finish completed, success: %r, elapsed: %r, "
            "offline ZIP enabled: %r, offline ZIP created: %r, experiment ID: %s",
            finished_successfully,
            elapsed,
            create_offline_zip,
            archive_created,
            self.online_streamer.experiment_key,
        )

        return finished_successfully

    def _try_to_upload_offline_zip_file(self, offline_zip_file: str):
        if self.offline_zip_uploader is not None:
            offline_upload_start = time.time()
            LOGGER.debug(
                "FallbackStreamer: trying to upload offline ZIP file: %r",
                offline_zip_file,
            )
            try:
                self.offline_zip_uploader(offline_zip_file)

                elapsed = time.time() - offline_upload_start
                LOGGER.debug(
                    "FallbackStreamer: offline ZIP uploaded successfully, elapsed: %r",
                    elapsed,
                )
            except Exception as ex:
                LOGGER.error("Failed to upload offline ZIP file: %r", ex, exc_info=True)

    def _create_offline_archive_and_clean(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool = True,
    ) -> bool:
        try:
            self._create_offline_archive(
                experiment_key=experiment_key,
                workspace=workspace,
                project_name=project_name,
                tags=tags,
                comet_config=comet_config,
                _join=_join,
            )
            return True
        except Exception:
            LOGGER.error(
                FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE, exc_info=True
            )
            return False
        finally:
            # make sure to clean up
            self._clean_offline_data()

    def _abort_offline_and_clean_offline(self, _join):
        if self.offline_streamer is not None:
            LOGGER.debug("Aborting offline streamer before cleaning")
            self.offline_streamer.abort_and_wait(self.terminate_timeout, _join=_join)

        self._clean_offline_data()

    def _clean_offline_data(self) -> None:
        if self.temp_dir is not None:
            try:
                LOGGER.debug(
                    "Cleaning collected offline data from dir: %r", self.temp_dir
                )
                shutil.rmtree(self.temp_dir)
            except Exception:
                LOGGER.debug(
                    "Failed to remove collected offline data from temporary directory: %r",
                    self.temp_dir,
                    exc_info=True,
                )

        LOGGER.debug(
            "Cleaning collected temporary files being uploaded by online streamer"
        )
        self.online_streamer._clean_file_uploads()

    def _create_offline_archive(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool,
    ) -> None:
        if self.offline_streamer is None or self.offline_streamer_disabled_or_failed:
            LOGGER.error(FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE_ERROR)
            return

        finished_successfully = self.offline_streamer.wait_for_finish(_join=_join)
        if not finished_successfully:
            LOGGER.error(FALLBACK_STREAMER_FAILED_TO_WRITE_OFFLINE_DATA_ERROR)

        # create offline experiment archive
        #
        offline_utils.write_experiment_meta_file(
            tempdir=self.temp_dir,
            experiment_key=experiment_key,
            workspace=workspace,
            project_name=project_name,
            tags=tags,
            start_time=self.start_time,
            stop_time=self.stop_time,
            resume_strategy=RESUME_STRATEGY_CREATE,
        )

        # adjust offline directory path if appropriate
        self.offline_directory, _ = get_offline_data_dir_path(
            comet_config=comet_config,
            offline_directory=self.offline_directory,
            logger=LOGGER,
        )

        # create offline archive into offline directory
        random_string = "".join(random.choice(string.ascii_letters) for _ in range(6))
        offline_archive_file_name = "%s-%s.zip" % (experiment_key, random_string)
        (
            self.offline_archive_file,
            self.offline_directory,
        ) = offline_utils.create_experiment_archive(
            offline_directory=self.offline_directory,
            offline_archive_file_name=offline_archive_file_name,
            data_dir=self.temp_dir,
            logger=LOGGER,
        )

    @property
    def counter(self) -> int:
        return self.online_streamer._counter

    @counter.setter
    def counter(self, new_value: int) -> None:
        self._counter = new_value

    @property
    def msg_waiting_timeout(self) -> float:
        return self.online_streamer.msg_waiting_timeout

    @property
    def file_upload_waiting_timeout(self) -> float:
        return self.online_streamer.file_upload_waiting_timeout

    @property
    def file_upload_read_timeout(self) -> float:
        return self.online_streamer.message_handler.file_upload_read_timeout

    @property
    def message_batch_compress(self) -> bool:
        return self.online_streamer.message_handler.message_batch_compress

    @property
    def retry_incidents_manager(self) -> RetryIncidentsManager:
        return self.online_streamer.retry_manager

    @message_batch_compress.setter
    def message_batch_compress(self, compress: bool) -> None:
        self.online_streamer._message_batch_compress = compress
