# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import json
import logging
import os.path
import shutil
import sqlite3
import tempfile
import threading
from collections import namedtuple
from enum import IntEnum, unique
from typing import Callable, Dict, List, Optional

from .json_encoder import NestedEncoder
from .logging_messages import RESILIENCE_NOT_INITIALIZED_ERROR
from .messages import (
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
    MessageCallbacks,
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
    WebSocketMessage,
)
from .messages_utils import convert_upload_in_memory_to_file_message

DEFAULT_DB_FILE = "comet_messages.db"

LOGGER = logging.getLogger(__name__)


@unique
class MessageStatus(IntEnum):
    registered = 1
    delivered = 2
    failed = 3


@unique
class ManagerStatus(IntEnum):
    undefined = 1
    initialized = 2
    closed = 3
    error = 4


DBMessage = namedtuple("DBMessage", ["id", "type", "json", "status"])

ReplayCallback = Callable[[BaseMessage], None]


class ReplayManager:
    def __init__(
        self, db_file: Optional[str] = None, conn: sqlite3.Connection = None
    ) -> None:
        self.status = ManagerStatus.undefined
        self.tmp_dir = tempfile.mkdtemp()
        if db_file is None:
            db_file = os.path.join(self.tmp_dir, DEFAULT_DB_FILE)

        self.db_file = db_file
        # open DB connection if appropriate
        if conn is None:
            conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.conn = conn

        self.__lock__ = threading.RLock()
        self._create_db_schema()

        self.message_callbacks: Dict[int, MessageCallbacks] = {}
        self.message_files: Dict[int, str] = {}

    def _create_db_schema(self):
        try:
            with self.__lock__:
                with self.conn:
                    self.conn.execute(
                        """CREATE TABLE messages
                                            (message_id INTEGER NOT NULL PRIMARY KEY,
                                            status INTEGER NOT NULL,
                                            message_type TEXT NOT NULL,
                                            message_json TEXT NOT NULL)"""
                    )
                    self.status = ManagerStatus.initialized
        except Exception as ex:
            msg = "Database schema creation failed, reason: %r" % ex
            self._mark_as_db_failed(msg)
            LOGGER.debug(msg, exc_info=True)

    def close(self) -> None:
        if self.closed:
            return

        with self.__lock__:
            self.status = ManagerStatus.closed

            try:
                LOGGER.debug("Closing messages DB connection")
                self.conn.close()
            except Exception:
                LOGGER.debug("Failed to close messages DB connection", exc_info=True)

            # delete temporary data
            if self.tmp_dir is not None:
                try:
                    LOGGER.debug("Cleaning temporary data dir: %r", self.tmp_dir)
                    shutil.rmtree(self.tmp_dir)
                except Exception:
                    LOGGER.debug(
                        "Failed to clean temporary data dir: %r",
                        self.tmp_dir,
                        exc_info=True,
                    )

    def register_message(
        self, message: BaseMessage, status: MessageStatus = MessageStatus.registered
    ) -> None:
        if not self.initialized:
            LOGGER.debug("Not initialized - register message ignored")
            return

        with self.__lock__:
            if self.closed:
                LOGGER.warning("Already closed - register message ignored")
                return

            message_json = self._preprocess_registered_message(message)
            # insert into DB
            values = (
                message.message_id,
                status,
                message.type,
                message_json,
            )
            try:
                with self.conn:
                    self.conn.execute("INSERT INTO messages VALUES (?,?,?,?)", values)
            except Exception as ex:
                msg = (
                    "register_message: failed to insert message into DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)

    def register_messages(
        self,
        messages: List[BaseMessage],
        status: MessageStatus = MessageStatus.registered,
    ):
        if not self.initialized:
            LOGGER.debug("Not initialized - register messages list ignored")
            return

        with self.__lock__:
            if self.closed:
                LOGGER.warning("Already closed - register messages list ignored")
                return

            values = []
            for message in messages:
                message_json = self._preprocess_registered_message(message)
                values.append(
                    (
                        message.message_id,
                        status,
                        message.type,
                        message_json,
                    )
                )

            try:
                with self.conn:
                    self.conn.executemany(
                        "INSERT INTO messages VALUES (?,?,?,?)", values
                    )
            except Exception as ex:
                msg = (
                    "register_messages: failed to insert messages into DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)

    def _clean_message_leftovers(self, message_id: int):
        if message_id in self.message_callbacks:
            del self.message_callbacks[message_id]

        # Cleanup message file
        if message_id in self.message_files:
            try:
                os.remove(self.message_files[message_id])
            except Exception:
                LOGGER.debug(
                    "Failed to remove temporary file: %r of the message: %d",
                    self.message_files[message_id],
                    message_id,
                    exc_info=True,
                )

    def _preprocess_registered_message(self, message: BaseMessage):
        if message.message_id is None:
            raise ValueError("Message ID expected")

        if isinstance(message, UploadInMemoryMessage):
            message = convert_upload_in_memory_to_file_message(message, self.tmp_dir)
            self.message_files[message.message_id] = message.file_path

        # save callbacks to be used later
        callbacks = message.get_message_callbacks()
        if callbacks is not None:
            self.message_callbacks[message.message_id] = callbacks

        return json.dumps(
            message.to_db_message_dict(),
            sort_keys=True,
            separators=(",", ":"),
            cls=NestedEncoder,
        )

    def update_messages_batch(
        self, message_ids: List[int], status: MessageStatus
    ) -> None:
        if not self.initialized:
            LOGGER.debug(
                "Not initialized - messages batch update ignored, size: %d, status: %r",
                len(message_ids),
                status,
            )
            return

        with self.__lock__:
            if self.closed:
                LOGGER.warning(
                    "Already closed - messages batch update ignored, size: %d, status: %r",
                    len(message_ids),
                    status,
                )
                return

            try:
                with self.conn:
                    if status == MessageStatus.delivered:
                        db_ids = [(message_id,) for message_id in message_ids]
                        c = self.conn.executemany(
                            "DELETE FROM messages WHERE message_id = ?",
                            db_ids,
                        )
                        LOGGER.debug(
                            "Deleted %d DB message records for %d delivered messages",
                            c.rowcount,
                            len(message_ids),
                        )
                        # delete saved message callbacks and leftovers
                        for message_id in message_ids:
                            self._clean_message_leftovers(message_id)
                    else:
                        db_status_ids = [
                            (status, message_id) for message_id in message_ids
                        ]
                        c = self.conn.executemany(
                            "UPDATE messages SET status = ? WHERE message_id = ?",
                            db_status_ids,
                        )
                        LOGGER.debug(
                            "Updated %d DB message records for %d messages to have status: %r",
                            c.rowcount,
                            len(message_ids),
                            status,
                        )
            except Exception as ex:
                msg = (
                    "update_messages_batch: failed to update messages batch in the DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)

    def update_message(self, message_id: int, status: MessageStatus) -> None:
        if not self.initialized:
            LOGGER.debug(
                "Not initialized - message update ignored, id: %d, status: %r",
                message_id,
                status,
            )
            return

        with self.__lock__:
            if self.closed:
                LOGGER.warning(
                    "Already closed - message update ignored, id: %d, status: %r",
                    message_id,
                    status,
                )
                return
            try:
                with self.conn:
                    if status == MessageStatus.delivered:
                        # remove delivered messages
                        self.conn.execute(
                            "DELETE FROM messages WHERE message_id = ?", (message_id,)
                        )
                        # delete saved message callbacks and leftovers
                        self._clean_message_leftovers(message_id)
                    else:
                        self.conn.execute(
                            "UPDATE messages SET status = ? WHERE message_id = ?",
                            (status, message_id),
                        )
            except Exception as ex:
                msg = (
                    "update_message: failed to update message in the DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)

    def replay_failed_messages(self, replay_callback: ReplayCallback) -> int:
        if not self.initialized:
            LOGGER.debug("Not initialized - messages replay ignored")
            return 0

        with self.__lock__:
            if self.closed:
                LOGGER.warning("Already closed - messages replay ignored")
                return 0

            try:
                db_messages = self._fetch_failed_messages()
                if len(db_messages) == 0:
                    return 0
            except Exception as ex:
                msg = (
                    "replay_failed_messages: failed to fetch failed messages from DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)
                return 0

            messages_ids = [(message.id,) for message in db_messages]
            # update DB records to mark failed messages as in progress
            try:
                with self.conn:
                    c = self.conn.executemany(
                        "UPDATE messages SET status = %d WHERE message_id = ?"
                        % MessageStatus.registered,
                        messages_ids,
                    )
                    LOGGER.debug(
                        "Updated %d DB message records for %d failed messages",
                        c.rowcount,
                        len(db_messages),
                    )
            except Exception as ex:
                msg = (
                    "replay_failed_messages: failed to update messages in the DB, reason: %r"
                    % ex
                )
                self._mark_as_db_failed(msg)
                LOGGER.debug(msg, exc_info=True)
        return self._replay_messages(
            db_messages=db_messages, replay_callback=replay_callback
        )

    def _replay_messages(
        self, db_messages: List[DBMessage], replay_callback: ReplayCallback
    ) -> int:
        LOGGER.debug("Replaying %d failed messages to streamer", len(db_messages))
        for message in db_messages:
            if self.closed:
                return 0

            try:
                base_message = db_message_to_message(message)

                callbacks = self.message_callbacks.get(message.id, None)
                if callbacks is not None:
                    base_message.set_message_callbacks(callbacks)

                replay_callback(base_message)
            except Exception:
                LOGGER.error("Failed to replay message: %r", message)

        return len(db_messages)

    def get_message(self, message_id: int) -> Optional[BaseMessage]:
        db_message = self.get_db_message(message_id)
        if db_message is not None:
            return db_message_to_message(db_message)
        else:
            return None

    def get_db_message(self, message_id: int) -> Optional[DBMessage]:
        with self.conn:
            c = self.conn.execute(
                "SELECT message_id, message_type, message_json, status FROM messages WHERE message_id = ?",
                (message_id,),
            )
            row = c.fetchone()
            if row is not None:
                return DBMessage(id=row[0], type=row[1], json=row[2], status=row[3])
            else:
                return None

    @property
    def closed(self) -> bool:
        return self.status == ManagerStatus.closed

    @property
    def initialized(self) -> bool:
        return self.status == ManagerStatus.initialized

    @property
    def failed(self) -> bool:
        return self.status == ManagerStatus.error

    def _fetch_failed_messages(self) -> List[DBMessage]:
        messages_db = []
        for row in self.conn.execute(
            "SELECT message_id, message_type, message_json FROM messages WHERE status = ?",
            (MessageStatus.failed,),
        ):
            messages_db.append(
                DBMessage(
                    id=row[0], type=row[1], json=row[2], status=MessageStatus.failed
                )
            )

        return messages_db

    def _mark_as_db_failed(self, message: str):
        self.status = ManagerStatus.error
        LOGGER.error(RESILIENCE_NOT_INITIALIZED_ERROR, message)


def db_message_to_message(db_message: DBMessage) -> BaseMessage:
    message_dict = json.loads(db_message.json)
    if db_message.type == CloudDetailsMessage.type:
        message = CloudDetailsMessage.from_db_message_dict(message_dict)
    elif db_message.type == FileNameMessage.type:
        message = FileNameMessage.from_db_message_dict(message_dict)
    elif db_message.type == GitMetadataMessage.type:
        message = GitMetadataMessage.from_db_message_dict(message_dict)
    elif db_message.type == GpuStaticInfoMessage.type:
        message = GpuStaticInfoMessage.from_db_message_dict(message_dict)
    elif db_message.type == HtmlMessage.type:
        message = HtmlMessage.from_db_message_dict(message_dict)
    elif db_message.type == HtmlOverrideMessage.type:
        message = HtmlOverrideMessage.from_db_message_dict(message_dict)
    elif db_message.type == InstalledPackagesMessage.type:
        message = InstalledPackagesMessage.from_db_message_dict(message_dict)
    elif db_message.type == LogDependencyMessage.type:
        message = LogDependencyMessage.from_db_message_dict(message_dict)
    elif db_message.type == LogOtherMessage.type:
        message = LogOtherMessage.from_db_message_dict(message_dict)
    elif db_message.type == MetricMessage.type:
        message = MetricMessage.from_db_message_dict(message_dict)
    elif db_message.type == ModelGraphMessage.type:
        message = ModelGraphMessage.from_db_message_dict(message_dict)
    elif db_message.type == OsPackagesMessage.type:
        message = OsPackagesMessage.from_db_message_dict(message_dict)
    elif db_message.type == ParameterMessage.type:
        message = ParameterMessage.from_db_message_dict(message_dict)
    elif db_message.type == RegisterModelMessage.type:
        message = RegisterModelMessage.from_db_message_dict(message_dict)
    elif db_message.type == RemoteAssetMessage.type:
        message = RemoteAssetMessage.from_db_message_dict(message_dict)
    elif db_message.type == RemoteModelMessage.type:
        message = RemoteModelMessage.from_db_message_dict(message_dict)
    elif db_message.type == StandardOutputMessage.type:
        message = StandardOutputMessage.from_db_message_dict(message_dict)
    elif db_message.type == SystemDetailsMessage.type:
        message = SystemDetailsMessage.from_db_message_dict(message_dict)
    elif db_message.type == SystemInfoMessage.type:
        message = SystemInfoMessage.from_db_message_dict(message_dict)
    elif db_message.type == UploadFileMessage.type:
        message = UploadFileMessage.from_db_message_dict(message_dict)
    elif db_message.type == Log3DCloudMessage.type:
        message = Log3DCloudMessage.from_db_message_dict(message_dict)
    elif db_message.type == WebSocketMessage.type:
        message = WebSocketMessage.from_db_message_dict(message_dict)
    else:
        raise ValueError("Unsupported message type: %r" % db_message.type)

    return message
