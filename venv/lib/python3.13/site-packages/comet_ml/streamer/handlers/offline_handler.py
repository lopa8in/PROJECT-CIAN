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
import os
import shutil
import tempfile
import threading
from os.path import splitext
from typing import Any, AnyStr, Callable, Dict, Optional, Type

from ...config import OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME
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
from ...messages_utils import convert_upload_in_memory_to_file_message
from ...utils import compact_json_dump
from .base_handler import BaseMessageHandler, MessageHandlerType
from .handler_context import HandlerContext


class OfflineMessageHandler(BaseMessageHandler):
    """
    Handles offline messages during operation by storing them locally and providing
    fallback mechanisms.

    This class is designed to store messages in a temporary directory for offline
    processing, employ a fallback mode if necessary, and provide support for
    serializing messages to JSON format. It extends the `BaseMessageHandler`
    functionality and is particularly useful in environments where network
    connectivity might be unreliable or intermittent. The class is equipped with
    methods for message handling, writing data to disk, and converting in-memory
    uploads to file-based uploads.

    Attributes:
        tmp_dir (str): Directory for storing temporary files.
        wait_timeout (int): Timeout value for processing or waiting operations.
        on_error (Callable): Callback function invoked when an error occurs,
            receiving error details in a string format.
        fallback_mode (bool): Determines if fallback mode is active, impacting how
            files are handled during offline message processing.
    """

    def __init__(
        self,
        tmp_dir: AnyStr,
        wait_timeout: int,
        on_error_callback: Callable[[str], None],
        fallback_mode: bool,
        write_lock: threading.RLock,
    ) -> None:
        super().__init__()

        self.tmp_dir = tmp_dir
        self.wait_timeout = wait_timeout
        self.on_error = on_error_callback
        self.fallback_mode = fallback_mode

        self.__write_lock__ = write_lock

        self._message_handlers: Optional[Dict[Type[BaseMessage], Callable]] = None

        self.file = open(
            os.path.join(self.tmp_dir, OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME), "w"
        )

    def close(self):
        # Close the messages files once we are sure we won't write in it anymore
        self.file.close()

    def _write(
        self, json_line_message: Dict[str, Any], context: HandlerContext
    ) -> None:
        if not context.message_loop_active:
            # an offline handler was aborted
            return

        with self.__write_lock__:
            compact_json_dump(json_line_message, self.file)
            self.file.write("\n")
            self.file.flush()

    def _process_upload_message(
        self, message: UploadFileMessage, context: HandlerContext
    ) -> None:
        # Create the file on disk with the same extension if set
        ext = splitext(message.file_path)[1]

        if ext:
            suffix = ".%s" % ext
        else:
            suffix = ""

        tmp_file = tempfile.NamedTemporaryFile(
            dir=self.tmp_dir, suffix=suffix, delete=False
        )
        tmp_file.close()

        # do not remove an original file in fallback mode, i.e., when offline streamer is just a backup for online streamer
        if self.fallback_mode:
            # fallback mode
            if message.clean:
                # just copy to keep it after a message would be cleaned by online streamer
                shutil.copy(message.file_path, tmp_file.name)
        else:
            # Offline mode
            if message.clean:
                # Clean by moving the original file to the newly created file
                shutil.move(message.file_path, tmp_file.name)
            else:
                # copy to our tmp dir
                shutil.copy(message.file_path, tmp_file.name)

        msg_json = message.to_message_dict()
        # update file_path directly in JSON representation to avoid side effects
        msg_json["file_path"] = os.path.basename(tmp_file.name)
        data = {"type": UploadFileMessage.type, "payload": msg_json}
        self._write(data, context=context)

    def _process_upload_in_memory_message(
        self, message: UploadInMemoryMessage, context: HandlerContext
    ) -> None:
        # We need to convert the in-memory file to a file one
        new_message = convert_upload_in_memory_to_file_message(message, self.tmp_dir)
        self._process_upload_message(new_message, context=context)

    def _process_message(self, message: BaseMessage, context: HandlerContext):
        msg_json = message.to_message_dict()

        data = {"type": message.type, "payload": msg_json}
        self._write(data, context=context)

    def create_message_handlers(self) -> Dict[Type[BaseMessage], MessageHandlerType]:
        return {
            UploadFileMessage: self._process_upload_message,
            UploadInMemoryMessage: self._process_upload_in_memory_message,
            RemoteAssetMessage: self._process_message,
            MetricMessage: self._process_message,
            ParameterMessage: self._process_message,
            OsPackagesMessage: self._process_message,
            ModelGraphMessage: self._process_message,
            SystemDetailsMessage: self._process_message,
            CloudDetailsMessage: self._process_message,
            FileNameMessage: self._process_message,
            HtmlMessage: self._process_message,
            LogOtherMessage: self._process_message,
            HtmlOverrideMessage: self._process_message,
            InstalledPackagesMessage: self._process_message,
            GpuStaticInfoMessage: self._process_message,
            GitMetadataMessage: self._process_message,
            SystemInfoMessage: self._process_message,
            StandardOutputMessage: self._process_message,
            LogDependencyMessage: self._process_message,
            RegisterModelMessage: self._process_message,
            RemoteModelMessage: self._process_message,
            Log3DCloudMessage: self._process_message,
        }
