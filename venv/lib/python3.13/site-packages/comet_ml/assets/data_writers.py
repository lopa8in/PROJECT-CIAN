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
import requests

from .. import file_downloader, gs_bucket_info, s3_bucket_info
from .._typing import IO, Optional
from ..connection.connection_helpers import write_stream_response_to_file


class AssetDataWriter(object):
    def write(self, file: IO[bytes]) -> None:
        pass


class AssetDataWriterFromResponse(AssetDataWriter):
    def __init__(
        self,
        response: requests.Response,
        monitor: Optional[file_downloader.FileDownloadSizeMonitor] = None,
    ) -> None:
        self._response = response
        self.monitor = monitor

    def write(self, file: IO[bytes]) -> None:
        write_stream_response_to_file(self._response, file, self.monitor)


class AssetDataWriterFromS3(AssetDataWriter):
    def __init__(
        self,
        s3_uri: str,
        version_id: Optional[str],
        monitor: Optional[file_downloader.FileDownloadSizeMonitor] = None,
    ) -> None:
        self._s3_uri = s3_uri
        self._version_id = version_id
        self.monitor = monitor

    def write(self, file: IO[bytes]) -> None:
        callback = None
        if self.monitor is not None:
            callback = self.monitor.monitor_callback

        s3_bucket_info.download_s3_file(
            s3_uri=self._s3_uri,
            file_object=file,
            callback=callback,
            version_id=self._version_id,
        )


class AssetDataWriterFromGCS(AssetDataWriter):
    def __init__(
        self,
        gs_uri: str,
        version_id: Optional[str],
        monitor: Optional[file_downloader.FileDownloadSizeMonitor] = None,
    ) -> None:
        self._gs_uri = gs_uri
        self._version_id = version_id
        self.monitor = monitor

    def write(self, file: IO[bytes]) -> None:
        callback = None
        if self.monitor is not None:
            callback = self.monitor.monitor_callback

        gs_bucket_info.download_gs_file(
            gs_uri=self._gs_uri,
            file_object=file,
            callback=callback,
            version_id=self._version_id,
        )
