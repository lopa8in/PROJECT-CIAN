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
from __future__ import print_function

import logging
import math
from typing import Callable, List, Optional, Tuple, Union

from .connection.connection_upload import (
    upload_asset_item,
    upload_file,
    upload_file_like,
    upload_remote_asset,
    upload_s3_multipart_file,
    upload_s3_multipart_file_like,
    upload_thumbnail,
)
from .file_upload_limits_guard import FileUploadLimitsGuard
from .file_upload_size_monitor import UploadSizeMonitor
from .logging_messages import (
    FILE_UPLOAD_MANAGER_FAILED_TO_SUBMIT_ALREADY_CLOSED,
    FILE_UPLOAD_MANAGER_MONITOR_FIRST_MESSAGE,
    FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION,
    FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION_UNKOWN_ETA,
    FILE_UPLOAD_MANAGER_MONITOR_WAITING_BACKEND_ANSWER,
)
from .s3.multipart_upload.multipart_upload_options import MultipartUploadOptions
from .thread_pool import Future, get_thread_pool
from .upload_options import (
    AssetItemUploadOptions,
    FileLikeUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
    ThumbnailUploadOptions,
    UploadOptions,
)
from .utils import format_bytes, get_time_monotonic

RemainingUploadData = Tuple[int, int, int]

LOGGER = logging.getLogger(__name__)


class FileUploadManager(object):
    def __init__(
        self,
        worker_cpu_ratio: int,
        s3_upload_options: MultipartUploadOptions,
        worker_count: Optional[int] = None,
    ) -> None:
        self.upload_results: List[UploadResult] = []

        pool_size, cpu_count, self._executor = get_thread_pool(
            worker_cpu_ratio, worker_count
        )
        self.s3_upload_options = s3_upload_options

        self.limits_guard = FileUploadLimitsGuard()

        self.closed = False

        if not self.s3_upload_options.direct_s3_upload_enabled:
            LOGGER.debug(
                "Direct S3 upload disabled due to unsupported backend version."
            )
        else:
            LOGGER.debug("Direct S3 upload enabled.")

        LOGGER.debug(
            "FileUploadManager instantiated with %d threads, %d CPUs, %d worker_cpu_ratio, %s worker_count",
            pool_size,
            cpu_count,
            worker_cpu_ratio,
            worker_count,
        )

    def _use_s3_direct_upload(
        self, options: Union[FileUploadOptions, FileLikeUploadOptions]
    ) -> bool:
        return self.s3_upload_options.has_direct_s3_upload_enabled_for(
            upload_type=options.upload_type,
            file_size=options.estimated_size,
        )

    def upload_file_thread(self, options: FileUploadOptions) -> None:

        if self._use_s3_direct_upload(options):
            self._initiate_upload(
                options=options,
                uploader=upload_s3_multipart_file,
                s3_multipart=True,
                use_limits_guard=False,
            )
        else:
            self._initiate_upload(options=options, uploader=upload_file)

    def upload_file_like_thread(self, options: FileLikeUploadOptions) -> None:
        if self._use_s3_direct_upload(options):
            self._initiate_upload(
                options=options,
                uploader=upload_s3_multipart_file_like,
                s3_multipart=True,
                use_limits_guard=False,
            )
        else:
            self._initiate_upload(options=options, uploader=upload_file_like)

    def upload_asset_item_thread(self, options: AssetItemUploadOptions) -> None:
        self._initiate_upload(
            options=options,
            uploader=upload_asset_item,
            use_limits_guard=False,
        )

    def upload_thumbnail_thread(self, options: ThumbnailUploadOptions) -> None:
        self._initiate_upload(
            options=options,
            uploader=upload_thumbnail,
            use_limits_guard=False,
        )

    def upload_remote_asset_thread(self, options: RemoteAssetsUploadOptions) -> None:
        self._initiate_upload(
            options=options,
            uploader=upload_remote_asset,
            use_limits_guard=False,
        )

    def _initiate_upload(
        self,
        options: UploadOptions,
        uploader: Callable,
        s3_multipart: bool = False,
        use_limits_guard: bool = True,
    ) -> None:
        if self.closed:
            LOGGER.warning(FILE_UPLOAD_MANAGER_FAILED_TO_SUBMIT_ALREADY_CLOSED, options)
            return

        monitor = UploadSizeMonitor()
        if options.estimated_size is not None:
            monitor.total_size = options.estimated_size

        kwargs = {"_monitor": monitor, "options": options}
        if s3_multipart is True:
            kwargs["multipart_options"] = self.s3_upload_options

        if use_limits_guard is True:
            kwargs["upload_limits_guard"] = self.limits_guard

        future = self._executor.submit(uploader, **kwargs)
        async_result = UploadResult(
            future=future, critical=options.critical, monitor=monitor
        )
        self.upload_results.append(async_result)

    def all_done(self) -> bool:
        return all(result.ready() for result in self.upload_results)

    def remaining_data(self) -> RemainingUploadData:
        remaining_uploads = 0
        remaining_bytes_to_upload = 0
        total_size = 0
        for result in self.upload_results:
            monitor = result.monitor
            if monitor.total_size is None or monitor.bytes_read is None:
                continue

            if result.ready() is True:
                continue

            total_size += monitor.total_size
            remaining_uploads += 1

            remaining_bytes_to_upload += monitor.total_size - monitor.bytes_read

        return remaining_uploads, remaining_bytes_to_upload, total_size

    def remaining_uploads(self) -> int:
        status_list = [result.ready() for result in self.upload_results]
        return status_list.count(False)

    def close(self) -> None:
        self._executor.close()
        self.closed = True

    def join(self) -> None:
        self._executor.join()

    def has_failed(self) -> bool:
        """Returns True if:
        * at least one critical file uploads has failed
        * at least one critical file upload is not finished yet, caller must handle the timeout itself
        """
        for result in self.upload_results:
            if not result.critical:
                continue

            if not result.ready():
                return True
            elif not result.successful():
                return True

        return False


class FileUploadManagerMonitor(object):
    def __init__(self, file_upload_manager: FileUploadManager) -> None:
        self.file_upload_manager = file_upload_manager
        self.last_remaining_bytes = 0
        self.last_remaining_uploads_display = None

    def log_remaining_uploads(self) -> None:
        uploads, remaining_bytes, total_size = self.file_upload_manager.remaining_data()

        current_time = get_time_monotonic()

        if remaining_bytes == 0:
            LOGGER.info(
                FILE_UPLOAD_MANAGER_MONITOR_WAITING_BACKEND_ANSWER,
            )
        elif self.last_remaining_uploads_display is None:
            LOGGER.info(
                FILE_UPLOAD_MANAGER_MONITOR_FIRST_MESSAGE,
                uploads,
                format_bytes(remaining_bytes),
                format_bytes(total_size),
            )
        else:
            uploaded_bytes = self.last_remaining_bytes - remaining_bytes
            time_elapsed = current_time - self.last_remaining_uploads_display
            upload_speed = uploaded_bytes / time_elapsed

            # Avoid 0 division if no bytes were uploaded in the last period
            if uploaded_bytes <= 0:
                # avoid negative upload speed
                if upload_speed < 0:
                    upload_speed = 0

                LOGGER.info(
                    FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION_UNKOWN_ETA,
                    uploads,
                    format_bytes(remaining_bytes),
                    format_bytes(total_size),
                    format_bytes(upload_speed),
                )

            else:
                # Avoid displaying 0s, also math.ceil returns a float in Python 2.7
                remaining_time = str(int(math.ceil(remaining_bytes / upload_speed)))

                LOGGER.info(
                    FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION,
                    uploads,
                    format_bytes(remaining_bytes),
                    format_bytes(total_size),
                    format_bytes(upload_speed),
                    remaining_time,
                )

        self.last_remaining_bytes = remaining_bytes
        self.last_remaining_uploads_display = current_time

    def all_done(self) -> bool:
        return self.file_upload_manager.all_done()


class UploadResult(object):
    def __init__(
        self, future: Future, critical: bool, monitor: UploadSizeMonitor
    ) -> None:
        self.future = future
        self.critical = critical
        self.monitor = monitor

    def ready(self) -> bool:
        """Allows to check if wrapped Future successfully finished"""
        return self.future.done()

    def successful(self) -> bool:
        """Allows to check if wrapped Future completed without raising an exception"""
        return self.future.successful()
