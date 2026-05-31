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
from typing import Optional


class S3UploadError(Exception):
    def __init__(self, reason: str, due_connection_error: bool = False):
        self.reason = reason
        self.due_connection_error = due_connection_error

    def __str__(self):
        msg = "S3 file upload failed, reason:%s, due connection error: %s"
        return msg % (self.reason, self.due_connection_error)


class S3UploadFileError(S3UploadError):
    def __init__(
        self,
        file: str,
        reason: str,
        retry_attempts: Optional[int] = None,
        due_connection_error: bool = False,
    ):
        super().__init__(reason, due_connection_error)
        self.file = file
        self.retry_attempts = retry_attempts

    def __str__(self):
        msg = "S3 file upload failed, file: %r. Reason: %r, due connection error: %s"
        return msg % (self.file, self.reason, self.due_connection_error)


class S3UploadErrorFileIsTooLarge(S3UploadFileError):
    def __init__(self, file: str, reason: str):
        super().__init__(file, reason)


class S3UploadErrorFileIsEmpty(S3UploadFileError):
    def __init__(self, file: str, reason: str):
        super().__init__(file, reason)
