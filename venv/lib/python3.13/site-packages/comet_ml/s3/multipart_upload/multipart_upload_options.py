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
from .upload_types import DIRECT_S3_UPLOAD_TYPES


class MultipartUploadOptions:
    def __init__(
        self,
        file_size_threshold: int,
        upload_expires_in: int,
        direct_s3_upload_enabled: bool = False,
    ):
        self.file_size_threshold = file_size_threshold
        self.upload_expires_in = upload_expires_in
        self.direct_s3_upload_enabled = direct_s3_upload_enabled

    def has_direct_s3_upload_enabled_for(
        self, upload_type: str, file_size: int
    ) -> bool:
        if not self.direct_s3_upload_enabled:
            return False

        return (
            upload_type in DIRECT_S3_UPLOAD_TYPES
            and file_size >= self.file_size_threshold
        )
