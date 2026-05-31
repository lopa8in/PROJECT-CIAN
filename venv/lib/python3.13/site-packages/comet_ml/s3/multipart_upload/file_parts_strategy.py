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
import math
import os
from typing import IO

from .upload_error import S3UploadErrorFileIsEmpty, S3UploadErrorFileIsTooLarge

# Constants defining existing AWS S3 limits (https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html)
MAX_SUPPORTED_PARTS_NUMBER = 10000
MIN_FILE_PART_SIZE = 5 * 1024 * 1024
MAX_FILE_PART_SIZE = 5 * 1024 * 1024 * 1024


class BaseFilePartsStrategy(object):
    def __init__(
        self,
        file: str,
        file_size: int,
        max_file_part_size: int = MIN_FILE_PART_SIZE,
    ):
        self.file = file
        self.file_size = file_size

        if max_file_part_size < MIN_FILE_PART_SIZE:
            self.max_file_part_size = MIN_FILE_PART_SIZE
        elif max_file_part_size > MAX_FILE_PART_SIZE:
            self.max_file_part_size = MAX_FILE_PART_SIZE
        else:
            self.max_file_part_size = max_file_part_size

    def calculate(self) -> int:
        if self.file_size == 0:
            raise S3UploadErrorFileIsEmpty(self.file, "Can not upload empty file")

        parts_number = math.ceil(self.file_size / self.max_file_part_size)

        if parts_number > MAX_SUPPORTED_PARTS_NUMBER:
            parts_number = MAX_SUPPORTED_PARTS_NUMBER

        # check that we are still in part size limits
        part_size = math.ceil(self.file_size / parts_number)
        if part_size > MAX_FILE_PART_SIZE:
            raise S3UploadErrorFileIsTooLarge(
                self.file,
                "File is too large to be uploaded, file size: %d" % self.file_size,
            )

        if part_size < MIN_FILE_PART_SIZE:
            self.max_file_part_size = MIN_FILE_PART_SIZE
        else:
            self.max_file_part_size = part_size

        return parts_number


class FilePartsStrategy(BaseFilePartsStrategy):
    def __init__(
        self,
        file_path: str,
        file_size: int = None,
        max_file_part_size: int = MIN_FILE_PART_SIZE,
    ):
        if file_size is None:
            file_size = os.path.getsize(file_path)

        super(FilePartsStrategy, self).__init__(
            file=file_path, file_size=file_size, max_file_part_size=max_file_part_size
        )


class FileLikePartsStrategy(BaseFilePartsStrategy):
    def __init__(
        self,
        file_name: str,
        file_like: IO,
        file_size: int,
        max_file_part_size: int = MIN_FILE_PART_SIZE,
    ):
        super(FileLikePartsStrategy, self).__init__(
            file=file_name, file_size=file_size, max_file_part_size=max_file_part_size
        )
        self.file_like = file_like
