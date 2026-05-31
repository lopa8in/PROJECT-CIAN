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
import logging
import os.path
from typing import Any, Optional

from comet_ml.logging_messages import BOTO3_IMPORT_FAILED
from comet_ml.upload_callback.callback import UploadCallback

LOGGER = logging.getLogger(__name__)


def get_s3_uploader(bucket: str, key: Optional[str] = None, **kwargs) -> UploadCallback:
    """Return an upload callback that can be used to upload offline archive
    when an online Experiment has fallback to offline.

    See also: [comet_ml.Experiment.set_offline_zip_uploader][]

    The AWS credentials must be present either via environment variables or in
    the user home directory in format supported by `boto3` client as described
    at:
    [https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html]()

    Args:
        bucket: The bucket name to which the PUT action was initiated.
        key (Optional[str], optional): Object key for which the PUT action was
            initiated. If not provided the automatically generated file name
            will be used.
        kwargs: The additional parameters supported by `boto3` client as
            described here:
            [https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html]()

    Returns:
        A callback to be used to upload offline experiment data to S3.

    For example:
    ```python
    from comet_ml import start, get_s3_uploader

    upload_callback = get_s3_uploader(bucket="my_bucket")
    experiment = start()
    experiment.set_offline_zip_uploader(upload_callback)
    ```
    """
    try:
        import boto3
    except ImportError as ex:
        LOGGER.warning(
            "boto3 is not installed or cannot be imported",
            exc_info=True,
            extra={"show_traceback": True},
        )
        raise ex
    s3_client = boto3.client("s3")
    return _create_s3_uploader(s3_client=s3_client, bucket=bucket, key=key, **kwargs)


def _create_s3_uploader(
    s3_client: Any, bucket: str, key: Optional[str] = None, **kwargs
) -> UploadCallback:
    def s3_uploader(file: str):
        if key is None:
            file_key = os.path.basename(file)
        else:
            file_key = key
        try:
            with open(file, "rb") as f:
                response = s3_client.put_object(
                    Body=f, Bucket=bucket, Key=file_key, **kwargs
                )
                LOGGER.info("Upload to S3 completed, got response: %r", response)
        except Exception:
            LOGGER.error(
                "Upload to S3 failed, bucket: %r, key: %r, extra arguments: %r",
                bucket,
                file_key,
                kwargs,
                exc_info=True,
            )

    return s3_uploader
