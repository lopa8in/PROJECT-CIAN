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
import time
from typing import Any, Dict, List, NamedTuple, Optional

import requests
from requests import Response, Session

from ...config import UPLOAD_FILE_MAX_RETRIES, UPLOAD_FILE_RETRY_BACKOFF_FACTOR
from ...connection.connection_helpers import calculate_backoff_time
from .upload_error import S3UploadError

MAX_UPLOAD_START_ATTEMPTS = UPLOAD_FILE_MAX_RETRIES
MAX_S3_PART_UPLOAD_ATTEMPTS = UPLOAD_FILE_MAX_RETRIES
MAX_UPLOAD_COMPLETE_ATTEMPTS = UPLOAD_FILE_MAX_RETRIES


LOGGER = logging.getLogger(__name__)


class UploadResult(NamedTuple):
    response: Optional[Response]
    retry_attempts: int
    failed: bool
    has_connection_error: bool


class UploadRetryStrategyOp(object):
    """A strategy to perform upload operations with defined retry attempt counts
    for different stages of S3 direct upload.

    Args:
        max_upload_start_attempts: number of attempts for upload-start call.
        max_upload_complete_attempts: number of attempts for S3 AWS part upload
        max_s3_file_part_upload_attempts: number of attempts for S3 AWS part upload
    """

    def __init__(
        self,
        max_upload_start_attempts: int,
        max_upload_complete_attempts: int,
        max_s3_file_part_upload_attempts: int,
        retry_backoff_factor: float,
    ):
        self.max_upload_start_attempts = max_upload_start_attempts
        self.max_upload_complete_attempts = max_upload_complete_attempts
        self.max_s3_file_part_upload_attempts = max_s3_file_part_upload_attempts
        self.retry_backoff_factor = retry_backoff_factor

    @classmethod
    def default_upload_retry_strategy(cls):
        return UploadRetryStrategyOp(
            max_upload_start_attempts=MAX_UPLOAD_START_ATTEMPTS,
            max_upload_complete_attempts=MAX_UPLOAD_COMPLETE_ATTEMPTS,
            max_s3_file_part_upload_attempts=MAX_S3_PART_UPLOAD_ATTEMPTS,
            retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        )

    def start_multipart_upload(
        self,
        session: Session,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, Any],
    ) -> UploadResult:
        result = _request_with_retries(
            session=session,
            method="POST",
            url=url,
            json_payload=payload,
            data=None,
            headers=headers,
            max_retries=self.max_upload_start_attempts,
            retry_backoff_factor=self.retry_backoff_factor,
        )
        _raise_for_comet_bad_status_or_failure(
            result=result, operation="start S3 direct upload"
        )

        return result

    def complete_multipart_upload(
        self,
        session: Session,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, Any],
    ) -> UploadResult:
        result = _request_with_retries(
            session=session,
            method="POST",
            url=url,
            json_payload=payload,
            data=None,
            headers=headers,
            max_retries=self.max_upload_complete_attempts,
            retry_backoff_factor=self.retry_backoff_factor,
        )
        _raise_for_comet_bad_status_or_failure(
            result=result, operation="complete S3 direct upload"
        )

        return result

    def upload_s3_file_part(
        self,
        session: Session,
        url: str,
        file_data: bytes,
    ) -> UploadResult:
        # retrying also on 500 response status (see description: https://comet-ml.atlassian.net/browse/CM-10420)
        result = _request_with_retries(
            session=session,
            method="PUT",
            url=url,
            json_payload=None,
            data=file_data,
            headers=None,
            max_retries=self.max_s3_file_part_upload_attempts,
            retry_backoff_factor=self.retry_backoff_factor,
            retry_on_bad_response_status=True,
            allowed_retry_status_codes=[500],
        )
        return result


def _raise_for_comet_bad_status_or_failure(result: UploadResult, operation: str):
    if result.failed and result.response is not None:
        LOGGER.debug(
            "Bad response for %s, status: %d, text: %r",
            operation,
            result.response.status_code,
            result.response.text,
        )
        result.response.raise_for_status()

    if result.failed:
        if result.response is not None:
            LOGGER.warning(
                "Bad response received when trying to %s, code: %d, text: %r",
                operation,
                result.response.status_code,
                result.response.text,
            )
        raise S3UploadError(
            reason="Failed to %s due to recurrent connection error with Comet backend"
            % operation,
            due_connection_error=result.has_connection_error,
        )


def _request_with_retries(
    session: Session,
    method: str,
    url: str,
    json_payload: Optional[Dict[str, Any]],
    data: Optional[bytes],
    headers: Optional[Dict[str, Any]],
    max_retries: int,
    retry_backoff_factor: float,
    retry_on_bad_response_status: bool = False,
    allowed_retry_status_codes: Optional[List[int]] = None,
) -> UploadResult:
    retry_attempt = 0
    response = None
    while retry_attempt < max_retries:
        try:
            response = session.request(
                method=method,
                url=url,
                json=json_payload,
                data=data,
                headers=headers,
            )
            failed = False
        except (ConnectionError, requests.ConnectionError):
            failed = True
            response = None
            LOGGER.debug(
                "ConnectionError when do %r for URL %r. Attempt: %d of %d.",
                method,
                url,
                retry_attempt,
                max_retries,
                exc_info=True,
            )

        retry_attempt += 1

        if response is not None and response.status_code != 200:
            failed = True
            if not retry_on_bad_response_status:
                # do not retry on bad status code - fail immediately
                return UploadResult(
                    response=response,
                    retry_attempts=retry_attempt,
                    failed=True,
                    has_connection_error=False,
                )
            elif (
                allowed_retry_status_codes is not None
                and response.status_code not in allowed_retry_status_codes
            ):
                # status_code isn't in allowed retry status codes - fail immediately
                return UploadResult(
                    response=response,
                    retry_attempts=retry_attempt,
                    failed=True,
                    has_connection_error=False,
                )

        if not failed:
            return UploadResult(
                response=response,
                retry_attempts=retry_attempt,
                failed=False,
                has_connection_error=False,
            )

        if failed and retry_attempt < max_retries:
            backoff_time = calculate_backoff_time(
                backoff_factor=retry_backoff_factor, retry_attempt=retry_attempt
            )
            LOGGER.debug(
                "Failed to do %r, URL %r. Attempt: %d of %d. Retrying after: %r seconds",
                method,
                url,
                retry_attempt,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)

    # connection error was not resolved
    return UploadResult(
        response=response,
        retry_attempts=retry_attempt,
        failed=True,
        has_connection_error=response is None,
    )
