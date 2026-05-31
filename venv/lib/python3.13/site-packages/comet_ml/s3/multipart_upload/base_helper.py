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
from typing import Any, Dict, List, Optional, Union

from requests import Response, Session

from ...utils import encode_metadata
from .retry_strategy import UploadRetryStrategyOp

LOGGER = logging.getLogger(__name__)


class MultipartUploadMetadata:
    __slots__ = ["request_id", "parts_urls"]

    def __init__(self, request_id: str, parts_urls: List[str]):
        self.request_id = request_id
        self.parts_urls = parts_urls


class S3MultipartBaseHelper(object):
    def __init__(
        self,
        upload_start_url: str,
        upload_complete_url: str,
        parameters: Dict[str, Any],
        headers: Dict[str, Any],
        upload_retry_strategy: UploadRetryStrategyOp,
        expires_in: int,
        metadata: Optional[Dict[Any, Any]] = None,
    ):
        self.expires_in = expires_in
        self.parameters = parameters
        self.headers = headers
        self.metadata = metadata
        self.upload_start_url = upload_start_url
        self.upload_complete_url = upload_complete_url
        self.upload_retry_strategy_op = upload_retry_strategy

    def start_multipart_upload(
        self, session: Session, parts_number: int
    ) -> MultipartUploadMetadata:
        LOGGER.debug(
            "Requesting multipart start with url: %r, parts number: %d",
            self.upload_start_url,
            parts_number,
        )
        payload = {
            "startUploadParams": {
                "cloudType": "AWS",
                "preSignUrlTimeoutSec": self.expires_in,
                "numOfFileParts": parts_number,
            }
        }
        payload.update(self.parameters)
        if self.metadata is not None:
            encoded_metadata = encode_metadata(self.metadata)
            payload["metadata"] = encoded_metadata

        LOGGER.debug("Multipart start request body: %r", json.dumps(payload))

        result = self.upload_retry_strategy_op.start_multipart_upload(
            session=session,
            url=self.upload_start_url,
            payload=payload,
            headers=self.headers,
        )
        LOGGER.debug(
            "Got start multipart upload response - attempts made: %d, status: %d"
            % (result.retry_attempts, result.response.status_code)
        )

        # parse response
        response_json = result.response.json()

        LOGGER.debug("Start multipart upload response body: %r", response_json)

        request_id = response_json["requestId"]
        signed_urls = response_json["preSignUrl"]

        return MultipartUploadMetadata(request_id=request_id, parts_urls=signed_urls)

    def complete_multipart_upload(
        self,
        session: Session,
        upload_metadata: MultipartUploadMetadata,
        parts: List[Dict[str, Union[str, int]]],
        file_size: int,
        succeed: bool = True,
    ) -> Response:
        LOGGER.debug(
            "Requesting multipart complete with url: %r, request id: %r, parts number: %d",
            self.upload_complete_url,
            upload_metadata.request_id,
            len(parts),
        )
        payload = {
            "requestId": upload_metadata.request_id,
            "succeed": succeed,
            "fileSize": file_size,
        }
        if "apiKey" in self.parameters:
            payload["apiKey"] = self.parameters["apiKey"]

        if succeed:
            payload["uploadedFileParts"] = parts

        LOGGER.debug("Request body: %r", json.dumps(payload))

        result = self.upload_retry_strategy_op.complete_multipart_upload(
            session=session,
            url=self.upload_complete_url,
            payload=payload,
            headers=self.headers,
        )
        LOGGER.debug(
            "Got complete multipart upload response - attempts made: %d, status: %d, text: %s"
            % (result.retry_attempts, result.response.status_code, result.response.text)
        )

        return result.response
