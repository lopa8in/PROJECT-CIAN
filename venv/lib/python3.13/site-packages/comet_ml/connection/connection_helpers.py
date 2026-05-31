# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import math
from http import HTTPStatus
from typing import IO, Any, Dict, List, Optional

import requests

from ..batch_utils import MessageBatchItem
from ..constants import (
    PAYLOAD_EXPERIMENT_KEY,
    PAYLOAD_LOCAL_TIMESTAMP,
    PAYLOAD_OFFSET,
    PAYLOAD_OUTPUT,
    PAYLOAD_OUTPUT_LINES,
    PAYLOAD_RUN_CONTEXT,
    PAYLOAD_STDERR,
)
from ..exceptions import BACKEND_CUSTOM_ERROR, CometRestApiException
from ..file_downloader import FileDownloadSizeMonitor
from ..messages import StandardOutputMessage
from .http_session import API_KEY_HEADER

# the maximal retry backoff time
BACKOFF_MAX = 120


def split_output_by_breaks_and_length(
    output: str, max_line_length: int, keep_ends: bool = True
) -> List[str]:
    lines = []
    for line in output.splitlines(keepends=keep_ends):
        split_lines = split_line_by_length(line, max_line_length)
        lines.extend(split_lines)

    return lines


def split_line_by_length(line: str, max_line_length: int) -> List[str]:
    lines = []
    while len(line) > max_line_length:
        lines.append(line[:max_line_length])
        line = line[max_line_length:]

    if len(line) > 0:
        lines.append(line)

    return lines


def format_stdout_message_batch_items(
    batch_items: List[MessageBatchItem],
    timestamp: int,
    experiment_key: str,
    stderr: bool,
    max_line_length: int,
) -> Optional[Dict[str, Any]]:
    stdout_lines = []
    timestamp = int(timestamp * 1000)  # the Java format - milliseconds since epoch

    for item in batch_items:
        if not isinstance(item.message, StandardOutputMessage):
            continue

        if stderr != item.message.stderr:
            # different message type than requested
            continue

        lines = split_line_by_length(item.message.output, max_line_length)
        if len(lines) == 0:
            continue

        # message_id increases monotonically, but we need gaps in the sequence to allow inserting
        # new values between existing ones. To achieve this, we expand the numeric range by a factor of 1.5
        offset = int(math.ceil(item.message.message_id * 1.5))

        for i, line in enumerate(lines):
            if i > 0:
                offset += 1

            stdout_lines.append(
                {
                    PAYLOAD_STDERR: stderr,
                    PAYLOAD_OUTPUT: line,
                    PAYLOAD_LOCAL_TIMESTAMP: timestamp,
                    PAYLOAD_OFFSET: offset,
                }
            )

    if len(stdout_lines) == 0:
        return None

    payload = {
        PAYLOAD_EXPERIMENT_KEY: experiment_key,
        PAYLOAD_OUTPUT_LINES: stdout_lines,
        PAYLOAD_RUN_CONTEXT: None,
    }
    return payload


def format_message_batch_items(
    batch_items: List[MessageBatchItem], experiment_key: str
) -> Dict[str, Any]:
    """Encodes a list of messages into batch body dictionary to be used for batch endpoints"""
    messages_arr = []
    for item in batch_items:
        messages_arr.append(item.message.to_batch_message_dict())

    batch_body = {
        PAYLOAD_EXPERIMENT_KEY: experiment_key,
        "values": messages_arr,
    }
    return batch_body


def format_remote_assets_batch_items(
    batch_items: List[MessageBatchItem],
) -> Dict[str, Any]:
    messages_arr = []
    for item in batch_items:
        messages_arr.append(item.message.to_batch_message_dict())

    batch_body = {"remoteArtifactAssetRequests": messages_arr}
    return batch_body


def calculate_backoff_time(backoff_factor: float, retry_attempt: int) -> float:
    if retry_attempt <= 1:
        return 1

    backoff_value = backoff_factor * (2 ** (retry_attempt - 1))
    return min(BACKOFF_MAX, backoff_value)


def get_backend_custom_error_msg(response: requests.Response) -> Optional[str]:
    try:
        data = response.json()
        code = data.get("sdk_error_code")
        if code == BACKEND_CUSTOM_ERROR:
            return data.get("msg")
    except Exception:
        return None


def raise_for_status_code(response: requests.Response):
    if response.status_code >= HTTPStatus.BAD_REQUEST:
        raise CometRestApiException(response.request.method, response=response)


def get_clientlib_params(
    experiment_id: str, project_id: str, api_key: str
) -> Dict[str, str]:
    return {"experimentId": experiment_id, "projectId": project_id, "apiKey": api_key}


def get_clientlib_headers(experiment_id: str) -> Dict[str, str]:
    return {"X-COMET-DEBUG-EXPERIMENT-KEY": experiment_id}


def get_api_key_header(api_key: str) -> Dict[str, str]:
    return {API_KEY_HEADER: api_key}


def write_stream_response_to_file(
    responses: requests.Response,
    file_object: IO[bytes],
    monitor: Optional[FileDownloadSizeMonitor] = None,
) -> None:
    for chunk in responses.iter_content(chunk_size=1024 * 1024):
        bytes_written = file_object.write(chunk)
        if monitor:
            monitor.monitor_callback(bytes_written)
