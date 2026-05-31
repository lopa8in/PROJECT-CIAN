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
import io
import logging
import os
import tempfile
from typing import Optional
from urllib.parse import urlparse

from comet_ml.parallel_utils import makedirs_synchronized

import requests

from .._typing import PreparedRequest
from ..assets.data_writers import AssetDataWriter, AssetDataWriterFromResponse
from ..connection.http_session import get_cached_comet_http_session
from ..exceptions import ArtifactDownloadException, CometRestApiException
from ..file_downloader import FileDownloadSizeMonitor
from ..file_utils import file_sha1sum, io_sha1sum
from ..logging_messages import (
    ARTIFACT_ASSET_DOWNLOAD_FAILED_REPR,
    ARTIFACT_DOWNLOAD_FILE_OVERWRITTEN,
    ARTIFACT_DOWNLOAD_READ_FILE_EXCEPTION,
    ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_ALREADY_EXISTS_EXCEPTION,
    ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_EXCEPTION,
    ASSET_WRITE_ERROR,
)

LOGGER = logging.getLogger(__name__)


def download_artifact_asset(
    prepared_request: PreparedRequest,
    timeout: float,
    asset_id: str,
    artifact_repr: str,
    artifact_str: str,
    asset_logical_path: str,
    asset_path: str,
    overwrite: str,
    verify_tls: bool,
    _monitor: Optional[FileDownloadSizeMonitor] = None,
) -> None:
    try:
        retry_session = get_cached_comet_http_session(
            retry=True,
            verify_tls=verify_tls,
            api_key=prepared_request.api_key,
        )

        response: requests.Response = retry_session.get(
            url=prepared_request.url,
            params=prepared_request.json,
            headers=prepared_request.headers,
            stream=True,
        )

        if response.status_code != 200:
            response.close()
            raise CometRestApiException("GET", response)
    except Exception:
        raise ArtifactDownloadException(
            ARTIFACT_ASSET_DOWNLOAD_FAILED_REPR % (asset_id, artifact_repr)
        )

    try:
        _write_artifact_asset_data_to_disk(
            artifact_str=artifact_str,
            asset_id=asset_id,
            asset_logical_path=asset_logical_path,
            asset_path=asset_path,
            overwrite=overwrite,
            writer=AssetDataWriterFromResponse(response=response, monitor=_monitor),
        )
    finally:
        try:
            response.close()
        except Exception:
            LOGGER.debug(
                "Error closing artifact asset download response", exc_info=True
            )
            pass


def download_cloud_storage_artifact_asset(
    data_writer: AssetDataWriter,
    asset_id: str,
    artifact_repr: str,
    artifact_str: str,
    asset_logical_path: str,
    asset_path: str,
    overwrite: str,
    _monitor: Optional[FileDownloadSizeMonitor] = None,
) -> None:
    try:
        data_writer.monitor = _monitor
        _write_artifact_asset_data_to_disk(
            artifact_str=artifact_str,
            asset_id=asset_id,
            asset_logical_path=asset_logical_path,
            asset_path=asset_path,
            overwrite=overwrite,
            writer=data_writer,
        )
    except Exception:
        LOGGER.debug("Error writing S3/GCS artifact asset file on path", exc_info=True)
        raise ArtifactDownloadException(
            ARTIFACT_ASSET_DOWNLOAD_FAILED_REPR % (asset_id, artifact_repr)
        )
    return None


def _write_artifact_asset_data_to_disk(
    artifact_str: str,
    asset_id: str,
    asset_logical_path: str,
    asset_path: str,
    overwrite: str,
    writer: AssetDataWriter,
) -> None:
    if os.path.isfile(asset_path):
        if overwrite == "OVERWRITE":
            LOGGER.warning(
                ARTIFACT_DOWNLOAD_FILE_OVERWRITTEN,
                asset_path,
                asset_logical_path,
                artifact_str,
            )
        elif overwrite == "PRESERVE":
            # TODO: Print LOG message if content is different when we have the SHA1 stored the
            # backend
            return
        else:
            # Download the file to a temporary file
            # TODO: Just compare the checksums
            try:
                existing_file_checksum = file_sha1sum(asset_path)
            except Exception:
                LOGGER.debug("Error computing sha1sum", exc_info=True)
                raise ArtifactDownloadException(
                    ARTIFACT_DOWNLOAD_READ_FILE_EXCEPTION % asset_path
                )

            try:
                with tempfile.NamedTemporaryFile() as f:
                    writer.write(file=f)

                    # Flush to be sure that everything is written
                    f.flush()
                    f.seek(0)

                    # Compute checksums
                    asset_checksum = io_sha1sum(f)

            except Exception:
                LOGGER.debug("Error write tmpfile to compute checksum", exc_info=True)
                raise ArtifactDownloadException(
                    ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_EXCEPTION
                    % (asset_id, asset_path)
                )

            if asset_checksum != existing_file_checksum:
                raise ArtifactDownloadException(
                    ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_ALREADY_EXISTS_EXCEPTION
                    % (asset_id, asset_path)
                )

            return None
    else:
        try:
            dirpart = os.path.dirname(asset_path)
            makedirs_synchronized(dirpart, exist_ok=True)
        except Exception:
            LOGGER.debug("Error creating directories", exc_info=True)
            raise ArtifactDownloadException(
                ASSET_WRITE_ERROR
                % (
                    asset_id,
                    asset_path,
                )
            )

    try:
        with io.open(asset_path, "wb") as f:
            writer.write(file=f)
    except Exception:
        LOGGER.debug("Error writing file on path", exc_info=True)
        raise ArtifactDownloadException(
            ASSET_WRITE_ERROR
            % (
                asset_id,
                asset_path,
            )
        )


def is_remote_uri_downloadable(remote_uri: str) -> bool:
    """
    Determines if a given remote URI is downloadable.

    This function checks the scheme of a provided URI to determine if it uses
    a supported protocol for downloading. Currently supported protocols are
    "s3" and "gs".
    """
    o = urlparse(remote_uri)
    return o.scheme == "s3" or o.scheme == "gs"
