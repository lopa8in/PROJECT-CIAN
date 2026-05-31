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

"""
Author: Guy Avnon

This module contains asset processing methods.

"""
import logging
from urllib.parse import urlparse

from .. import exceptions, file_uploader, gs_bucket_info, s3_bucket_info
from .._typing import Any, List, Optional, Tuple
from ..logging_messages import (
    FAILED_TO_ADD_REMOTE_MODEL_SYNC_ASSET,
    SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_MODEL,
)

ALWAYS_OVERWRITE = True
UNUSED = None
LOGGER = logging.getLogger(__name__)


def remote_model(
    uri: str, max_synced_objects: int, metadata: Any, raise_on_error: bool = False
) -> Tuple[bool, Optional[str], List[file_uploader.PreprocessedRemoteAsset]]:
    url_scheme = _get_uri_scheme(uri, raise_on_error=raise_on_error)

    preprocessed_assets = []

    try:
        if url_scheme == "s3":
            preprocessed_assets = s3_bucket_info.preprocess_remote_s3_assets(
                remote_uri=uri,
                logical_path=UNUSED,
                overwrite=ALWAYS_OVERWRITE,
                upload_type=UNUSED,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
        elif url_scheme == "gs":
            preprocessed_assets = gs_bucket_info.preprocess_remote_gs_assets(
                remote_uri=uri,
                logical_path=UNUSED,
                overwrite=ALWAYS_OVERWRITE,
                upload_type=UNUSED,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
        else:
            LOGGER.debug(SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_MODEL, uri)
            return (
                False,
                SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_MODEL % uri,
                preprocessed_assets,
            )

        return True, None, preprocessed_assets

    except exceptions.LogAssetException as lax:
        if raise_on_error:
            raise lax
        else:
            LOGGER.warning(lax.backend_err_msg, exc_info=True)
            error_message = lax.backend_err_msg
    except Exception as ex:
        if raise_on_error:
            raise ex
        else:
            LOGGER.warning(FAILED_TO_ADD_REMOTE_MODEL_SYNC_ASSET, uri, exc_info=True)
            error_message = FAILED_TO_ADD_REMOTE_MODEL_SYNC_ASSET % uri

    return False, error_message, preprocessed_assets


def _get_uri_scheme(uri: str, raise_on_error: bool) -> Optional[str]:
    try:
        parsed_url = urlparse(uri)
        return parsed_url.scheme
    except Exception as e:
        if raise_on_error:
            raise e
        else:
            LOGGER.warning(
                "Failed to parse remote model's URI '%s'", uri, exc_info=True
            )
            return None
