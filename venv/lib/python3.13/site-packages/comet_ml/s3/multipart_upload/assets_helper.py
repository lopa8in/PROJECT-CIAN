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
import urllib.parse
from typing import Any, Dict, Optional

from .base_helper import S3MultipartBaseHelper
from .retry_strategy import UploadRetryStrategyOp

LOGGER = logging.getLogger(__name__)

SUFFIX_ASSET_UPLOAD_START = "asset/upload-start"
SUFFIX_ASSET_UPLOAD_COMPLETE = "asset/upload-complete"


class S3MultipartAssetsHelper(S3MultipartBaseHelper):
    def __init__(
        self,
        base_url: str,
        parameters: Dict[str, Any],
        headers: Dict[str, Any],
        upload_retry_strategy: UploadRetryStrategyOp,
        expires_in: int,
        asset_metadata: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__(
            upload_start_url=urllib.parse.urljoin(base_url, SUFFIX_ASSET_UPLOAD_START),
            upload_complete_url=urllib.parse.urljoin(
                base_url, SUFFIX_ASSET_UPLOAD_COMPLETE
            ),
            parameters=parameters,
            headers=headers,
            upload_retry_strategy=upload_retry_strategy,
            metadata=asset_metadata,
            expires_in=expires_in,
        )
