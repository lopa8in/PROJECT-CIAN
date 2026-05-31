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
from typing import Any, Dict, NamedTuple, Optional

from comet_ml.config import DEFAULT_ASSET_UPLOAD_SIZE_LIMIT, DEFAULT_UPLOAD_SIZE_LIMIT
from comet_ml.constants import ASSET_TYPE_DEFAULT, ASSET_TYPE_VIDEO
from comet_ml.utils import log_once_at_level

LOGGER = logging.getLogger(__name__)


class ExperimentHandshakeResponse(NamedTuple):
    run_id: str
    project_id: Optional[str]
    is_github: bool
    focus_link: Optional[str]
    last_offset: int
    upload_limit: int
    video_upload_limit: int
    asset_upload_limit: int
    feature_toggles: Dict[str, bool]
    web_asset_url: Optional[str]
    web_image_url: Optional[str]
    api_asset_url: Optional[str]
    api_image_url: Optional[str]
    experiment_name: Optional[str]
    s3_direct_access_enabled: bool
    artifact_name_max_length: Optional[int]
    asset_name_max_length: Optional[int]
    image_name_max_length: Optional[int]


def _preprocess_upload_limit(
    value: Optional[int], upload_type: str, default_value: int
) -> int:
    if isinstance(value, int) and value > 0:
        # The limit is given in Mb, convert it back in bytes
        return value * 1024 * 1024
    else:
        LOGGER.debug(
            "Fallback to default %s upload size limit, %r value is invalid",
            upload_type,
            value,
        )
        return default_value


def parse_experiment_handshake_response(
    res_body: Dict[str, Any]
) -> ExperimentHandshakeResponse:
    run_id: str = res_body["runId"]

    project_id: Optional[str] = res_body.get("project_id", None)

    is_github = bool(res_body.get("githubEnabled", False))

    focus_link: Optional[str] = res_body.get("focusUrl", None)

    last_offset = res_body.get("lastOffset", 0)

    s3_direct_access_enabled = res_body.get("s3DirectAccessEnabled", False)

    # Upload limit
    upload_limit = _preprocess_upload_limit(
        res_body.get("upload_file_size_limit_in_mb", None),
        upload_type="",
        default_value=DEFAULT_UPLOAD_SIZE_LIMIT,
    )

    video_upload_limit = _preprocess_upload_limit(
        res_body.get("uploadVideoMaxSizeMB", None),
        upload_type=ASSET_TYPE_VIDEO,
        default_value=DEFAULT_UPLOAD_SIZE_LIMIT,
    )

    asset_upload_limit = _preprocess_upload_limit(
        res_body.get("asset_upload_file_size_limit_in_mb", None),
        upload_type=ASSET_TYPE_DEFAULT,
        default_value=DEFAULT_ASSET_UPLOAD_SIZE_LIMIT,
    )

    artifact_name_max_length = res_body.get("artifactNameMaxLength", None)
    asset_name_max_length = res_body.get("assetNameMaxLength", None)
    image_name_max_length = res_body.get("imageNameMaxLength", None)

    res_msg = res_body.get("msg")
    if res_msg:
        log_once_at_level(logging.INFO, res_msg)

    # Parse feature toggles
    feature_toggles: Dict[str, bool] = {}
    LOGGER.debug("Raw feature toggles %r", res_body.get("featureToggles", []))
    for toggle in res_body.get("featureToggles", []):
        try:
            feature_toggles[toggle["name"]] = bool(toggle["enabled"])
        except (KeyError, TypeError):
            LOGGER.debug("Invalid feature toggle: %s", toggle, exc_info=True)
    LOGGER.debug("Parsed feature toggles %r", feature_toggles)

    # Parse URL prefixes
    web_asset_url: Optional[str] = res_body.get("cometWebAssetUrl", None)
    web_image_url: Optional[str] = res_body.get("cometWebImageUrl", None)
    api_asset_url: Optional[str] = res_body.get("cometRestApiAssetUrl", None)
    api_image_url: Optional[str] = res_body.get("cometRestApiImageUrl", None)

    experiment_name: Optional[str] = res_body.get("name", None)

    return ExperimentHandshakeResponse(
        run_id=run_id,
        project_id=project_id,
        is_github=is_github,
        focus_link=focus_link,
        last_offset=last_offset,
        upload_limit=upload_limit,
        video_upload_limit=video_upload_limit,
        asset_upload_limit=asset_upload_limit,
        feature_toggles=feature_toggles,
        web_asset_url=web_asset_url,
        web_image_url=web_image_url,
        api_asset_url=api_asset_url,
        api_image_url=api_image_url,
        experiment_name=experiment_name,
        s3_direct_access_enabled=s3_direct_access_enabled,
        artifact_name_max_length=artifact_name_max_length,
        asset_name_max_length=asset_name_max_length,
        image_name_max_length=image_name_max_length,
    )
