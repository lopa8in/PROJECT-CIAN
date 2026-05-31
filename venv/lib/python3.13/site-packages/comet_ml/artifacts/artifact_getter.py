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
import json
import logging
from typing import Dict, Optional

from ..artifacts.logged_artifact import LoggedArtifact
from ..config import Config
from ..connection import RestApiClient
from ..exceptions import (
    ArtifactNotFinalException,
    ArtifactNotFound,
    CometRestApiException,
    GetArtifactException,
    Unauthorized,
)
from ..logging_messages import (
    ARTIFACT_GET_FAILED_TO_DECODE_METADATA_WARNING,
    ARTIFACT_GET_GENERAL_EXCEPTION,
    ARTIFACT_GET_NOT_FINAL_EXCEPTION,
    ARTIFACT_GET_NOT_FOUND_EXCEPTION,
    ARTIFACT_GET_UNAUTHORIZED_EXCEPTION,
)
from ..summary import Summary

LOGGER = logging.getLogger(__name__)


def get_artifact(
    rest_api_client: RestApiClient,
    get_artifact_params: Dict[str, Optional[str]],
    experiment_id: str,
    summary: Summary,
    config: Config,
    raise_on_warning: bool = False,
) -> LoggedArtifact:
    """
    Retrieves artifact details using the given parameters, processes metadata, and
    returns a LoggedArtifact object. Handles specific exceptions related to the
    artifacts and logs warnings for metadata decoding issues.

    Args:
        rest_api_client :
            An instance of RestApiClient to interact with the API.
        get_artifact_params :
            Dictionary containing parameters required for fetching the artifact
            details.
        experiment_id :
            The experiment ID associated with the request.
        summary :
            A summary object to include in the resulting LoggedArtifact instance.
        config :
            Configuration object used for the artifact creation.
        raise_on_warning :
            Flag to determine whether to raise an exception for warnings, by default
            False.

    Returns:
        The resulting LoggedArtifact object with detailed metadata and attributes.

    Raises:
        GetArtifactException: Raised for general exceptions or unauthorized access errors.
        ArtifactNotFound: Raised if the requested artifact is not found.
        ArtifactNotFinalException: Raised if the requested artifact's version is not finalized.
    """
    try:
        result = rest_api_client.get_artifact_version_details(**get_artifact_params)
    except Unauthorized as e:
        raise GetArtifactException(
            ARTIFACT_GET_UNAUTHORIZED_EXCEPTION
            % (get_artifact_params.get("name"), get_artifact_params.get("workspace"))
        ) from e
    except CometRestApiException as e:
        if e.sdk_error_code == 624523:
            raise ArtifactNotFound(
                ARTIFACT_GET_NOT_FOUND_EXCEPTION % get_artifact_params
            ) from e
        if e.sdk_error_code == 90403 or e.sdk_error_code == 90402:
            raise ArtifactNotFinalException(
                ARTIFACT_GET_NOT_FINAL_EXCEPTION % get_artifact_params
            ) from e

        raise e
    except Exception as ex:
        raise GetArtifactException(ARTIFACT_GET_GENERAL_EXCEPTION) from ex

    artifact_name = result["artifact"]["artifactName"]
    artifact_version = result["artifactVersion"]
    artifact_metadata = result["metadata"]
    if artifact_metadata:
        try:
            artifact_metadata = json.loads(artifact_metadata)
        except Exception as e:
            if raise_on_warning:
                raise e
            LOGGER.warning(
                ARTIFACT_GET_FAILED_TO_DECODE_METADATA_WARNING,
                artifact_name,
                artifact_version,
                exc_info=True,
            )
            artifact_metadata = None

    return LoggedArtifact(
        aliases=result["alias"],
        artifact_id=result["artifact"]["artifactId"],
        artifact_name=artifact_name,
        artifact_tags=result["artifact"]["tags"],
        artifact_type=result["artifact"]["artifactType"],
        artifact_version_id=result["artifactVersionId"],
        config=config,
        experiment_key=experiment_id,  # TODO: Remove ME
        metadata=artifact_metadata,
        rest_api_client=rest_api_client,
        size=result["sizeInBytes"],
        source_experiment_key=result["experimentKey"],
        summary=summary,
        version_tags=result["tags"],
        version=artifact_version,
        workspace=result["artifact"]["workspaceName"],
    )
