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
from typing import NamedTuple, Optional


class ArtifactName(NamedTuple):
    """
    Represents the name and related details of an artifact.

    This class encapsulates the details of an artifact, including its workspace,
    name, and version or alias. It is designed to provide an easy-to-use structure
    for managing artifact identities and their associated metadata.
    """

    workspace: Optional[str]
    name: str
    version_or_alias: Optional[str]


def parse_artifact_name(
    artifact_name: str,
) -> ArtifactName:
    """
    Parses an artifact name string into its components: workspace, artifact name, and
    version or alias.

    The provided artifact name string is expected to follow a specific format where it
    can optionally include a workspace and a version or alias. This function splits the
    artifact name into its parts for further processing.

    Args:
        artifact_name:
            The name of the artifact, which may include workspace and version or alias
            information in the format "workspace/name:version" or "name:version".

    Returns:
        A tuple containing the following:
        - workspace:
            The workspace name, or None if no workspace is specified.
        - name:
            The name of the artifact.
        - version_or_alias:
            The version or alias of the artifact, or None if not specified.
    """

    name_parts = artifact_name.split("/")

    # First parse the workspace
    if len(name_parts) == 1:
        workspace = None
        artifact_name_version = name_parts[0]
    else:
        workspace = name_parts[0]
        artifact_name_version = name_parts[1]

    name_version_parts = artifact_name_version.split(":", 1)

    if len(name_version_parts) == 1:
        artifact_name = name_version_parts[0]
        version_or_alias = None
    else:
        artifact_name = name_version_parts[0]
        version_or_alias = name_version_parts[1]

    return ArtifactName(
        workspace=workspace, name=artifact_name, version_or_alias=version_or_alias
    )
