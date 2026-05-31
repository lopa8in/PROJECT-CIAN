# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import glob
import os
from typing import List, Optional

from comet_ml.artifacts.logged_artifact import LoggedArtifact

from .metadata import (
    DATASET_METADATA_FILENAME,
    DataSetMetadata,
    read_dataset_metadata,
    write_dataset_metadata,
)


def dataset_download_dir(
    logged_artifact: LoggedArtifact, base_directory: Optional[str]
) -> str:
    if base_directory is None:
        base_directory = os.getcwd()

    return os.path.join(
        base_directory,
        os.path.normpath(logged_artifact.name),
        str(logged_artifact.version),
    )


def validate_cached_dataset_assets(
    logged_artifact: LoggedArtifact, dataset_metadata: DataSetMetadata
) -> bool:
    if len(dataset_metadata.assets) == 0:
        return False

    downloadable_assets = set(
        [asset.id for asset in logged_artifact.assets if asset.is_asset_downloadable()]
    )
    if len(downloadable_assets) != len(dataset_metadata.assets):
        return False

    return len(downloadable_assets.intersection(set(dataset_metadata.assets))) == len(
        dataset_metadata.assets
    )


def check_cached_dataset_files(
    dataset_path: str,
    logged_artifact: LoggedArtifact,
) -> bool:
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        return False

    # check dataset metadata
    dataset_metadata = read_dataset_metadata(dataset_path)
    if (
        dataset_metadata.name != logged_artifact.name
        or dataset_metadata.version != logged_artifact.version
    ):
        return False

    # validate dataset assets
    assets_valid = validate_cached_dataset_assets(logged_artifact, dataset_metadata)
    if not assets_valid:
        return False

    # check dataset files
    files = list_dataset_files(dataset_path, pathname="**", recursive=True)
    return len(files) == len(dataset_metadata.assets)


def list_dataset_files(
    dataset_directory: str,
    pathname: str,
    recursive: bool,
) -> List[str]:
    return [
        file
        for file in glob.glob(
            os.path.join(dataset_directory, pathname), recursive=recursive
        )
        if os.path.isfile(file) and os.path.basename(file) != DATASET_METADATA_FILENAME
    ]


def download_and_list_dataset_files(
    logged_artifact: LoggedArtifact,
    download_directory: str,
    pathname: str,
    recursive: bool,
) -> List[str]:
    artifact = logged_artifact.download(download_directory)
    artifact.version = logged_artifact.version
    write_dataset_metadata(download_directory, artifact)

    return list_dataset_files(
        download_directory, pathname=pathname, recursive=recursive
    )
