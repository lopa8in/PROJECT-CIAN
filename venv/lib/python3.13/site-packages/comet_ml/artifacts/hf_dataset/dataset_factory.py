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
import logging
import os
from typing import List, Optional

from comet_ml.artifacts.logged_artifact import LoggedArtifact
from comet_ml.logging_messages import (
    HF_DATASET_DOWNLOADING_DATASET,
    HF_DATASET_USING_CACHED,
)

from .helpers import (
    check_cached_dataset_files,
    dataset_download_dir,
    download_and_list_dataset_files,
    list_dataset_files,
)

LOGGER = logging.getLogger(__name__)


def get_hf_dataset(
    logged_artifact: LoggedArtifact,
    download_directory: Optional[str],
    pathname: str,
    recursive: bool,
    use_cached_dataset: bool,
) -> List[str]:
    """
    Retrieve and prepare a Hugging Face dataset for usage.

    This function manages the process of fetching and returning the necessary
    dataset files from a Comet dataset artifact. The process includes
    checking for cached files, downloading new ones if required, and listing
    all relevant dataset files based on the provided parameters.

    Arguments:
        logged_artifact: Represents the dataset artifact to be retrieved.
            It includes metadata essential for locating and managing the dataset.
        download_directory: Optional path specifying the base directory for
            downloading the dataset. If not provided, a default directory path
            might be used internally.
        pathname: Specifies the path specifications to filter or locate specific
            files in the dataset. Can include directory paths or filename masks.
        recursive: Boolean indicating whether to include files from subdirectories
            recursively. If true, all files matching the pathname in subdirectories
            will also be considered.
        use_cached_dataset: Boolean that determines whether previously downloaded
            dataset files should be used if available. If true and the files are
            found in the cache, downloading is skipped.

    Returns:
        List of strings representing the file paths included in the dataset,
        filtered and processed based on the provided parameters.
    """
    download_directory = dataset_download_dir(
        logged_artifact, base_directory=download_directory
    )
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    dataset_already_downloaded = use_cached_dataset and check_cached_dataset_files(
        download_directory, logged_artifact
    )
    if dataset_already_downloaded:
        LOGGER.info(HF_DATASET_USING_CACHED, download_directory)
        return list_dataset_files(
            download_directory, pathname=pathname, recursive=recursive
        )

    LOGGER.info(
        HF_DATASET_DOWNLOADING_DATASET,
        logged_artifact.name,
        download_directory,
    )

    return download_and_list_dataset_files(
        logged_artifact=logged_artifact,
        download_directory=download_directory,
        pathname=pathname,
        recursive=recursive,
    )
