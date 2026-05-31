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
import json
import os
import tempfile
from collections import namedtuple
from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

import semantic_version

from ..api import APIExperiment
from ..assets.data_writers import AssetDataWriterFromGCS, AssetDataWriterFromS3
from ..cloud_storage_utils import (
    META_ERROR_MESSAGE,
    META_FILE_SIZE,
    META_SYNCED,
    META_VERSION_ID,
)
from ..config import Config, get_check_tls_certificate
from ..connection.connection_api import RestApiClient
from ..constants import ASSET_TYPE_DEFAULT
from ..exceptions import ArtifactAssetNotFound, ArtifactDownloadException
from ..file_downloader import FileDownloadManager, FileDownloadManagerMonitor
from ..file_uploader import (
    PreprocessedAsset,
    PreprocessedFileAsset,
    PreprocessedRemoteAsset,
    PreprocessedSyncedRemoteAsset,
)
from ..logging_messages import (
    ARTIFACT_ASSET_DOWNLOAD_FAILED,
    ARTIFACT_DOWNLOAD_CANNOT_DOWNLOAD_ASSET_EXCEPTION,
    ARTIFACT_DOWNLOAD_CANNOT_GET_ASSETS_LIST_EXCEPTION,
    ARTIFACT_DOWNLOAD_FINISHED,
    ARTIFACT_DOWNLOAD_START_MESSAGE,
    ASSET_DOWNLOAD_FAILED_WITH_ERROR,
    UNSUPPORTED_URI_SYNCED_REMOTE_ASSET,
)
from ..summary import Summary
from ..utils import ImmutableDict, wait_for_done
from .artifact import Artifact
from .artifact_assets_downloader import (
    download_artifact_asset,
    download_cloud_storage_artifact_asset,
    is_remote_uri_downloadable,
)
from .helpers import validate_overwrite_strategy
from .logged_artifact_asset import LoggedArtifactAsset

LOGGER = getLogger(__name__)


class LoggedArtifact:
    """
    Represents a logged artifact within a workspace and provides utility methods to
    interact with its assets, versions, and metadata.

    This class serves as a representation of an artifact logged in a workspace, enabling
    users to retrieve associated assets, download them, and query metadata or other
    details about the artifact. To create or fetch this object, utilize the
    `comet_ml.Experiment.get_artifact` method rather than instantiating this class directly.

    Note:
        You shouldn't try to create this object by hand, please use
        [comet_ml.Experiment.get_artifact()][] instead to retrieve an artifact.
    """

    def __init__(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_id: str,
        artifact_version_id: str,
        workspace: str,
        rest_api_client: RestApiClient,
        experiment_key: str,
        version: str,
        aliases: List[str],
        artifact_tags: List[str],
        version_tags: List[str],
        size: int,
        metadata: Dict[str, Any],
        source_experiment_key: str,
        summary: Summary,
        config: Config,
    ) -> None:
        # Artifact fields
        self._artifact_type = artifact_type
        self._name = artifact_name
        self._artifact_id = artifact_id
        self._artifact_version_id = artifact_version_id

        self._version = semantic_version.Version(version)
        self._aliases = frozenset(aliases)
        self._rest_api_client = rest_api_client
        self._workspace = workspace
        self._artifact_tags = frozenset(artifact_tags)
        self._version_tags = frozenset(version_tags)
        self._size = size
        self._source_experiment_key = source_experiment_key
        self._experiment_key = experiment_key  # TODO: Remove ME
        self._summary = summary
        self._config = config

        self._raw_assets_list = None

        if metadata is not None:
            self._metadata = ImmutableDict(metadata)
        else:
            self._metadata = ImmutableDict()

    def _raw_assets(self):
        """Returns the artifact version ID assets"""
        return self._rest_api_client.get_artifact_files(
            workspace=self._workspace,
            name=self._name,
            version=str(self.version),
        )["files"]

    def _to_logged_artifact_asset(
        self, raw_artifact_asset: Dict[str, Any]
    ) -> LoggedArtifactAsset:
        if "remote" in raw_artifact_asset:
            remote = raw_artifact_asset["remote"]
        else:
            remote = (
                raw_artifact_asset["link"] is not None
            )  # TODO: Remove me after October 1st

        asset_metadata = raw_artifact_asset.get("metadata")
        if asset_metadata is not None and isinstance(asset_metadata, str):
            try:
                asset_metadata = json.loads(asset_metadata)
            except Exception:
                LOGGER.warning(
                    "Failed to parse metadata for artifact asset %s",
                    raw_artifact_asset["fileName"],
                    exc_info=True,
                )
                asset_metadata = None

        return LoggedArtifactAsset(
            remote=remote,
            logical_path=raw_artifact_asset["fileName"],
            size=raw_artifact_asset["fileSize"],
            link=raw_artifact_asset["link"],
            metadata=asset_metadata,
            asset_type=raw_artifact_asset["type"],
            id=raw_artifact_asset["assetId"],
            artifact_version_id=self._artifact_version_id,
            artifact_id=self._artifact_id,
            source_experiment_key=self._source_experiment_key,
            verify_tls=self._config.get_bool(
                None, "comet.internal.check_tls_certificate"
            ),
            rest_api_client=self._rest_api_client,
            download_timeout=self._config.get_int(None, "comet.timeout.file_download"),
            logged_artifact_repr=self.__repr__(),
            logged_artifact_str=self.__str__(),
            experiment_key=self._experiment_key,
        )

    @property
    def assets(self) -> List[LoggedArtifactAsset]:
        """
        The list of `LoggedArtifactAsset` that have been logged with this `LoggedArtifact`.
        """
        artifact_version_assets = []

        if self._raw_assets_list is None:
            self._raw_assets_list = self._raw_assets()

        for asset in self._raw_assets_list:
            artifact_version_assets.append(self._to_logged_artifact_asset(asset))

        return artifact_version_assets

    @property
    def remote_assets(self) -> List[LoggedArtifactAsset]:
        """
        The list of remote `LoggedArtifactAsset` that have been logged with this `LoggedArtifact`.
        """
        artifact_version_assets = []

        for asset in self._raw_assets():
            if "remote" in asset:
                remote = asset["remote"]
            else:
                remote = asset["link"] is not None  # TODO: Remove me after October 1st

            if not remote:
                continue

            artifact_version_assets.append(self._to_logged_artifact_asset(asset))

        return artifact_version_assets

    def get_asset(self, asset_logical_path) -> LoggedArtifactAsset:
        """
        Returns the LoggedArtifactAsset object matching the given asset_logical_path or raises an Exception
        """
        for asset in self._raw_assets():
            if asset["fileName"] == asset_logical_path:
                return self._to_logged_artifact_asset(asset)

        raise ArtifactAssetNotFound(asset_logical_path, self)

    def download(
        self,
        path: Optional[str] = None,
        overwrite_strategy: Union[bool, str] = False,
        sync_mode: bool = True,
    ) -> Artifact:
        """
        Download the current Artifact Version assets to a given directory (or the local directory by
        default).

        This method downloads assets and remote assets that were synced from a compatible cloud
        object storage (AWS S3 or GCP GCS). Other non-remote assets are not downloaded and you can
        access their link with the `artifact.assets` property.

        Args:
            path (str): Where to download artifact version assets. If not provided,
                a temporary path will be used, the root path can be accessed through the Artifact
                object which is returned by download under the `.download_local_path` attribute.
            overwrite_strategy (bool | str): One of the three possible strategies to handle
                conflict when trying to download an artifact version asset to a path with an
                existing file. See below for allowed values. Default is False or "FAIL".
            sync_mode (bool): Enables download of remote assets from the cloud storage platforms
                (AWS S3, GCP GS).

        Returns:
            Artifact: The Artifact downloaded object

        Note:
            Overwrite strategy allowed values:

            - False or "FAIL": If a file already exists and its content is different, raise the
                `comet_ml.exceptions.ArtifactDownloadException`.
            - "PRESERVE": If a file already exists
                and its content is different, show a WARNING but preserve the existing content.
            - True or "OVERWRITE": If a file already exists and its content is different, replace it by the
                asset version asset.
        """

        if path is None:
            root_path = tempfile.mkdtemp()
        else:
            root_path = path

        overwrite_strategy = validate_overwrite_strategy(overwrite_strategy)

        new_artifact_assets: Dict[str, PreprocessedAsset] = {}
        new_artifact_asset_ids = set()

        try:
            if self._raw_assets_list is None:
                self._raw_assets_list = self._raw_assets()
        except Exception:
            raise ArtifactDownloadException(
                ARTIFACT_DOWNLOAD_CANNOT_GET_ASSETS_LIST_EXCEPTION % self
            )

        worker_cpu_ratio = self._config.get_int(
            None, "comet.internal.file_upload_worker_ratio"
        )
        worker_count = self._config.get_raw(None, "comet.internal.worker_count")
        download_manager = FileDownloadManager(
            worker_cpu_ratio=worker_cpu_ratio, worker_count=worker_count
        )

        file_download_timeout = self._config.get_int(
            None, "comet.timeout.file_download"
        )
        verify_tls = get_check_tls_certificate(self._config)

        download_result_holder = namedtuple(
            "_download_result_holder",
            [
                "download_result",
                "asset_filename",
                "asset_path",
                "asset_metadata",
                "asset_id",
                "asset_synced",
                "asset_type",
                "asset_overwrite_strategy",
                "asset_remote_uri",
            ],
        )
        results: List[download_result_holder] = list()

        self_repr = repr(self)
        self_str = str(self)

        for asset in self._raw_assets_list:
            asset_metadata = asset["metadata"]
            if asset_metadata is not None:
                asset_metadata = json.loads(asset["metadata"])

            if "remote" in asset:
                asset_remote = asset["remote"]
            else:
                asset_remote = (
                    asset["link"] is not None
                )  # TODO: Remove me after October 1st

            remote_uri = asset.get("link", None)
            asset_filename = asset["fileName"]
            asset_id = asset["assetId"]
            asset_path = os.path.join(root_path, asset_filename)
            asset_synced = False
            asset_sync_error = None
            asset_type = asset.get("type", ASSET_TYPE_DEFAULT)
            if asset_metadata is not None:
                if META_SYNCED in asset_metadata:
                    asset_synced = asset_metadata[META_SYNCED]
                if META_ERROR_MESSAGE in asset_metadata:
                    asset_sync_error = asset_metadata[META_ERROR_MESSAGE]

            if asset_remote is True:
                # check if sync_mode is not enabled or asset was not synced properly
                if sync_mode is False or asset_synced is False:
                    # check if error is in metadata - failed to sync during upload due cloud storage error
                    if asset_sync_error is not None and sync_mode is True:
                        # raise error only if sync_mode==True
                        raise ArtifactDownloadException(
                            ASSET_DOWNLOAD_FAILED_WITH_ERROR
                            % (asset_filename, asset_sync_error)
                        )

                    # We don't download plain remote assets
                    new_artifact_assets[asset_filename] = PreprocessedRemoteAsset(
                        remote_uri=remote_uri,
                        overwrite=False,
                        upload_type=asset_type,
                        metadata=asset_metadata,
                        step=None,
                        asset_id=asset_id,
                        logical_path=asset_filename,
                        size=len(asset["link"]),
                    )
                    new_artifact_asset_ids.add(asset_id)
                    self._summary.increment_section("downloads", "artifact assets")
                else:
                    # check that asset is from supported cloud storage if sync_mode enabled
                    # and asset was synced during Artifact upload
                    o = urlparse(remote_uri)
                    if is_remote_uri_downloadable(remote_uri):
                        # register download from AWS S3 or GCS
                        if META_FILE_SIZE in asset_metadata:
                            asset_file_size = asset_metadata[META_FILE_SIZE]
                        else:
                            asset_file_size = 0

                        version_id = None
                        if META_VERSION_ID in asset_metadata:
                            version_id = asset_metadata[META_VERSION_ID]

                        if o.scheme == "s3":
                            data_writer = AssetDataWriterFromS3(
                                s3_uri=remote_uri, version_id=version_id
                            )
                        else:
                            data_writer = AssetDataWriterFromGCS(
                                gs_uri=remote_uri, version_id=version_id
                            )

                        result = download_manager.download_file_async(
                            download_cloud_storage_artifact_asset,
                            data_writer=data_writer,
                            estimated_size=asset_file_size,
                            asset_id=asset_id,
                            artifact_repr=self_repr,
                            artifact_str=self_str,
                            asset_logical_path=asset_filename,
                            asset_path=asset_path,
                            overwrite=overwrite_strategy,
                        )
                        results.append(
                            download_result_holder(
                                download_result=result,
                                asset_filename=asset_filename,
                                asset_path=asset_path,
                                asset_metadata=asset_metadata,
                                asset_id=asset_id,
                                asset_synced=asset_synced,
                                asset_type=asset_type,
                                asset_overwrite_strategy=overwrite_strategy,
                                asset_remote_uri=remote_uri,
                            )
                        )
                    else:
                        # unsupported URI scheme for synced asset
                        raise ArtifactDownloadException(
                            UNSUPPORTED_URI_SYNCED_REMOTE_ASSET % remote_uri
                        )
            else:
                prepared_request = (
                    self._rest_api_client._prepare_experiment_asset_request(
                        asset_id=asset_id,
                        experiment_key=self._experiment_key,
                        artifact_version_id=asset["artifactVersionId"],
                    )
                )

                # register asset to be downloaded
                result = download_manager.download_file_async(
                    download_artifact_asset,
                    prepared_request=prepared_request,
                    timeout=file_download_timeout,
                    verify_tls=verify_tls,
                    asset_id=asset_id,
                    artifact_repr=self_repr,
                    artifact_str=self_str,
                    asset_logical_path=asset_filename,
                    asset_path=asset_path,
                    overwrite=overwrite_strategy,
                    estimated_size=asset["fileSize"],
                )

                results.append(
                    download_result_holder(
                        download_result=result,
                        asset_filename=asset_filename,
                        asset_path=asset_path,
                        asset_metadata=asset_metadata,
                        asset_id=asset_id,
                        asset_synced=asset_synced,
                        asset_type=asset_type,
                        asset_overwrite_strategy=overwrite_strategy,
                        asset_remote_uri=remote_uri,
                    )
                )

        # Forbid new usage
        download_manager.close()

        # Wait for download manager to complete registered file downloads
        if not download_manager.all_done():
            monitor = FileDownloadManagerMonitor(download_manager)

            LOGGER.info(
                ARTIFACT_DOWNLOAD_START_MESSAGE,
                self._workspace,
                self._name,
                self._version,
            )

            wait_for_done(
                check_function=monitor.all_done,
                timeout=self._config.get_int(None, "comet.timeout.artifact_download"),
                progress_callback=monitor.log_remaining_downloads,
                sleep_time=15,
            )

        # iterate over download results and create file assets descriptors
        try:
            for result in results:
                try:
                    result.download_result.get(file_download_timeout)

                    new_asset_size = os.path.getsize(result.asset_path)
                except Exception:
                    # display failed message
                    LOGGER.error(
                        ARTIFACT_ASSET_DOWNLOAD_FAILED,
                        result.asset_filename,
                        self._workspace,
                        self._name,
                        self._version,
                        exc_info=True,
                    )

                    raise ArtifactDownloadException(
                        ARTIFACT_DOWNLOAD_CANNOT_DOWNLOAD_ASSET_EXCEPTION
                        % (result.asset_filename, self_repr)
                    )

                self._summary.increment_section(
                    "downloads",
                    "artifact assets",
                    size=new_asset_size,
                )

                if result.asset_synced is False:
                    # downloaded local asset
                    new_artifact_assets[result.asset_filename] = PreprocessedFileAsset(
                        local_path_or_data=result.asset_path,
                        upload_type=result.asset_type,
                        logical_path=result.asset_filename,
                        metadata=result.asset_metadata,
                        overwrite=result.asset_overwrite_strategy,
                        step=None,
                        asset_id=result.asset_id,
                        grouping_name=None,  # TODO: FIXME?
                        extension=None,  # TODO: FIXME?
                        size=new_asset_size,
                        copy_to_tmp=False,
                    )
                else:
                    # downloaded synced remote asset from cloud storage (AWS S3, GCS)
                    new_artifact_assets[result.asset_filename] = (
                        PreprocessedSyncedRemoteAsset(
                            remote_uri=result.asset_remote_uri,
                            overwrite=result.asset_overwrite_strategy,
                            upload_type=result.asset_type,
                            metadata=result.asset_metadata,
                            step=None,
                            asset_id=result.asset_id,
                            logical_path=result.asset_filename,
                            size=new_asset_size,
                            local_path=result.asset_path,
                        )
                    )

                new_artifact_asset_ids.add(result.asset_id)

            # display success message
            LOGGER.info(
                ARTIFACT_DOWNLOAD_FINISHED, self._workspace, self._name, self._version
            )
        finally:
            download_manager.join()

        return Artifact._from_logged_artifact(
            name=self._name,
            artifact_type=self._artifact_type,
            assets=new_artifact_assets,
            root_path=root_path,
            asset_ids=new_artifact_asset_ids,
        )

    def get_source_experiment(
        self,
        api_key: Optional[str] = None,
        cache: bool = True,
    ) -> APIExperiment:
        """
        Returns an APIExperiment object pointing to the experiment that created this artifact version, assumes that the API key is set else-where.
        """
        return APIExperiment(
            api_key=api_key,
            cache=cache,
            previous_experiment=self._source_experiment_key,
        )

    def update_artifact_tags(self, new_artifact_tags: Sequence[str]) -> None:
        """
        Update the logged artifact tags
        """
        new_artifact_tags_list = list(new_artifact_tags)

        self._rest_api_client.update_artifact(
            self._artifact_id,
            tags=new_artifact_tags,
        )

        self._artifact_tags = frozenset(new_artifact_tags_list)

    def update_version_tags(self, new_version_tags: Sequence[str]) -> None:
        """
        Update the logged artifact version tags
        """
        new_version_tags_list = list(new_version_tags)

        self._rest_api_client.update_artifact_version(
            self._artifact_version_id,
            version_tags=new_version_tags_list,
        )

        self._version_tags = frozenset(new_version_tags_list)

    def update_aliases(self, new_aliases: Sequence[str]) -> None:
        """
        Update the logged artifact tags
        """
        new_aliases_list = list(new_aliases)

        self._rest_api_client.update_artifact_version(
            self._artifact_version_id,
            version_aliases=new_aliases_list,
        )

        self._aliases = frozenset(new_aliases_list)

    # Public properties
    @property
    def name(self):
        """
        The logged artifact name.
        """
        return self._name

    @property
    def artifact_type(self):
        """
        The logged artifact type.
        """
        return self._artifact_type

    @property
    def version(self) -> semantic_version.Version:
        """
        The logged artifact version, as a SemanticVersion. See
        https://python-semanticversion.readthedocs.io/en/latest/reference.html#semantic_version.Version
        for reference
        """
        return self._version

    @property
    def workspace(self):
        """
        The logged artifact workspace name.
        """
        return self._workspace

    @property
    def aliases(self):
        """
        The set of logged artifact aliases.
        """
        return self._aliases

    @property
    def metadata(self):
        """
        The logged artifact metadata.
        """
        return self._metadata

    @property
    def version_tags(self):
        """
        The set of logged artifact version tags.
        """
        return self._version_tags

    @property
    def artifact_tags(self):
        """
        The set of logged artifact tags.
        """
        return self._artifact_tags

    @property
    def size(self):
        """
        The total size of logged artifact version; it is the sum of all the artifact version assets.
        """
        return self._size

    @property
    def source_experiment_key(self):
        """
        The experiment key of the experiment that created this LoggedArtifact.
        """
        return self._source_experiment_key

    def __str__(self):
        return "<%s '%s/%s:%s'>" % (
            self.__class__.__name__,
            self._workspace,
            self._name,
            self._version,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(artifact_name=%r, artifact_type=%r, workspace=%r, version=%r, aliases=%r, artifact_tags=%r, version_tags=%r, size=%r, source_experiment_key=%r)"
            % (
                self._name,
                self._artifact_type,
                self._workspace,
                self._version,
                self._aliases,
                self._artifact_tags,
                self._version_tags,
                self._size,
                self._source_experiment_key,
            )
        )
