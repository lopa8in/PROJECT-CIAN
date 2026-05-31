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
import os
import tempfile
from typing import Any, Dict
from urllib.parse import urlparse

from ..assets.data_writers import AssetDataWriterFromGCS, AssetDataWriterFromS3
from ..cloud_storage_utils import META_SYNCED, META_VERSION_ID
from ..connection.connection_api import RestApiClient
from .artifact_asset import ArtifactAsset
from .artifact_assets_downloader import (
    download_artifact_asset,
    download_cloud_storage_artifact_asset,
    is_remote_uri_downloadable,
)
from .helpers import validate_overwrite_strategy


class LoggedArtifactAsset:
    """
    Represent assets logged to an Artifact
    """

    __slots__ = (
        "_remote",
        "_logical_path",
        "_size",
        "_link",
        "_metadata",
        "_asset_type",
        "_id",
        "_artifact_version_id",
        "_artifact_id",
        "_source_experiment_key",
        "_rest_api_client",
        "_download_timeout",
        "_logged_artifact_repr",
        "_logged_artifact_str",
        "_experiment_key",
        "_verify_tls",
    )

    def __init__(
        self,
        remote: bool,
        logical_path: str,
        size: int,
        link: str,
        metadata: Dict[str, Any],
        asset_type: str,
        id: str,
        artifact_version_id: str,
        artifact_id: str,
        source_experiment_key: str,
        verify_tls: bool,
        rest_api_client: RestApiClient = None,
        download_timeout: float = None,
        logged_artifact_repr: str = None,
        logged_artifact_str: str = None,
        experiment_key: str = None,
    ) -> None:
        self._remote = remote
        self._logical_path = logical_path
        self._size = size
        self._link = link
        self._metadata = metadata
        self._asset_type = asset_type
        self._id = id
        self._artifact_version_id = artifact_version_id
        self._artifact_id = artifact_id
        self._source_experiment_key = source_experiment_key

        self._rest_api_client = rest_api_client
        self._download_timeout = download_timeout
        self._verify_tls = verify_tls
        self._logged_artifact_repr = logged_artifact_repr
        self._logged_artifact_str = logged_artifact_str
        self._experiment_key = experiment_key

    @property
    def remote(self) -> bool:
        """Is the asset a remote asset or not, boolean"""
        return self._remote

    @property
    def logical_path(self) -> str:
        """Asset relative logical_path, str or None"""
        return self._logical_path

    @property
    def size(self) -> int:
        """Asset size if the asset is a non-remote asset, int"""
        return self._size

    @property
    def link(self) -> str:
        """Asset remote link if the asset is remote, str or None"""
        return self._link

    @property
    def metadata(self) -> Dict[str, Any]:
        """Asset metadata, dict"""
        return self._metadata

    @property
    def asset_type(self) -> str:
        """Asset type, str"""
        return self._asset_type

    @property
    def id(self) -> str:
        """Asset unique id, str"""
        return self._id

    @property
    def artifact_version_id(self) -> str:
        """Artifact version id, str"""
        return self._artifact_version_id

    @property
    def artifact_id(self) -> str:
        """Artifact id, str"""
        return self._artifact_id

    @property
    def source_experiment_key(self) -> str:
        """The experiment key of the experiment that logged this asset, str"""
        return self._source_experiment_key

    def is_remote_and_synced(self) -> bool:
        """Is the asset synced or not?"""
        if not self._remote:
            return False

        if self._metadata is None:
            return False

        asset_synced = self._metadata.get(META_SYNCED)
        return asset_synced is not None and asset_synced is True

    def is_asset_downloadable(self) -> bool:
        """
        Determines if an asset is downloadable.

        This method checks specific conditions to determine whether the asset can
        be downloaded. If the asset is not remote, it is deemed downloadable. For
        remote assets, it additionally checks if the asset is both remote and
        synced, as well as if the remote URI is available for download.

        Returns:
            bool: True if the asset is downloadable, False otherwise.
        """
        if not self._remote:
            return True

        if not self.is_remote_and_synced():
            return False

        return is_remote_uri_downloadable(self._link)

    def __repr__(self) -> str:
        return (
            "%s(remote=%r, logical_path=%r, size=%r, link=%r, metadata=%r, asset_type=%r, id=%r, artifact_version_id=%r, artifact_id=%r, source_experiment_key=%r, synced'%r)"
            % (
                self.__class__.__name__,
                self._remote,
                self._logical_path,
                self._size,
                self._link,
                self._metadata,
                self._asset_type,
                self._id,
                self._artifact_version_id,
                self._artifact_id,
                self._source_experiment_key,
                self.is_remote_and_synced(),
            )
        )

    def __eq__(self, other: "LoggedArtifactAsset") -> bool:
        return (
            self._remote == other._remote
            and self._logical_path == other._logical_path
            and self._size == other._size
            and self._link == other._link
            and self._metadata == other._metadata
            and self._asset_type == other._asset_type
            and self._id == other._id
            and self._artifact_version_id == other._artifact_version_id
            and self._artifact_id == other._artifact_id
            and self._source_experiment_key == other._source_experiment_key
        )

    def __lt__(self, other: "LoggedArtifactAsset") -> bool:
        return self._logical_path < other._logical_path

    def download(
        self,
        local_path: str = None,  # if None, downloads to a tmp path
        logical_path: str = None,
        overwrite_strategy=False,
    ) -> ArtifactAsset:
        """
        Download the asset to a given full path or directory

        Returns:
            The artifact asset downloaded

        Args:
            local_path: the root folder to which to download.
                if None, will download to a tmp path if str will be either a root local path or a
                full local path
            logical_path: the path relative to the root local_path to use. If None and
                local_path==None then no relative path is used, a file would just be a tmp path on
                local disk. If None and local_path!=None then the local_path will be treated as a
                root path, and the asset's logical_path will be appended to the root path to form a
                full local path. If "" or False then local_path will be used as a full path
                (local_path can also be None)
            overwrite_strategy: can be False, "FAIL", "PRESERVE" or "OVERWRITE"
                and follows the same semantics for overwrite strategy as artifact.download()
        """
        if local_path is None:
            root_path = tempfile.mkdtemp()
        else:
            root_path = local_path

        if logical_path is None:
            asset_filename = self._logical_path
        else:
            asset_filename = logical_path

        result_asset_path = os.path.join(root_path, asset_filename)

        if self.is_remote_and_synced() and is_remote_uri_downloadable(self._link):
            # download remote synced asset from cloud storage
            o = urlparse(self._link)

            if o.scheme == "s3":
                data_writer = AssetDataWriterFromS3(
                    s3_uri=self._link, version_id=self._metadata.get(META_VERSION_ID)
                )
            else:
                data_writer = AssetDataWriterFromGCS(
                    gs_uri=self._link, version_id=self._metadata.get(META_VERSION_ID)
                )

            download_cloud_storage_artifact_asset(
                data_writer=data_writer,
                asset_id=self._id,
                artifact_repr=self._logged_artifact_repr,
                artifact_str=self._logged_artifact_str,
                asset_logical_path=asset_filename,
                asset_path=result_asset_path,
                overwrite=validate_overwrite_strategy(overwrite_strategy),
            )
        else:
            # download from REST API
            prepared_request = self._rest_api_client._prepare_experiment_asset_request(
                asset_id=self._id,
                experiment_key=self._experiment_key,
                artifact_version_id=self._artifact_version_id,
            )

            download_artifact_asset(
                prepared_request=prepared_request,
                timeout=self._download_timeout,
                asset_id=self._id,
                artifact_repr=self._logged_artifact_repr,
                artifact_str=self._logged_artifact_str,
                asset_logical_path=asset_filename,
                asset_path=result_asset_path,
                overwrite=validate_overwrite_strategy(overwrite_strategy),
                verify_tls=self._verify_tls,
            )

        return ArtifactAsset(
            remote=False,
            logical_path=self._logical_path,
            size=self._size,
            link=None,
            metadata=self._metadata,
            asset_type=self._asset_type,
            local_path_or_data=result_asset_path,
        )
