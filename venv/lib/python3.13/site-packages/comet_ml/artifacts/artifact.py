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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import semantic_version

from ..cloud_storage_utils import META_ERROR_MESSAGE, META_SYNCED
from ..constants import ASSET_TYPE_DEFAULT
from ..debug import debug_helpers
from ..exceptions import ArtifactConflictingAssetLogicalPath, LogAssetException
from ..file_uploader import (
    FileUpload,
    FolderUpload,
    MemoryFileUpload,
    PreprocessedAsset,
    PreprocessedAssetFolder,
    PreprocessedRemoteAsset,
    PreprocessedSyncedRemoteAsset,
    dispatch_user_file_upload,
    preprocess_asset_file,
    preprocess_asset_folder,
    preprocess_asset_memory_file,
    preprocess_remote_asset,
)
from ..gs_bucket_info import preprocess_remote_gs_assets
from ..logging_messages import (
    ARTIFACT_ASSET_TYPE_DEPRECATED_WARNING,
    ARTIFACT_FAILED_TO_PARSE_REMOTE_ASSET_URI_WARNING,
    FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
    SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_ARTIFACT,
)
from ..s3_bucket_info import preprocess_remote_s3_assets
from ..utils import generate_guid
from ..validation.metadata_validator import validate_metadata
from .artifact_asset import ArtifactAsset

LOGGER = logging.getLogger(__name__)


class Artifact:
    """
    Comet Artifacts allow keeping track of assets beyond any particular experiment. You can keep
    track of Artifact versions, create many types of assets, manage them, and use them in any
    step in your ML pipelines---from training to production deployment.

    Artifacts live in a Comet Project are identified by their name and version string number.

    Example how to log an artifact with an asset:

    ```python
    from comet_ml import Artifact, start

    experiment = start()
    artifact = Artifact("Artifact-Name", "Artifact-Type")
    artifact.add("local-file")

    experiment.log_artifact(artifact)
    experiment.end()
    ```

    Example how to get and download all artifact assets:

    ```python
    from comet_ml import start

    experiment = start()
    artifact = experiment.get_artifact("Artifact-Name", WORKSPACE, PROJECT_NAME)

    artifact.download("/data/input")
    ```

    The artifact is created on the frontend only when calling [comet_ml.CometExperiment.log_artifact][]

    Attributes:
        name: The name of the artifact.
        artifact_type: The type of the artifact, e.g., 'dataset' or 'model'. Defaults to 'data'.
        version: The specific version of the artifact. If not provided, it is auto-generated.
        aliases: A set of aliases associated with the artifact version for easy reference.
        metadata: Additional data or annotations attached to the artifact version.
        version_tags: Tags associated with the artifact version for categorization.
    """

    def __init__(
        self,
        name: str,
        artifact_type: Optional[str] = None,
        version: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
        metadata: Any = None,
        version_tags: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Create a new Artifact object with the given name and optional version number. If no version
        is provided, one will be generated automatically. Also, you can provide aliases and tags as
        well as additional metadata to attach to the artifact version.

        Args:
            name: The artifact name. Exceeding 100 character lengths will cause an exception.
            artifact_type: The artifact-type, for example, `dataset`.
            version: The version number to create. If not provided, a new version number
                will be created automatically.
            aliases: Some aliases to attach to the future Artifact
                Version. The aliases list is converted into a set for de-duplication.
            metadata: Some additional data to attach to the future Artifact Version. Must
                be a JSON-encodable dict.
        """

        # Artifact fields
        self.name = name

        # Upsert fields
        if artifact_type is None:
            self.artifact_type = "data"
        else:
            self.artifact_type = artifact_type

        if version is None:
            self.version = None
        else:
            self.version = semantic_version.Version(version)

        self.version_tags: Set[str] = set()
        if version_tags is not None:
            self.version_tags = set(version_tags)

        self.aliases: Set[str] = set()
        if aliases is not None:
            self.aliases = set(aliases)

        self.metadata = validate_metadata(metadata, raise_on_invalid=True)

        self._assets: Dict[str, PreprocessedAsset] = {}

        # The set of assets IDs that was already downloaded through LoggedArtifact.download
        self._downloaded_asset_ids: Set[str] = set()

        self._download_local_path: Optional[str] = None

    @classmethod
    def _from_logged_artifact(
        cls,
        name: str,
        artifact_type: str,
        assets: Dict[str, PreprocessedAsset],
        root_path: str,
        asset_ids: Set[str],
    ) -> "Artifact":
        new_artifact = cls(name, artifact_type)
        new_artifact._assets = assets
        new_artifact._download_local_path = root_path
        new_artifact._downloaded_asset_ids = asset_ids

        return new_artifact

    def add(
        self,
        local_path_or_data: Any,
        logical_path: Optional[str] = None,
        overwrite: bool = False,
        copy_to_tmp: bool = True,  # if local_path_or_data is a file pointer
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a local asset to the current pending artifact object.

        Args:
            local_path_or_data (str | File-like):Either a file/directory path of the files you want
                to log, or a file-like asset.
            logical_path (str): A custom file name to be displayed. If not
                provided the filename from the `local_path_or_data` argument will be used.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            copy_to_tmp (bool): If `local_path_or_data` is a file-like object, then this flag determines
                if the file is first copied to a temporary file before upload. If
                `copy_to_tmp` is False, then it is sent directly to the cloud.
            metadata (dict): Some additional data to attach to the the audio asset. Must be a
                JSON-encodable dict.
        """
        if local_path_or_data is None:
            raise TypeError("local_path_or_data cannot be None")

        dispatched = dispatch_user_file_upload(local_path_or_data)

        if not isinstance(dispatched, (FileUpload, FolderUpload, MemoryFileUpload)):
            raise ValueError(
                "Invalid file_data %r, must either be a valid file-path or an IO object"
                % local_path_or_data
            )

        if isinstance(dispatched, FileUpload):
            asset_id = generate_guid()
            preprocessed = preprocess_asset_file(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                file_name=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                asset_id=asset_id,
                copy_to_tmp=copy_to_tmp,
            )
        elif isinstance(dispatched, FolderUpload):
            preprocessed = preprocess_asset_folder(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                logical_path=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
            )
        else:
            preprocessed = preprocess_asset_memory_file(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                file_name=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
            )

        if isinstance(preprocessed, PreprocessedAssetFolder):
            self._add_preprocessed_folder(preprocessed)
        else:
            self._add_preprocessed(preprocessed)

    def add_remote(
        self,
        uri: str,
        logical_path: Optional[str] = None,
        overwrite: bool = False,
        asset_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync_mode: bool = True,
        max_synced_objects: int = 10000,
    ) -> None:
        """
        Add a remote asset to the current pending artifact object. A Remote Asset is an asset but
        its content is not uploaded and stored on Comet. Rather a link for its location is stored, so
        you can identify and distinguish between two experiment using different version of a dataset
        stored somewhere else.

        Args:
            uri (str): The remote asset location, there is no imposed format and it could be a
                private link.
            logical_path (str): The "name" of the remote asset, could be a dataset
                name, a model file name.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            asset_type (str): Define the type of the asset - Deprecated.
            metadata (dict): Some additional data to attach to the remote asset.
                Must be a JSON-encodable dict.
            sync_mode (bool): If True and the URI begins with s3://, Comet attempts to list all
                objects in the given bucket and path. Each object will be logged as a separate
                remote asset. If object versioning is enabled on the S3 bucket, Comet also logs each
                object version to be able to download the exact version. If False, Comet just logs a
                single remote asset with the provided URI as the remote URI. Default is True.
            max_synced_objects (int): When sync_mode is True and the URI begins with s3://, set the
                maximum number of S3 objects to log. If there are more matching S3 objects than
                max_synced_objects, a warning will be displayed and the provided URI will be logged
                as a single remote asset.
        """
        if asset_type:
            debug_helpers.log_warning_or_raise(
                ARTIFACT_ASSET_TYPE_DEPRECATED_WARNING, logger=LOGGER
            )

        asset_type = None

        if sync_mode is True:
            url_scheme = None
            try:
                o = urlparse(uri)
                url_scheme = o.scheme
            except Exception as e:
                debug_helpers.log_warning_or_raise(
                    ARTIFACT_FAILED_TO_PARSE_REMOTE_ASSET_URI_WARNING,
                    uri,
                    exc_info=True,
                    logger=LOGGER,
                    original_exception=e,
                )

            error_message = None
            success = False
            if url_scheme == "s3":
                success, error_message = self._add_s3_assets(
                    uri=uri,
                    max_synced_objects=max_synced_objects,
                    logical_path=logical_path,
                    overwrite=overwrite,
                    asset_type=asset_type,
                    metadata=metadata,
                )
            elif url_scheme == "gs":
                success, error_message = self._add_gs_assets(
                    uri=uri,
                    max_synced_objects=max_synced_objects,
                    logical_path=logical_path,
                    overwrite=overwrite,
                    asset_type=asset_type,
                    metadata=metadata,
                )
            else:
                # log debug warning
                LOGGER.debug(SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_ARTIFACT, uri)

            if success is True:
                # to avoid logging this artifact as plain artifact beneath
                return

            # append error message to the metadata
            if error_message is not None:
                # add to metadata
                if metadata is None:
                    metadata = dict()
                metadata[META_ERROR_MESSAGE] = error_message
                metadata[META_SYNCED] = False

        # process asset as usually
        preprocessed = preprocess_remote_asset(
            remote_uri=uri,
            logical_path=logical_path,
            overwrite=overwrite,
            upload_type=asset_type,
            metadata=metadata,
        )
        self._add_preprocessed(preprocessed)

    def _add_gs_assets(
        self,
        uri: str,
        max_synced_objects: int,
        logical_path: Optional[str],
        overwrite: bool,
        asset_type: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            preprocessed_assets = preprocess_remote_gs_assets(
                remote_uri=uri,
                logical_path=logical_path,
                overwrite=overwrite,
                upload_type=asset_type,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
            for asset in preprocessed_assets:
                self._add_preprocessed(asset)

            # success - no error
            return True, None

        except LogAssetException as lax:
            debug_helpers.log_warning_or_raise(
                lax.backend_err_msg,
                logger=LOGGER,
                original_exception=lax,
                exc_info=True,
            )
            error_message = lax.backend_err_msg
        except Exception as e:
            debug_helpers.log_warning_or_raise(
                FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
                uri,
                exc_info=True,
                logger=LOGGER,
                original_exception=e,
            )
            error_message = FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET % uri

        return False, error_message

    def _add_s3_assets(
        self,
        uri: str,
        max_synced_objects: int,
        logical_path: Optional[str],
        overwrite: bool,
        asset_type: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            preprocessed_assets = preprocess_remote_s3_assets(
                remote_uri=uri,
                logical_path=logical_path,
                overwrite=overwrite,
                upload_type=asset_type,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
            for asset in preprocessed_assets:
                self._add_preprocessed(asset)

            # success - no error
            return True, None

        except LogAssetException as lax:
            debug_helpers.log_warning_or_raise(
                lax.backend_err_msg,
                logger=LOGGER,
                original_exception=lax,
                exc_info=True,
            )
            error_message = lax.backend_err_msg
        except Exception as e:
            debug_helpers.log_warning_or_raise(
                FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
                uri,
                logger=LOGGER,
                original_exception=e,
                exc_info=True,
            )
            error_message = FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET % uri

        return False, error_message

    def _preprocessed_user_input(self, preprocessed: PreprocessedAsset) -> Any:
        if isinstance(preprocessed, PreprocessedRemoteAsset):
            return preprocessed.remote_uri
        else:
            return preprocessed.local_path_or_data

    def _add_preprocessed(self, preprocessed: PreprocessedAsset) -> None:
        preprocessed_logical_path = preprocessed.logical_path

        if preprocessed_logical_path in self._assets:
            # Allow the overriding of an asset inherited from a downloaded version
            if (
                self._assets[preprocessed_logical_path].asset_id
                in self._downloaded_asset_ids
            ):
                self._downloaded_asset_ids.remove(
                    self._assets[preprocessed_logical_path].asset_id
                )
                self._assets[preprocessed_logical_path] = preprocessed
            else:
                raise ArtifactConflictingAssetLogicalPath(
                    self._preprocessed_user_input(
                        self._assets[preprocessed_logical_path]
                    ),
                    self._preprocessed_user_input(preprocessed),
                    preprocessed_logical_path,
                )
        else:
            self._assets[preprocessed_logical_path] = preprocessed

    def _add_preprocessed_folder(
        self, preprocessed_folder: PreprocessedAssetFolder
    ) -> None:

        for preprocessed_asset_file in preprocessed_folder:
            self._add_preprocessed(preprocessed_asset_file)

    def __str__(self) -> str:
        return "%s(%r, artifact_type=%r)" % (
            self.__class__.__name__,
            self.name,
            self.artifact_type,
        )

    def __repr__(self) -> str:
        return (
            "%s(name=%r, artifact_type=%r, version=%r, aliases=%r, version_tags=%s)"
            % (
                self.__class__.__name__,
                self.name,
                self.artifact_type,
                self.version,
                self.aliases,
                self.version_tags,
            )
        )

    @property
    def assets(self) -> List[ArtifactAsset]:
        """
        The list of `ArtifactAssets` that have been logged with this `Artifact`.
        """
        artifact_version_assets = []

        for asset in self._assets.values():

            if isinstance(asset, PreprocessedRemoteAsset):
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=True,
                        logical_path=asset.logical_path,
                        # Semantically remote files have a 0 size, but we are still counting
                        # the size for upload progress
                        size=0,
                        link=asset.remote_uri,
                        metadata=asset.metadata,
                        asset_type=asset.upload_type,
                        local_path_or_data=None,
                        asset_id=asset.asset_id,
                    )
                )
            elif isinstance(asset, PreprocessedSyncedRemoteAsset):
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=True,
                        logical_path=asset.logical_path,
                        size=asset.size,
                        link=asset.remote_uri,
                        metadata=asset.metadata,
                        asset_type=asset.upload_type,
                        local_path_or_data=asset.local_path,
                        asset_id=asset.asset_id,
                    )
                )
            else:
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=False,
                        logical_path=asset.logical_path,
                        size=asset.size,
                        link=None,
                        metadata=asset.metadata,
                        asset_type=None,
                        local_path_or_data=asset.local_path_or_data,
                        asset_id=asset.asset_id,
                    )
                )

        return artifact_version_assets

    @property
    def download_local_path(self) -> Optional[str]:
        """If the Artifact object was returned by `LoggedArtifact.download`, returns the root path
        where the assets have been downloaded. Else, returns None.
        """
        return self._download_local_path
