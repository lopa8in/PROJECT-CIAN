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

import functools
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Union

from comet_ml import cloud_storage_utils, semantic_version, utils

from ...api_objects import setup_client
from ...api_objects.api_objects_helpers import experiment_key_from_uri
from ...assets import data_writers
from ...config import get_api_key, get_config
from ...constants import ASSET_TYPE_DEFAULT
from ...exceptions import CometException, RemoteModelDownloadException
from ...file_downloader import (
    DownloadResultHolder,
    FileDownloadManager,
    FileDownloadManagerMonitor,
)
from ...logging_messages import (
    ASSET_DOWNLOAD_FAILED_WITH_ERROR,
    DEPRECATED_EXPAND_MODEL_DOWNLOAD,
    MODEL_REMOTE_ASSET_DOWNLOAD_UNSUPPORTED_URI,
    REMOTE_MODEL_DOWNLOAD_FAILED,
    REMOTE_MODEL_DOWNLOAD_FINISHED,
    REMOTE_MODEL_DOWNLOAD_START_MESSAGE,
)
from .assets_list import AssetsList
from .model_helpers import (
    download_cloud_storage_remote_model_asset,
    download_model_asset,
    download_s3_model_asset,
    has_direct_s3_download_enabled,
)

LOGGER = logging.getLogger(__name__)


class Model:
    """
    Model is an API object implementing various methods to manipulate models in the model registry.
    """

    def __init__(
        self, workspace: str, model_name: str, *, api_key: Optional[str] = None
    ):
        """
        Creates a new Model object that can be used to manipulate a model in the model registry.

        Args:
            workspace: Name of the workspace to which the model belongs
            model_name: Name of the model stored in the model registry
            api_key: Your API key obtained from comet.com.
                If not specified, api_key will be obtained from the comet configuration or environment variables.
        """

        self._workspace = workspace
        self._model_name = model_name

        self._config = get_config()
        self._api_key = get_api_key(api_key, self._config)

    @classmethod
    def from_registry(
        cls, workspace: str, model_name: str, *, api_key: Optional[str] = None
    ):
        """
        Obtain a Model object from the model registry.

        Args:
            workspace: Name of the workspace to which the model belongs.
            model_name: Name of the model stored in the model registry.
            api_key: Your API key obtained from comet.com.
                If not specified, api_key will be obtained from the comet configuration or environment variables.
        """
        model = Model(workspace, model_name, api_key=api_key)
        if not Model.__internal_api_compatible_backend__(model._client):
            actual_backend = model._client.get_api_backend_version()
            message = "{} API object only works if backend version is at least {} (actual backend version: {})".format(
                cls.__name__,
                cls.minimal_backend(),
                actual_backend,
            )
            raise CometException(message)
        model._load_compact_details()
        return model

    @classmethod
    def minimal_backend(cls) -> semantic_version.SemanticVersion:
        config = get_config()
        return semantic_version.SemanticVersion.parse(
            config["comet.novel_model_registry_api.minimum_backend_version"]
        )

    @property
    @functools.lru_cache()
    def _client(self):
        client = setup_client.setup(api_key=self._api_key, use_cache=False)
        return client

    @classmethod
    @functools.lru_cache()
    def __internal_api_compatible_backend__(cls, client):
        actual_backend = client.get_api_backend_version()
        if actual_backend is None:
            raise CometException("could not parse backend version")
        return actual_backend >= cls.minimal_backend()

    def _load_compact_details(self):
        params = {"workspaceName": self._workspace, "modelName": self._model_name}
        data = self._client.get_from_endpoint("registry-model/compact-details", params)
        self._registry_model_id = data["registryModelId"]

    def tags(self, version: str):
        """
        Returns the tags for a given version of the model.

        Args:
            version: The model version.
        """
        details = self.get_details(version)
        return details["tags"]

    def status(self, version: str):
        """
        Returns the status for a given version of the model, e.g. "Production"

        Args:
            version: The model version.
        """
        details = self.get_details(version)
        return details["status"]

    def __repr__(self):
        return "Model(%s, %s)" % (repr(self._workspace), repr(self._model_name))

    def delete_tag(self, version: str, tag: str):
        """
        Deletes a tag from a given version of the model

        Args:
            version: the model version
            tag: the tag to delete
        """
        self._client.delete_registry_model_version_stage(
            self._workspace, self._model_name, version, stage=tag
        )

    def add_tag(self, version: str, tag: str):
        """
        Add a tag to a given version of the model

        Args:
            version: the model version
            tag: the tag to add
        """
        self._enforce_type("version", version, str)
        self._enforce_type("tag", tag, str)
        self._client.add_registry_model_version_stage(
            self._workspace, self._model_name, version, stage=tag
        )

    def _enforce_type(self, name, thing, expected_type):
        actual_type = type(thing)
        if actual_type is not expected_type:
            raise TypeError(
                '"{name}" must be of type {expected_type}, not {actual_type}'.format(
                    name=name, expected_type=expected_type, actual_type=actual_type
                )
            )

    def set_status(self, version, status):
        """
        Set the status of a given version of the model

        Args:
            version (str): the model version
            status (str): one of the allowed status values, e.g. "Production"

        See also: The
        [comet_ml.API.model_registry_allowed_status_values][]
        on the API class.
        """
        self._enforce_type("status", status, str)
        model_item_id = self.get_details(version)["registryModelItemId"]
        payload = {"status": status, "modelItemId": model_item_id}
        self._client.post_from_endpoint(
            "write/registry-model/item/status", payload=payload
        )

    @property
    def name(self):
        """
        Returns the model name
        """
        return self._model_name

    def find_versions(self, version_prefix="", status=None, tag=None):
        """
        Return a list of matching versions for the model, sorted in descending order (latest version
        is first).

        Args:
            version_prefix (str): If specified, return only those versions that start with
                version_prefix, e.g. "3" may find "3.2" but not "4.0", and "2.1" will find "2.1.0"
                and "2.1.1" but not "2.0.0" or "2.2.3".
            status (str): If specified, return only versions with the given status.
            tag (str): If specified, return only versions with the given tag.
        """
        response = self._client.get_from_endpoint(
            "registry-model/items",
            params={
                "workspaceName": self._workspace,
                "modelName": self._model_name,
                "tag": tag,
                "status": status,
                "versionPrefix": version_prefix,
            },
        )
        items = response["items"]
        return [item["version"] for item in items]

    def get_version_history(self, version: str):
        """
        Return the history of changes for a given Model version. This method returns a dictionary of
        list of changes per day, see below for an example:

        Args:
            version: the model version

        Example:
            Running the code sample:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            model = api.get_model("my-workspace", "my-sklearn-model")
            model.get_version_history("2.31.0")
            ```

            will return the dictionary:
            ```json
            {
                'November 16, 2023': [{'actionType': 'MODEL_VERSION_STATUS_REQUEST_APPROVED',
                            'comment': '',
                            'newValue': {'changedBy': 'lothiraldan',
                                        'registryModelStatus': 'Development'},
                            'previousValue': {'changedBy': 'user',
                                            'registryModelStatus': 'None'},
                            'registryModelId': 'bg0HklLmLbokkv8VIA5MzR40F',
                            'registryModelItemActualPathParams': None,
                            'registryModelItemId': '7dylnoIk8sX5sCojyBRnMNFYv',
                            'registryModelItemVersion': '2.31.0',
                            'registryModelName': 'my-sklearn-model',
                            'userAvatarLink': 'https://github.com/user.png?size=30',
                            'userName': 'user'},
                        {'actionType': 'MODEL_VERSION_STATUS_CHANGED',
                            'comment': '',
                            'newValue': {'changedBy': 'user',
                                        'registryModelStatus': 'Development'},
                            'previousValue': {'changedBy': 'user',
                                            'registryModelStatus': 'None'},
                            'registryModelId': 'bg0HklLmLbokkv8VIA5MzR40F',
                            'registryModelItemActualPathParams': None,
                            'registryModelItemId': '7dylnoIk8sX5sCojyBRnMNFYv',
                            'registryModelItemVersion': '2.31.0',
                            'registryModelName': 'my-sklearn-model',
                            'userAvatarLink': 'https://github.com/user.png?size=30',
                            'userName': 'user'},
                        {'actionType': 'MODEL_VERSION_STATUS_REQUEST_CHANGE',
                            'comment': '',
                            'newValue': {'registryModelStatus': 'Development'},
                            'previousValue': {'registryModelStatus': 'None'},
                            'registryModelId': 'bg0HklLmLbokkv8VIA5MzR40F',
                            'registryModelItemActualPathParams': None,
                            'registryModelItemId': '7dylnoIk8sX5sCojyBRnMNFYv',
                            'registryModelItemVersion': '2.31.0',
                            'registryModelName': 'my-sklearn-model',
                            'userAvatarLink': 'https://github.com/user.png?size=30',
                            'userName': 'user'}],
                'June 6, 2023': [{'actionType': 'MODEL_VERSION_DOWNLOADED',
                    'comment': None,
                    'newValue': {},
                    'previousValue': {},
                    'registryModelId': 'bg0HklLmLbokkv8VIA5MzR40F',
                    'registryModelItemActualPathParams': None,
                    'registryModelItemId': '7dylnoIk8sX5sCojyBRnMNFYv',
                    'registryModelItemVersion': '2.31.0',
                    'registryModelName': 'my-sklearn-model',
                    'userAvatarLink': 'https://github.com/user.png?size=30',
                    'userName': 'user'},
                    {'actionType': 'MODEL_VERSION_CREATED',
                    'comment': None,
                    'newValue': {},
                    'previousValue': {},
                    'registryModelId': 'bg0HklLmLbokkv8VIA5MzR40F',
                    'registryModelItemActualPathParams': None,
                    'registryModelItemId': '7dylnoIk8sX5sCojyBRnMNFYv',
                    'registryModelItemVersion': '2.31.0',
                    'registryModelName': 'my-sklearn-model',
                    'userAvatarLink': 'https://github.com/user.png?size=30',
                    'userName': 'user'}]}
            ```
        """
        COMPATIBLE_BACKEND_VERSION = "3.7.28"
        if (
            self._client.get_api_backend_version()
            < semantic_version.SemanticVersion.parse(COMPATIBLE_BACKEND_VERSION)
        ):
            raise CometException(
                "Model.get_version_history method is supported on backend version %s and above"
                % COMPATIBLE_BACKEND_VERSION
            )

        registry_model_item_id = self.get_details(version)["registryModelItemId"]

        response = self._client.post_from_endpoint(
            "registry-model/history/item",
            payload={"registryModelItemId": registry_model_item_id, "groupedBy": "DAY"},
        )

        return response.json()

    def download(
        self,
        version: str,
        output_folder: Optional[Union[pathlib.Path, str]] = None,
        expand: bool = True,
    ) -> None:
        """
        Download the files for a given version of the model. This method downloads assets and remote
        assets that were synced from a compatible cloud object storage (AWS S3 or GCP GCS). Other
        remote assets are not downloaded and you can access their uri with the
        [comet_ml.Model.get_assets][] method.

        Args:
            version: the model version
            output_folder: files will be saved in this folder. If not provided, will download to a
                temporary directory.
            expand (DEPRECATED): if True (the default), model files will be saved to the given folder. If False,
                it has no difference from True.
        """

        if not expand:
            LOGGER.warning(DEPRECATED_EXPAND_MODEL_DOWNLOAD)

        root_path = tempfile.mkdtemp() if output_folder is None else output_folder

        try:
            raw_assets = self._get_assets(version)
        except Exception:
            raise RemoteModelDownloadException(
                "Cannot get asset list for Model %r" % self
            )

        LOGGER.debug(
            "Model assets download - direct S3 assets download enabled: %r",
            raw_assets.s3_download_enabled,
        )

        worker_cpu_ratio = self._config.get_int(
            None, "comet.internal.file_upload_worker_ratio"
        )
        worker_count = self._config.get_raw(None, "comet.internal.worker_count")

        download_manager = FileDownloadManager(
            worker_cpu_ratio=worker_cpu_ratio, worker_count=worker_count
        )

        results: List[DownloadResultHolder] = list()

        for asset in raw_assets:
            asset_filename = asset.get("fileName", "")
            local_asset_path = os.path.join(root_path, asset_filename)

            asset_id = asset["assetId"]
            asset_type = asset.get("type", ASSET_TYPE_DEFAULT)
            asset_metadata = asset["metadata"]
            if asset_metadata is not None:
                asset_metadata = json.loads(asset["metadata"])

            remote_uri = asset.get("link", None)

            if "remote" in asset:
                remote_asset = asset["remote"]
            else:
                remote_asset = False

            if remote_asset:
                if remote_uri is None or remote_uri == "":
                    LOGGER.warning(
                        "Empty link found in remote model asset data - skipping asset download. Asset: %r",
                        asset,
                    )
                    # skip download because of an empty link
                    continue

                result = self._start_remote_asset_download(
                    download_manager=download_manager,
                    asset_metadata=asset_metadata,
                    asset_filename=asset_filename,
                    remote_uri=remote_uri,
                    asset_id=asset_id,
                    asset_type=asset_type,
                    local_asset_path=local_asset_path,
                )
            elif raw_assets.s3_download_enabled:
                remote_uri = asset.get("s3Link", None)
                if remote_uri is None or remote_uri == "":
                    LOGGER.warning(
                        "Empty S3 link found in model asset data - skipping asset download. Asset: %r",
                        asset,
                    )
                    # skip download because of an empty link
                    continue

                result = self._start_s3_asset_direct_download(
                    download_manager=download_manager,
                    asset_metadata=asset_metadata,
                    asset_filename=asset_filename,
                    asset_file_size=asset.get("fileSize", 0),
                    remote_uri=remote_uri,
                    asset_id=asset_id,
                    asset_type=asset_type,
                    local_asset_path=local_asset_path,
                )
            else:
                # fallback to downloading assets from Comet by link
                if remote_uri is None or remote_uri == "":
                    LOGGER.warning(
                        "Empty link found in model asset data - skipping asset download. Asset: %r",
                        asset,
                    )
                    # skip download because of an empty link
                    continue

                result = self._start_asset_download(
                    download_manager=download_manager,
                    asset_metadata=asset_metadata,
                    asset_filename=asset_filename,
                    asset_file_size=asset.get("fileSize", 0),
                    remote_uri=remote_uri,
                    asset_id=asset_id,
                    asset_type=asset_type,
                    local_asset_path=local_asset_path,
                )

            if result is not None:
                results.append(result)

        download_manager.close()

        if not download_manager.all_done():
            monitor = FileDownloadManagerMonitor(download_manager)

            LOGGER.info(
                REMOTE_MODEL_DOWNLOAD_START_MESSAGE,
                self._workspace,
                self._model_name,
                version,
            )

            utils.wait_for_done(
                check_function=monitor.all_done,
                timeout=self._config.get_int(
                    None, "comet.timeout.remote_model_download"
                ),
                progress_callback=monitor.log_remaining_downloads,
                sleep_time=15,
            )

        try:
            for result in results:
                self._verify_result(result, version)

            LOGGER.info(
                REMOTE_MODEL_DOWNLOAD_FINISHED,
                self._workspace,
                self._model_name,
                version,
            )
        finally:
            download_manager.join()

        LOGGER.info("Downloaded asset files is in %r folder.", root_path)

    def _start_s3_asset_direct_download(
        self,
        download_manager: FileDownloadManager,
        asset_filename: str,
        asset_file_size: int,
        remote_uri: str,
        asset_id: str,
        asset_type: str,
        local_asset_path: str,
        asset_metadata: Dict[str, Any],
    ) -> DownloadResultHolder:
        LOGGER.debug(
            "Downloading S3 asset, ID: %r, URI: %r, file name: %r, size: %d, local path: %r, metadata: %r",
            asset_id,
            remote_uri,
            asset_filename,
            asset_file_size,
            local_asset_path,
            asset_metadata,
        )

        download_result = download_manager.download_file_async(
            download_func=download_s3_model_asset,
            estimated_size=asset_file_size,
            url=remote_uri,
            asset_id=asset_id,
            remote_model_repr=repr(self),
            asset_path=local_asset_path,
        )
        return DownloadResultHolder(
            download_result=download_result,
            asset_filename=asset_filename,
            asset_path=local_asset_path,
            asset_metadata=asset_metadata,
            asset_id=asset_id,
            asset_synced=True,
            asset_type=asset_type,
            asset_remote_uri=remote_uri,
        )

    def _start_asset_download(
        self,
        download_manager: FileDownloadManager,
        asset_filename: str,
        asset_file_size: int,
        remote_uri: str,
        asset_id: str,
        asset_type: str,
        local_asset_path: str,
        asset_metadata: Dict[str, Any],
    ) -> DownloadResultHolder:
        LOGGER.debug(
            "Downloading asset, ID: %r, URI: %r, file name: %r, size: %d, local path: %r, metadata: %r",
            asset_id,
            remote_uri,
            asset_filename,
            asset_file_size,
            local_asset_path,
            asset_metadata,
        )

        experiment_key = experiment_key_from_uri(remote_uri)
        download_result = download_manager.download_file_async(
            download_func=download_model_asset,
            estimated_size=asset_file_size,
            asset_id=asset_id,
            remote_model_repr=repr(self),
            asset_path=local_asset_path,
            rest_api_client=self._client,
            experiment_key=experiment_key,
        )
        return DownloadResultHolder(
            download_result=download_result,
            asset_filename=asset_filename,
            asset_path=local_asset_path,
            asset_metadata=asset_metadata,
            asset_id=asset_id,
            asset_synced=True,
            asset_type=asset_type,
            asset_remote_uri=remote_uri,
        )

    def _start_remote_asset_download(
        self,
        download_manager: FileDownloadManager,
        asset_metadata: Dict[str, Any],
        asset_filename: str,
        remote_uri: str,
        asset_id: str,
        asset_type: str,
        local_asset_path: str,
    ) -> Optional[DownloadResultHolder]:
        LOGGER.debug(
            "Downloading remote asset, ID: %r, URI: %r, file name: %r, local path: %r, metadata: %r",
            asset_id,
            remote_uri,
            asset_filename,
            local_asset_path,
            asset_metadata,
        )

        asset_synced = False
        asset_sync_error = None

        if asset_metadata is not None:
            if cloud_storage_utils.META_SYNCED in asset_metadata:
                asset_synced = asset_metadata[cloud_storage_utils.META_SYNCED]
            if cloud_storage_utils.META_ERROR_MESSAGE in asset_metadata:
                asset_sync_error = asset_metadata[
                    cloud_storage_utils.META_ERROR_MESSAGE
                ]

        if asset_synced is False and asset_sync_error is not None:
            raise RemoteModelDownloadException(
                ASSET_DOWNLOAD_FAILED_WITH_ERROR % (asset_filename, asset_sync_error)
            )

        parsed_url = utils.urlparse(remote_uri)
        if parsed_url.scheme not in ["s3", "gs"]:
            LOGGER.warning(MODEL_REMOTE_ASSET_DOWNLOAD_UNSUPPORTED_URI % remote_uri)
            return None

        if cloud_storage_utils.META_FILE_SIZE in asset_metadata:
            asset_file_size = asset_metadata[cloud_storage_utils.META_FILE_SIZE]
        else:
            asset_file_size = 0

        version_id = None
        if cloud_storage_utils.META_VERSION_ID in asset_metadata:
            version_id = asset_metadata[cloud_storage_utils.META_VERSION_ID]

        if parsed_url.scheme == "s3":
            data_writer = data_writers.AssetDataWriterFromS3(
                s3_uri=remote_uri, version_id=version_id
            )
        else:
            data_writer = data_writers.AssetDataWriterFromGCS(
                gs_uri=remote_uri, version_id=version_id
            )

        download_result = download_manager.download_file_async(
            download_func=download_cloud_storage_remote_model_asset,
            estimated_size=asset_file_size,
            data_writer=data_writer,
            asset_id=asset_id,
            remote_model_repr=repr(self),
            asset_path=local_asset_path,
        )
        return DownloadResultHolder(
            download_result=download_result,
            asset_filename=asset_filename,
            asset_path=local_asset_path,
            asset_metadata=asset_metadata,
            asset_id=asset_id,
            asset_synced=asset_synced,
            asset_type=asset_type,
            asset_remote_uri=remote_uri,
        )

    def get_assets(self, version: str) -> List[dict]:
        """
        Returns the assets list for the given version. Remote assets have the key `remote` set to `True`.

        Args:
            version: The model version.

        ```python
        >>> from comet_ml.api import API
        >>> api = API()
        >>> model = api.get_model("my-workspace", "my-model-name")
        >>> model.get_assets()
        [
            {
                'fileName': 'file',
                'fileSize': 0,
                'runContext': None,
                'step': None,
                'remote': True,
                'link': 's3://bucket/dir/file',
                'compressedAssetLink': 's3://bucket/dir/file',
                's3Link': None,
                'createdAt': 1700131519059,
                'dir': 'models/my-model',
                'canView': False,
                'audio': False,
                'video': False,
                'histogram': False,
                'image': False,
                'type': 'model-element',
                'metadata': '',
                'assetId': '6ce04b4331bd4f7d9eb56a0d876ead72',
                'tags': [],
                'curlDownload': 'curl "..." > file',
                'experimentKey': '...'
            },
            {
                'fileName': 'model_metadata.json',
                'fileSize': 49,
                'runContext': None,
                'step': None,
                'remote': False,
                'link': '...',
                'compressedAssetLink': '...',
                's3Link': '...',
                'createdAt': 1700131496454,
                'dir': 'models/my-model',
                'canView': False,
                'audio': False,
                'video': False,
                'histogram': False,
                'image': False,
                'type': 'model-element',
                'metadata': None,
                'assetId': 'd4fcc9ef32394ea0956c1725c0c98604',
                'tags': [],
                'curlDownload': 'curl "..." -H"Authorization: <Your Api Key>" > model_metadata.json',
                'experimentKey': '...'
            }
        ]
        ```
        """
        return self._get_assets(version)

    def _get_assets(self, version: str) -> AssetsList:
        ONLY_ONE_VERSION = 0
        params = {
            "workspaceName": self._workspace,
            "modelName": self._model_name,
            "version": version,
        }
        data = self._client.get_from_endpoint(
            "registry-model/item/download-instructions", params
        )
        s3_download_enabled = has_direct_s3_download_enabled(download_instructions=data)

        return AssetsList(
            seq=data["versions"][ONLY_ONE_VERSION]["assets"],
            s3_download_enabled=s3_download_enabled,
        )

    def _verify_result(self, result: DownloadResultHolder, version: str):
        file_download_timeout = self._config.get_int(
            None, "comet.timeout.file_download"
        )

        try:
            result.download_result.get(file_download_timeout)
        except Exception:
            LOGGER.error(
                REMOTE_MODEL_DOWNLOAD_FAILED,
                result.asset_filename,
                self._workspace,
                self._model_name,
                version,
                exc_info=True,
            )

            raise RemoteModelDownloadException(
                "Cannot download Asset %s for Remote Model %s"
                % (result.asset_filename, repr(self))
            )

    @classmethod
    def __internal_api__register__(
        cls,
        experiment_id: str,
        model_name,
        version: str,
        workspace: str,
        registry_name: str,
        public: bool,
        description: str,
        comment: str,
        tags: List[str],
        status: str,
        *,
        api_key=None
    ):
        payload = {
            "experimentKey": experiment_id,
            "experimentModelName": model_name,
            "registryModelName": registry_name,
            "registryModelDescription": description,
            "version": version,
            "versionComment": comment,
            "tags": tags,
            "status": status,
            "publicModel": public,
        }
        model = Model(workspace, registry_name, api_key=api_key)
        model._client.post_from_endpoint(
            "write/registry-model/item/create", payload=payload
        )
        return model

    def get_details(self, version: str):
        """
        Returns a dict with various details about the given model version.

        The exact details returned may vary by backend version, but they include e.g. experimentKey, comment, createdAt timestamp, updatedAt timestamp.

        Args:
            version: the model version
        """
        params = {
            "workspaceName": self._workspace,
            "modelName": self._model_name,
            "version": version,
        }
        response = self._client.get_from_endpoint(
            "registry-model/item/details", params=params
        )
        return response
