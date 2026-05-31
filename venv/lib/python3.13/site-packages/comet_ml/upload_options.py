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
from typing import IO, Any, Callable, Dict, List, Optional, Union

from .assets import asset_item
from .config import UPLOAD_FILE_MAX_RETRIES, UPLOAD_FILE_RETRY_BACKOFF_FACTOR


class UploadOptions:
    def __init__(
        self,
        api_key: str,
        project_id: str,
        experiment_id: str,
        upload_endpoint: str,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        estimated_size: int = 0,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
        log_connection_error_as_debug: bool = True,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.experiment_id = experiment_id
        self.upload_endpoint = upload_endpoint
        self.estimated_size = estimated_size
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.critical = critical
        self.additional_params = additional_params
        self.metadata = metadata
        self.clean = clean
        self.on_asset_upload = on_asset_upload
        self.on_failed_asset_upload = on_failed_asset_upload
        self.log_connection_error_as_debug = log_connection_error_as_debug


class UploadOptionsWithRetry(UploadOptions):
    def __init__(
        self,
        api_key: str,
        project_id: str,
        experiment_id: str,
        upload_endpoint: str,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        estimated_size: int = 0,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
        max_retries=UPLOAD_FILE_MAX_RETRIES,
        retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        log_connection_error_as_debug: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            log_connection_error_as_debug=log_connection_error_as_debug,
        )
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor


class FileUploadOptions(UploadOptionsWithRetry):
    def __init__(
        self,
        file_path: str,
        upload_type: str,
        base_url: str,
        api_key: str,
        project_id: Optional[str],
        experiment_id: str,
        upload_endpoint: str,
        estimated_size: int,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
        max_retries=UPLOAD_FILE_MAX_RETRIES,
        retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        log_connection_error_as_debug: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            log_connection_error_as_debug=log_connection_error_as_debug,
        )
        self.file_path = file_path
        self.upload_type = upload_type
        self.base_url = base_url


class FileLikeUploadOptions(UploadOptionsWithRetry):
    def __init__(
        self,
        file_like: IO,
        upload_type: str,
        base_url: str,
        api_key: str,
        project_id: str,
        experiment_id: str,
        upload_endpoint: str,
        estimated_size: int,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
        max_retries=UPLOAD_FILE_MAX_RETRIES,
        retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        log_connection_error_as_debug: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            log_connection_error_as_debug=log_connection_error_as_debug,
        )
        self.file_like = file_like
        self.upload_type = upload_type
        self.base_url = base_url


class RemoteAssetsUploadOptions(UploadOptions):
    def __init__(
        self,
        remote_uri: str,
        upload_type: str,
        api_key: str,
        project_id: str,
        experiment_id: str,
        upload_endpoint: str,
        estimated_size: int,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
        log_connection_error_as_debug: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            log_connection_error_as_debug=log_connection_error_as_debug,
        )
        self.remote_uri = remote_uri
        self.upload_type = upload_type


class AssetItemUploadOptions(UploadOptionsWithRetry):
    def __init__(
        self,
        asset_id: str,
        asset_item: asset_item.AssetItem,
        api_key: str,
        project_id: Optional[str],
        experiment_id: str,
        upload_endpoint: str,
        estimated_size: int,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        all_items: List[asset_item.AssetItem],
        upload_type: str,
        asset_name: str,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        max_retries=UPLOAD_FILE_MAX_RETRIES,
        retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        log_connection_error_as_debug: bool = True,
        on_asset_upload: Optional[Callable[[Any], None]] = None,
        on_failed_asset_upload: Optional[Callable[[Any], None]] = None,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            log_connection_error_as_debug=log_connection_error_as_debug,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
        )

        self.asset_id = asset_id
        self.asset_item = asset_item
        self.all_items = all_items
        self.upload_type = upload_type
        self.asset_name = asset_name


class ThumbnailUploadOptions(UploadOptionsWithRetry):
    def __init__(
        self,
        thumbnail_path: str,
        api_key: str,
        project_id: Optional[str],
        experiment_id: str,
        upload_endpoint: str,
        estimated_size: int,
        timeout: float,
        verify_tls: bool,
        critical: bool,
        additional_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clean: bool = True,
        max_retries=UPLOAD_FILE_MAX_RETRIES,
        retry_backoff_factor=UPLOAD_FILE_RETRY_BACKOFF_FACTOR,
        log_connection_error_as_debug: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            project_id=project_id,
            experiment_id=experiment_id,
            upload_endpoint=upload_endpoint,
            estimated_size=estimated_size,
            timeout=timeout,
            verify_tls=verify_tls,
            critical=critical,
            additional_params=additional_params,
            metadata=metadata,
            clean=clean,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            log_connection_error_as_debug=log_connection_error_as_debug,
        )

        self.thumbnail_path = thumbnail_path


AvailableUploadOptions = Union[
    FileUploadOptions,
    FileLikeUploadOptions,
    RemoteAssetsUploadOptions,
    AssetItemUploadOptions,
    ThumbnailUploadOptions,
]
