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
from typing import Optional, Union

from .convert_utils import data_to_fp
from .file_uploader import is_user_text
from .file_utils import write_file_like_to_tmp_file
from .messages import (
    Log3DCloudMessage,
    RemoteAssetMessage,
    UploadFileMessage,
    UploadInMemoryMessage,
)
from .upload_options import (
    AssetItemUploadOptions,
    AvailableUploadOptions,
    FileLikeUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
)


def convert_upload_in_memory_to_file_message(
    message: UploadInMemoryMessage, tmp_dir: str
) -> UploadFileMessage:
    if is_user_text(message.file_like):
        file_like = data_to_fp(message.file_like)
    else:
        file_like = message.file_like

    tmp_file = write_file_like_to_tmp_file(file_like, tmp_dir)

    return UploadFileMessage(
        file_path=tmp_file,
        upload_type=message.upload_type,
        additional_params=message.additional_params,
        metadata=message.metadata,
        clean=True,
        size=message._size,
        critical=message._critical,
        message_id=message.message_id,
        on_asset_upload=message._on_asset_upload,
        on_failed_asset_upload=message._on_failed_asset_upload,
    )


def create_upload_message_from_options(
    upload_options: AvailableUploadOptions,
    message_id: int,
) -> Optional[
    Union[
        UploadFileMessage, UploadInMemoryMessage, RemoteAssetMessage, Log3DCloudMessage
    ]
]:
    """
    Create an upload message from upload options for retry purposes.

    Args:
        upload_options: The original upload options from a failed upload
        message_id: The message ID to use for the retry message

    Returns:
        An appropriate upload message (UploadFileMessage, UploadInMemoryMessage, or RemoteAssetMessage)
    """

    if isinstance(upload_options, FileUploadOptions):
        # File upload message
        return UploadFileMessage(
            file_path=upload_options.file_path,
            upload_type=upload_options.upload_type,
            additional_params=upload_options.additional_params,
            metadata=upload_options.metadata,
            size=upload_options.estimated_size,
            clean=upload_options.clean,
            critical=upload_options.critical,
            on_asset_upload=getattr(upload_options, "on_asset_upload", None),
            on_failed_asset_upload=getattr(
                upload_options, "on_failed_asset_upload", None
            ),
            message_id=message_id,
        )
    elif isinstance(upload_options, FileLikeUploadOptions):
        # File-like upload message
        return UploadInMemoryMessage(
            file_like=upload_options.file_like,
            upload_type=upload_options.upload_type,
            additional_params=upload_options.additional_params,
            metadata=upload_options.metadata,
            size=upload_options.estimated_size,
            critical=upload_options.critical,
            on_asset_upload=getattr(upload_options, "on_asset_upload", None),
            on_failed_asset_upload=getattr(
                upload_options, "on_failed_asset_upload", None
            ),
            message_id=message_id,
        )
    elif isinstance(upload_options, RemoteAssetsUploadOptions):
        # Remote asset upload message
        return RemoteAssetMessage(
            remote_uri=upload_options.remote_uri,
            upload_type=upload_options.upload_type,
            additional_params=upload_options.additional_params,
            metadata=upload_options.metadata,
            size=upload_options.estimated_size,
            critical=upload_options.critical,
            on_asset_upload=getattr(upload_options, "on_asset_upload", None),
            on_failed_asset_upload=getattr(
                upload_options, "on_failed_asset_upload", None
            ),
            message_id=message_id,
        )
    elif isinstance(upload_options, AssetItemUploadOptions):
        # 3D cloud upload message
        return Log3DCloudMessage(
            name=upload_options.asset_name,
            asset_id=upload_options.asset_id,
            upload_type=upload_options.upload_type,
            items=upload_options.all_items,
            thumbnail_path=None,
            metadata=upload_options.metadata,
            step=None,
            message_id=message_id,
        )

    return None
