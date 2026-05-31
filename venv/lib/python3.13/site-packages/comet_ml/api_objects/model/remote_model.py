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
from logging import getLogger

from comet_ml import file_uploader, messages
from comet_ml._typing import Any, Callable, Dict, List, Optional
from comet_ml.api_objects.model import Model

LOGGER = getLogger(__name__)


class RemoteModel(Model):
    def __init__(
        self,
        workspace: str,
        model_name: str,
        api_key: Optional[str] = None,
        remote_assets: Optional[List[file_uploader.PreprocessedRemoteAsset]] = None,
    ):
        super().__init__(workspace, model_name, api_key=api_key)
        self._remote_assets = remote_assets if remote_assets is not None else []

    def append_asset(self, asset: file_uploader.PreprocessedRemoteAsset):
        self._remote_assets.append(asset)

    def to_message(
        self, on_model_upload: Callable, on_failed_model_upload: Callable
    ) -> messages.RemoteModelMessage:
        return messages.RemoteModelMessage(
            model_name=self._model_name,
            remote_assets=self._convert_remote_assets_to_json(),
            on_model_upload=on_model_upload,
            on_failed_model_upload=on_failed_model_upload,
        )

    def _convert_remote_assets_to_json(self) -> List[Dict[str, Any]]:
        remote_assets_json = []
        for remote_asset in self._remote_assets:
            remote_assets_json.append(remote_asset.to_remote_model_asset_json())

        return remote_assets_json
