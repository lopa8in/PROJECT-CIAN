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
from threading import Event

from .constants import (
    ASSET_TYPE_CURVE,
    ASSET_TYPE_EMBEDDINGS,
    ASSET_TYPE_HISTOGRAM_3D,
    ASSET_TYPE_MODEL_ELEMENT,
    ASSET_TYPE_NOTEBOOK,
    ASSET_TYPE_TEXT_SAMPLE,
    ASSET_TYPE_VIDEO,
    ASSET_TYPE_VISUALIZATION,
)


class FileUploadLimitsGuard:
    def __init__(self) -> None:
        self._image_upload_limit_reached = Event()
        self._embedding_upload_limit_reached = Event()
        self._histogram_upload_limit_reached = Event()
        self._text_samples_limit_reached = Event()
        self._model_element_limit_reached = Event()
        self._curve_limit_reached = Event()
        self._notebook_limit_reached = Event()
        self._video_limit_reached = Event()

        self._guards_map = {
            ASSET_TYPE_VISUALIZATION: self._image_upload_limit_reached,
            ASSET_TYPE_EMBEDDINGS: self._embedding_upload_limit_reached,
            ASSET_TYPE_HISTOGRAM_3D: self._histogram_upload_limit_reached,
            ASSET_TYPE_TEXT_SAMPLE: self._text_samples_limit_reached,
            ASSET_TYPE_MODEL_ELEMENT: self._model_element_limit_reached,
            ASSET_TYPE_CURVE: self._curve_limit_reached,
            ASSET_TYPE_NOTEBOOK: self._notebook_limit_reached,
            ASSET_TYPE_VIDEO: self._video_limit_reached,
        }

    def has_upload_limit_exceeded(self, upload_type: str) -> bool:
        limit_guard = self._guards_map.get(upload_type)
        if limit_guard is None:
            return False
        return limit_guard.is_set()

    def image_upload_limit_exceeded(self) -> None:
        self._image_upload_limit_reached.set()

    def embedding_upload_limit_exceeded(self) -> None:
        self._embedding_upload_limit_reached.set()

    def histogram_upload_limit_exceeded(self) -> None:
        self._histogram_upload_limit_reached.set()

    def text_samples_limit_exceeded(self) -> None:
        self._text_samples_limit_reached.set()

    def model_element_limit_exceeded(self) -> None:
        self._model_element_limit_reached.set()

    def curve_limit_exceeded(self) -> None:
        self._curve_limit_reached.set()

    def notebook_limit_exceeded(self) -> None:
        self._notebook_limit_reached.set()

    def video_limit_exceeded(self) -> None:
        self._video_limit_reached.set()
