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

import json
import logging
import pathlib
import tempfile
from types import ModuleType
from typing import Any

import comet_ml
import comet_ml._reporting
import comet_ml.connection
from comet_ml import model_downloader
from comet_ml.model_downloader.uri import parse

import torch

from . import constants, model_metadata
from .types import ModelStateDict

LOGGER = logging.getLogger(__name__)


def from_disk(
    MODEL_URI: str, map_location: Any, pickle_module: ModuleType, **torch_load_args
) -> ModelStateDict:
    dirpath = parse.filepath(MODEL_URI)
    model_state_dict = _read_with_metadata_checks(
        dirpath, map_location, pickle_module, **torch_load_args
    )

    return model_state_dict


def from_remote(
    MODEL_URI: str, map_location: Any, pickle_module: ModuleType, **torch_load_args
) -> ModelStateDict:
    with tempfile.TemporaryDirectory() as tmp_dirpath:
        model_downloader.download(MODEL_URI, tmp_dirpath)
        model_state_dict = _read_with_metadata_checks(
            tmp_dirpath, map_location, pickle_module, **torch_load_args
        )

    return model_state_dict


def track_usage(model_uri: str) -> None:
    config = comet_ml.get_config()

    comet_ml.connection.Reporting.report(
        config=config,
        api_key=comet_ml.get_api_key(None, config),
        event_name=comet_ml._reporting.PYTORCH_MODEL_LOADING_EXPLICIT_CALL,
        err_msg=model_uri,
    )


def _read_with_metadata_checks(
    dirpath: str, map_location: Any, pickle_module: ModuleType, **torch_load_args
) -> ModelStateDict:
    comet_model_metadata_path = pathlib.Path(
        dirpath, constants.COMET_MODEL_METADATA_FILENAME
    )
    with open(comet_model_metadata_path) as stream_in:
        comet_model_metadata = json.load(stream_in)

    model_metadata.warn_if_has_mismatches_with_environment(
        comet_model_metadata, pickle_module
    )

    model_state_dict = torch.load(
        pathlib.Path(dirpath, constants.MODEL_DATA_DIRECTORY, constants.MODEL_FILENAME),
        map_location=map_location,
        pickle_module=pickle_module,
        **torch_load_args
    )

    return model_state_dict
