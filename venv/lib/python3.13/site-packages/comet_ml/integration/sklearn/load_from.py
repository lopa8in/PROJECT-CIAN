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

import io
import json
import pathlib
import tempfile
from typing import Any

from comet_ml import model_downloader
from comet_ml.model_downloader.uri import parse

from . import comet_metadata
from .persistence import dispatcher


def disk(model_uri: str, **load_kwargs) -> Any:
    directory = parse.filepath(model_uri)
    model = _read_model(directory, **load_kwargs)

    return model


def remote(model_uri: str, **load_kwargs) -> Any:
    with tempfile.TemporaryDirectory() as directory:
        model_downloader.download(model_uri, directory)
        model = _read_model(directory, **load_kwargs)

    return model


def _read_model(directory: str, **load_kwargs) -> Any:
    with io.open(pathlib.Path(directory, comet_metadata.FILE_NAME)) as stream:
        metadata = json.load(stream)

    comet_metadata.warn_if_mismatches_with_environment(metadata)

    persistence_module_name = metadata["model_metadata"]["scikit-learn"][
        "persistence_module"
    ]

    loader = dispatcher.get_loader(
        persistence_module_name=persistence_module_name, **load_kwargs
    )

    model_path = pathlib.Path(metadata["model_metadata"]["scikit-learn"]["model_path"])
    with io.open(directory / model_path, mode="rb") as stream:
        model = loader.load(stream)

    return model
