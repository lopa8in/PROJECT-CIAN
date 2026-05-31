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

import logging
from typing import TYPE_CHECKING, Dict

import comet_ml
from comet_ml import logging_messages

import sklearn

if TYPE_CHECKING:  # pragma: no cover
    from .persistence import dumper

FILE_NAME = "CometModel"

LOGGER = logging.getLogger(__name__)


def collect(dumper: "dumper.Dumper") -> Dict[str, str]:
    sklearn_metadata = {}

    comet_model_metadata = {
        "format": "scikit-learn",
        "comet_sdk_version": comet_ml.__version__,
        "model_metadata": {"scikit-learn": sklearn_metadata},
    }

    sklearn_metadata["sklearn_version"] = sklearn.__version__
    sklearn_metadata["persistence_module"] = dumper.module_name
    sklearn_metadata["persistence_module_version"] = dumper.module_version
    sklearn_metadata["model_path"] = str(
        "model-data/comet-sklearn-model.{}".format(dumper.file_extension)
    )

    return comet_model_metadata


def warn_if_mismatches_with_environment(metadata: Dict[str, str]):
    metadata_sklearn_version = metadata["model_metadata"]["scikit-learn"][
        "sklearn_version"
    ]

    if metadata_sklearn_version != sklearn.__version__:
        LOGGER.warning(
            logging_messages.SKLEARN_INTEGRATION_SKLEARN_VERSION_MISMATCH_WARNING,
            metadata_sklearn_version,
            sklearn.__version__,
        )
