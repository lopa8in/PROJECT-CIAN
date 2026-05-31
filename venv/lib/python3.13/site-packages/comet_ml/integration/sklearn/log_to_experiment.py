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
import logging
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from . import comet_metadata

if TYPE_CHECKING:  # pragma: no cover
    import comet_ml

    from .persistence import dumper

LOGGER = logging.getLogger(__name__)


def model(
    experiment: "comet_ml.CometExperiment",
    model_name: str,
    model: Any,
    destination_filename: str,
    metadata: Optional[Dict],
    dumper_: "dumper.Dumper",
) -> None:
    model_temp_file = tempfile.NamedTemporaryFile()

    try:
        dumper_.dump(model, model_temp_file)
    except Exception:
        LOGGER.debug("Failed to dump model", exc_info=True)
        model_temp_file.close()
        raise

    model_temp_file.seek(0)
    upload_callback = _get_upload_callback(model_temp_file)
    experiment._log_model(
        model_name,
        model_temp_file,
        file_name=destination_filename,
        metadata=metadata,
        critical=True,
        on_model_upload=upload_callback,
        on_failed_model_upload=upload_callback,
    )


def _get_upload_callback(
    temporary_file: tempfile.NamedTemporaryFile,
) -> Callable[[Any], None]:
    return lambda response: temporary_file.close()


def comet_model_metadata(
    experiment: "comet_ml.CometExperiment",
    model_name: str,
    metadata: Dict[str, Any],
) -> None:
    experiment._log_model(
        model_name,
        io.StringIO(json.dumps(metadata)),
        file_name=comet_metadata.FILE_NAME,
        critical=True,
    )
