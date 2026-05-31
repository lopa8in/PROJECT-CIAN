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
import json
import logging
import os
from typing import Any, Dict, List, NamedTuple

import semantic_version

from ..artifact import Artifact

DATASET_METADATA_FILENAME = "comet_hf_dataset_metadata.json"


LOGGER = logging.getLogger(__name__)


class DataSetMetadata(NamedTuple):
    name: str
    version: semantic_version.Version
    assets: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = self._asdict()
        d["version"] = str(d["version"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSetMetadata":
        return cls(
            name=d["name"],
            version=semantic_version.Version(d["version"]),
            assets=d["assets"],
        )


def read_dataset_metadata(dataset_path: str) -> DataSetMetadata:
    metadict = json.load(open(os.path.join(dataset_path, DATASET_METADATA_FILENAME)))
    LOGGER.debug("Read dataset metadata: %s", metadict)
    return DataSetMetadata.from_dict(metadict)


def write_dataset_metadata(dataset_path: str, artifact: Artifact) -> None:
    assets = []
    for asset in artifact.assets:
        if asset.local_path_or_data is not None:
            assets.append(asset.id)

    dataset_metadata = DataSetMetadata(
        name=artifact.name,
        version=artifact.version,
        assets=assets,
    )

    metafile = os.path.join(dataset_path, DATASET_METADATA_FILENAME)
    with open(metafile, "w") as f:
        json.dump(dataset_metadata.to_dict(), f)

    LOGGER.debug(
        "Wrote dataset metadata: %s to: [%s] file.", dataset_metadata, metafile
    )
