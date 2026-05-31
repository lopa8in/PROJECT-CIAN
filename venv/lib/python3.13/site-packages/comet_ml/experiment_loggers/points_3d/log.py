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

import json
import logging
import platform
import tempfile

from ... import constants, messages, summary
from ..._typing import Any, Callable, Dict, List, Optional, Point3D
from ...assets import asset_item
from ...constants import ASSET_TYPE_ASSET_ITEM
from . import randomize, thumbnail

LOGGER = logging.getLogger(__name__)
RANDOMIZE_MEMORY_SIZE = 1000000  # 1 MB RAM
SHUFFLE_FILE_SPLITS = 20  # Number of files to split into, if too big for memory


def log(
    scene_name: str,
    points: Optional[List[Point3D]],
    boxes: Optional[List[Dict[str, Any]]],
    metadata: Optional[Dict],
    step: Optional[int],
    points_3d_upload_limits: Dict[str, int],
    summary: summary.Summary,
    enqueue_message_callback: Callable,
):
    if not points and not boxes:
        LOGGER.warning("Either points or boxes are required.")
        return None

    min_max_x = [float("inf"), float("-inf")]
    min_max_y = [float("inf"), float("-inf")]
    min_max_z = [float("inf"), float("-inf")]

    items = []
    randomized_points_file = None
    boxes_file = None
    points_length_limit = points_3d_upload_limits["maxPoints"]
    boxes_length_limit = points_3d_upload_limits["maxBoxes"]

    if points:
        count = 0
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as points_file:
            for point in points:
                min_max_x = min(point[0], min_max_x[0]), max(point[0], min_max_x[1])
                min_max_y = min(point[1], min_max_y[0]), max(point[1], min_max_y[1])
                min_max_z = min(point[2], min_max_z[0]), max(point[2], min_max_z[1])

                points_file.write(json.dumps(point))
                points_file.write("\n")
                count += 1

                if count > points_length_limit:
                    LOGGER.warning(
                        "Provided points length is over limit, limiting points to %s"
                        % str(points_length_limit)
                    )
                    break

        if platform.system() == "Windows":  # TODO: CM-9581
            items.append(
                asset_item.AssetItem(
                    points_file.name,
                    constants.POINTS_FILE_NAME,
                    "points",
                )
            )
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as randomized_points_file:
                randomize.shuffle(
                    points_file.name,
                    randomized_points_file.name,
                    RANDOMIZE_MEMORY_SIZE,
                    SHUFFLE_FILE_SPLITS,
                )

                items.append(
                    asset_item.AssetItem(
                        randomized_points_file.name,
                        constants.POINTS_FILE_NAME,
                        "points",
                    )
                )

    if boxes:
        count = 0
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as boxes_file:
            for box in boxes:
                for segment_points in box["segments"]:
                    for point in segment_points:
                        min_max_x = min(point[0], min_max_x[0]), max(
                            point[0], min_max_x[1]
                        )
                        min_max_y = min(point[1], min_max_y[0]), max(
                            point[1], min_max_y[1]
                        )
                        min_max_z = min(point[2], min_max_z[0]), max(
                            point[2], min_max_z[1]
                        )

                boxes_file.write(json.dumps(box))
                boxes_file.write("\n")
                count += 1

                if count > boxes_length_limit:
                    LOGGER.warning(
                        "Provided boxes length is over limit, limiting boxes to %s"
                        % str(boxes_length_limit)
                    )
                    break

            items.append(
                asset_item.AssetItem(
                    boxes_file.name, constants.BOXES_FILE_NAME, "boxes"
                )
            )

    if metadata is None:
        metadata = {}

    metadata["xRange"] = list(min_max_x)
    metadata["yRange"] = list(min_max_y)
    metadata["zRange"] = list(min_max_z)

    thumbnail_path = thumbnail.create_thumbnail(
        randomized_points_file.name if randomized_points_file else None,
        boxes_file.name if boxes_file else None,
        45,
        0,
        45,
        min_max_x,
        min_max_y,
        min_max_z,
    )

    if thumbnail_path is None:
        return None

    log_3d_cloud_message = messages.Log3DCloudMessage(
        scene_name,
        upload_type=ASSET_TYPE_ASSET_ITEM,
        items=items,
        thumbnail_path=thumbnail_path,
        metadata=metadata,
        step=step,
    )

    enqueue_message_callback(log_3d_cloud_message)
    summary.increment_section("uploads", "3d point cloud")
