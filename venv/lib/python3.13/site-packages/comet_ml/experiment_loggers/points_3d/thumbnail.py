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
import tempfile
from collections import defaultdict

from ..._typing import Any, Tuple
from . import math_3d

LOGGER = logging.getLogger(__name__)


def create_thumbnail(
    points_filename, boxes_filename, x, y, z, min_max_x, min_max_y, min_max_z
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        LOGGER.error(
            "The Python library Pillow is required to generate a 3D Cloud thumbnail"
        )
        return None

    size = (250, 250)
    background_color = (51, 51, 77)

    image = Image.new("RGB", size, background_color)
    canvas = ImageDraw.Draw(image)

    midpoint = [
        (min_max_x[0] + min_max_x[1]) / 2,
        (min_max_y[0] + min_max_y[1]) / 2,
        (min_max_z[0] + min_max_z[1]) / 2,
    ]

    x_range = abs(min_max_x[0] - min_max_x[1])
    y_range = abs(min_max_y[0] - min_max_y[1])

    scale = min(
        size[0] / (x_range if x_range != 0 else 1),
        size[1] / (y_range if y_range != 0 else 1),
    )
    transform = math_3d.identity()
    transform = math_3d.matmul(
        transform, math_3d.translate_xyz(*[-n for n in midpoint])
    )
    transform = math_3d.matmul(transform, math_3d.rotate_z(z))
    transform = math_3d.matmul(transform, math_3d.rotate_x(x))
    transform = math_3d.matmul(transform, math_3d.rotate_y(y))
    transform = math_3d.matmul(transform, math_3d.scale_xyz(scale, scale, scale))
    transform = math_3d.matmul(
        transform, math_3d.translate_xyz(size[0] / 2, size[1] / 2, 0)
    )

    fcanvas = defaultdict(lambda: None)
    if points_filename:
        with open(points_filename) as fp:
            line = fp.readline()
            while line:
                data = json.loads(line)
                point = data[:3]
                if len(data) > 3:
                    color = tuple([int(round(c)) for c in data[3:]])
                    if not _is_color_valid(color, "points"):
                        return None
                else:
                    color = (255, 255, 255)
                math_3d.draw_point_fake(size, fcanvas, transform, point, color)
                line = fp.readline()

    if fcanvas:
        for x, y in fcanvas:
            color = fcanvas[(x, y)]["color"]
            canvas.point((x, y), fill=color)

    if boxes_filename:
        with open(boxes_filename) as fp:
            line = fp.readline()
            while line:
                data = json.loads(line)
                if "color" in data and data["color"]:
                    color = tuple(data["color"])
                    if not _is_color_valid(color, "boxes"):
                        return None
                else:
                    color = (255, 255, 255)
                for points in data["segments"]:
                    point1 = points[0]
                    for point2 in points[1:]:
                        math_3d.draw_line(
                            size, canvas, transform, point1, point2, color
                        )
                        point1 = point2
                line = fp.readline()

    tmp_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    image.save(tmp_file, "png")

    return tmp_file.name


def _is_color_valid(primary_color: Tuple, shape: str):
    for color in primary_color:
        if not 0 <= color <= 255:
            LOGGER.error(
                "%s is not a valid color for %s, please choose a number between 0 to 255."
                % (color, shape)
            )
            return False

    return True
