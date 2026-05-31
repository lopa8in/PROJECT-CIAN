# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import logging
import numbers

from ._typing import Any, Dict, List, Optional, Point, Tuple
from .validation.helpers import flatten_list, validate_box, validate_polygon_points

LOGGER = logging.getLogger(__name__)


class AnnotationBase:
    """
    Base class for all annotations
    """

    def __init__(
        self,
        label: str,
        score: Optional[float] = None,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        if len(label) == 0:
            raise ValueError("label must be a non-zero length string")
        if score is not None and not isinstance(score, numbers.Number):
            raise ValueError("score must be a number if given")
        if id is not None and not isinstance(id, str):
            raise ValueError("id must be a str if given")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict if given")

        self.label = label
        self.score = score
        self.id = id
        self.metadata = metadata

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        retval = {"label": self.label}
        if self.score is not None:
            retval["score"] = self.score
        if self.id is not None:
            retval["id"] = self.id
        if self.metadata is not None:
            retval["metadata"] = self.metadata
        return retval


class Box(AnnotationBase):
    """
    A data structure containing an annotation for
    a list of bounding boxes.
    """

    def __init__(
        self,
        box: Tuple[float, float, float, float],
        label: str,
        score: Optional[float] = None,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a bounding box annotation.

        Args:
            box (tuple of int): a tuple of (x, y, width, height).
            label (str): A name; could be class name (like "dog") or specific item.
            score (float, optional): A value between 0 and 1.
            id (str, optional): A user-given ID of boxed item(s).
            metadata (dict, optional): Additional metadata to be associated with
                these boxed items.
        """
        super().__init__(label, score, id, metadata)
        self.boxes = [validate_box(box)]

    def to_dict(self) -> Dict[str, Any]:
        retval = super().to_dict()
        retval["boxes"] = self.boxes
        return retval


class Polygon(AnnotationBase):
    """
    A data structure containing an annotation for
    a collection of polygons.
    """

    def __init__(
        self,
        points: List[Point],
        label: str,
        score: Optional[float] = None,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a polygon annotation. A polygon must contain at
        least three (x, y) points. These values are given in pixels in
        association with the original image size.

        Args:
            points (list of Point): a list of (x, y) pairs.
            label (str): A name; could be class name (like "dog") or specific item.
            score (float, optional): A value between 0 and 1.
            id (str, optional): A user-given ID for polygon.
            metadata (dict, optional): Additional metadata to be associated with
                the polygon.

        """
        super().__init__(label, score, id, metadata)
        self.points = [validate_polygon_points(points)]

    def to_dict(self) -> Dict[str, Any]:
        retval = super().to_dict()
        retval["points"] = self.points
        return retval


class Layer:
    """
    An annotation layer for images.
    """

    def __init__(
        self,
        boxes: Optional[List[Box]] = None,
        polygons: Optional[List[Polygon]] = None,
        name: str = None,
    ):
        """
        Create an annotation layer, composed of Boxes or Polygons.

        Args:
            boxes (list of Box, optional): list of Boxes
            polygons (list of Polygon, optional): list of Polygons
            name (str): the name of the layer

        Note either boxes or points must be given.
        """
        if boxes:
            if not all(isinstance(box, Box) for box in boxes):
                raise ValueError("Layer boxes takes a list Boxes")
            if len(boxes) == 0:
                raise ValueError("Layer boxes, if given, must contain at least one box")

        if polygons is not None:
            if not all(isinstance(polygon, Polygon) for polygon in polygons):
                raise ValueError("Layer polygons takes a list Polygons")
            if len(polygons) == 0:
                raise ValueError(
                    "Layer polygons, if given, must contain at least one polygon"
                )

        if boxes is None and polygons is None:
            raise ValueError("Layer must contain either boxes or polygons")

        if not isinstance(name, str):
            raise ValueError("Layer name must be a string")
        if len(name) == 0:
            raise ValueError("Layer name must be a non-zero length string")

        self.annotations = []
        if boxes is not None:
            self.annotations.extend(boxes)
        if polygons:
            self.annotations.extend(polygons)
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data": [annotation.to_dict() for annotation in self.annotations],
        }
