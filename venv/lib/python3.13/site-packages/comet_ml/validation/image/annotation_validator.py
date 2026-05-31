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
from typing import Any, Dict, List, Optional, Union

from ...annotations import Layer
from ..helpers import flatten_list
from ..result import ValidationResult
from ..validator import Validator
from .annotation_validator_helpers import (
    has_data_fields,
    has_layer_names,
    validate_data_section,
)


class ImageAnnotationValidator(Validator):
    """Annotations contains a list of annotation layers,
    where each layer contains {"name": layer_name, "data": list_of_annotations}.
    Each annotation can support bounding boxes, regions or both.
    annotations = [
        {
         "name": "Predictions",
         "data": [
            {
            "boxes": [[x, y, w, h]],
            "points": [[x1, y1, x2, y2, x3, y3 ...], ...],
            "label": // (str, required),
            "score": // (number, may be None)
            "id": // (str, optional),
            "metadata": {},
            }
         ]
        }
    ]
    """

    def __init__(self, annotations: List[Union[Dict[str, Any], Layer]]):
        if annotations is not None:
            for i, annotation in enumerate(annotations):
                if isinstance(annotation, Layer):
                    annotations[i] = annotation.to_dict()

        self.annotations = annotations
        self.validation_result: Optional[ValidationResult] = None

    def validate(self) -> ValidationResult:
        try:
            self.validation_result = ValidationResult(failed=False)
            if self.annotations is None or len(self.annotations) == 0:
                # empty annotations is not an error
                return self.validation_result

            if not has_layer_names(self.annotations):
                self.validation_result.register_failures(
                    ["Layer's 'name' is missing from one or all of image annotations"]
                )

            if not has_data_fields(self.annotations):
                self.validation_result.register_failures(
                    ["Layer's 'data' is missing from one or all of image annotations"]
                )
            else:
                validate_data_failures = self._validate_data()
                if len(validate_data_failures) > 0:
                    self.validation_result.register_failures(validate_data_failures)

        except Exception as e:
            self.validation_result = ValidationResult(
                failure_reasons=["Unexpected validation error: %r" % e]
            )

        return self.validation_result

    def _validate_data(self) -> List[str]:
        failures = []
        data_sections = flatten_list([a["data"] for a in self.annotations])
        results = [validate_data_section(d) for d in data_sections]
        for r in results:
            if r.failed():
                failures.extend(r.failure_reasons)

        return failures
