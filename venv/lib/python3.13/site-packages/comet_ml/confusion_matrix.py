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
from __future__ import print_function

import logging
import random
from collections import defaultdict

from ._typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from .convert_utils import (
    HAS_NUMPY,
    convert_tensor_to_numpy,
    convert_to_bytes,
    convert_to_list,
    convert_to_scalar,
    numpy,
)
from .logging_messages import (
    CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_PREDICTOR_WARNING,
    CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_TRUE_WARNING,
    CONFUSION_MATRIX_CONVERT_TO_MATRIX_FAILED_ERROR,
    CONFUSION_MATRIX_ERROR_RESULTING_LENGTH,
    CONFUSION_MATRIX_ERROR_WRONG_LENGTH,
    CONFUSION_MATRIX_EXAMPLE_DICT_INVALID_FORMAT,
    CONFUSION_MATRIX_EXAMPLE_INVALID_TYPE,
    CONFUSION_MATRIX_EXAMPLE_NONE,
    CONFUSION_MATRIX_IMAGE_HASH_FAILED_INFO,
    CONFUSION_MATRIX_IMAGE_INDEX_EXPERIMENT_NOT_SET_INFO,
    CONFUSION_MATRIX_IMAGE_INDEX_FAILED_TO_GENERATE_IMAGE_INFO,
    CONFUSION_MATRIX_IMAGE_INDEX_IMAGES_NOT_SET_INFO,
    CONFUSION_MATRIX_INDEX_TO_EXAMPLE_ERROR,
    CONFUSION_MATRIX_NUM_LABELS_NOT_EQUAL_CATEGORIES_EXCEPTION,
    CONFUSION_MATRIX_SHOULD_BE_TWO_DIMENSIONAL_EXCEPTION,
    CONFUSION_MATRIX_TRUE_AND_PREDICTED_REQUIRED_EXCEPTION,
    CONFUSION_MATRIX_TRUE_PREDICTED_OR_MATRIX_EXCEPTION,
)
from .utils import is_list_like, log_once_at_level

LOGGER = logging.getLogger(__name__)
LOG_ONCE_CACHE = set()  # type: Set[str]


def convert_to_matrix(matrix: Any, dtype: Optional[type] = None) -> List:
    """
    Convert an unknown item into a list of lists of scalars
    and ensure type is dtype (if given).
    """
    # First, convert it to numpy if possible:
    if hasattr(matrix, "numpy"):  # pytorch tensor
        matrix = convert_tensor_to_numpy(matrix)
    elif hasattr(matrix, "eval"):  # tensorflow tensor
        matrix = matrix.eval()

    # Next, convert to lists of scalars:
    if hasattr(matrix, "tolist"):  # numpy array
        if len(matrix.shape) != 2:
            raise ValueError(CONFUSION_MATRIX_SHOULD_BE_TWO_DIMENSIONAL_EXCEPTION)
        return matrix.tolist()
    else:
        # assume it is something we can iterate over:
        return [convert_to_list(row, dtype=dtype) for row in matrix]


def convert_to_matrix_dict(matrix: Any) -> Dict:
    """
    Convert a matrix into a sparse representation using
    dict[(x,y)] = value where value > 0.
    """
    matrix_dict = {}
    matrix = convert_to_matrix(matrix, int)
    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            if matrix[x][y] > 0:
                matrix_dict[(x, y)] = matrix[x][y]
    return matrix_dict


class ConfusionMatrix(object):
    """
    Data structure for holding a confusion matrix of values and their
    labels.
    """

    def __init__(
        self,
        y_true=None,
        y_predicted=None,
        labels=None,
        matrix=None,
        title="Confusion Matrix",
        row_label="Actual Category",
        column_label="Predicted Category",
        max_examples_per_cell=25,
        max_categories=25,
        winner_function=None,
        index_to_example_function=None,
        cache=True,
        selected=None,
        images=None,
        images_annotations=None,
        experiment=None,
        examples=None,
        **kwargs  # keyword args for index_to_example_function
    ):
        """
        Create the ConfusionMatrix data structure.

        Args:
            y_true: (optional) list of vectors representing the targets, or a list
                of integers representing the correct label. If
                not provided, then matrix may be provided.
            y_predicted: (optional) list of vectors representing predicted
                values, or a list of integers representing the output. If
                not provided, then matrix may be provided.
            images: (optional) a list of data that can be passed to
                CometExperiment.log_image()
            images_annotations: (optional) Specifies annotation layers for images.
                This parameter expects a list of lists, where each inner list defines an
                annotation layer, potentially containing bounding boxes and/or polygons.

                For a comprehensive understanding of how annotations are utilized and structured,
                please consult the documentation for `Experiment.log_image()`.

                **Important Note for Custom `index_to_example_function` Implementations:**
                If you provide your own implementation for `index_to_example_function` (bypassing the default),
                it is critical to ensure that every `images_annotations` entry is consistently passed alongside its corresponding image within each `log_image()` call.
                Furthermore, the metadata associated with these annotations must precisely align with the expectations and requirements of the `Experiment.log_image()` call.
            labels: (optional) a list of strings that name of the
                columns and rows, in order. By default, it will be
                "0" through the number of categories (e.g., rows/columns).
            matrix: (optional) the confusion matrix (list of lists).
                Must be square, if given. If not given, then it is
                possible to provide y_true and y_predicted.
            examples: (optional) if matrix is given you may also give
                examples, a dictionary of {(x, y): [index1, index2, ...]}
                where x is column and y is row of matrix
            title: (optional) a custom name to be displayed. By
                default, it is "Confusion Matrix".
            row_label: (optional) label for rows. By default, it is
                "Actual Category".
            column_label: (optional) label for columns. By default,
                it is "Predicted Category".
            max_examples_per_cell: (optional) maximum number of
                examples per cell. By default, it is 25.
            max_categories: (optional) max number of columns and rows to
                use. By default, it is 25.
            winner_function: (optional) a function that takes in an
                entire list of rows of patterns, and returns
                the winning category for each row. By default, it is argmax.
            index_to_example_function: (optional) a function
                that takes an index and returns either
                a number, a string, a URL, or a {"sample": str,
                "assetId": str} dictionary. See below for more info.
                By default, the function returns a number representing
                the index of the example.
            cache: (optional) should the results of index_to_example_function
                be cached and reused? By default, cache is True.
            selected: (optional) None, or list of selected category
                indices. These are the rows/columns that will be shown. By
                default, select is None. If the number of categories is
                greater than max_categories, and selected is not provided,
                then selected will be computed automatically by selecting
                the most confused categories.
            experiment: (optional) passed automatically when using
                [comet_ml.CometExperiment.create_confusion_matrix][].
            kwargs: (optional) any extra keywords and their values will
                be passed onto the index_to_example_function.

        Note:
            The matrix is [row][col] and [real][predicted] order. That way, the
            data appears as it is display in the confusion matrix in the user
            interface on comet.com.

        Note:
            Uses winner_function to compute winning categories for
            y_true and y_predicted, if they are vectors.

        Example:

        ```python
        >>> import comet_ml
        # Typically, you can log the y_true/y_predicted or matrix:

        >>> experiment = comet_ml.start()

        # If you have a y_true and y_predicted:
        >>> y_predicted = model.predict(x_test)
        >>> experiment.log_confusion_matrix(y_true, y_predicted)

        # Or, if you have the categories for y_true or y_predicted
        # you can just pass those in:
        >>> experiment.log_confusion_matrix([0, 1, 2, 3],
                                            [2, 2, 2, 2]) # guess 2 for all

        # Or, if you have already computed the matrix:
        >>> experiment.log_confusion_matrix(labels=["one", "two", "three"],
                                            matrix=[[10, 0, 0],
                                                    [ 0, 9, 1],
                                                    [ 1, 1, 8],])

        # However, if you want to reuse examples from previous runs,
        # you can reuse a ConfusionMatrix instance. You might want to
        # do this if you are creating a series of confusion matrices
        # during the training of a model.
        # See https://staging.comet.com/docs/quick-start/ for a tutorial.

        >>> cm = ConfusionMatrix()
        >>> y_predicted = model.predict(x_test)
        >>> cm.compute_matrix(y_true, y_predicted)
        >>> experiment.log_confusion_matrix(matrix=cm)

        # Log again, using previously cached values:
        >>> y_predicted = model.predict(x_test)
        >>> cm.compute_matrix(y_true, y_predicted)
        >>> experiment.log_confusion_matrix(matrix=cm)
        ```

        For more details and example uses, please see:

        [Log Confusion Matrix](/docs/v2/guides/experiment-management/log-data/confusion-matrix/)

        or:

        [comet_ml.CometExperiment.log_confusion_matrix][]
        """
        self.images = images
        self.images_annotations = images_annotations
        self.experiment = experiment
        if self.images is not None:
            if index_to_example_function is None:
                index_to_example_function = self.image_index_to_example_function

        if y_true is not None and y_predicted is not None:
            if matrix is not None:
                raise ValueError(CONFUSION_MATRIX_TRUE_PREDICTED_OR_MATRIX_EXCEPTION)
            # else fine
        elif y_true is None and y_predicted is None:
            pass  # fine
        elif y_true is None or y_predicted is None:
            raise ValueError(CONFUSION_MATRIX_TRUE_AND_PREDICTED_REQUIRED_EXCEPTION)

        if winner_function is not None:
            self.winner_function = winner_function
        else:
            self.winner_function = self.default_winner_function

        if index_to_example_function is not None:
            self.index_to_example_function = index_to_example_function
        else:
            self.index_to_example_function = self.default_index_to_example_function

        self.labels = labels
        self.title = title
        self.row_label = row_label
        self.column_label = column_label
        self.max_examples_per_cell = max_examples_per_cell
        self.max_categories = max_categories
        self.selected = sorted(selected) if selected is not None else None
        self.use_cache = cache
        self.clear_cache()
        self.clear()
        self._need_init = True
        self._example_matrix = {}
        self._dimension = None  # type: Optional[int]

        if y_true is not None and y_predicted is not None:
            self.compute_matrix(
                y_true,
                y_predicted,
                images=images,
                images_annotations=images_annotations,
                **kwargs
            )
        elif matrix is not None:
            if self._need_init:
                self.initialize()
            try:
                self._matrix = convert_to_matrix_dict(matrix)
                self._dimension = len(matrix)
                if examples:
                    self._process_examples(examples, **kwargs)
            except Exception:
                LOGGER.error(
                    CONFUSION_MATRIX_CONVERT_TO_MATRIX_FAILED_ERROR,
                    exc_info=True,
                )

    def need_init(self):
        """
        Method to call when you want to reset the confusion matrix.
        Doesn't reset cache.

        Note:
            This method is called automatically when the confusion matrix is logged.
        """
        self._need_init = True

    def clear(self):
        """
        Clear the matrices and type.
        """
        self._need_init = True
        self.type = None
        self._example_matrix = None
        self._matrix = None
        self._dimension = None  # type: Optional[int]
        self.images = None
        self.images_annotations = None

    def clear_cache(self):
        """
        Clear the caches.
        """
        # Set of indices (ints):
        self._cache = set()  # type: Set[int]
        # Map index (int) -> example
        self._cache_example = {}  # type: Dict[int, Any]

    def initialize(self):
        """
        Initialize the confusion matrix.
        """
        self._need_init = False
        self._matrix = {}
        self._example_matrix = {}

    def default_winner_function(self, ndarray):
        """
        A default winner function. Takes a list
        of patterns to apply winner function to.

        Args:
            ndarray: a 2-D matrix where rows are the patterns

        Returns a list of winning categories.
        """
        if HAS_NUMPY:

            def winner(ndarray):
                return numpy.argmax(ndarray, axis=1)

        else:
            # numpy is faster, but not required
            log_once_at_level(
                logging.INFO,
                "numpy not installed; using a slower "
                + "winner_function for confusion matrix",
            )

            def winner(ndarray):
                # Even if the following code is doing two iterations on the
                # list, most of the computation is done by C code
                return [array.index(max(array)) for array in ndarray]

        return winner(ndarray)

    def default_index_to_example_function(self, index, **kwargs):
        """
        User-provided function.

        Args:
            index: the index of the pattern being tested
            kwargs: additional keyword arguments for an overridden method

        Returns:
            an integer representing the winning category
            a string representing an example
            a string representing an URL (starts with "http")
            a dictionary containing keys "sample" and "assetId"

        The return dictionary is used to link a confusion matrix cell
        with a Comet asset. In this function, you can create an asset
        and return a dictionary, like so:

        ```python
        # Example index_to_example_function
        def index_to_example_function(index):
            # x_test is user's inputs (just an example):
            image_array = x_test[index]
            # Make an asset name:
            image_name = "confusion-matrix-%05d.png" % index
            # Make an asset:
            results = experiment.log_image(
                image_array, name=image_name, image_shape=(28, 28, 1)
            )
            # Return the example name and assetId
            return {"sample": image_name, "assetId": results["imageId"]}

        # Then, pass it to ConfusionMatrix(), or log_confusion_matrix()
        ```
        """
        return index

    def image_index_to_example_function(
        self, index: int, **kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        The internal index_to_example function used when
        passing in images to the compute_matrix() method.

        Args:
            index: the index of the pattern being tested
            kwargs: additional keyword arguments for [comet_ml.CometExperiment.log_image]

        Returns:
            a dictionary containing keys "sample" and "assetId"
        """
        if self.images is None:
            log_once_at_level(
                logging.INFO, CONFUSION_MATRIX_IMAGE_INDEX_IMAGES_NOT_SET_INFO
            )
            return None

        if self.experiment is None:
            log_once_at_level(
                logging.INFO,
                CONFUSION_MATRIX_IMAGE_INDEX_EXPERIMENT_NOT_SET_INFO,
            )
            return None

        image_array = self.images[index]
        image_annotations = (
            self.images_annotations[index] if self.images_annotations else None
        )

        image_name = "confusion-matrix-%05d.png" % index
        result = self.experiment.log_image(
            image_array, name=image_name, annotations=image_annotations, **kwargs
        )
        if result is None:
            log_once_at_level(
                logging.INFO, CONFUSION_MATRIX_IMAGE_INDEX_FAILED_TO_GENERATE_IMAGE_INFO
            )
            return None

        return {
            "index": index,
            "sample": image_name,
            "assetId": result["imageId"],
        }

    def _set_type_from_example(self, example):
        """
        Take the cached example and set the global
        confusion matrix type.

        Args:
            example: an int or dict
        """
        if isinstance(example, int):
            self.type = "integer"
        elif isinstance(example, dict):
            if example["assetId"] is not None:
                self.type = "image"
            elif example["sample"].startswith("http"):
                self.type = "link"
            else:
                self.type = "string"
        else:
            raise TypeError("unknown example type: %r" % example)

    def _process_new_example(self, example, index):
        """
        Turn the user's return value into a proper example.  Sets the type
        based on user's value.

        Args:
            example: a new example from user function
            index: the index of the example

        Side-effect: saves in cache if possible.
        """
        if example is None:
            LOGGER.info(
                CONFUSION_MATRIX_EXAMPLE_NONE,
                self.index_to_example_function,
                index,
            )
            return None
        elif isinstance(example, int):
            self.type = "integer"
        elif isinstance(example, str):
            if example.startswith("http"):
                self.type = "link"
            else:
                self.type = "string"
            example = {
                "index": index,  # index
                "sample": example,  # example
                "assetId": None,  # assetId
            }
        elif isinstance(example, dict):
            # a dict of index (int), assetId (string), example (string)
            if "sample" not in example or "assetId" not in example:
                LOGGER.warning(
                    CONFUSION_MATRIX_EXAMPLE_DICT_INVALID_FORMAT,
                    self.index_to_example_function,
                    index,
                )
                return None
            # Add the index, in case not already in:
            if "index" not in example:
                example["index"] = index
            # Set the confusion matrix type:
            if "type" in example:
                self.type = example["type"]
                # Remove from dict:
                del example["type"]
            else:  # default type
                self.type = "image"
        else:
            LOGGER.warning(
                CONFUSION_MATRIX_EXAMPLE_INVALID_TYPE,
                self.index_to_example_function,
                type(example),
                index,
            )
            return None
        if self.type != "integer" and self.use_cache:
            self._put_example_in_cache(index, example)
        return example

    def _index_to_example(self, index, **kwargs):
        """
        Wrapper around user function/cache.

        Args:
            index: the index of the example
            kwargs: passed to user function
        """
        if self.use_cache and self._example_in_cache(index):
            example = self._get_example_from_cache(index)
            self._set_type_from_example(example)
            return example

        try:
            example = self.index_to_example_function(index, **kwargs)
        except Exception:
            log_once_at_level(
                logging.ERROR,
                CONFUSION_MATRIX_INDEX_TO_EXAMPLE_ERROR,
                self.index_to_example_function,
                index,
                exc_info=True,
                extra={"show_traceback": True},
            )
            example = index
        example = self._process_new_example(example, index)

        return example

    def _get_cache_key(self, index):
        """
        Given an index, return a globally-unique
        key for the cache.
        """
        if self.images is not None:
            try:
                image = self.images[index]
                array = convert_tensor_to_numpy(image)
                array_bytes = convert_to_bytes(array)
                return hash(array_bytes)
            except Exception:
                log_once_at_level(
                    logging.INFO, CONFUSION_MATRIX_IMAGE_HASH_FAILED_INFO, exc_info=True
                )
                return None
        else:
            return index

    def _get_example_from_cache(self, index):
        """
        Get a example from the example cache.

        Args:
            index: the index of example
        """
        key = self._get_cache_key(index)
        return self._cache_example[key]

    def _example_in_cache(self, index):
        """
        Is the example in the example cache?

        Args:
            index: the index of example
        """
        key = self._get_cache_key(index)
        return key in self._cache_example

    def _put_example_in_cache(self, index, example):
        """
        Put a example in the example cache.

        Args:
            index: the index of example
            example: the processed example
        """
        key = self._get_cache_key(index)
        self._cache_example[key] = example

    def _example_from_list(
        self, indices: Set[int], x: int, y: int, **kwargs: Any
    ) -> Set[int]:
        """
        Example from indices so that it is no more than max length.
        Use previous indices from cache.

        Args:
            indices: the indices of the patterns to example from
            x: the column of example cell
            y: the row of example cell
            kwargs: keyword args to pass to user function
        """
        key_map = {self._get_cache_key(index): index for index in indices}
        keys = list(key_map.keys())

        if len(keys) <= self.max_examples_per_cell:
            retval = keys
        else:
            retval = list(self._cache.intersection(keys))
            # If you need more:
            retval += [
                keys.pop(random.randint(0, len(keys) - 1))
                for i in range(self.max_examples_per_cell - len(retval))
            ]

        # Return minimum needed:
        previous = len(self._example_matrix.get((x, y), []))
        retval = set(retval[: self.max_examples_per_cell - previous])

        new_ones = retval - self._cache
        if self.index_to_example_function is not None:
            examples = []
            for key in retval:
                index = key_map[key]
                example = self._index_to_example(index, **kwargs)
                if example is not None:
                    examples.append(example)

            if len(examples) > 0:
                # Allow to accumulate
                if (x, y) in self._example_matrix:
                    self._example_matrix[(x, y)].extend(examples)
                else:
                    self._example_matrix[(x, y)] = examples

        # Update the ones sent:
        if self.use_cache:
            self._cache.update(new_ones)

        return retval

    def _get_labels(self):
        if self.labels is None:
            if self.selected is not None:
                labels = [str(label) for label in self.selected]
            elif self._dimension is not None:
                labels = [str(label) for label in range(self._dimension)]
            else:
                labels = []
        elif self.selected is not None:
            labels = [
                str(label)
                for (i, label) in enumerate(self.labels)
                if i in self.selected
            ]
        else:
            labels = self.labels
        return labels

    def _get_matrix(self, labels: Optional[List[str]] = None) -> List[Any]:
        matrix = None
        if self._matrix is not None:
            if self.selected is not None:
                matrix = [
                    [self._matrix.get((row, col), 0) for col in self.selected]
                    for row in self.selected
                ]
            else:
                matrix = self._expand_dict(self._matrix, default=0, labels=labels)
        return matrix

    def _get_example_matrix(self, labels: Optional[List[str]] = None) -> List[Any]:
        smatrix = None
        if self._example_matrix is not None and len(self._example_matrix) > 0:
            smatrix = self._expand_dict(
                self._example_matrix, default=None, labels=labels
            )
            if self.selected is not None:
                smatrix = [
                    [smatrix[row][col] for col in self.selected]
                    for row in self.selected
                ]
        return smatrix

    def _expand_dict(
        self,
        matrix_dict: Dict,
        default: Union[int, None] = None,
        labels: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Expand the dictionary representation into a full matrix.
        """
        n = self._dimension
        if labels is not None and len(labels) > self._dimension:
            n = len(labels)
        matrix = [[default for y in range(n)] for x in range(n)]
        for x, y in matrix_dict:
            matrix[x][y] = matrix_dict[(x, y)]
        return matrix

    def compute_matrix(
        self,
        y_true,
        y_predicted,
        index_to_example_function=None,
        images=None,
        images_annotations=None,
        **kwargs
    ):
        """
        Compute the confusion matrix.

        Args:
            y_true: list of vectors representing the targets, or a list
                of integers representing the correct label
            y_predicted: list of vectors representing predicted
                values, or a list of integers representing the output
            images: (optional) a list of data that can be passed to
                [comet_ml.CometExperiment.log_image][]
            images_annotations: (optional) Specifies annotation layers for images.
                This parameter expects a list of lists, where each inner list defines an
                annotation layer, potentially containing bounding boxes and/or polygons.

                For a comprehensive understanding of how annotations are utilized and structured,
                please consult the documentation for `Experiment.log_image()`.

                **Important Note for Custom `index_to_example_function` Implementations:**
                If you provide your own implementation for `index_to_example_function` (bypassing the default),
                it is critical to ensure that every `images_annotations` entry is consistently passed alongside its corresponding image within each `log_image()` call.
                Furthermore, the metadata associated with these annotations must precisely align with the expectations and requirements of the `Experiment.log_image()` call.
            index_to_example_function: (optional) a function
                that takes an index and returns either
                a number, a string, a URL, or a {"sample": str,
                "assetId": str} dictionary. See below for more info.
                By default, the function returns a number representing
                the index of the example.

        Note:
            Uses winner_function to compute winning categories for
            y_true and y_predicted, if they are vectors.
        """
        self.images = images
        self.images_annotations = images_annotations
        if self.images is not None:
            if index_to_example_function is None:
                index_to_example_function = self.image_index_to_example_function

        if len(y_true) != len(y_predicted):
            raise ValueError(
                CONFUSION_MATRIX_ERROR_WRONG_LENGTH % (len(y_true), len(y_predicted))
            )

        if is_list_like(y_true[0]):
            # Winner function must work to map it to a category:
            xs = self.winner_function(y_true)  # type: Iterable[Any]
            x_dimension = len(y_true[0])
        else:
            xs = y_true
            x_dimension = max(y_true) + 1

        if is_list_like(y_predicted[0]):
            # Winner function must work to map it to a category:
            ys = self.winner_function(y_predicted)  # type: Iterable[Any]
            y_dimension = len(y_predicted[0])
        else:
            ys = y_predicted
            y_dimension = max(y_predicted) + 1

        if len(xs) != len(ys):
            LOGGER.error(CONFUSION_MATRIX_ERROR_RESULTING_LENGTH, len(xs), len(ys))
            return

        self._dimension = max(x_dimension, y_dimension)

        if index_to_example_function is not None:
            self.index_to_example_function = index_to_example_function

        # Create initial confusion matrix
        if self._need_init:
            self.initialize()

        examples = defaultdict(set)  # type: Dict[Tuple[int, int], Set[int]]
        for i, (raw_x, raw_y) in enumerate(zip(xs, ys)):
            try:
                x = convert_to_scalar(raw_x, dtype=int)  # type: int
            except TypeError:
                LOGGER.warning(
                    CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_TRUE_WARNING, raw_x
                )
                continue

            try:
                y = convert_to_scalar(raw_y, dtype=int)  # type: int
            except TypeError:
                LOGGER.warning(
                    CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_PREDICTOR_WARNING, raw_y
                )
                continue

            # Add count to cell:
            self._matrix[(x, y)] = self._matrix.get((x, y), 0) + 1
            # Add index to cell:
            examples[(x, y)].add(i)

        self._process_examples(examples, **kwargs)

    def _process_examples(self, examples, **kwargs):
        # Example all cells that have items (reuse from cache/other cells if possible):
        for key in examples:
            x, y = key
            if (self.selected is None) or (x in self.selected and y in self.selected):
                self._example_from_list(examples[key], x, y, **kwargs)

    def to_json(self):
        """
        Return the associated confusion matrix as the JSON to
        upload.
        """
        if (
            (self._matrix is not None)
            and (self._dimension > self.max_categories)
            and (self.selected is None)
        ):
            # If there is a matrix, and it is bigger than max, and selected is None
            # then we will automatically select those with most confusion:
            correct_counts = [
                (i, self._matrix.get((i, i), 0)) for i in range(self._dimension)
            ]
            ordered_rows = sorted(correct_counts, key=lambda pair: pair[1])
            self.selected = [row[0] for row in ordered_rows[: self.max_categories]]

        labels = self._get_labels()
        matrix = self._get_matrix(labels)
        smatrix = self._get_example_matrix(labels)

        if smatrix is None:
            self.type = None

        if matrix is not None:
            if len(matrix) != len(labels):
                raise ValueError(
                    CONFUSION_MATRIX_NUM_LABELS_NOT_EQUAL_CATEGORIES_EXCEPTION
                )

        return {
            "version": 1,
            "title": self.title,
            "labels": labels,
            "matrix": matrix,
            "rowLabel": self.row_label,
            "columnLabel": self.column_label,
            "maxSamplesPerCell": self.max_examples_per_cell,
            "sampleMatrix": smatrix,
            "type": self.type,
        }

    def display(self, space=4):
        """
        Display an ASCII version of the confusion matrix.

        Args:
            space: (optional) column width
        """

        def format(string):
            print(("%" + str(space) + "s") % str(string)[: space - 1], end="")

        json_format = self.to_json()
        total_width = len(json_format["matrix"]) * space
        row_label = json_format["rowLabel"] + (" " * total_width)
        format(row_label[0])
        format("")
        print(json_format["title"].center(total_width))
        format(row_label[1])
        format("")
        print(json_format["columnLabel"].center(total_width))
        format(row_label[2])
        format("")
        for row in range(len(json_format["matrix"])):
            format(json_format["labels"][row])
        print()
        format(row_label[3])
        for row in range(len(json_format["matrix"])):
            format(json_format["labels"][row])
            for col in range(len(json_format["matrix"][row])):
                format(json_format["matrix"][row][col])
            print()
            format(row_label[row + 4])
        print()
