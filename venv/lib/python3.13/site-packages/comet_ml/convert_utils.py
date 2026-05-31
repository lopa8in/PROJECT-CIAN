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
import csv
import inspect
import io
import json
import logging
import math
import numbers
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from ._typing import AudioNormalization, StringResult
from .constants import ASSET_TYPE_DATAFRAME, ASSET_TYPE_DEFAULT
from .exceptions import StringConversionException, StringValueTruncatedException
from .file_utils import get_file_extension
from .logging_messages import (
    CONVERT_DATAFRAME_INVALID_FORMAT,
    CONVERT_TABLE_INVALID_FORMAT,
    CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_COLORMAP_INFO,
    CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_EXCEPTION,
    CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_INFO,
    CONVERT_UTILS_CONVERT_MODEL_TO_STRING_FAILED_RETURN,
    CONVERT_UTILS_CONVERT_MODEL_TO_STRING_TF_TO_JSON_FAILED_WARNING,
    CONVERT_UTILS_CONVERT_TO_LIST_SHOULD_BE_ONE_DIMENSIONAL_EXCEPTION,
    CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING,
    CONVERT_UTILS_CONVERT_TO_STRING_WARNING,
    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_DICTIONARY_VALUE_WARNING,
    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_STR_VALUE_WARNING,
    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_SIZE,
    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_VALUE,
    CONVERT_UTILS_DATA_TO_FP_ERROR,
    CONVERT_UTILS_FAST_FLATTEN_NO_NUMPY_INFO,
    CONVERT_UTILS_IMAGE_BACKGROUND_COLOR_NO_ALPHA_CHANNEL_EXCEPTION,
    CONVERT_UTILS_INSTALL_NUMPY_SCIPY_ERROR,
    CONVERT_UTILS_INSTALL_PIL_ERROR,
    CONVERT_UTILS_INSTALL_PIL_NUMPY_ERROR,
    CONVERT_UTILS_INSTALL_PIL_NUMPY_MATPLOTLIB_ERROR,
    CONVERT_UTILS_INVALID_IMAGE_FILE_TYPE_ERROR,
    CONVERT_UTILS_NO_NUMPY_ERROR,
    CONVERT_UTILS_NUMPY_NOT_FOUND_WARNING,
    CONVERT_UTILS_PIL_CONVERT_ERROR,
    CONVERT_UTILS_PREPARE_DATAFRAME_WARNING,
    CONVERT_UTILS_WRONG_TYPE_EXCEPTION,
    DATAFRAME_CONVERSION_ERROR,
    EXPERIMENT_LOG_IMAGE_GPU_TENSOR_NOT_SUPPORTED,
    METRIC_ARRAY_WARNING,
)
from .utils import is_list_like, log_once_at_level

LOGGER = logging.getLogger(__name__)

INFINITY = float("inf")

TRUNCATED_INDICATOR = " [truncated]"

try:
    import numpy

    HAS_NUMPY = True
except ImportError:
    LOGGER.warning(CONVERT_UTILS_NUMPY_NOT_FOUND_WARNING)
    HAS_NUMPY = False
    numpy = None


def convert_to_bytes(item: Any) -> bytes:
    """
    Convert the item into bytes. The conversion may not
    be reversable.
    """
    if hasattr(item, "tobytes"):
        # Numpy arrays, etc.
        return item.tobytes()
    else:
        str_item = str(item)
        return str_item.encode(encoding="utf-8", errors="xmlcharrefreplace")


def convert_tensor_to_numpy(tensor: Any) -> Any:
    """
    Convert from various forms of pytorch tensors
    to numpy arrays.

    Note: torch tensors can have both "detach" and "numpy"
    methods, but numpy() alone will fail if tensor.requires_grad
    is True.
    """
    if hasattr(tensor, "detach"):  # pytorch tensor with attached gradient
        tensor = tensor.detach()

    if hasattr(tensor, "numpy"):  # pytorch tensor
        tensor = tensor.numpy()

    return tensor


def convert_to_scalar(user_data: Any, dtype: Optional[Type] = None) -> Any:
    """
    Try to ensure that the given user_data is converted back to a
    Python scalar, and of proper type (if given).
    """
    # Fast-path for types and class, we currently does not apply any conversion
    # to classes
    if inspect.isclass(user_data):

        if dtype and not isinstance(user_data, dtype):
            raise TypeError(CONVERT_UTILS_WRONG_TYPE_EXCEPTION % (user_data, dtype))

        return user_data

    # First, try to convert tensor to numpy objects
    try:
        if hasattr(user_data, "numpy"):
            user_data = user_data.numpy()
    except Exception:
        LOGGER.debug(
            "Failed to convert tensor %r to numpy object",
            user_data,
            exc_info=True,
        )

    # Then try to convert numpy object to a Python scalar
    try:
        if hasattr(user_data, "item") and callable(user_data.item):
            user_data = user_data.item()
    except Exception:
        LOGGER.debug(
            "Failed to convert object %r to Python scalar",
            user_data,
            exc_info=True,
        )

    if dtype is not None and not isinstance(user_data, dtype):
        raise TypeError(CONVERT_UTILS_WRONG_TYPE_EXCEPTION % (user_data, dtype))

    return user_data


def convert_to_list(
    items: Any, dtype: Any = None, multidimensional: bool = False
) -> List[Any]:
    """
    Take an unknown item and convert to a list of scalars
    and ensure type is dtype, if given.
    """
    # First, convert it to numpy if possible:
    if hasattr(items, "numpy"):  # pytorch tensor
        items = convert_tensor_to_numpy(items)
    elif hasattr(items, "eval"):  # tensorflow tensor
        items = items.eval()

    # Next, handle numpy array:
    if hasattr(items, "tolist"):
        if not multidimensional and len(items.shape) != 1:
            raise ValueError(
                CONVERT_UTILS_CONVERT_TO_LIST_SHOULD_BE_ONE_DIMENSIONAL_EXCEPTION
            )
        result = items.tolist()  # type: List[Any]
        return result
    else:
        # assume it is something with numbers in it:
        return [convert_to_scalar(item, dtype=dtype) for item in items]


def fix_special_floats(value: Any, _inf=INFINITY, _neginf=-INFINITY) -> Any:
    """Fix out of bounds floats (like infinity and -infinity) and Not A
    Number.
    Returns either a fixed value that could be JSON encoded or the original
    value.
    """

    try:
        value = convert_tensor_to_numpy(value)

        # Check if the value is Nan, equivalent of math.isnan
        if math.isnan(value):
            return "NaN"

        elif value == _inf:
            return "Infinity"

        elif value == _neginf:
            return "-Infinity"

    except Exception:
        # Value cannot be compared
        return value

    return value


def is_tensor_on_gpu(tensor: Any) -> bool:
    if hasattr(tensor, "is_cuda"):
        return tensor.is_cuda

    return False


def image_data_to_file_like_object(
    image_data: Union[IO[bytes], Any],
    file_name: Optional[str],
    image_format: str,
    image_scale: float,
    image_shape: Optional[Sequence[int]],
    image_colormap: Optional[str],
    image_minmax: Optional[Sequence[float]],
    image_channels: str,
) -> Union[IO[bytes], None, Any]:
    """
    Ensure that the given image_data is converted to a file_like_object ready
    to be uploaded
    """
    try:
        import PIL.Image
    except ImportError:
        PIL = None

    # Conversion from standard objects to image
    # Allow file-like objects, numpy arrays, etc.
    if hasattr(image_data, "numpy"):  # pytorch tensor
        if is_tensor_on_gpu(image_data):
            LOGGER.error(EXPERIMENT_LOG_IMAGE_GPU_TENSOR_NOT_SUPPORTED)
            return None

        array = convert_tensor_to_numpy(image_data)
        fp = array_to_image_fp(
            array,
            image_format,
            image_scale,
            image_shape,
            image_colormap,
            image_minmax,
            image_channels,
        )

        return fp
    elif hasattr(image_data, "eval"):  # tensorflow tensor
        array = image_data.eval()
        fp = array_to_image_fp(
            array,
            image_format,
            image_scale,
            image_shape,
            image_colormap,
            image_minmax,
            image_channels,
        )

        return fp
    elif PIL is not None and isinstance(image_data, PIL.Image.Image):  # PIL.Image
        # filename tells us what format to use:
        registered_image_extensions = PIL.Image.registered_extensions()
        supported_extensions = [
            extension
            for extension, registered_format in registered_image_extensions.items()
            if registered_format in PIL.Image.OPEN
        ]

        if (
            file_name is not None
            and "." in file_name
            and get_file_extension(file_name) in supported_extensions
        ):
            _, image_format = file_name.rsplit(".", 1)
        fp = image_to_fp(image_data, image_format)

        return fp
    elif image_data.__class__.__name__ == "ndarray":  # numpy array
        fp = array_to_image_fp(
            image_data,
            image_format,
            image_scale,
            image_shape,
            image_colormap,
            image_minmax,
            image_channels,
        )

        return fp
    elif hasattr(image_data, "read"):  # file-like object
        return image_data
    elif isinstance(image_data, (tuple, list)):  # list or tuples
        if not HAS_NUMPY:
            LOGGER.error(CONVERT_UTILS_NO_NUMPY_ERROR)
            return None
        array = numpy.array(image_data)
        fp = array_to_image_fp(
            array,
            image_format,
            image_scale,
            image_shape,
            image_colormap,
            image_minmax,
            image_channels,
        )
        return fp
    else:
        LOGGER.error(CONVERT_UTILS_INVALID_IMAGE_FILE_TYPE_ERROR, type(image_data))
        if PIL is None:
            LOGGER.error(CONVERT_UTILS_INSTALL_PIL_ERROR)
        return None


def array_to_image_fp(
    array: Any,
    image_format: str,
    image_scale: float,
    image_shape: Optional[Sequence[int]],
    image_colormap: Optional[str],
    image_minmax: Optional[Sequence[float]],
    image_channels: str,
) -> Optional[IO[bytes]]:
    """
    Convert a numpy array to an in-memory image
    file pointer.
    """
    image = array_to_image(
        array, image_scale, image_shape, image_colormap, image_minmax, image_channels
    )
    if not image:
        return None
    return image_to_fp(image, image_format)


def array_to_image(
    array: Any,
    image_scale: float = 1.0,
    image_shape: Optional[Sequence[int]] = None,
    image_colormap: Optional[str] = None,
    image_minmax: Optional[Sequence[float]] = None,
    image_channels: Optional[str] = None,
    mode: Optional[str] = None,
) -> Optional[Any]:
    """
    Convert a numpy array to an in-memory image.
    """
    try:
        import matplotlib
        import numpy
        import PIL.Image
    except ImportError:
        LOGGER.error(
            CONVERT_UTILS_INSTALL_PIL_NUMPY_MATPLOTLIB_ERROR,
            exc_info=True,
        )
        return None

    array = numpy.array(array)

    # Handle image transformations here
    # End up with a 0-255 PIL Image
    if image_minmax is not None:
        minmax = image_minmax
    else:  # auto minmax
        min_array, max_array = array.min(), array.max()
        if min_array == max_array:
            min_array = min_array - 0.5
            max_array = max_array + 0.5
        min_array = math.floor(min_array)
        max_array = math.ceil(max_array)
        minmax = [min_array, max_array]

    # if a shape is given, try to reshape it:
    if image_shape is not None:
        try:
            # array shape is opposite of image size(width, height)
            if len(image_shape) == 2:
                array = array.reshape(image_shape[1], image_shape[0])
            elif len(image_shape) == 3:
                array = array.reshape(image_shape[1], image_shape[0], image_shape[2])
            else:
                raise Exception(
                    CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_EXCEPTION
                    % image_shape
                )
        except Exception:
            LOGGER.info(
                CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_INFO, exc_info=True
            )

    if image_channels == "first" and len(array.shape) == 3:
        array = numpy.moveaxis(array, 0, -1)
    # If 3D, but last array is flat, make it 2D:
    if len(array.shape) == 3:
        if array.shape[-1] == 1:
            array = array.reshape((array.shape[0], array.shape[1]))
        elif array.shape[0] == 1:
            array = array.reshape((array.shape[1], array.shape[2]))
    elif len(array.shape) == 1:
        # if 1D, make it 2D:
        array = numpy.array([array])

    # Ok, now let's colorize and scale
    if image_colormap is not None:
        # Need to be in range (0,1) for colormapping:
        array = rescale_array(array, minmax, (0, 1), "float")
        try:
            if hasattr(matplotlib.cm, "get_cmap"):
                # old versions of matplotlib before v3.7
                cm_hot = matplotlib.cm.get_cmap(image_colormap)
            else:
                cm_hot = matplotlib.colormaps[image_colormap]
            array = cm_hot(array)
        except Exception:
            LOGGER.info(
                CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_COLORMAP_INFO, exc_info=True
            )
        # rescale again:
        array = rescale_array(array, (0, 1), (0, 255), "uint8")

        # Convert to RGBA:
        try:
            image = PIL.Image.fromarray(array, "RGBA")
        except (TypeError, ValueError) as e:
            array_str, array_shape = prepare_array_to_print(array)
            LOGGER.error(
                CONVERT_UTILS_PIL_CONVERT_ERROR.format(array_str, array_shape, str(e))
            )
            return None

    else:
        # Rescale (0, 255)
        array = rescale_array(array, minmax, (0, 255), "uint8")

        try:
            image = PIL.Image.fromarray(array)
        except (TypeError, ValueError) as e:
            array_str, array_shape = prepare_array_to_print(array)
            LOGGER.error(
                CONVERT_UTILS_PIL_CONVERT_ERROR.format(array_str, array_shape, str(e))
            )
            return None

    if image_scale != 1.0:
        image = image.resize(
            (int(image.size[0] * image_scale), int(image.size[1] * image_scale))
        )

    # Put in a standard mode:
    if mode:
        image = image.convert(mode)
    elif image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGB")
    return image


def dataset_to_sprite_image(
    matrix: Any,
    size: Sequence[int],
    preprocess_function: Optional[Callable] = None,
    transparent_color: Optional[Tuple[int, int, int]] = None,
    background_color_function: Optional[Callable] = None,
) -> Any:
    """
    Convert a dataset (array of arrays) into a giant image of
    images (a sprite sheet).

    Args:
        matrix: array of vectors or Images
        size: (width, height) of each thumbnail image
        preprocess_function: function to preprocess image values
        transparent_color: color to mark as transparent
        background_color_function: function that takes index, returns a color

    Returns: image
    """
    try:
        from PIL import Image
    except ImportError:
        LOGGER.error(CONVERT_UTILS_INSTALL_PIL_ERROR)
        return None

    length = len(matrix)
    sprite_size = math.ceil(math.sqrt(length))

    sprite_image = Image.new(
        mode="RGBA",
        size=(int(sprite_size * size[0]), int(sprite_size * size[1])),
        color=(0, 0, 0, 0),
    )
    if preprocess_function is not None:
        matrix = preprocess_function(matrix)
    for i, array in enumerate(matrix):
        if isinstance(array, Image.Image):
            image = array
        else:
            image = array_to_image(
                array,
                image_scale=1.0,
                image_shape=size,
                image_colormap=None,
                image_minmax=(0, 255),
                image_channels="last",
                mode="RGBA",
            )

            if image is None:
                return None

        if transparent_color is not None:
            image = image_transparent_color(image, transparent_color, threshold=1)
        if background_color_function is not None:
            color = background_color_function(i)
            image = image_background_color(image, color)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        location = (int((i % sprite_size) * size[0]), int((i // sprite_size) * size[1]))
        sprite_image.paste(image, location)
    return sprite_image


def image_background_color(image: Any, color: Tuple[int, int, int]) -> Any:
    """
    Given an image with some transparency, add a background color to it.

    Args:
        image: a PIL image
        color: a red, green, blue color tuple
    """
    try:
        from PIL import Image
    except ImportError:
        LOGGER.error(CONVERT_UTILS_INSTALL_PIL_ERROR)
        return None

    if image.mode != "RGBA":
        raise ValueError(
            CONVERT_UTILS_IMAGE_BACKGROUND_COLOR_NO_ALPHA_CHANNEL_EXCEPTION
        )

    new_image = Image.new("RGB", image.size, color)
    new_image.paste(image, (0, 0), image)
    return new_image


def image_transparent_color(
    image: Any, color: Tuple[int, int, int], threshold: int = 1
) -> Any:
    """
    Given a color, make that be the transparent color.

    Args:
        image: a PIL image
        color: a red, green, blue color tuple
        threshold: the max difference in each color component
    """
    try:
        import numpy
        from PIL import Image
    except ImportError:
        LOGGER.error(CONVERT_UTILS_INSTALL_PIL_NUMPY_ERROR)
        return

    image = image.convert("RGBA")
    array = numpy.array(numpy.asarray(image))
    r, g, b, a = numpy.rollaxis(array, axis=-1)
    mask = (
        (numpy.abs(r - color[0]) < threshold)
        & (numpy.abs(g - color[1]) < threshold)
        & (numpy.abs(b - color[2]) < threshold)
    )
    array[mask, 3] = 0
    return Image.fromarray(array, mode="RGBA")


def image_to_fp(image: Any, image_format: str) -> IO[bytes]:
    """
    Convert a PIL.Image into an in-memory file
    pointer.
    """
    if image_format.upper() == "JPG":
        # fix image format to be compatible with PIL library
        image_format = "JPEG"

    fp = io.BytesIO()
    image.save(fp, format=image_format)  # save the content to fp
    fp.seek(0)
    return fp


def rescale_array(array, old_range, new_range, dtype):
    """
    Given a numpy array in an old_range, rescale it
    into new_range, and make it an array of dtype.
    """
    if not HAS_NUMPY:
        LOGGER.error(CONVERT_UTILS_NO_NUMPY_ERROR)
        return

    old_min, old_max = old_range
    if array.min() < old_min or array.max() > old_max:
        ## truncate:
        array = numpy.clip(array, old_min, old_max)
    new_min, new_max = new_range
    old_delta = float(old_max - old_min)
    new_delta = float(new_max - new_min)
    if old_delta == 0:
        return ((array - old_min) + (new_min + new_max) / 2.0).astype(dtype)
    else:
        return (new_min + (array - old_min) * new_delta / old_delta).astype(dtype)


def _field_names_from_headers(
    columns: int, headers: Union[Sequence[str], bool]
) -> List[str]:
    field_names = []
    if isinstance(headers, bool):
        # headers specified as True - create dummy column names
        for i in range(1, columns + 1):
            field_names.append("column%s" % i)
    else:
        field_names = headers

    # make sure to limit field names length to specified number of columns
    return field_names[:columns]


def table_to_fp(
    tabular_data: Any, headers: Union[Sequence[str], bool], format: str
) -> io.StringIO:
    if isinstance(headers, bool):
        add_headers = headers
    else:
        add_headers = True
    fp = io.StringIO()

    if format == "csv":
        dialect = csv.excel
    else:
        dialect = csv.excel_tab

    if isinstance(tabular_data[0], (numbers.Number, str)):
        columns_count = 1
    else:
        columns_count = len(flatten(tabular_data[0]))

    field_names = None
    if add_headers:
        field_names = _field_names_from_headers(columns_count, headers=headers)
        csv_writer = csv.DictWriter(fp, dialect=dialect, fieldnames=field_names)
        csv_writer.writeheader()
    else:
        csv_writer = csv.writer(fp, dialect=dialect)

    for row in tabular_data:
        if (
            add_headers
            and field_names is not None
            and len(field_names) == columns_count
        ):
            columns = [row] if columns_count == 1 else flatten(row)
            row_to_write = {}
            for i, column in enumerate(columns):
                row_to_write[field_names[i]] = column
        else:
            row_to_write = [row] if columns_count == 1 else flatten(row)
        csv_writer.writerow(row_to_write)

    fp.seek(0)
    return fp


def data_to_fp(data: Union[bytes, str, Any]) -> Optional[IO]:
    if isinstance(data, str):
        fp = io.StringIO()
        fp.write(data)
    elif isinstance(data, bytes):
        fp = io.BytesIO()
        fp.write(data)
    else:
        fp = io.StringIO()
        try:
            json.dump(data, fp)
        except Exception:
            LOGGER.error(CONVERT_UTILS_DATA_TO_FP_ERROR, exc_info=True)
            return None
    fp.seek(0)
    return fp


def write_numpy_array_as_wav(
    numpy_array: Any,
    sample_rate: int,
    file_object: IO[bytes],
    normalize_input: AudioNormalization,
) -> None:
    """Convert a numpy array to a WAV file using the given sample_rate and
    write it to the file object
    """
    try:
        import numpy
        from scipy.io.wavfile import write
    except ImportError:
        LOGGER.error(CONVERT_UTILS_INSTALL_NUMPY_SCIPY_ERROR)
        return

    if normalize_input == AudioNormalization.PCM_16_BIT:
        # normalize input data to fit int16
        array_max = numpy.max(numpy.abs(numpy_array))
        numpy_array = numpy.int16(numpy_array / array_max * 32767)

    write(file_object, sample_rate, numpy_array)


def lazy_flatten(iterable):
    if hasattr(iterable, "flatten"):
        iterable = iterable.flatten()
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, (numbers.Number, str)):
            yield value
        else:
            if hasattr(value, "flatten"):
                value = value.flatten()  # type: ignore
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


def flatten(items):
    """
    Given a nested list or a numpy array,
    return the data flattened.
    """
    if isinstance(items, (numbers.Number, str)):
        return items
    return list(lazy_flatten(items))


def fast_flatten(items):
    """
    Given a nested list or a numpy array,
    return the data flattened.
    """
    if isinstance(items, (numbers.Number, str)):
        return items

    try:
        items = convert_tensor_to_numpy(items)
    except Exception:
        LOGGER.debug("unable to convert tensor; continuing", exc_info=True)

    if HAS_NUMPY:
        try:
            # Vector, Matrix conversion:
            items = numpy.array(items, dtype=float)
            # Return numpy array:
            return items.reshape(-1)
        except Exception:
            try:
                # Uneven conversion, 2 deep:
                items = numpy.array([numpy.array(item) for item in items], dtype=float)
                return items.reshape(-1)
            except Exception:
                # Fall through
                LOGGER.debug(
                    "numpy unable to convert items in fast_flatten", exc_info=True
                )
                return numpy.array(flatten(items))
    else:
        log_once_at_level(
            logging.INFO,
            CONVERT_UTILS_FAST_FLATTEN_NO_NUMPY_INFO,
        )
        return flatten(items)


def convert_dict_to_string(user_data: Mapping, raise_on_warning: bool = False) -> str:
    try:
        return json.dumps(user_data, sort_keys=True)
    except TypeError:
        retval = {}
        for key in user_data:
            try:
                value = convert_to_string(
                    user_data[key], raise_on_warning=raise_on_warning
                )
            except Exception:
                value = str(user_data[key])
            retval[key] = value
        return str(retval)


def convert_to_string(
    user_data: Any, source: Optional[str] = None, raise_on_warning: bool = False
) -> str:
    """
    Given an object, return it as a string.
    """
    if isinstance(user_data, Mapping):
        return convert_dict_to_string(user_data, raise_on_warning=raise_on_warning)

    # hydra ConfigNode special case:
    if hasattr(user_data, "node"):
        return convert_dict_to_string(user_data.node, raise_on_warning=raise_on_warning)

    if hasattr(user_data, "numpy"):
        user_data = convert_tensor_to_numpy(user_data)

    if isinstance(user_data, bytes) and not isinstance(user_data, str):
        user_data = user_data.decode("utf-8")

    if isinstance(user_data, str):
        return user_data

    if source is not None:
        try:
            user_data = repr(user_data)
        except Exception:
            LOGGER.debug("Cannot get user_data repr", exc_info=True)
        finally:
            message = CONVERT_UTILS_CONVERT_TO_STRING_WARNING % (source, user_data)
            if raise_on_warning:
                raise StringConversionException(message)
            else:
                log_once_at_level(logging.WARNING, message)

    return str(user_data)


def convert_to_string_truncated(
    user_data: Any,
    size: int,
    source: Optional[str] = None,
    raise_on_warning: bool = False,
) -> StringResult:
    value = convert_to_string(user_data, source, raise_on_warning=raise_on_warning)
    full_value = value
    truncated = False
    if len(value) > size:
        truncated = True
        if size < len(TRUNCATED_INDICATOR):
            value = value[:size]
        else:
            value = value[: size - len(TRUNCATED_INDICATOR)] + TRUNCATED_INDICATOR
    return StringResult(value, truncated, full_value=full_value)


def convert_to_string_key(
    user_data: Any, max_length: int, raise_on_warning: bool = False
) -> StringResult:
    res = convert_to_string_truncated(
        user_data, size=max_length, raise_on_warning=raise_on_warning
    )
    if res.truncated and raise_on_warning:
        raise StringValueTruncatedException(
            CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING
            % (max_length, res.full_value)
        )
    else:
        return convert_to_string_truncated(user_data, size=max_length)


def convert_to_string_value(
    user_data: Any,
    max_length: int,
    source: Optional[str] = None,
    raise_on_warning: bool = False,
) -> str:
    res = convert_to_string_truncated(
        user_data, max_length, source=source, raise_on_warning=raise_on_warning
    )
    if res.truncated:
        message = CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING % (
            max_length,
            res.full_value,
        )
        if raise_on_warning:
            raise StringValueTruncatedException(message)
        else:
            LOGGER.warning(message)

    return res.value


def convert_model_to_string(model: Any) -> str:
    """
    Given a model of some kind, convert to a string.
    """
    if type(model).__name__ == "Graph":  # Tensorflow Graph Definition
        try:
            from google.protobuf import json_format

            graph_def = model.as_graph_def()
            model = json_format.MessageToJson(graph_def, sort_keys=True)
        except Exception:
            LOGGER.warning(
                CONVERT_UTILS_CONVERT_MODEL_TO_STRING_TF_TO_JSON_FAILED_WARNING,
                exc_info=True,
            )

    if hasattr(model, "to_json"):
        # First, try with sorted keys:
        try:
            model = model.to_json(sort_keys=True)
        except Exception:
            model = model.to_json()
    elif hasattr(model, "to_yaml"):
        model = model.to_yaml()

    try:
        return str(model)
    except Exception:
        LOGGER.warning(CONVERT_UTILS_CONVERT_MODEL_TO_STRING_FAILED_RETURN)
        return CONVERT_UTILS_CONVERT_MODEL_TO_STRING_FAILED_RETURN


def convert_object_to_dictionary(obj: Any) -> Dict[str, Any]:
    """
    This function takes an object and turns it into
    a dictionary. It turns all properties (including
    computed properties) into a {property_name: string, ...}
    dictionary.
    """
    # hydra ConfigStore special case:
    if obj.__class__.__module__.startswith("hydra.core") and hasattr(obj, "repo"):
        return obj.repo

    dic = {}
    for attr in dir(obj):
        # Python 2 exposed some "internal" functions attributed as `func_X`. They were renamed in
        # `__X__` in Python 3.
        if attr.startswith("__") or attr.startswith("to_") or attr.startswith("func_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        try:
            dic[attr] = str(value)
        except Exception:
            pass
    return dic


def prepare_dataframe(
    dataframe: Any, asset_format: Optional[str], **kwargs: Optional[dict]
) -> Optional[io.StringIO]:
    """
    Log a pandas dataframe.
    """
    fp = io.StringIO()
    if asset_format == "json":
        dataframe.to_json(fp, **kwargs)
    elif asset_format == "csv":
        dataframe.to_csv(fp, **kwargs)
    elif asset_format == "md":
        dataframe.to_markdown(fp, **kwargs)
    elif asset_format == "html":
        dataframe.to_html(fp, **kwargs)
    else:
        LOGGER.warning(
            CONVERT_UTILS_PREPARE_DATAFRAME_WARNING,
            asset_format,
        )
        return None

    fp.seek(0)
    return fp


def check_is_pandas_dataframe(dataframe: Any) -> Optional[bool]:
    """
    Is it like a dataframe? For example, does it have
    to_markdown(), to_html() methods?
    """
    try:
        from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
    except ImportError:
        return None

    return isinstance(dataframe, (ABCDataFrame, ABCSeries))


def check_is_matplotlib_figure(figure: Any) -> bool:
    return figure.__class__.__name__ == "Figure" and hasattr(figure, "axes")


def convert_log_table_input_to_io(
    filename: str,
    tabular_data: Optional[Any],
    headers: Union[Sequence[str], bool],
    format_kwargs: Dict[str, Any],
    catch_exception: bool = True,
) -> Optional[Tuple[IO, str]]:
    # Get the extension
    if "." in filename:
        format = filename.rsplit(".", 1)[-1]
    else:
        format = ""

    if check_is_pandas_dataframe(tabular_data):
        if format not in ["json", "csv", "md", "html"]:
            if catch_exception:
                LOGGER.error(CONVERT_DATAFRAME_INVALID_FORMAT, format)
                return None
            else:
                raise ValueError(CONVERT_DATAFRAME_INVALID_FORMAT % format)

        try:
            dataframe_fp = prepare_dataframe(tabular_data, format, **format_kwargs)
        except Exception:
            if catch_exception is False:
                raise

            LOGGER.error(DATAFRAME_CONVERSION_ERROR, format, exc_info=True)
            return None

        if dataframe_fp:
            return dataframe_fp, ASSET_TYPE_DATAFRAME
        else:
            return None

    else:
        if format not in ["tsv", "csv"]:
            if catch_exception:
                LOGGER.error(CONVERT_TABLE_INVALID_FORMAT, format)
                return None
            else:
                raise ValueError(CONVERT_TABLE_INVALID_FORMAT % format)

        fp = table_to_fp(tabular_data, headers, format=format)
        return fp, ASSET_TYPE_DEFAULT


def convert_user_input_to_metric_value(
    user_input: Any, max_length: int, raise_on_warning: bool = False
) -> Union[int, float, str]:
    value = convert_to_scalar(user_input)

    if isinstance(user_input, Mapping):
        if raise_on_warning:
            raise TypeError(
                CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_DICTIONARY_VALUE_WARNING
            )
        else:
            log_once_at_level(
                logging.WARNING,
                CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_DICTIONARY_VALUE_WARNING,
            )
        value = convert_dict_to_string(user_input, raise_on_warning=raise_on_warning)
    elif is_list_like(value):
        # Try to get the first value of the Array
        try:
            if len(value) != 1:
                raise TypeError(
                    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_SIZE % value
                )

            if not isinstance(value[0], (int, float)) or isinstance(value, bool):
                raise TypeError(
                    CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_VALUE % value
                )

            value = value[0]
        except TypeError as ex:
            if raise_on_warning:
                raise ex
            else:
                LOGGER.warning(METRIC_ARRAY_WARNING, value)

            value = convert_to_string_value(
                value, max_length=max_length, raise_on_warning=raise_on_warning
            )
    elif isinstance(value, str):
        if raise_on_warning:
            raise TypeError(
                CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_STR_VALUE_WARNING
            )
        else:
            log_once_at_level(
                logging.WARNING,
                CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_STR_VALUE_WARNING,
            )
        value = convert_to_string_value(
            value, max_length=max_length, raise_on_warning=raise_on_warning
        )
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        value = convert_to_string_value(
            value,
            source="metric value",
            max_length=max_length,
            raise_on_warning=raise_on_warning,
        )

    return value


def try_convert_string_to_float(value: str) -> Union[float, str]:
    try:
        value = float(value)
    except ValueError:
        pass

    return value


def prepare_array_to_print(array: "numpy.ndarray") -> Tuple[str, str]:
    """
    Make and cut ndarray string representation to 5 lines maximum:
    2 lines from the beginning + ... + 2 lines from the end
    """

    array_str_lines = str(array).splitlines()

    if len(array_str_lines) > 5:
        line_with_indent = array_str_lines[1]
        indent_length = len(line_with_indent) - len(line_with_indent.lstrip())
        indent = " " * indent_length
        array_str_lines = array_str_lines[:2] + [indent + "..."] + array_str_lines[-2:]

    array_str = "\n".join(array_str_lines)
    shape_str = str(array.shape)

    return array_str, shape_str


def try_move_to_cpu(tensor: Any) -> Any:
    """
    Attempts to move the given Tensor object to the CPU if it has a 'cpu' method. This
    function checks if the object has the attribute 'cpu', and if so, invokes the
    method to return the object moved to the CPU. If the 'cpu' attribute is not
    present, the object is returned unchanged.

    Args:
        tensor: The Tensor object to attempt to move to CPU.

    Returns:
        The new Tensor object detached and moved to CPU if a 'cpu' method exists; otherwise, the
        unchanged object.
    """
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        return tensor.cpu()
    return tensor
