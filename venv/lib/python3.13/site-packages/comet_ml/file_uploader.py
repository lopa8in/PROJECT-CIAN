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

""" This module handles syncing git repos with the backend. Used for pull
request features."""

import io
import logging
import os
import pathlib
import shutil
import tempfile
from typing import NamedTuple, Tuple

from . import file_utils
from ._typing import (
    IO,
    Any,
    AudioNormalization,
    Callable,
    Dict,
    MemoryUploadable,
    Optional,
    TemporaryFilePath,
    Union,
    UserText,
    ValidFilePath,
)
from .constants import (
    ASSET_TYPE_AUDIO,
    ASSET_TYPE_DEFAULT,
    ASSET_TYPE_GIT_PATCH,
    ASSET_TYPE_VIDEO,
    ASSET_TYPE_VISUALIZATION,
    SUPPORTED_VIDEO_FORMATS,
)
from .convert_utils import (
    data_to_fp,
    image_data_to_file_like_object,
    write_numpy_array_as_wav,
)
from .exceptions import AssetIsTooBig, CometException
from .file_utils import (
    get_file_extension,
    is_valid_file_path,
    write_file_like_to_tmp_file,
)
from .logging_messages import (
    EXTENSION_NOT_FOUND,
    EXTENSION_NOT_SUPPORTED,
    FILE_UPLOADER_ASSET_JSON_FAILED_ERROR,
    FILE_UPLOADER_AUDIO_NUMPY_REQUIRED_EXCEPTION,
    FILE_UPLOADER_AUDIO_SAMPLE_RATE_NONE_EXCEPTION,
    FILE_UPLOADER_AUDIO_SAMPLE_RATE_ZERO_EXCEPTION,
    FILE_UPLOADER_AUDIO_UNSUPPORTED_TYPE_EXCEPTION,
    FILE_UPLOADER_IMAGE_COVERT_ERROR,
    FILE_UPLOADER_IMAGE_COVERT_EXCEPTION,
    FILE_UPLOADER_NUMPY_MISSING_WARNING,
    FILE_UPLOADER_PREPROCESS_ASSET_MEMORY_FILE_NAME_REQUIRED_EXCEPTION,
    FILE_UPLOADER_PREPROCESS_REMOTE_ASSET_FILE_NAME_ERROR,
    FILE_UPLOADER_PROCESS_UPLOAD_BY_FILEPATH_COPY_WARNING,
    FILE_UPLOADER_SAVE_MATPLOTLIB_FAILED_WARNING,
    FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_EMPTY_EXCEPTION,
    FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_UNSUPPORTED,
    FILE_UPLOADER_SAVE_PLOTLY_FAILED_WARNING,
    FILE_UPLOADER_TOTAL_LEN_EXCEPTION,
    FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT,
    LOG_AUDIO_TOO_BIG,
    LOG_FIGURE_TOO_BIG,
    LOG_IMAGE_TOO_BIG,
    LOG_VIDEO_TOO_BIG,
    UPLOAD_ASSET_TOO_BIG,
    UPLOAD_FILE_OS_ERROR,
)
from .messages import RemoteAssetMessage, UploadFileMessage, UploadInMemoryMessage
from .utils import (
    encode_asset_data_to_json,
    encode_metadata,
    generate_guid,
    parse_remote_uri,
)
from .validation.metadata_validator import validate_metadata

LOGGER = logging.getLogger(__name__)

try:
    import numpy
except ImportError:
    LOGGER.warning(FILE_UPLOADER_NUMPY_MISSING_WARNING)
    pass

try:
    from plotly.graph_objects import Figure as PlotlyFigure
except ImportError:
    PlotlyFigure = None

try:
    from matplotlib.axes import Axes as MatplotlibAxes
except ImportError:
    MatplotlibAxes = None

OptionalUploadMessageType = Optional[Union[UploadInMemoryMessage, UploadFileMessage]]


def is_user_text(val: Any) -> bool:
    return isinstance(val, (str, bytes))


# Requests accepts either a file-object (IO, StringIO and BytesIO), a file path, string.
# We also accepts specific inputs for each logging method


def check_max_file_size(file_path: str, max_upload_size: int, too_big_msg: str) -> int:
    """Check if a file identified by its file path is bigger than the maximum
    allowed upload size. Raises AssetIsTooBig if the file is greater than the
    upload limit.
    """

    # Check the file size before reading it
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_upload_size:
            raise AssetIsTooBig(file_path, file_size, max_upload_size)

        return file_size

    except OSError:
        LOGGER.error(too_big_msg, file_path, exc_info=True)
        raise


def save_matplotlib_figure(figure: Optional[Any] = None, format: str = "svg") -> str:
    """Try saving either the current global pyplot figure or the given one
    and return None in case of error.
    """
    # Get the right figure to upload
    if figure is None:
        import matplotlib.pyplot

        # Get current global figure
        figure = matplotlib.pyplot.gcf()

    if hasattr(figure, "gcf"):
        # The pyplot module was passed as figure
        figure = figure.gcf()

    # check if this is not empty matplotlib figure
    if hasattr(figure, "get_axes") and len(figure.get_axes()) == 0:
        raise TypeError(FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_EMPTY_EXCEPTION)

    if hasattr(figure, "savefig"):
        # Save the file to a tmp_file but don't delete it, the file uploader
        # thread will take care of it
        tmp_file = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        figure.savefig(tmp_file, format=format, bbox_inches="tight")
        tmp_file.flush()
        tmp_file.close()

        return tmp_file.name
    else:
        raise TypeError(FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_UNSUPPORTED % figure)


def save_plotly_figure(figure: Optional[Any], format: str) -> str:
    """
    Save the Plotly Figure as an image.
    """
    # Save the file to a tmp_file but don't delete it, the file uploader
    # thread will take care of it
    tmp_file = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
    figure.write_image(tmp_file, format=format)
    tmp_file.flush()
    tmp_file.close()
    return tmp_file.name


def total_len(o: Any) -> int:
    if hasattr(o, "__len__"):
        return len(o)

    if hasattr(o, "getvalue"):
        # e.g. BytesIO, cStringIO.StringIO
        return len(o.getvalue())

    if hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
        except OSError:
            pass
        else:
            return os.fstat(fileno).st_size

    raise NotImplementedError(FILE_UPLOADER_TOTAL_LEN_EXCEPTION, o, o.__class__)


class AssetUploadUserInput(object):
    def __init__(self, user_input: Any) -> None:
        self.user_input = user_input


class FileUpload(AssetUploadUserInput):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

        try:
            self.size = os.path.getsize(file_path)
        except OSError:
            LOGGER.debug("Error retrieving file size for %r", file_path)
            self.size = 0


class MemoryFileUpload(AssetUploadUserInput):
    def __init__(self, user_input: Any) -> None:
        super().__init__(user_input)

        try:
            self.size = total_len(user_input)
        except Exception:
            LOGGER.debug("Error retrieving size for %r", user_input)
            self.size = 0


class UserTextFileUpload(AssetUploadUserInput):
    def __init__(self, user_input: UserText) -> None:
        super().__init__(user_input)

        try:
            self.size = total_len(user_input)
        except Exception:
            LOGGER.debug("Error retrieving size for %r", user_input)
            self.size = 0


class ObjectToConvertFileUpload(AssetUploadUserInput):
    def __init__(self, user_input: Any) -> None:
        super().__init__(user_input)
        self.size = 0


class FolderUpload(AssetUploadUserInput):
    pass


def _convert_pathlib_input(user_input: Any) -> Any:
    if isinstance(user_input, pathlib.Path):
        return str(user_input.expanduser())
    elif isinstance(user_input, (str, bytes)):
        return os.path.expanduser(user_input)

    return user_input


def dispatch_user_file_upload(
    user_input: Any,
) -> Union[
    FileUpload,
    FolderUpload,
    MemoryFileUpload,
    UserTextFileUpload,
    ObjectToConvertFileUpload,
]:

    user_input = _convert_pathlib_input(user_input)

    if isinstance(user_input, ValidFilePath) or is_valid_file_path(user_input):
        return FileUpload(user_input)
    elif hasattr(user_input, "read"):  # Support Python 2 legacy StringIO
        return MemoryFileUpload(user_input)
    elif os.path.isdir(user_input):
        return FolderUpload(user_input)
    elif is_user_text(user_input):
        return UserTextFileUpload(user_input)
    else:
        return ObjectToConvertFileUpload(user_input)


class BaseUploadProcessor(object):

    TOO_BIG_MSG = ""
    UPLOAD_TYPE = ""

    def __init__(
        self,
        user_input: Any,
        upload_limit: int,
        url_params: Optional[Dict[str, Optional[Any]]],
        metadata: Optional[Dict[str, Any]],
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: str,
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.user_input = user_input
        self.url_params = url_params
        self.metadata = validate_metadata(metadata)
        self.upload_limit = upload_limit
        self.error_message_identifier = error_message_identifier
        self.tmp_dir = tmp_dir
        self.file_size = None  # type: Optional[int]
        self.critical = critical
        self.on_asset_upload = on_asset_upload
        self.on_failed_asset_upload = on_failed_asset_upload

        self.copy_to_tmp = copy_to_tmp

        LOGGER.debug("%r created with %r", self, self.__dict__)

    def process(self) -> OptionalUploadMessageType:
        user_input = _convert_pathlib_input(self.user_input)

        if isinstance(user_input, ValidFilePath) or is_valid_file_path(user_input):
            return self.process_upload_by_filepath(user_input)
        elif hasattr(user_input, "read"):  # Support Python 2 legacy StringIO
            return self.process_io_object(user_input)
        elif is_user_text(user_input):
            return self.process_user_text(user_input)
        else:
            return self.process_upload_to_be_converted(user_input)

    # Dispatched user input method, one method per supported type in general. By
    # default, these methods raise an exception, implement them for supported
    # input type per upload type

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:
        raise TypeError(FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT % type(upload_filepath))

    def process_upload_to_be_converted(
        self, user_input: Any
    ) -> OptionalUploadMessageType:
        raise TypeError(FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT % type(user_input))

    def process_io_object(self, io_object: IO) -> OptionalUploadMessageType:
        raise TypeError(FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT % type(io_object))

    def process_user_text(self, user_text: UserText) -> OptionalUploadMessageType:
        raise TypeError(FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT % user_text)

    # Low-level common code, once we have either an IO object or a filepath to upload

    def _process_upload_by_filepath(
        self, user_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:
        try:
            self.file_size = check_max_file_size(
                user_filepath, self.upload_limit, self.TOO_BIG_MSG
            )
        except AssetIsTooBig as exc:
            if self.error_message_identifier is None:
                error_message_identifier = exc.file_path
            else:
                error_message_identifier = self.error_message_identifier

            LOGGER.error(
                self.TOO_BIG_MSG, error_message_identifier, exc.file_size, exc.max_size
            )
            return None
        except Exception:
            LOGGER.debug("Error while checking the file size", exc_info=True)
            return None

        upload_filepath = self._handle_in_memory_file_upload(user_filepath)

        # If we failed to copy the file, abort
        if not upload_filepath:
            return None

        LOGGER.debug(
            "File upload message %r, type %r, params %r",
            upload_filepath,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        # Clean only temporary files
        if isinstance(upload_filepath, TemporaryFilePath):
            clean = True
        else:
            clean = False

        if self.copy_to_tmp and not isinstance(upload_filepath, TemporaryFilePath):
            LOGGER.warning(
                FILE_UPLOADER_PROCESS_UPLOAD_BY_FILEPATH_COPY_WARNING,
                upload_filepath,
            )

        upload_message = UploadFileMessage(
            upload_filepath,
            self.UPLOAD_TYPE,
            self.url_params,
            self.metadata,
            size=self.file_size,
            clean=clean,
            critical=self.critical,
            on_asset_upload=self.on_asset_upload,
            on_failed_asset_upload=self.on_failed_asset_upload,
        )

        return upload_message

    def _handle_in_memory_file_upload(
        self, upload_filepath: ValidFilePath
    ) -> Union[None, ValidFilePath, TemporaryFilePath]:
        # If we cannot remove the uploaded file or need the file content will
        # be frozen to the time the upload call is made, pass copy_to_tmp with
        # True value
        if self.copy_to_tmp is True and not isinstance(
            upload_filepath, TemporaryFilePath
        ):
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            tmp_file.close()
            LOGGER.debug(
                "Copying %s to %s because of copy_to_tmp",
                upload_filepath,
                tmp_file.name,
            )
            try:
                shutil.copyfile(upload_filepath, tmp_file.name)
            except (OSError, IOError):
                LOGGER.error(UPLOAD_FILE_OS_ERROR, upload_filepath, exc_info=True)
                return None
            upload_filepath = TemporaryFilePath(tmp_file.name)

        return upload_filepath

    def _process_upload_io(self, io_object: IO) -> OptionalUploadMessageType:
        if self.copy_to_tmp:
            LOGGER.debug("Saving IO to tmp_file because of copy_to_tmp")
            # Convert the file-like to a temporary file on disk
            file_path = write_file_like_to_tmp_file(io_object, self.tmp_dir)
            self.copy_to_tmp = False

            # TODO it would be easier to use the same field name for a file or a figure upload
            if "fileName" in self.url_params and self.url_params["fileName"] is None:
                self.url_params["fileName"] = os.path.basename(file_path)

            if "figName" in self.url_params and self.url_params["figName"] is None:
                self.url_params["figName"] = os.path.basename(file_path)

            return self._process_upload_by_filepath(TemporaryFilePath(file_path))

        LOGGER.debug(
            "File-like upload message %r, type %r, params %r",
            io_object,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        return UploadInMemoryMessage(
            io_object,
            self.UPLOAD_TYPE,
            self.url_params,
            self.metadata,
            size=0,  # TODO: Replace by pre-processing
            critical=self.critical,
            on_asset_upload=self.on_asset_upload,
            on_failed_asset_upload=self.on_failed_asset_upload,
        )

    def _process_upload_text(self, user_text: UserText) -> OptionalUploadMessageType:
        if self.copy_to_tmp:
            # TODO: Be more efficient here
            io_object = data_to_fp(user_text)

            if not io_object:
                # We couldn't convert to an io_object
                return None

            file_path = write_file_like_to_tmp_file(io_object, self.tmp_dir)

            return self._process_upload_by_filepath(TemporaryFilePath(file_path))

        LOGGER.debug(
            "Text upload message %r, type %r, params %r",
            user_text,
            self.UPLOAD_TYPE,
            self.url_params,
        )

        return UploadInMemoryMessage(
            user_text,
            self.UPLOAD_TYPE,
            self.url_params,
            self.metadata,
            size=0,  # TODO: Replace by pre-processing
            critical=self.critical,
            on_asset_upload=self.on_asset_upload,
            on_failed_asset_upload=self.on_failed_asset_upload,
        )


class AssetUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG

    def __init__(
        self,
        user_input: Any,
        upload_type: str,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, str]],
        upload_limit: int,
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: Optional[str],
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.UPLOAD_TYPE = upload_type

        super(AssetUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:

        if self.url_params["fileName"] is None:
            self.url_params["fileName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        return self._process_upload_by_filepath(upload_filepath)

    def process_io_object(self, io_object: IO) -> OptionalUploadMessageType:
        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_io(io_object)

    def process_user_text(self, user_text: UserText) -> OptionalUploadMessageType:
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class FigureUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_FIGURE_TOO_BIG
    UPLOAD_TYPE = ASSET_TYPE_VISUALIZATION

    def __init__(
        self,
        user_input: Any,
        upload_limit: int,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, str]],
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: str,
        critical: bool,
        format: str,
        svg_size_limit: int,
        fallback_format: str = "png",
        upload_type: Optional[str] = None,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        super(FigureUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )
        if upload_type is not None:
            self.UPLOAD_TYPE = upload_type
        self._format = format
        self._fallback_format = fallback_format
        self._svg_size_limit = svg_size_limit

    def process_upload_to_be_converted(
        self, user_input: Any
    ) -> Optional[UploadFileMessage]:
        filename = self._save_figure_to_tmp(user_input)
        if filename is None:
            return None

        if self._format == "auto":
            self.url_params["extension"] = get_file_extension(filename)
        else:
            self.url_params["extension"] = self._format

        return self._process_upload_by_filepath(TemporaryFilePath(filename))

    def _save_figure_to_tmp(self, user_input) -> Optional[str]:
        if self._format == "auto":
            figure_format = "svg"
        else:
            figure_format = self._format

        if PlotlyFigure is not None and isinstance(user_input, PlotlyFigure):
            try:
                filename = save_plotly_figure(user_input, figure_format)
            except Exception:
                LOGGER.warning(
                    FILE_UPLOADER_SAVE_PLOTLY_FAILED_WARNING,
                    exc_info=True,
                )
                # An error occurred
                return None
        else:
            try:
                if MatplotlibAxes is not None and isinstance(
                    user_input, MatplotlibAxes
                ):
                    # support for a seaborn library which uses Axes classes
                    user_input = user_input.get_figure()

                filename = save_matplotlib_figure(user_input, figure_format)
            except Exception as e:
                LOGGER.warning(
                    FILE_UPLOADER_SAVE_MATPLOTLIB_FAILED_WARNING, e, exc_info=True
                )
                # An error occurred
                return None

        # check if the resulting figure is larger than the limit allowed for SVG and
        # fallback to the _fallback_format if appropriate
        file_size = os.path.getsize(filename)
        if figure_format == "svg" and file_size > self._svg_size_limit:
            self._format = self._fallback_format
            LOGGER.info(
                "Trying to log a figure that is exceeding allowed SVG file size limit (%d > %d). Falling back to %s format.",
                file_size,
                self._svg_size_limit,
                self._fallback_format.upper(),
            )
            return self._save_figure_to_tmp(user_input)

        return filename


class ImageUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_IMAGE_TOO_BIG
    UPLOAD_TYPE = ASSET_TYPE_VISUALIZATION

    def __init__(
        self,
        user_input: Any,
        name: Optional[str],
        overwrite: bool,
        image_format: str,
        image_scale: float,
        image_shape: Optional[Tuple[int, int]],
        image_colormap: Optional[str],
        image_minmax: Optional[Tuple[int, int]],
        image_channels: str,
        upload_limit: int,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, Any]],
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: str,
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.image_format = image_format
        self.image_scale = image_scale
        self.image_shape = image_shape
        self.image_colormap = image_colormap
        self.image_minmax = image_minmax
        self.image_channels = image_channels
        super().__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:

        if self.url_params["figName"] is None:
            self.url_params["figName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        return self._process_upload_by_filepath(upload_filepath)

    def process_upload_to_be_converted(
        self, user_input: Any
    ) -> OptionalUploadMessageType:
        try:
            image_object = image_data_to_file_like_object(
                user_input,
                self.name,
                self.image_format,
                self.image_scale,
                self.image_shape,
                self.image_colormap,
                self.image_minmax,
                self.image_channels,
            )
        except Exception as exc:
            LOGGER.error(FILE_UPLOADER_IMAGE_COVERT_EXCEPTION, exc, exc_info=True)
            return None

        if not image_object:
            LOGGER.error(FILE_UPLOADER_IMAGE_COVERT_ERROR, exc_info=True)
            return None

        return self._process_upload_io(image_object)

    def process_io_object(self, io_object: IO) -> OptionalUploadMessageType:
        extension = get_file_extension(self.name)
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_io(io_object)

    def process_user_text(self, user_text: UserText) -> OptionalUploadMessageType:
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class VideoUploadProcessor(BaseUploadProcessor):
    TOO_BIG_MSG = LOG_VIDEO_TOO_BIG
    UPLOAD_TYPE = ASSET_TYPE_VIDEO
    SUPPORTED_TYPES = SUPPORTED_VIDEO_FORMATS

    def __init__(
        self,
        user_input: Any,
        name: Optional[str],
        format: Optional[str],
        upload_limit: int,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, Any]],
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: str,
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.name = name
        super().__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )
        # make sure we pre-process file path before trying to get its extension
        user_input = _convert_pathlib_input(user_input)
        extension = file_utils.get_extension(
            format=format, user_input=user_input, file_name=name
        )

        if not extension:
            raise CometException(EXTENSION_NOT_FOUND)

        if extension.upper() not in self.SUPPORTED_TYPES:
            raise CometException(
                EXTENSION_NOT_SUPPORTED % (extension, ", ".join(self.SUPPORTED_TYPES))
            )

        self.url_params["extension"] = extension

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:
        if self.url_params["fileName"] is None:
            self.url_params["fileName"] = os.path.basename(upload_filepath)

        return self._process_upload_by_filepath(upload_filepath)

    def process_io_object(self, io_object: IO) -> OptionalUploadMessageType:
        return self._process_upload_io(io_object)


class AudioUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = LOG_AUDIO_TOO_BIG
    UPLOAD_TYPE = ASSET_TYPE_AUDIO

    def __init__(
        self,
        user_input: Any,
        sample_rate: Optional[int],
        normalize_data: AudioNormalization,
        upload_limit: int,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, str]],
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: str,
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.normalize_data = normalize_data

        super(AudioUploadProcessor, self).__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:
        if self.url_params["fileName"] is None:
            self.url_params["fileName"] = os.path.basename(upload_filepath)

        self.url_params["extension"] = get_file_extension(upload_filepath)

        # The file has not been sampled
        self.url_params["sampleRate"] = None

        return self._process_upload_by_filepath(upload_filepath)

    def process_upload_to_be_converted(
        self, user_input: Any
    ) -> OptionalUploadMessageType:

        try:
            if not isinstance(user_input, numpy.ndarray):
                raise TypeError(
                    FILE_UPLOADER_AUDIO_UNSUPPORTED_TYPE_EXCEPTION % type(user_input)
                )
        except NameError:
            # Numpy is not available
            raise TypeError(FILE_UPLOADER_AUDIO_NUMPY_REQUIRED_EXCEPTION)

        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        if self.sample_rate is None:
            raise TypeError(FILE_UPLOADER_AUDIO_SAMPLE_RATE_NONE_EXCEPTION)

        if self.sample_rate == 0:
            raise TypeError(FILE_UPLOADER_AUDIO_SAMPLE_RATE_ZERO_EXCEPTION)

        # Send the sampling rate to the backend
        self.url_params["sampleRate"] = self.sample_rate

        # And save it in the metadata too
        self.metadata["sample_rate"] = self.sample_rate

        # Write to a file directly to avoid temporary IO copy when we know it
        # will ends up on the file-system anyway
        if self.copy_to_tmp:
            tmp_file = tempfile.NamedTemporaryFile(delete=False)

            write_numpy_array_as_wav(
                user_input,
                self.sample_rate,
                tmp_file,
                normalize_input=self.normalize_data,
            )

            tmp_file.close()

            return self._process_upload_by_filepath(TemporaryFilePath(tmp_file.name))
        else:
            io_object = io.BytesIO()

            write_numpy_array_as_wav(
                user_input,
                self.sample_rate,
                io_object,
                normalize_input=self.normalize_data,
            )

            return self._process_upload_io(io_object)

    def process_user_text(self, user_text: UserText) -> OptionalUploadMessageType:
        LOGGER.error(UPLOAD_FILE_OS_ERROR, user_text)
        return None


class AssetDataUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG

    def __init__(
        self,
        user_input: Any,
        upload_type: str,
        url_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, str]],
        upload_limit: int,
        copy_to_tmp: bool,
        error_message_identifier: Any,
        tmp_dir: Optional[str],
        critical: bool,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
    ) -> None:
        self.UPLOAD_TYPE = upload_type
        super().__init__(
            user_input,
            upload_limit,
            url_params,
            metadata,
            copy_to_tmp,
            error_message_identifier,
            tmp_dir,
            critical,
            on_asset_upload,
            on_failed_asset_upload,
        )

    def process_upload_to_be_converted(
        self, user_input: Any
    ) -> OptionalUploadMessageType:
        # We have an object which is neither an IO object, neither a str nor bytes
        try:
            converted = encode_asset_data_to_json(user_input)
        except Exception:
            LOGGER.error(FILE_UPLOADER_ASSET_JSON_FAILED_ERROR, exc_info=True)
            return None

        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_text(converted)

    def process_user_text(self, user_text: UserText) -> OptionalUploadMessageType:
        extension = get_file_extension(self.url_params["fileName"])
        if extension is not None:
            self.url_params["extension"] = extension

        return self._process_upload_text(user_text)


class GitPatchUploadProcessor(BaseUploadProcessor):

    TOO_BIG_MSG = UPLOAD_ASSET_TOO_BIG
    UPLOAD_TYPE = ASSET_TYPE_GIT_PATCH

    def process_upload_by_filepath(
        self, upload_filepath: ValidFilePath
    ) -> Optional[UploadFileMessage]:
        return self._process_upload_by_filepath(upload_filepath)


class PreprocessedRemoteAsset(NamedTuple):
    remote_uri: str
    overwrite: bool
    upload_type: str
    metadata: Dict[str, Any]
    step: Optional[int]
    asset_id: str
    logical_path: str
    size: int

    def to_message(
        self,
        critical: bool,
        on_asset_upload: Callable,
        on_failed_asset_upload: Callable,
        experiment_url_params: Dict[str, Any],
    ) -> RemoteAssetMessage:
        return _remote_asset_to_message(
            asset=self,
            critical=critical,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            experiment_url_params=experiment_url_params,
        )

    def to_remote_model_asset_json(self) -> Dict[str, str]:
        fields = {
            "assetId": self.asset_id,
            "remoteUrl": self.remote_uri,
            "fileName": self.logical_path,
        }
        if self.metadata is not None:
            encoded_metadata = encode_metadata(self.metadata)
            if encoded_metadata:
                fields["metadata"] = encoded_metadata

        return fields


def preprocess_remote_asset(
    remote_uri: Any,
    logical_path: Optional[str],
    overwrite: bool,
    upload_type: Optional[str],
    metadata: Optional[Dict[str, str]],
    asset_id: Optional[str] = None,
    step: Optional[int] = None,
) -> PreprocessedRemoteAsset:

    if asset_id is None:
        asset_id = generate_guid()
    asset_id = asset_id

    if logical_path is None:
        # Try to parse the URI to see if we can extract a useful file name
        logical_path = parse_remote_uri(remote_uri)
        if not logical_path:
            LOGGER.info(
                FILE_UPLOADER_PREPROCESS_REMOTE_ASSET_FILE_NAME_ERROR,
                remote_uri,
            )
            logical_path = "remote"

    try:
        size = len(remote_uri)
    except Exception:
        LOGGER.debug("Couldn't compute size for remote uri %r", remote_uri)
        size = 0

    return PreprocessedRemoteAsset(
        remote_uri=remote_uri,
        overwrite=overwrite,
        upload_type=upload_type,
        metadata=validate_metadata(metadata),
        step=step,
        asset_id=asset_id,
        logical_path=logical_path,
        size=size,
    )


class PreprocessedSyncedRemoteAsset(NamedTuple):
    """Remote asset which was synced against supported cloud storage platforms"""

    remote_uri: str
    overwrite: bool
    upload_type: str
    metadata: Dict[str, Any]
    step: Optional[int]
    asset_id: str
    logical_path: str
    size: int
    local_path: str  # the path to the downloaded file

    def to_message(
        self,
        critical: bool,
        on_asset_upload: Callable,
        on_failed_asset_upload: Callable,
        experiment_url_params: Dict[str, Any],
    ) -> RemoteAssetMessage:
        return _remote_asset_to_message(
            asset=self,
            critical=critical,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
            experiment_url_params=experiment_url_params,
        )


def _remote_asset_to_message(
    asset: Union[PreprocessedRemoteAsset, PreprocessedSyncedRemoteAsset],
    critical: bool,
    on_asset_upload: Callable,
    on_failed_asset_upload: Callable,
    experiment_url_params: Dict[str, Any],
) -> RemoteAssetMessage:
    url_params = {
        "assetId": asset.asset_id,
        "fileName": asset.logical_path,
        "isRemote": True,
        "overwrite": asset.overwrite,
        "step": asset.step,
    }

    # If the asset type is more specific, include the
    # asset type as "type" in query parameters:
    if asset.upload_type is not None and asset.upload_type != ASSET_TYPE_DEFAULT:
        url_params["type"] = asset.upload_type

    url_params.update(experiment_url_params)

    return RemoteAssetMessage(
        remote_uri=asset.remote_uri,
        upload_type=asset.upload_type,
        additional_params=url_params,
        metadata=asset.metadata,
        size=asset.size,
        critical=critical,
        on_asset_upload=on_asset_upload,
        on_failed_asset_upload=on_failed_asset_upload,
    )


class PreprocessedFileAsset(NamedTuple):
    local_path_or_data: str
    upload_type: str
    logical_path: str
    metadata: Dict[str, Any]
    overwrite: bool
    copy_to_tmp: bool
    step: Optional[int]
    asset_id: str
    grouping_name: Optional[str]
    extension: Optional[str]
    size: int

    def to_message(
        self,
        critical: bool,
        on_asset_upload: Callable,
        on_failed_asset_upload: Callable,
        clean: bool,
        experiment_url_params: Dict[str, Any],
    ) -> UploadFileMessage:
        url_params = {
            "assetId": self.asset_id,
            "extension": self.extension,
            "fileName": self.logical_path,
            "overwrite": self.overwrite,
        }

        # If the asset type is more specific, include the
        # asset type as "type" in query parameters:
        if self.upload_type != ASSET_TYPE_DEFAULT:
            url_params["type"] = self.upload_type

        if self.grouping_name is not None:
            url_params["groupingName"] = self.grouping_name

        url_params.update(experiment_url_params)

        return UploadFileMessage(
            file_path=self.local_path_or_data,
            upload_type=self.upload_type,
            additional_params=url_params,
            metadata=self.metadata,
            clean=clean,
            critical=critical,
            size=self.size,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
        )

    def copy(
        self, new_local_path: str, new_copy_to_tmp: bool
    ) -> "PreprocessedFileAsset":
        return PreprocessedFileAsset(
            local_path_or_data=new_local_path,
            upload_type=self.upload_type,
            logical_path=self.logical_path,
            metadata=self.metadata,
            overwrite=self.overwrite,
            copy_to_tmp=new_copy_to_tmp,
            step=self.step,
            asset_id=self.asset_id,
            grouping_name=self.grouping_name,
            extension=self.extension,
            size=self.size,
        )


def preprocess_asset_file(
    dispatched: FileUpload,
    upload_type: str,
    file_name: Optional[str],
    metadata: Optional[Dict[str, str]],
    overwrite: bool,
    copy_to_tmp: bool,
    step: Optional[int] = None,
    grouping_name: Optional[str] = None,
    asset_id: Optional[str] = None,
) -> PreprocessedFileAsset:
    upload_filepath = dispatched.user_input

    if asset_id is None:
        asset_id = generate_guid()
    asset_id = asset_id

    if file_name is None:
        file_name = os.path.basename(upload_filepath)

    extension = get_file_extension(upload_filepath)

    size = dispatched.size

    return PreprocessedFileAsset(
        local_path_or_data=upload_filepath,
        upload_type=upload_type,
        logical_path=file_name,
        metadata=validate_metadata(metadata),
        overwrite=overwrite,
        copy_to_tmp=copy_to_tmp,
        step=step,
        asset_id=asset_id,
        grouping_name=grouping_name,
        extension=extension,
        size=size,
    )


class PreprocessedAssetFolder(list):
    pass


def preprocess_asset_folder(
    dispatched: FolderUpload,
    upload_type: str,
    logical_path: Optional[str],
    metadata: Optional[Dict[str, str]],
    overwrite: bool,
    copy_to_tmp: bool,
    step: Optional[int] = None,
    grouping_name: Optional[str] = None,
) -> PreprocessedAssetFolder:
    folder_asset = PreprocessedAssetFolder()
    user_input_path = dispatched.user_input

    if logical_path is None:
        logical_path = pathlib.Path(user_input_path).name
    else:
        logical_path = pathlib.Path(logical_path)

    for asset_folder, subs, files in os.walk(user_input_path):
        for file_name in files:
            full_asset_path = pathlib.Path(asset_folder) / file_name
            asset_file_name = logical_path / full_asset_path.relative_to(
                user_input_path
            )

            if not full_asset_path.is_file():
                continue

            dispatched = dispatch_user_file_upload(full_asset_path)

            asset_file = preprocess_asset_file(
                dispatched=dispatched,
                upload_type=upload_type,
                file_name=str(asset_file_name),
                metadata=metadata,
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
                grouping_name=grouping_name,
                step=step,
            )
            folder_asset.append(asset_file)

    return folder_asset


class PreprocessedMemoryFileAsset(NamedTuple):
    local_path_or_data: MemoryUploadable
    upload_type: str
    logical_path: str
    metadata: Dict[str, Any]
    overwrite: bool
    copy_to_tmp: bool
    step: int
    asset_id: str
    grouping_name: str
    extension: str
    size: int

    def to_message(
        self,
        critical: bool,
        on_asset_upload: Callable,
        on_failed_asset_upload: Callable,
        experiment_url_params: Dict[str, Any],
        clean: bool = False,
    ) -> UploadInMemoryMessage:

        # TODO: Clean is ignored but kept to keep the to_message API consistent with
        # PreprocessedFileAsset.to_message
        url_params = {
            "assetId": self.asset_id,
            "extension": self.extension,
            "fileName": self.logical_path,
            "overwrite": self.overwrite,
        }

        # If the asset type is more specific, include the
        # asset type as "type" in query parameters:
        if self.upload_type != ASSET_TYPE_DEFAULT:
            url_params["type"] = self.upload_type

        if self.grouping_name is not None:
            url_params["groupingName"] = self.grouping_name

        url_params.update(experiment_url_params)

        return UploadInMemoryMessage(
            file_like=self.local_path_or_data,
            upload_type=self.upload_type,
            additional_params=url_params,
            metadata=self.metadata,
            size=self.size,
            critical=critical,
            on_asset_upload=on_asset_upload,
            on_failed_asset_upload=on_failed_asset_upload,
        )

    def to_preprocessed_file_asset(self, new_local_path, new_copy_to_tmp):
        return PreprocessedFileAsset(
            local_path_or_data=new_local_path,
            upload_type=self.upload_type,
            logical_path=self.logical_path,
            metadata=self.metadata,
            overwrite=self.overwrite,
            copy_to_tmp=new_copy_to_tmp,
            step=self.step,
            asset_id=self.asset_id,
            grouping_name=self.grouping_name,
            extension=self.extension,
            size=self.size,
        )


def preprocess_asset_memory_file(
    dispatched: MemoryFileUpload,
    upload_type: str,
    file_name: Optional[str],
    metadata: Optional[Dict[str, str]],
    overwrite: bool,
    copy_to_tmp: bool,
    step: Optional[int] = None,
    grouping_name: Optional[str] = None,
    asset_id: Optional[str] = None,
) -> PreprocessedMemoryFileAsset:
    file_like = dispatched.user_input

    if asset_id is None:
        asset_id = generate_guid()
    asset_id = asset_id

    extension = get_file_extension(file_name)

    size = dispatched.size

    # XXX: Previously when no file_name was given and copy_to_tmp was set to True, we would use the
    # temporary file basename, given random file name. Now we raise an Exception in all cases
    if file_name is None:
        raise TypeError(
            FILE_UPLOADER_PREPROCESS_ASSET_MEMORY_FILE_NAME_REQUIRED_EXCEPTION
        )

    return PreprocessedMemoryFileAsset(
        local_path_or_data=file_like,
        upload_type=upload_type,
        logical_path=file_name,
        metadata=validate_metadata(metadata),
        overwrite=overwrite,
        copy_to_tmp=copy_to_tmp,
        step=step,
        asset_id=asset_id,
        grouping_name=grouping_name,
        extension=extension,
        size=size,
    )


PreprocessedAsset = Union[
    PreprocessedFileAsset,
    PreprocessedMemoryFileAsset,
    PreprocessedRemoteAsset,
    PreprocessedSyncedRemoteAsset,
    PreprocessedAssetFolder,
]


def is_asset_remote(asset: PreprocessedAsset) -> bool:
    if isinstance(asset, PreprocessedRemoteAsset) or isinstance(
        asset, PreprocessedSyncedRemoteAsset
    ):
        return True
    return False


def handle_in_memory_file_upload(
    tmp_dir: str, upload_filepath: ValidFilePath, raise_on_error: bool = False
) -> Union[None, ValidFilePath, TemporaryFilePath]:
    if not isinstance(upload_filepath, TemporaryFilePath):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
        tmp_file.close()
        LOGGER.debug(
            "Copying file %r to the temporary file: %r", upload_filepath, tmp_file.name
        )
        try:
            shutil.copyfile(upload_filepath, tmp_file.name)
        except (OSError, IOError) as e:
            if raise_on_error:
                raise e
            else:
                LOGGER.error(UPLOAD_FILE_OS_ERROR, upload_filepath, exc_info=True)
            return None

        upload_filepath = TemporaryFilePath(tmp_file.name)

    return upload_filepath
