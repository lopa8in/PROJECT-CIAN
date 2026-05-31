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

import calendar
import getpass
import io
import json
import logging
import os
import os.path
import platform
import random
import string
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

from ._jupyter import _in_ipython_environment
from .exceptions import StringValueTruncatedException

if sys.version_info[:2] >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

from urllib.parse import urlencode, urljoin, urlparse, urlsplit, urlunparse, urlunsplit

from requests.models import PreparedRequest

from .json_encoder import NestedEncoder, NestedNumpyOrTensorEncoder
from .logging_messages import (
    CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING,
    MISSING_PANDAS_OR_YDATA_PROFILING,
)

LOGGER = logging.getLogger(__name__)
LOG_ONCE_CACHE: Set[str] = set()

AWS_LAMBDA_ENVIRONMENT_MARKER_KEY = "LAMBDA_TASK_ROOT"

if hasattr(time, "monotonic"):
    get_time_monotonic = time.monotonic
else:
    # Python2 just won't have accurate time durations
    # during clock adjustments, like leap year, etc.
    get_time_monotonic = time.time


def log_once_at_level(
    logging_level: int, message: str, *args: Any, **kwargs: Any
) -> None:
    """
    Log the given message once at the given level then at the DEBUG
    level on further calls.

    This is a global log-once-per-session, as opposed to the
    log-once-per-experiment.
    """
    global LOG_ONCE_CACHE

    if message not in LOG_ONCE_CACHE:
        LOG_ONCE_CACHE.add(message)
        LOGGER.log(logging_level, message, *args, **kwargs)
    else:
        LOGGER.debug(message, *args, **kwargs)


def remove_log_once_at_level(message: str) -> None:
    """
    Clears the specified message from the global log-once-per-session cache, enabling it to be logged at the original
    level again when ``log_once_at_level()`` is subsequently invoked.
    """
    global LOG_ONCE_CACHE

    LOG_ONCE_CACHE.discard(message)


def merge_url(url: str, params: Dict[Any, Any]) -> str:
    """
    Given an URL that might have query strings,
    combine with additional query strings.

    Args:
        url - a url string (perhaps with a query string)
        params - a dict of additional query key/values

    Returns: a string
    """
    req = PreparedRequest()
    req.prepare_url(url, params)
    return req.url


def url_join(base: str, *parts: Any, **kwargs: Any) -> str:
    """Given a base and url parts (for example [workspace, project, id]) returns a full URL"""
    # TODO: Enforce base to have a scheme and netloc?
    result = base

    for part in parts[:-1]:
        if not part.endswith("/"):
            raise ValueError("Intermediary part not ending with /")

        result = urljoin(result, part)

    result = urljoin(result, parts[-1])

    if kwargs:
        # merge the url with additional query args:
        result = merge_url(result, kwargs)

    return result


def is_iterable(value: Any) -> bool:
    try:
        iter(value)
        return True

    except (TypeError, AttributeError):
        return False


def is_list_like(value: Any) -> bool:
    """Check if the value is a list-like"""
    if is_iterable(value) and not isinstance(value, str):
        return True

    else:
        return False


def to_utf8(str_or_bytes: AnyStr) -> AnyStr:
    if hasattr(str_or_bytes, "decode"):
        return str_or_bytes.decode("utf-8", errors="replace")

    return str_or_bytes


def local_timestamp() -> int:
    """Return a timestamp in a format expected by the backend (milliseconds)"""
    now = datetime.now(timezone.utc)
    timestamp_in_seconds = calendar.timegm(now.timetuple()) + (now.microsecond / 1e6)
    return timestamp_milliseconds(timestamp_in_seconds)


def timestamp_milliseconds(timestamp_in_seconds: float) -> int:
    return int(timestamp_in_seconds * 1000)


def seconds_to_datetime_str(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).isoformat()


def wait_for_done(
    check_function: Callable,
    timeout: float,
    progress_callback: Optional[Callable] = None,
    sleep_time: float = 1,
) -> None:
    """Wait up to TIMEOUT seconds for the check function to return True"""
    end_time = time.time() + timeout
    while check_function() is False and time.time() < end_time:
        if progress_callback is not None:
            progress_callback()
        # Wait a max of sleep_time, but keep checking to see if
        # check_function is done. Allows wait_for_empty to end
        # before sleep_time has elapsed:
        end_sleep_time = time.time() + sleep_time
        while check_function() is False and time.time() < end_sleep_time:
            time.sleep(sleep_time / 20.0)


def read_unix_packages(
    package_status_file: str = "/var/lib/dpkg/status",
) -> Optional[List[str]]:
    if os.path.isfile(package_status_file):
        package = None
        os_packages = []
        with open(package_status_file, "rb") as fp:
            for binary_line in fp:
                line = binary_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("Package: "):
                    package = line[9:]
                if line.startswith("Version: "):
                    version = line[9:]
                    if package is not None:
                        os_packages.append((package, version))
                    package = None
        os_packages_list = sorted(
            [("%s=%s" % (package, version)) for (package, version) in os_packages]
        )
        return os_packages_list
    else:
        return None


ONE_KBYTE = float(1024)
ONE_MBYTE = float(1024 * 1024)
ONE_GBYTE = float(1024 * 1024 * 1024)


def format_bytes(size: float) -> str:
    """
    Given a size in bytes, return a sort string representation.
    """
    if size >= ONE_GBYTE:
        return "%.2f %s" % (size / ONE_GBYTE, "GB")
    elif size >= ONE_MBYTE:
        return "%.2f %s" % (size / ONE_MBYTE, "MB")
    elif size >= ONE_KBYTE:
        return "%.2f %s" % (size / ONE_KBYTE, "KB")
    else:
        return "%d %s" % (size, "bytes")


def encode_metadata(metadata: Optional[Dict[Any, Any]]) -> Optional[str]:
    if metadata is None:
        return None

    if type(metadata) is not dict:
        LOGGER.info("invalid metadata, expecting dict type", exc_info=True)
        return None

    if metadata == {}:
        return None

    try:
        json_encoded = json.dumps(
            metadata, separators=(",", ":"), sort_keys=True, cls=NestedEncoder
        )
        return json_encoded
    except Exception:
        LOGGER.info("invalid metadata, expecting JSON-encodable object", exc_info=True)
        return None


def encode_asset_data_to_json(asset_data: Any) -> str:
    return json.dumps(asset_data, separators=(",", ":"), cls=NestedNumpyOrTensorEncoder)


def get_comet_version() -> str:
    try:
        return importlib_metadata.version("comet_ml")
    except importlib_metadata.PackageNotFoundError:
        return "Please install comet with `pip install comet_ml`"


def get_user() -> str:
    try:
        return getpass.getuser()
    except KeyError:
        # We are in a system with no user, like Docker container with custom UID
        return "unknown"
    except Exception:
        LOGGER.debug(
            "Unknown exception getting the user from the system", exc_info=True
        )
        return "unknown"


def valid_ui_tabs(
    tab: Optional[str] = None, preferred: bool = False
) -> Union[str, List[str]]:
    """
    List of valid UI tabs in browser.
    """
    mappings = {
        "artifacts": "artifacts",
        "assets": "assets",
        "audio": "audio",
        "charts": "chart",
        "code": "code",
        "confusion-matrices": "confusionMatrix",
        "graphics": "images",
        "histograms": "histograms",
        "installed-packages": "installedPackages",
        "metrics": "metrics",
        "notes": "notes",
        "parameters": "params",
        "system-metrics": "systemMetrics",
        "text": "text",
        "output": "stdout",
        "panels": "panels",
        "graph": "graph",
        "other": "other",
        "html": "html",
    }
    preferred_names = list(mappings.keys())
    # Additional keys:
    mappings.update(
        {
            "assetStorage": "assets",
            "chart": "chart",
            "confusion-matrix": "confusionMatrix",
            "confusionMatrix": "confusionMatrix",
            "images": "images",
            "installedPackages": "installedPackages",
            "params": "params",
            "systemMetrics": "systemMetrics",
            "graph-def": "graph",
            "graph-definition": "graph",
            "stdout": "stdout",
        }
    )
    if preferred:
        return preferred_names
    elif tab is None:
        return list(mappings.keys())
    elif tab in mappings:
        return mappings[tab]
    else:
        raise ValueError("invalid tab name; tab should be in %r" % preferred_names)


def shape(data: Any) -> List[int]:
    """
    Given a nested list or a numpy array,
    return the shape.
    """
    if hasattr(data, "shape"):
        return list(data.shape)
    else:
        try:
            length = len(data)
            return [length] + shape(data[0])
        except TypeError:
            return []


def tensor_length(data: Any) -> int:
    """
    Get the length of a tensor/list.
    """
    if hasattr(data, "shape"):
        return data.shape[0]
    else:
        try:
            length = len(data)
        except TypeError:
            length = 0
    return length


def truncate_to_length_or_raise(
    value: Optional[str],
    length: Optional[int],
    raise_on_truncate: bool = False,
) -> (str, bool):
    out_value, truncated = truncate_to_length(value, length=length)
    if truncated and raise_on_truncate:
        raise StringValueTruncatedException(
            CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING % (length, value)
        )

    return out_value, truncated


def truncate_to_length(value: Optional[str], length: Optional[int]) -> (str, bool):
    if value is None or length is None:
        return value, False

    if len(value) > length:
        return value[:length], True

    return value, False


def proper_registry_model_name(name: str) -> str:
    """
    A proper registry model name is:
        * lowercase
        * replaces all non-alphanumeric with dashes
        * removes leading and trailing dashes
        * limited to 1 dash in a row
    """
    name = "".join([(char if char.isalnum() else "-") for char in name])
    while name.startswith("-"):
        name = name[1:]
    while name.endswith("-"):
        name = name[:-1]
    name = name.lower()
    while "--" in name:
        name = name.replace("--", "-")
    return name


def is_interactive() -> bool:
    """
    Returns True if in interactive mode
    """
    return bool(getattr(sys, "ps1", sys.flags.interactive))


def get_dataframe_profile_html(
    dataframe: Any, minimal: Optional[bool]
) -> Optional[str]:
    """
    Log a pandas dataframe profile.
    """
    dataframe_profiling = _import_dataframe_profiling()

    if dataframe_profiling is None:
        return None

    if minimal:  # earlier versions of pandas_profiling did not support minimal
        profile = dataframe_profiling.ProfileReport(dataframe, minimal=minimal)
    else:
        profile = dataframe_profiling.ProfileReport(dataframe)
    html = profile.to_html()  # type: str
    html = html.replace("http:", "https:")
    return html


def _import_dataframe_profiling():
    """
    Pandas_profiling was deprecated by its creators in favor
    of the ydata_profiling. Comet still supports usage of the old package.
    """
    try:
        import ydata_profiling

        return ydata_profiling
    except ImportError:
        try:
            import pandas_profiling

            return pandas_profiling
        except ImportError:
            LOGGER.warning(MISSING_PANDAS_OR_YDATA_PROFILING, exc_info=True)
            return None


def clean_string(val: str) -> str:
    if val:
        return "".join([char for char in val if char not in ["'", '"', " "]])
    else:
        return ""


def get_api_key_from_user() -> str:
    """
    Get the Comet API key from the user.
    """
    from .config import get_comet_url_override, get_config

    settings = get_config()
    client_url = get_comet_url_override(settings)
    root_url = sanitize_url(get_root_url(client_url))

    print(
        "Please paste your Comet API key from {root_url}api/my/settings/".format(
            root_url=root_url
        )
    )
    print("(api key may not show as you type)")
    api_key = clean_string(getpass.getpass("Comet API key: "))
    return api_key


def parse_remote_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None

    try:
        parsed = urlparse(uri)
        if parsed.path:
            # Split the path
            return parsed.path.split("/")[-1]
        else:
            return None
    except Exception:
        LOGGER.debug("Failure parsing URI %r", uri, exc_info=True)
        return None


class ImmutableDict(dict):
    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


class IterationProgressCallback(object):
    def __init__(self, iterable, callback, frequency):
        self._iterable = iterable
        self._callback = callback
        self._frequency = frequency

    def __iter__(self):
        last = get_time_monotonic()

        for obj in self._iterable:
            yield obj

            now = get_time_monotonic()
            diff = now - last
            if diff > self._frequency:
                last = now
                try:
                    self._callback()
                except Exception:
                    LOGGER.debug("Error calling the progress callback", exc_info=True)


def generate_guid() -> str:
    """Generate a GUID"""
    return uuid.uuid4().hex


def compact_json_dump(data: Any, fp: TextIO) -> None:
    json.dump(data, fp, sort_keys=True, separators=(",", ":"), cls=NestedEncoder)


def get_root_url(url: str) -> str:
    """Remove the path, params, query and fragment from a given URL"""
    parts = urlparse(url)
    scheme, netloc, path, params, query, fragment = parts

    return urlunparse((scheme, netloc, "", "", "", ""))


def sanitize_url(url: str, ending_slash: bool = True) -> str:
    """Sanitize URL, checking that it is a valid URL and ensure it contains an ending slash / or
    that it doesn't contain an ending slash depending on the value of ending_slash"""
    parts = urlparse(url)
    scheme, netloc, path, params, query, fragment = parts

    # TODO: Raise an exception if params, query and fragment are not empty?

    # Ensure the leading slash
    if ending_slash is True:
        if path and not path.endswith("/"):
            path = path + "/"
        elif not path and not netloc.endswith("/"):
            netloc = netloc + "/"
    else:
        if path and path.endswith("/"):
            path = path[:-1]

    return urlunparse((scheme, netloc, path, params, query, fragment))


def metric_name(name: str, prefix: Optional[str] = None) -> str:
    if prefix:
        return "%s.%s" % (prefix, name)
    else:
        return name


def subprocess_run_and_check(
    args: List[str],
    timeout: int,
    exception_warning_log_message: str,
    return_code_warning_log_message: str,
) -> Optional[bytes]:
    """Run the command passed as a List of string with the given timeout (ignored on Python 2.7).

    If the command ran but it's exit code is not 0, show the return_code_warning_log_message as a
    Warning log, logs the stdout/stderr in debug logs and return None.

    If any other error occurs, show the exception_warning_log_message as a Warning log and return
    None.
    """
    try:
        completed_process = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception:
        LOGGER.warning(exception_warning_log_message, exc_info=True)
        return None

    try:
        completed_process.check_returncode()
    except subprocess.CalledProcessError as exception:
        LOGGER.warning(return_code_warning_log_message, exc_info=True)
        message = "%s\n" % exception
        if exception.output is not None and len(exception.output) > 0:
            message += to_utf8(exception.output) + "\n"
        if exception.stderr is not None and len(exception.stderr) > 0:
            message += to_utf8(exception.stderr)

        LOGGER.warning(message)
        return None
    except Exception:
        LOGGER.warning(exception_warning_log_message, exc_info=True)
        return None

    return completed_process.stdout


def tag_experiment_with_commit_number(experiment):
    if sys.version_info.major == 3:
        commit = subprocess.check_output(
            "git rev-parse HEAD", shell=True, universal_newlines=True
        ).strip()
    else:
        commit = subprocess.check_output("git rev-parse HEAD", shell=True).strip()

    commit = commit[:8]
    experiment.add_tag(commit)


def tag_experiment_with_transport_layer_identifier(experiment):
    """Adds to the experiment TAG with transport layer identifier taken from environment"""
    experiment.add_tag("HTTP: %r" % True)


def is_aws_lambda_environment() -> bool:
    """Allows to check if we are executing in AWS lambda environment"""
    return AWS_LAMBDA_ENVIRONMENT_MARKER_KEY in os.environ


def find_logger_spec(logger_spec: str) -> str:
    if is_aws_lambda_environment():
        return "simple"

    if logger_spec == "default":
        if _in_ipython_environment():
            logger_spec = "simple"
        # MacOS default start mode is now spawn and fork is unsafe https://bugs.python.org/issue33725
        elif platform.system() == "Darwin":
            logger_spec = "simple"
        else:
            logger_spec = "native"
    else:
        logger_spec = logger_spec or None  # in case of ""

    return logger_spec


def optional_update(destination: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    the equivalent of destination.update(source),
    except that if a (key, value) pair in source is None,
    then this key is skipped.

    this essentially automates the pattern:
    if variable is not None:
        destination["variable"] = variable
    """
    for key, value in source.items():
        if value is not None:
            destination[key] = value


def read_git_patch_zip(patch_data: bytes) -> bytes:
    zip_patch = io.BytesIO(patch_data)
    archive = zipfile.ZipFile(zip_patch)
    return archive.read("git_diff.patch")


def compress_git_patch(git_patch: Any) -> Tuple[zipfile.ZipFile, str]:
    # Create a zip
    zip_dir = tempfile.mkdtemp()

    zip_path = os.path.join(zip_dir, "patch.zip")
    archive = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    archive.writestr("git_diff.patch", git_patch)
    archive.close()

    return archive, zip_path


def random_ascii_string(length: int = 6) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def masked_api_key(api_key: str) -> Optional[str]:
    if api_key is None:
        return None

    length = len(api_key)
    if length < 5:
        return api_key

    prefix = api_key[:5]
    return prefix + "*" * (length - 5)


def is_valid_experiment_key(experiment_key: str) -> bool:
    """Validate an experiment_key; returns True or False"""
    return (
        isinstance(experiment_key, str)
        and experiment_key.isalnum()
        and (32 <= len(experiment_key) <= 50)
    )


def create_asset_url(prefix, **query_arguments):
    if prefix is None:
        return None

    splitted = list(urlsplit(prefix))

    splitted[3] = urlencode(query_arguments)

    return urlunsplit(splitted)
