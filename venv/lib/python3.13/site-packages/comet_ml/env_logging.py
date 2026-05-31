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

"""
Author: Boris Feld

This module contains the functions dedicated to logging the environment
information

"""
import functools
import inspect
import json
import logging
import os
import platform
import socket
import sys

if sys.version_info[:2] > (3, 9):
    import importlib.metadata as importlib_metadata
else:
    # importlib_metadata for versions <= (3, 9) doesn't have names for distributions
    import pkg_resources

from io import StringIO
from urllib.parse import urlparse

import requests

from ._typing import Any, Dict, List, Optional, Tuple, Union
from .config import get_backend_address
from .logging_messages import (
    LOG_CONDA_ENV_ERROR,
    LOG_CONDA_ENV_RETURN_CODE,
    LOG_CONDA_INFO_ERRORS,
    LOG_CONDA_INFO_RETURN_CODE,
    LOG_CONDA_PACKAGES_ERRORS,
    LOG_CONDA_PACKAGES_RETURN_CODE,
)
from .messages import SystemDetailsMessage
from .utils import get_comet_version, get_user, subprocess_run_and_check

LOGGER = logging.getLogger(__name__)

PYTEST_ENV = "pytest" in sys.modules  # TODO: check if there are better alternatives

CONDA_ENV_TIMEOUT = 60


def get_pid() -> int:
    return os.getpid()


def get_hostname() -> str:
    return socket.gethostname()


def get_os() -> str:
    return platform.platform(aliased=True)


def get_os_type() -> str:
    return platform.system()


def get_python_version_verbose() -> str:
    return sys.version


def get_python_version() -> str:
    return platform.python_version()


def get_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = get_backend_address()
        parsed = urlparse(server_address)
        host = parsed.hostname
        port = parsed.port
        if port is None:
            port = {"http": 80, "https": 443}.get(parsed.scheme, 0)
        s.connect((host, port))
        addr = s.getsockname()[0]
        s.close()
        return addr

    except socket.error:
        LOGGER.warning("Failed to log ip", exc_info=True)
        return None


def get_command() -> List[str]:
    return sys.argv


def get_env_details_message() -> SystemDetailsMessage:
    os_type, _, os_release, _, machine, processor = platform.uname()

    return SystemDetailsMessage.create(
        context=None,
        command=get_command(),
        env=None,
        hostname=get_hostname(),
        ip=get_ip(),
        machine=machine,
        os_release=os_release,
        os_type=os_type,
        os=get_os(),
        pid=get_pid(),
        processor=processor,
        python_exe=sys.executable,
        python_version_verbose=get_python_version_verbose(),
        python_version=get_python_version(),
        user=get_user(),
    )


def _get_non_pii_env_details(comet_url: str) -> Dict[str, Any]:
    """Return environment details without PII, useful to send to external services like error tracking"""

    return {
        "pid": get_pid(),
        "os": get_os(),
        "os_type": get_os_type(),
        "python_version_verbose": get_python_version_verbose(),
        "python_version": get_python_version(),
        "user": get_user(),
        "python_exe": sys.executable,
        "sdk_version": get_comet_version(),
        "comet_url": comet_url,
    }


def get_caller_file_path() -> Optional[Tuple[str, str]]:
    """Returns the module name and file path from the first caller that isn't in the blacklisted
    list. If none is found, returns None
    """
    # Don't get source code from these libraries:
    ignore_list = (
        "comet_ml",
        "mlflow",
        "transformers",
        "ludwig",
        "pytorch_lightning",
        "metaflow",
    )

    for frame in inspect.stack(context=1):
        module = inspect.getmodule(frame[0])
        if module is not None:
            module_name = module.__name__

            if any(module_name.startswith(ignore_name) for ignore_name in ignore_list):
                continue

            filename = module.__file__.rstrip("cd")
            return module_name, filename

    # We didn't find any matching module
    LOGGER.debug("Didn't find any source code module")
    return None


def get_ipython_source_code() -> str:
    """
    Get the IPython source code from the history. Assumes that this
    method is run in a jupyter environment.

    Returns the command-history as a string that lead to this point.
    """
    import IPython

    ipy = IPython.get_ipython()
    env = ipy.ns_table["user_local"]
    source = StringIO()
    for n, code in enumerate(env["_ih"]):
        if code:
            source.write("# %%%% In [%s]:\n" % n)
            source.write(code)
            source.write("\n\n")
    return source.getvalue()


def get_ipython_cells():
    # type: () -> List[Tuple[Optional[str], Optional[Any]]]
    """
    Get the Notebook (ipython, colab, Jupyter, PyCharm, etc.) from the
    history. Assumes that this method is run in an ipython environment.

    Returns the command-history [(input,output), ...]

    """
    import IPython

    ipy = IPython.get_ipython()
    env = ipy.ns_table["user_local"]
    cells = []
    for n, code in enumerate(env["_ih"]):
        output = env["_oh"].get(n, None)
        if code or output:
            cells.append((code, output))
    return cells


def _format_text(text: Any) -> List[str]:
    """
    Format multi-line text in the ipython style.
    """
    if not isinstance(text, str):
        text = repr(text)

    lines = text.splitlines(True)  # type: List[str]
    # Remove potentially trailing newlines in the final element
    lines[-1] = lines[-1].rstrip("\r\n").rstrip("\n")
    return lines


def _format_output(output: Any, execution_count: int) -> Optional[Dict]:
    """
    Get the mime bundle, etc. for each output form.
    """
    if output is None:
        return None

    if hasattr(output, "_repr_mimebundle_"):
        return {
            "data": output._repr_mimebundle_(),
            "execution_count": execution_count,
            "metadata": {},
            "output_type": "execute_result",
        }
    else:
        return {
            "data": {"text/plain": _format_text(output)},
            "execution_count": execution_count,
            "metadata": {},
            "output_type": "execute_result",
        }


def create_notebook(cells: List[Tuple[Optional[str], Optional[Any]]]) -> Dict[str, Any]:
    """
    From a list of (input, output) tuples, create a JSON notebook.
    """
    final_cells = []
    for n, cell in enumerate(cells):
        new_cell = {
            "cell_type": "code",
            "execution_count": n + 1,
            "metadata": {},
            "source": _format_text(cell[0]),
        }

        if cell[1] is not None:
            new_cell["output_type"] = "execute_result"
            new_cell["outputs"] = [_format_output(cell[1], n + 1)]
        else:
            new_cell["outputs"] = []

        final_cells.append(new_cell)

    return {"nbformat": 4, "nbformat_minor": 0, "metadata": {}, "cells": final_cells}


def get_ipython_notebook() -> Dict[str, Any]:
    return create_notebook(get_ipython_cells())


def postprocess_gcp_cloud_metadata(cloud_metadata: Dict[str, Any]) -> Dict[str, Any]:

    # Attributes contains custom metadata and also contains Kubernetes config,
    # startup script and secrets, filter it out
    if "attributes" in cloud_metadata:
        del cloud_metadata["attributes"]

    return cloud_metadata


CLOUD_METADATA_MAPPING = {
    "AWS": {
        "url": "http://169.254.169.254/latest/dynamic/instance-identity/document",
        "headers": {},
    },
    "Azure": {
        "url": "http://169.254.169.254/metadata/instance?api-version=2019-08-15",
        "headers": {"Metadata": "true"},
    },
    "GCP": {
        "url": "http://169.254.169.254/computeMetadata/v1/instance/?recursive=true&alt=json",
        "headers": {"Metadata-Flavor": "Google"},
        "postprocess_function": postprocess_gcp_cloud_metadata,
    },
}


@functools.lru_cache(0 if PYTEST_ENV else 128)
def get_env_cloud_details(timeout: int = 1) -> Optional[Any]:
    for provider in CLOUD_METADATA_MAPPING.keys():
        try:
            params = CLOUD_METADATA_MAPPING[provider]
            response = requests.get(
                params["url"], headers=params["headers"], timeout=timeout
            )
            response.raise_for_status()
            response_data = response.json()

            postprocess_function = params.get("postprocess_function")
            if postprocess_function is not None:
                response_data = postprocess_function(response_data)

            return {"provider": provider, "metadata": response_data}
        except Exception as e:
            LOGGER.debug(
                "Not running on %s, couldn't retrieving metadata: %r", provider, e
            )

    return None


@functools.lru_cache(0 if PYTEST_ENV else 128)
def get_pip_packages(
    as_tuples: bool = False,
) -> Union[List[str], List[Tuple[str, str]]]:
    if sys.version_info[:2] <= (3, 9):
        result_tuples = sorted(
            [(package.key, package.version) for package in pkg_resources.working_set]
        )
    else:
        result_tuples = sorted(
            [
                (package.name.lower(), package.version)
                for package in importlib_metadata.distributions()
            ]
        )

    if as_tuples:
        return result_tuples

    return ["%s==%s" % (name, version) for name, version in result_tuples]


def _in_conda_environment() -> bool:
    return os.path.exists(os.path.join(sys.prefix, "conda-meta", "history"))


def _in_pydev_console() -> bool:
    return "PYDEVD_LOAD_VALUES_ASYNC" in os.environ


def _get_conda_env() -> Optional[str]:
    args = ["conda", "env", "export", "--json"]

    result = subprocess_run_and_check(
        args, CONDA_ENV_TIMEOUT, LOG_CONDA_ENV_ERROR, LOG_CONDA_ENV_RETURN_CODE
    )
    if result is None:
        return None

    try:
        decoded = json.loads(result)
        # Remove the variables key because it can contain secrets
        decoded.pop("variables", None)
        return json.dumps(decoded, indent=4, sort_keys=True)
    except Exception:
        LOGGER.warning(LOG_CONDA_ENV_ERROR, exc_info=True)
        return None


def _get_conda_explicit_packages() -> Optional[bytes]:
    args = ["conda", "list", "--explicit", "--md5"]
    return subprocess_run_and_check(
        args,
        CONDA_ENV_TIMEOUT,
        LOG_CONDA_PACKAGES_ERRORS,
        LOG_CONDA_PACKAGES_RETURN_CODE,
    )


def _get_conda_info() -> Optional[bytes]:
    args = ["conda", "info"]
    return subprocess_run_and_check(
        args, CONDA_ENV_TIMEOUT, LOG_CONDA_INFO_ERRORS, LOG_CONDA_INFO_RETURN_CODE
    )


def parse_log_other_instructions() -> Dict[str, str]:
    result = {}
    PREFIX = "comet_log_other_"
    PREFIX_LENGTH = len(PREFIX)

    for key, value in os.environ.items():
        lowered_key = key.lower()
        if lowered_key.startswith(PREFIX) and len(lowered_key) > PREFIX_LENGTH:
            name = lowered_key[PREFIX_LENGTH:]
            result[name] = value

    return result
