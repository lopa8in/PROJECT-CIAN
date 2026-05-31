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
import hashlib
import os
import pathlib
import random
import re
import tempfile
from typing import IO, Any, AnyStr, Generator, List, Optional, Tuple, Union

from ._typing import ValidFilePath


def io_sha1sum(io_object: IO[bytes]) -> str:
    sha1sum = hashlib.sha1()

    block = io_object.read(2**16)
    while len(block) != 0:
        sha1sum.update(block)
        block = io_object.read(2**16)

    return sha1sum.hexdigest()


def file_sha1sum(file_path: str) -> str:
    with open(file_path, "rb") as source:
        return io_sha1sum(source)


def get_extension(format: Optional[str], user_input: Any, file_name: Optional[str]):
    if format is not None:
        return format
    elif isinstance(user_input, ValidFilePath) or is_valid_file_path(user_input):
        return get_file_extension(user_input)
    elif hasattr(user_input, "read"):
        return get_file_extension(file_name)
    return None


def get_file_extension(file_path: AnyStr) -> Optional[str]:
    if file_path is None:
        return None

    ext = os.path.splitext(file_path)[1]
    if not ext:
        return None

    # Get rid of the leading "."
    return ext[1::]


def is_valid_file_path(file_path: Any) -> bool:
    """Check if the given argument is corresponding to a valid file path,
    ready for reading
    """
    try:
        if os.path.isfile(file_path):
            return True
        else:
            return False
    # We can receive lots of things as arguments
    except (TypeError, ValueError):
        return False


def write_file_like_to_tmp_file(file_like_object: Union[IO, str], tmpdir) -> str:
    # Copy of `shutil.copyfileobj` with binary / text detection

    buf = file_like_object.read(1)

    tmp_file = tempfile.NamedTemporaryFile(mode="w+b", dir=tmpdir, delete=False)

    encode = False

    # Detect binary/text
    if not isinstance(buf, bytes):
        encode = True
        buf = buf.encode("utf-8")

    tmp_file.write(buf)

    # Main copy loop
    while True:
        buf = file_like_object.read(16 * 1024)

        if not buf:
            break

        if encode:
            buf = buf.encode("utf-8")

        tmp_file.write(buf)

    tmp_file.close()

    return tmp_file.name


def list_assets_folder(
    folder: str, recursive: bool = False, extension_filter: Optional[List[str]] = None
) -> Generator[Tuple[str, str], None, None]:
    extension_filter_set = None

    if extension_filter is not None:
        extension_filter_set = set(extension_filter)

    if recursive:
        for dir_path, _, file_names in os.walk(folder):
            for file_name in file_names:

                if extension_filter_set:
                    file_extension = os.path.splitext(file_name)[1]

                    if file_extension not in extension_filter_set:
                        continue

                file_path = os.path.join(dir_path, file_name)
                yield file_name, file_path
    else:
        file_names = sorted(os.listdir(folder))
        for file_name in file_names:
            file_path = os.path.join(folder, file_name)
            if os.path.isfile(file_path):

                if extension_filter_set:
                    file_extension = os.path.splitext(file_name)[1]
                    if file_extension not in extension_filter_set:
                        continue

                yield file_name, file_path


def clean_and_check_root_relative_path(root: str, relative_path: str) -> str:
    """
    Given a root and a relative path, resolve the relative path to get an
    absolute path and make sure the resolved path is a child to root. Cases
    where it could not be the case would be if the `relative_path` contains `..`
    or if one part of the relative path is a symlink going above the root.

    Return the absolute resolved path and raises a ValueError if the root path
    is not absolute or if the resolved relative path goes above the root.
    """
    if not os.path.isabs(root):
        raise ValueError("Root parameter %r should an absolute path" % root)

    if not root.endswith(os.sep):
        root = root + os.sep

    real_root = os.path.realpath(root)

    joined_path = os.path.join(real_root, relative_path)
    resolved_path = os.path.realpath(joined_path)

    if not resolved_path.startswith(real_root):
        raise ValueError("Final path %r is outside of %r" % (resolved_path, real_root))

    return resolved_path


def check_if_path_relative_to_root(root: str, absolute_path: str) -> bool:
    if not os.path.isabs(root):
        raise ValueError("Root parameter %r should an absolute path" % root)

    root_full_path = os.path.realpath(root) + os.sep
    full_path = os.path.realpath(absolute_path)

    return full_path.startswith(root_full_path)


def safe_filename(filename: str) -> str:
    """
    Given a value, turn it into a valid filename.

    1. Remove the spaces
    2. Replace anything not alpha, '-', '_', or '.' with '_'
    3. Remove duplicate '_'
    """
    string = str(filename).strip().replace(" ", "_")
    string = re.sub(r"(?u)[^-\w.]", "_", string)
    return re.sub(r"_+", "_", string)


def make_template_filename(group: Optional[str] = None) -> str:
    if group is None:
        group = random.randint(0, 10000000)

    return "template_projector_config-%s.json" % group


def expand_user_home_path(path: Optional[Union[str, pathlib.Path]]) -> Optional[str]:
    """Expand home path staring with tilde to absolute path.
    If provided path doesn't start with tilde or is None just return it."""
    if path is None:
        return None

    if isinstance(path, pathlib.Path):
        return str(path.expanduser())

    if not path.startswith("~"):
        return path

    return os.path.expanduser(path)
