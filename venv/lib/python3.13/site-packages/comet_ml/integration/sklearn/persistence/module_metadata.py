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

from types import ModuleType
from typing import List, Optional

_SUPPORTED_MODULE_NAMES = {"pickle", "cloudpickle", "joblib"}

_FILE_EXTENSIONS = {
    "joblib": "joblib",
    "pickle": "pkl",
    "cloudpickle": "cpkl",
}

_VERSION_EXTRACTORS = {
    "joblib": lambda module: module.__version__,
    "pickle": lambda module: module.format_version,
    "cloudpickle": lambda module: module.__version__,
}

assert _SUPPORTED_MODULE_NAMES == set(_FILE_EXTENSIONS) == set(_VERSION_EXTRACTORS)


def supported_modules() -> List[str]:
    return list(_SUPPORTED_MODULE_NAMES)


def file_extension(persistence_module: ModuleType) -> str:
    name = persistence_module.__name__
    return _FILE_EXTENSIONS[name]


def version(persistence_module: ModuleType) -> Optional[str]:
    name = persistence_module.__name__
    if name in _VERSION_EXTRACTORS:
        return _VERSION_EXTRACTORS[name](persistence_module)

    return None
