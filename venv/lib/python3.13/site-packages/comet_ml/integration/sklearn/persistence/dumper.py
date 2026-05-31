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
from typing import Any, BinaryIO

from . import module_metadata


class Dumper:
    def __init__(self, persistence_module: ModuleType, **dump_kwargs):
        self._persistence_module = persistence_module
        self._dump_kwargs = dump_kwargs
        self._init_module_details()

    def _init_module_details(self) -> None:
        self.module_name = self._persistence_module.__name__
        self.module_version = module_metadata.version(self._persistence_module)
        self.file_extension = module_metadata.file_extension(self._persistence_module)

    def dump(self, value: Any, file: BinaryIO) -> None:
        self._persistence_module.dump(value, file, **self._dump_kwargs)
