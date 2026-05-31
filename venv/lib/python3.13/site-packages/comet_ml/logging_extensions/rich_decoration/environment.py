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

from . import rich_api_importer

try:
    Console = rich_api_importer.Console()
    _environment = Console(stderr=True)
except Exception:
    _environment = None


def width() -> int:
    return _environment._width


def height() -> int:
    return _environment._height


def no_color() -> bool:
    return _environment.no_color


def is_terminal() -> bool:
    return _environment.is_terminal


def is_jupyter() -> bool:
    return _environment.is_jupyter


def color_system() -> str:
    return _environment.color_system


def legacy_windows() -> bool:
    return _environment.legacy_windows
