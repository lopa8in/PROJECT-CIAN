# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

_debug_exceptions_raising = False


def has_enabled_debug_exception_raising() -> bool:
    return _debug_exceptions_raising


def _enable_debug_exception_raising() -> None:
    global _debug_exceptions_raising
    _debug_exceptions_raising = True


def _disable_debug_exception_raising() -> None:
    global _debug_exceptions_raising
    _debug_exceptions_raising = False
