# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************


import collections
from typing import List

_registry = collections.defaultdict(set)


def register(name: str, experiment_key: str):
    _registry[experiment_key].add(name)


def is_registered(name: str, experiment_key: str) -> bool:
    return name in _registry[experiment_key]


def ensure_registration(name: str, experiment_key: str) -> bool:
    if is_registered(name, experiment_key):
        return True

    register(name, experiment_key)
    return False


def registered_events(experiment_key: str) -> List[str]:
    return sorted(list(_registry[experiment_key]))


def clear():
    _registry.clear()
