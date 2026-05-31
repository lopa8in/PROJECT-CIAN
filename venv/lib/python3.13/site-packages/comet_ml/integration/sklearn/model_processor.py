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

from typing import Any


def process(model: Any) -> Any:
    model = _extract_best_estimator_if_exists(model)

    return model


def _extract_best_estimator_if_exists(model: Any) -> Any:
    estimator = getattr(model, "best_estimator_", None)

    if estimator is not None:
        return estimator

    return model
