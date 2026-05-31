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

import types

from ..uri import parse
from . import request_handlers

REQUEST_HANDLERS = types.MappingProxyType(
    {
        parse.RequestTypes.EXPERIMENT_BY_KEY: request_handlers.from_experiment_by_key,
        parse.RequestTypes.EXPERIMENT_BY_WORKSPACE: request_handlers.from_experiment_by_workspace,
        parse.RequestTypes.REGISTRY: request_handlers.from_registry,
    }
)
