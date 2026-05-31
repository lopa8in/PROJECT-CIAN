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

import inspect
import os
from typing import Any, Dict

from .config_class import Config

try:
    import streamlit as st
except ImportError:
    st = None


def _in_streamlit_environment():
    return st is not None and st.runtime.exists()


def find_page_name():
    stack = inspect.stack()
    for row in stack:
        frame, filename, line_number, function_name, lines, index = row
        basename = os.path.basename(filename)
        if basename.startswith("page_"):
            return basename


class StreamlitConfig(Config):
    def _get_override(self) -> Dict[str, Any]:
        key = find_page_name()
        if key:
            state = st.session_state.get("comet_config_override", {})
            return state.get(key, {})
        else:
            return {}
