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
import functools
import sys

import comet_ml

from . import environment, rich_api_importer


def stylize_text(text: str, color: str, **kwargs) -> str:
    if environment.legacy_windows():
        return text

    Style = rich_api_importer.Style()

    color = None if environment.no_color() else color
    style = Style(color=color, **kwargs)

    return style.render(text, color_system=environment.color_system())


def print_rule(title: str = "") -> None:
    Console = rich_api_importer.Console()
    Style = rich_api_importer.Style()

    if environment.is_jupyter():
        corrected_width = environment.width() - 1
        console = Console(
            stderr=True,
            force_jupyter=False,
            force_terminal=True,
            width=corrected_width,
            height=environment.height(),
        )
    else:
        console = Console(stderr=True)

    console.rule(title, style=Style(color="dark_cyan"))


def is_available() -> bool:
    if not _rich_enabled():
        return False

    try:
        rich_api_importer.rich()
    except Exception:
        return False

    return True


@functools.lru_cache(maxsize=0 if "pytest" in sys.modules else 1)
def _rich_enabled() -> bool:
    config = comet_ml.get_config()
    return config["comet.rich_output"]
