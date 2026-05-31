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

import urllib.parse


class URL:
    """
    This class is used to wrap url arguments in LOGGER.* calls.
    The reason why we need it is because we have a lot of %r log message
    arguments in the code for URLs.
    %r leads to __repr__ method called for url string which makes all control
    characters just visible symbols. We are suffering from it since we
    want to colorize links.

    This class emulates old behavior (surrounds urls with quotes)
    but with working colors.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return "'" + str(self) + "'"

    def __eq__(self, other):
        if not isinstance(other, URL):
            return False
        return self.value == other.value


def is_url(text: str) -> bool:
    try:
        result = urllib.parse.urlparse(text)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
