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


def setup(api_key, use_cache):
    # we need to import here to avoid circular imports
    from comet_ml.connection import connection_factory

    client = connection_factory.get_rest_api_client(
        "v2",
        api_key=api_key,
        use_cache=use_cache,
        headers={"X-COMET-SDK-SOURCE": "API"},
    )

    return client
