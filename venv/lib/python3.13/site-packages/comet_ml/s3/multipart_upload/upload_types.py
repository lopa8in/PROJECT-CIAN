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
from ...constants import (
    ASSET_TYPE_DEFAULT,
    ASSET_TYPE_MODEL_ELEMENT,
    ASSET_TYPE_VISUALIZATION,
)

ASSET_S3_UPLOAD_TYPE = ASSET_TYPE_DEFAULT
VISUALIZATION_S3_UPLOAD_TYPE = ASSET_TYPE_VISUALIZATION
MODEL_ELEMENT_S3_UPLOAD_TYPE = ASSET_TYPE_MODEL_ELEMENT

DIRECT_S3_UPLOAD_TYPES = [
    VISUALIZATION_S3_UPLOAD_TYPE,
    ASSET_S3_UPLOAD_TYPE,
    MODEL_ELEMENT_S3_UPLOAD_TYPE,
]
