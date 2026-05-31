# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

DEFAULT_PROJECT_NAME = "general"

SUPPORTED_VIDEO_FORMATS = ["MP4", "MOV", "WMV", "GIF"]

ASSET_TYPE_DEFAULT = "asset"

ASSET_TYPE_AUDIO = "audio"
ASSET_TYPE_VIDEO = "video"
ASSET_TYPE_EMBEDDINGS = "embeddings"
ASSET_TYPE_VISUALIZATION = "visualization"
ASSET_TYPE_HISTOGRAM_3D = "histogram3d"
ASSET_TYPE_TEXT_SAMPLE = "text-sample"
ASSET_TYPE_MODEL_ELEMENT = "model-element"
ASSET_TYPE_CURVE = "curve"
ASSET_TYPE_NOTEBOOK = "notebook"
ASSET_TYPE_TENSORFLOW_FILE = "tensorflow-file"
ASSET_TYPE_SOURCE_CODE = "source_code"
ASSET_TYPE_DATAFRAME_PROFILE = "dataframe-profile"
ASSET_TYPE_DATAFRAME = "dataframe"
ASSET_TYPE_CONFUSION_MATRIX = "confusion-matrix"
ASSET_TYPE_GIT_PATCH = "git-patch"
ASSET_TYPE_3D_POINTS = "3d-image"
ASSET_TYPE_ASSET_ITEM = "asset-item"

OTHER_KEY_CREATED_FROM = "Created from"

# Resume Strategy / Old Offline MODE
RESUME_STRATEGY_GET = "get"
RESUME_STRATEGY_GET_OR_CREATE = "get_or_create"
RESUME_STRATEGY_CREATE = "create"

DEPRECATED_OFFLINE_MODE_CREATE = "create"
DEPRECATED_OFFLINE_MODE_APPEND = "append"

DEPRECATED_OFFLINE_MODE_TO_RESUME_STRATEGY_MAP = {
    DEPRECATED_OFFLINE_MODE_APPEND: RESUME_STRATEGY_GET,
    DEPRECATED_OFFLINE_MODE_CREATE: RESUME_STRATEGY_CREATE,
}

# Conda related assets
CONDA_ENV_FILE_NAME = "conda-environment.yml"
CONDA_ENV_ASSET_TYPE = "conda-environment-definition"

CONDA_SPEC_FILE_NAME = "conda-spec.txt"
CONDA_SPEC_ASSET_TYPE = "conda-specification"

CONDA_INFO_FILE_NAME = "conda-info.yml"
ASSET_TYPE_CONDA_INFO = "conda-info"

# The experiment status response keys
STATUS_RESPONSE_IS_ALIVE_BEAT_DURATION_MILLIS = "is_alive_beat_duration_millis"
STATUS_RESPONSE_GPU_MONITOR_INTERVAL_MILLIS = "gpu_monitor_interval_millis"
STATUS_RESPONSE_CPU_MONITOR_INTERVAL_MILLIS = "cpu_monitor_interval_millis"
STATUS_RESPONSE_PARAMETER_UPDATE_INTERVAL_MILLIS = "parameter_update_interval_millis"
STATUS_RESPONSE_PENDING_RPCS = "pending_rpcs"
STATUS_RESPONSE_CODE = "code"
STATUS_RESPONSE_MESSAGE = "msg"

# The REST API constants
PAYLOAD_EXPERIMENT_KEY = "experimentKey"
PAYLOAD_OS_PACKAGES = "osPackages"
PAYLOAD_MODEL_GRAPH = "graph"
PAYLOAD_COMMAND = "command"
PAYLOAD_EXECUTABLE = "executable"
PAYLOAD_HOSTNAME = "hostname"
PAYLOAD_IP = "ip"
PAYLOAD_OS = "os"
PAYLOAD_OS_TYPE = "osType"
PAYLOAD_OS_RELEASE = "osRelease"
PAYLOAD_PID = "pid"
PAYLOAD_PYTHON_VERSION = "pythonVersion"
PAYLOAD_PYTHON_VERSION_VERBOSE = "pythonVersionVerbose"
PAYLOAD_MACHINE = "machine"
PAYLOAD_PROCESSOR = "processor"
PAYLOAD_ENV = "env"
PAYLOAD_USER = "user"
PAYLOAD_INSTALLED_PACKAGES = "installedPackages"
PAYLOAD_GPU_STATIC_INFO_LIST = "gpuStaticInfoList"
PAYLOAD_TOTAL_RAM = "totalRam"
PAYLOAD_USED_RAM = "usedRam"
PAYLOAD_FILE_PATH = "filePath"
PAYLOAD_HTML = "html"
PAYLOAD_OVERRIDE = "override"
PAYLOAD_TIMESTAMP = "timestamp"
PAYLOAD_PROVIDER = "provider"
PAYLOAD_METADATA = "metadata"
PAYLOAD_ADDITIONAL_SYSTEM_INFO_LIST = "logAdditionalSystemInfoList"
PAYLOAD_DEPENDENCIES = "logDependencies"
PAYLOAD_DEPENDENCY_NAME = "name"
PAYLOAD_DEPENDENCY_VERSION = "version"
PAYLOAD_STDERR = "stderr"
PAYLOAD_OUTPUT = "output"
PAYLOAD_LOCAL_TIMESTAMP = "localTimestamp"
PAYLOAD_OFFSET = "offset"
PAYLOAD_RUN_CONTEXT = "runContext"
PAYLOAD_OUTPUT_LINES = "outputLines"
PAYLOAD_MODEL_NAME = "modelName"
PAYLOAD_REMOTE_ASSETS = "remoteEntities"
PAYLOAD_API_KEY = "apiKey"
PAYLOAD_PARAMETER_NAME = "parameterName"
PAYLOAD_ASSET_NAME = "assetName"
BOXES_FILE_NAME = "boxes.jsonl"
POINTS_FILE_NAME = "points.jsonl"
