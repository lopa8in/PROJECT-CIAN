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

"""

This module contains useful types and mirror the typing module

"""
from enum import Enum
from typing import *  # noqa
from typing import (  # noqa
    IO,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

# isort: off


class ValidFilePath(str):
    """This type help marking a file_path as existing on disk as checked by `is_valid_file_path`"""

    pass


class TemporaryFilePath(ValidFilePath):
    """This type help marking a file_path as valid on disk as checked by `is_valid_file_path`"""

    pass


class BackendFeatureSupportByVersion(NamedTuple):
    """This type encapsulates info about result of minimum backend version check for support of specific feature"""

    feature_supported: bool
    min_backend_version_supported: str


class ColorWithShadesMap(NamedTuple):
    """This type encapsulates info about color map for panel"""

    primary: str
    light: str


class ExperimentCleanupResults(NamedTuple):
    """Encapsulates information about experiment cleaning"""

    initial_data_logger_cleaned: bool
    system_metrics_thread_cleaned: bool
    streamer_cleaned: bool
    std_logger_cleaned: bool

    def is_experiment_cleaned_successfully(self) -> bool:
        return (
            self.initial_data_logger_cleaned
            and self.system_metrics_thread_cleaned
            and self.streamer_cleaned
            and self.std_logger_cleaned
        )

    def __str__(self) -> str:
        return (
            f"\ninitial_data_logger_cleaned: {self.initial_data_logger_cleaned}"
            f"\nsystem_metrics_thread_cleaned: {self.system_metrics_thread_cleaned}"
            f"\nstreamer_cleaned: {self.streamer_cleaned}"
            f"\nstd_logger_cleaned: {self.std_logger_cleaned}"
        )


class PreparedRequest(NamedTuple):
    api_key: str
    url: str
    json: Dict[str, Any]
    headers: Dict[str, Any]


class AudioNormalization(Enum):
    """Defines types of normalization to be performed when logging audio data."""

    OFF = "OFF"
    PCM_16_BIT = "PCM_16_BIT"

    @classmethod
    def list_names(cls) -> List[str]:
        return [e.name for e in list(cls)]


class StringResult(NamedTuple):
    """Result of string conversion and possible truncation"""

    value: str
    truncated: bool
    full_value: str


UserText = Union[bytes, Text]  # noqa
MemoryUploadable = Union[IO, UserText]  # noqa
# With typing_extensions, we could use Literal["all"]
TensorflowInput = Union[int, str]  # noqa
Number = Union[int, float]

HeartBeatResponse = Tuple[int, Dict[str, Any], bool]
Point3D = Union[List[float], Tuple[float, float, float]]
OnMessageSentCallback = Callable[[int, bool, bool, Optional[str]], None]
OnMessagesBatchSentCallback = Callable[[List[int], bool, bool, Optional[str]], None]
PanelColorMap = Dict[str, ColorWithShadesMap]

Point = Union[List[float], Tuple[float, float]]


class ExperimentThrottledStatus(NamedTuple):
    """
    Represents the throttled status of an experiment.

    This class is a named tuple used to encapsulate information about whether an
    experiment is throttled. It includes a flag indicating throttling status, an
    optional message providing details about the throttling, and a list of optional
    reasons justifying the throttling.

    Attributes:
        throttled: A boolean indicating whether the experiment is throttled.
        message: An optional string providing details about the throttling status.
        reasons: An optional list of strings specifying reasons for throttling.
    """

    throttled: bool
    message: Optional[str]
    reasons: Optional[List[str]]
