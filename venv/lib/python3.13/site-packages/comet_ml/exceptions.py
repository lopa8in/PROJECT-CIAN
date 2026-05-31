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

"""
Author: Boris Feld

This module contains comet generated Exceptions

"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

from .logging_messages import (
    EMPTY_PROJECT_NAME,
    EXPERIMENT_START_MODE_UNSUPPORTED,
    INVALID_API_KEY,
    INVALID_WORKSPACE_NAME,
    MAX_NUMBER_OF_EXPERIMENTS_PER_WORKSPACE_REACHED,
    SERVICE_ACCOUNT_WORKSPACE_RESTRICTED_ERROR,
    UPLOAD_LIMIT_REACHED,
)
from .semantic_version import SemanticVersion

if TYPE_CHECKING:
    from .upload_options import AvailableUploadOptions


class CometException(Exception):
    """Base-class for all comet specific exceptions"""


class CometDebugException(CometException):
    pass


class NotParametrizedException(CometException):
    def __str__(self):
        return "Please call set_params or set_params_file before calling get_suggestion"


class ValidationError(CometException):
    pass


class AuthenticationError(CometException):
    pass


class NoMoreSuggestionsAvailable(CometException):
    def __str__(self):
        return "No more suggestions available!"


class StringConversionException(CometException):
    pass


class StringValueTruncatedException(StringConversionException):
    pass


class InvalidAPIKey(CometException):

    log_message = INVALID_API_KEY

    def __init__(self, api_key: str, backend_host: str):
        super(CometException, self).__init__()
        self.api_key = api_key
        self.args = (self.api_key, backend_host)


class InvalidWorkspace(CometException):

    log_message = INVALID_WORKSPACE_NAME

    def __init__(self, workspace):
        super(CometException, self).__init__()
        self.workspace = workspace
        self.args = (self.workspace,)


class ServiceAccountWorkspaceRestricted(CometException):

    log_message = SERVICE_ACCOUNT_WORKSPACE_RESTRICTED_ERROR

    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.args = (self.workspace,)


class ProjectNameEmpty(CometException):

    log_message = EMPTY_PROJECT_NAME

    def __init__(self):
        super(CometException, self).__init__()
        self.args = tuple()


class ProjectNameIsTooLong(CometException):
    def __init__(self, err_msg):
        super(CometException, self).__init__()
        # Use backend provided err_msg
        self.log_message = err_msg
        self.args = tuple()


class InvalidOfflineDirectory(CometException):
    def __init__(self, directory, reason):
        self.directory = directory
        self.reason = reason

    def __str__(self):
        msg = "Invalid offline directory: %s\nReason:%s"
        return msg % (self.directory, self.reason)


class InterruptedExperiment(KeyboardInterrupt):
    def __init__(self, username):
        self.username = username

    def __str__(self):
        msg = "The experiment has been stopped by user %s from Comet"
        return msg % self.username


class RPCFunctionAlreadyRegistered(CometException):
    def __init__(self, function_name):
        self.function_name = function_name

    def __str__(self):
        msg = "The callback name %r is already taken"
        return msg % self.function_name


class LambdaUnsupported(CometException):
    def __init__(self):
        pass

    def __str__(self):
        return "Lambda are not supported as remote actions"


class BadCallbackArguments(CometException):

    msg = "Remote action %r should accepts at least one argument named `experiment`"

    def __init__(self, callback):
        self.callback = callback

    def __str__(self):
        return self.msg % self.callback


class ExperimentAlreadyUploaded(CometException):
    msg = "Experiment with id %r was already uploaded. Please use ExistingExperiment() to log data into an existing run or Experiment() with a new id to log data to a new run."

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

    def __str__(self):
        return self.msg % self.experiment_id


class ViewOnlyAccessException(CometException):
    """Exception to mark access issues for the view-only accounts"""

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class MaxExperimentNumberReachedException(CometException):
    """Exception to mark that maximal number of experiments per specific workspace was reached"""

    def __init__(
        self, project_name: Optional[str] = None, workspace: Optional[str] = None
    ):
        if workspace is None:
            workspace = "default"
        if project_name is None:
            project_name = "default"
        self.msg = MAX_NUMBER_OF_EXPERIMENTS_PER_WORKSPACE_REACHED % (
            project_name,
            workspace,
        )

    def __str__(self) -> str:
        return self.msg


class OfflineExperimentUploadFailed(CometException):
    """Exception to mark issues with uploading of the offline experiments"""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class ExperimentNotFound(CometException):
    pass


class ModelNotFound(CometException):
    pass


class AssetIsTooBig(CometException):
    msg = "Asset %r size %d is greater than the upload limit %d"

    def __init__(self, file_path, file_size, max_size):
        self.file_path = file_path
        self.file_size = file_size
        self.max_size = max_size

    def __str__(self):
        return self.msg % (self.file_path, self.file_size, self.max_size)


class OptimizerException(CometException):
    pass


class InvalidOptimizerParameters(CometException):
    pass


class ExperimentDisabledException(CometException):
    pass


class ExperimentNotAlive(CometException):
    pass


class LogArtifactException(CometException):
    def __init__(
        self,
        backend_err_msg: Optional[str] = None,
        sdk_error_code: Optional[int] = None,
    ) -> None:
        self.backend_err_msg = backend_err_msg
        self.sdk_error_code = sdk_error_code

    def __repr__(self) -> str:
        if self.sdk_error_code is not None and self.backend_err_msg is not None:
            return "Log artifact failed with an error, %r [%s]" % (
                self.backend_err_msg,
                self.sdk_error_code,
            )
        elif self.backend_err_msg is not None:
            return "Log artifact failed with an error, %r" % self.backend_err_msg
        else:
            return "Log artifact failed with an error, check logs for details"

    def __str__(self) -> str:
        return self.__repr__()


class LogAssetException(CometException):
    def __init__(
        self,
        backend_err_msg: Optional[str] = None,
        sdk_error_code: Optional[int] = None,
    ) -> None:
        self.backend_err_msg = backend_err_msg
        self.sdk_error_code = sdk_error_code

    def __repr__(self) -> str:
        if self.sdk_error_code is not None and self.backend_err_msg is not None:
            return "Log asset failed with an error: %r [%s]" % (
                self.backend_err_msg,
                self.sdk_error_code,
            )
        elif self.backend_err_msg is not None:
            return "Log asset failed with an error: %r" % self.backend_err_msg
        else:
            return "Log asset failed with an error, check logs for details"

    def __str__(self) -> str:
        return self.__repr__()


class AnonymousModeException(CometException):
    def __init__(
        self,
        backend_err_msg: Optional[str] = None,
        sdk_error_code: Optional[int] = None,
    ) -> None:
        self.backend_err_msg = backend_err_msg
        self.sdk_error_code = sdk_error_code

    def __repr__(self):
        if self.sdk_error_code is not None:
            return "Anonymous mode failed with an error, %r [%s]" % (
                self.backend_err_msg,
                self.sdk_error_code,
            )

    def __str__(self):
        return self.__repr__()


class GetArtifactException(CometException):
    pass


class ArtifactNotFound(CometException):
    pass


class ArtifactAssetNotFound(CometException):
    def __init__(self, logical_path, logged_artifact):
        self.logical_path = logical_path
        self.logged_artifact = logged_artifact

    def __repr__(self):
        return "No asset found for logical_path %r in %r" % (
            self.logical_path,
            self.logged_artifact,
        )

    def __str__(self):
        return self.__repr__()


class ArtifactDownloadException(CometException):
    pass


class ArtifactNotFinalException(CometException):
    pass


class ArtifactConflictingAssetLogicalPath(CometException):
    def __init__(self, existing_asset, conflicting_asset, common_logical_path):
        # type (Any, str, Any) -> None
        self.existing_asset = existing_asset
        self.conflicting_asset = conflicting_asset
        self.common_logical_path = common_logical_path

    def __repr__(self):
        return (
            "Cannot add new asset (%r) with logical_path %r, an existing asset already exists with this logical_path (%r). To add this asset to this artifact you should use a new unique logical_path"
            % (self.conflicting_asset, self.common_logical_path, self.existing_asset)
        )

    def __str__(self):
        return self.__repr__()


class RemoteModelDownloadException(CometException):
    pass


class ExperimentCleaningException(CometException):
    pass


class CometRestApiException(CometException):
    def __init__(self, method: str, response: requests.Response):
        super().__init__()
        self.method = method
        self.response = response
        try:
            self.safe_json_response = (
                self.response.json()
            )  # type: Optional[Dict[Any, Any]]
            self.safe_msg = self.safe_json_response.get("msg", None)  # type: ignore[union-attr]
            self.sdk_error_code = self.safe_json_response.get("sdk_error_code", None)  # type: ignore[union-attr]
        except (ValueError, TypeError):
            self.safe_json_response = None
            self.safe_msg = None
            self.sdk_error_code = None

    def __str__(self):
        if self.safe_msg:
            return "%s %s failed with status code %s: %s" % (
                self.method,
                self.response.url,
                self.response.status_code,
                self.safe_msg,
            )
        else:
            return "%s %s failed with status code %s (%s)" % (
                self.method,
                self.response.url,
                self.response.status_code,
                self.response.content,
            )


class CometRestApiValueError(CometRestApiException):
    def __init__(self, method, message, response):
        super(CometRestApiValueError, self).__init__(method, response)
        self.message = message

    def __str__(self):
        return "%s %s %s; failed with status code %s" % (
            self.method,
            self.response.url,
            self.message,
            self.response.status_code,
        )


class PaymentRequired(CometRestApiException):
    def __str__(self):
        try:
            return self.response.json().get("msg", "")
        except ValueError:
            return None


class NotFound(CometRestApiException):
    def __str__(self):
        try:
            return self.response.json().get("msg", "")
        except ValueError:
            return None


class Unauthorized(CometRestApiException):

    def __init__(self, api_key: str, response: requests.Response):
        super().__init__(response.request.method, response)
        self.api_key = api_key

    def __str__(self):
        msg = super().__str__()
        return f"API key: {self.api_key}, reason: {msg}"


class QueryException(CometException):
    pass


class PersistenceModuleNotSupportedException(CometException):
    pass


class PersistenceModuleNotFoundException(CometException):
    pass


class BackendVersionTooOld(CometException):
    msg = """The API client requires backend version {minimal_backend_version} or higher.
Detected an incompatible backend version {backend_version} (at {backend_url}).
Please update your backend version or inform your administrator to update it.
In the mean-time, you can downgrade the python SDK to a version compatible with your backend with: `pip install -U 'comet-ml<3'`"""

    def __init__(
        self,
        backend_url,
        backend_version: SemanticVersion,
        minimal_backend_version: SemanticVersion,
    ):
        self.backend_url = backend_url
        self.backend_version = backend_version
        self.minimal_backend_version = minimal_backend_version

    def __str__(self):
        return self.msg.format(
            backend_url=self.backend_url,
            backend_version=self.backend_version,
            minimal_backend_version=self.minimal_backend_version,
        )


class UploadLimitReachedException(CometException):
    def __init__(self, asset_type: str):
        self.asset_type = asset_type

    def __str__(self):
        return UPLOAD_LIMIT_REACHED.format(asset_type=self.asset_type)


class ImagesUploadLimitReachedException(UploadLimitReachedException):
    ASSET_TYPE = "images"

    def __init__(self):
        super().__init__(asset_type=self.ASSET_TYPE)


class InvalidExperimentMode(CometException):
    pass


class InvalidExperimentModeUnsupported(InvalidExperimentMode):
    def __init__(self, mode: str, supported_modes: List[str]):
        self.message = EXPERIMENT_START_MODE_UNSUPPORTED % (mode, supported_modes)

    def __str__(self):
        return self.message


class SDKVersionIsTooOldException(CometException):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        if self.message is not None:
            return self.message
        else:
            return "Obsolete SDK version. Please update."


class BackendCustomError(CometException):
    pass


class ProjectConsideredLLM(CometException):
    pass


class FileUploadThrottledException(CometException):
    """Exception raised when backend throttles file upload (HTTP 429)"""

    def __init__(
        self,
        response: requests.Response,
        upload_options: "AvailableUploadOptions" = None,
    ) -> None:
        super().__init__()
        self.response = response
        self.upload_options = upload_options  # Store the original upload options
        upload_type = (
            upload_options.upload_type if upload_options is not None else "unknown"
        )
        self.args = (
            f"File upload throttled for type '{upload_type}' (HTTP 429)",
            response,
        )


"""
SDK Error Code mapping, names comes from the backend
"""

API_KEY_NOT_REGISTERED = 90212
NO_WRITE_ACCESS_TO_EXPERIMENT = 90210
EXPERIMENT_ALREADY_EXISTS = 90999
INVALID_OPTIMIZATION_ID = 5124123
NO_PROJECT_NAME_SPECIFIED = 98219
NON_EXISTING_TEAM = 90219
PROJECT_NAME_TOO_LONG = 1341241

# "ASSET UPLOAD LIMIT REACHED" ERROR CODES
IMAGE_UPLOAD_LIMIT_REACHED = 7000001
TEXT_SAMPLES_LIMIT_REACHED = 7000012
HISTOGRAM_LIMIT_REACHED = 7000013
MODEL_ELEMENT_LIMIT_REACHED = 7000014
CURVE_LIMIT_REACHED = 7000015
NOTEBOOK_LIMIT_REACHED = 7000016
EMBEDDINGS_LIMIT_REACHED = 7000017
VIDEO_LIMIT_REACHED = 7000018

ASSET_STATE_IMMUTABLE = 7000005
IMAGE_STATE_IMMUTABLE = 7000006
VIEW_ONLY_PERMISSIONS = 7000011
OBSOLETE_SDK_VERSION = 7000008
MAX_EXPERIMENTS_NUMBER_REACHED = 7000020
BACKEND_CUSTOM_ERROR = 7777777
PROJECT_CONSIDERED_LLM = 34323

INVALID_VERSION_STRING = 42001

SERVICE_ACCOUNT_WORKSPACE_RESTRICTED = 7000023

# Artifacts
ARTIFACT_VERSION_EXISTS = 90399
NO_ARTIFACT_FOUND = 624522
REUSE_DELETED_ARTIFACT_VERSION = 7000019
