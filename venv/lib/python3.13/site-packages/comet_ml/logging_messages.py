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

# Don't import comet_ml items here to avoid import cycles

MSG_FORMAT = "%(prefix)s %(message)s"

FILE_MSG_FORMAT = "[%(process)d-%(processName)s:%(thread)d] %(relativeCreated)d COMET %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"

DEBUG_FILE_MSG_FORMAT = "%(asctime)s [%(process)d-%(processName)s:%(thread)d] %(relativeCreated)d COMET %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"

GO_TO_DOCS_MSG = " \nFor more details, please refer to: https://www.comet.com/docs/v2/api-and-sdk/python-sdk/warnings-errors/"

INTERNET_CONNECTION_ERROR = (
    "Failed to establish a connection to Comet server. Please check your internet connection. "
    "Your experiment will not be logged" + GO_TO_DOCS_MSG
)

INVALID_WORKSPACE_NAME = "Workspace %s doesn't exist."

SERVICE_ACCOUNT_WORKSPACE_RESTRICTED_ERROR = (
    "Service account doesn't have access to workspace %s"
)

EMPTY_PROJECT_NAME = "project_name argument can't be empty."

FAILED_TO_CREATE_EXPERIMENT_COMET_ERROR = (
    "Failed to create experiment for unexpected reason: %s"
)

INVALID_API_KEY = (
    "The given API key '%s' is invalid on '%s', please check it against the dashboard. "
    "Your experiment will not be logged" + GO_TO_DOCS_MSG
)

METRIC_ARRAY_WARNING = (
    "Cannot safely convert %r object to a scalar value, using its string"
    " representation for logging. Resulting string might be invalid"
)

PARSING_ERR_MSG = """We failed to parse your parameter configuration file.

Type casting will be disabled for this run, please fix your configuration file.
"""

CASTING_ERROR_MESSAGE = """Couldn't cast parameter %r, returning raw value instead.
Please report it to comet.com and use `.raw(%r)` instead of `[%r]` in the meantime."""

LOG_ASSET_FOLDER_ERROR = (
    "We failed to read directory %r for uploading.\n"
    "Please double-check the file path, permissions, and that it is a directory."
)

LOG_ASSET_FOLDER_EMPTY = (
    "Directory %r is empty; no files were uploaded.\n"
    "Please double-check the directory path and the recursive parameter."
)

UPLOAD_FILE_OS_ERROR = (
    "We failed to read file %r for uploading.\n"
    "Please double-check the file path and permissions"
)

UPLOAD_ASSET_TOO_BIG = "Asset %s is bigger than the upload limit, %s > %s"

UPLOAD_LIMIT_REACHED = (
    "Experiment has reached the maximum number of {asset_type} allowed. "
    "No more {asset_type} would be uploaded. "
    "Please contact us if you would like to increase the limit."
)

LOG_IMAGE_TOO_BIG = "Image %s is bigger than the upload limit, %s > %s"

LOG_FIGURE_TOO_BIG = "Figure number %d is bigger than the upload limit, %s > %s"

LOG_AUDIO_TOO_BIG = "Audio %s is bigger than the upload limit, %s > %s"

LOG_VIDEO_TOO_BIG = "Video %s is bigger than the upload limit, %s > %s"

NATIVE_STD_WRAPPER_NOT_AVAILABLE = (
    "Native output logging mode is not available, falling back to basic output logging"
)

UNKNOWN_STD_WRAPPER_SPEC = (
    "Unknown output logging mode: %s, falling back to basic output logging"
)

CONSOLE_DEBUG_DEPRECATED = (
    "Switching the console logger level from DEBUG to INFO, as DEBUG is now deprecated."
)

EXPERIMENT_LIVE = "Experiment is live on comet.com %s\n"

EXPERIMENT_MESSAGE_QUEUE_FLUSH_PROMPT = (
    "Please wait for metadata to finish uploading (timeout is %s seconds)"
)

OFFLINE_EXPERIMENT_END = "To upload this offline experiment, run:\n    comet upload %s"

OFFLINE_EXPERIMENT_SAVE_ARCHIVE_INFO = "Begin archiving the offline data."

OFFLINE_EXPERIMENT_UPLOAD_ZIP_STARTED_INFO = "Trying to upload offline experiment's ZIP file (%r) using upload callback. This may take some time."

OFFLINE_EXPERIMENT_UPLOAD_ZIP_COMPLETED_INFO = (
    "Offline experiment's ZIP file was successfully uploaded using upload callback."
)

OFFLINE_EXPERIMENT_UPLOAD_ZIP_CALLBACK_ERROR = (
    "Failed to upload offline experiment's ZIP file using upload callback."
)

OFFLINE_SENDER_STARTS = "Starting the upload of the experiment"

OFFLINE_SENDER_ENDS = "The offline experiment has been uploaded on comet.com %s\n"

OFFLINE_SENDER_ENDS_PROCESSING = "Offline Experiment will now be processed---may take some time to appear on the Comet dashboard."

OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT = "Still saving offline stats to messages file before program termination (may take up to %d seconds)"

OFFLINE_SENDER_REMAINING_DATA_ITEMS_TO_WRITE = (
    "%d offline data items still in progress of writing to messages file"
)

OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA = (
    "Comet failed to write all data items to messages file (%s messages remaining)"
)

OFFLINE_DATA_DIR_DEFAULT_WARNING = (
    "Using %r path as offline directory. Pass 'offline_directory' parameter into constructor "
    "or set the 'COMET_OFFLINE_DIRECTORY' environment variable to manually choose where to store offline "
    "experiment archives."
)

OFFLINE_DATA_DIR_FAILED_WARNING = (
    "Failed to use offline data directory %r, the experiment will be saved in the temporary directory %r instead.\n"
    "Reason: %s"
)

OFFLINE_UPLOADING_EXPERIMENT_FILE_PROMPT = "Attempting to upload %r..."

OFFLINE_SUCCESS_UPLOADED_EXPERIMENTS = "Number of uploaded experiments: %d"

OFFLINE_FAILED_UPLOADED_EXPERIMENTS = "Number of failed experiment uploads: %d"

OFFLINE_UPLOAD_FAILED_UNEXPECTED_ERROR = "Upload failed for %r; unknown error"

OFFLINE_UPLOAD_FAILED_INVALID_API_KEY = (
    "Upload failed because of invalid Comet API key; please set COMET_API_KEY"
)

OFFLINE_UPLOADS_FAILED_DUE_TIMEOUT = (
    "Scheduled uploads failed due to timeout, remaining uploads: %d"
)

OFFLINE_FAILED_TO_REGISTER_MODEL_NO_MODEL_FILES = (
    "Failed to register model: %s, no model files was logged"
)

ADD_TAGS_ERROR = "Failed to add tag(s) %r to the experiment\n"

ADD_SYMLINK_ERROR = "Failed to create symlink to project:%s for experiment\n"

OFFLINE_EXPERIMENT_INVALID_WS_MSG = "An invalid message has been detected"

OFFLINE_EXPERIMENT_INVALID_METRIC_MSG = "An invalid metric message has been detected."

OFFLINE_EXPERIMENT_INVALID_UPLOAD_MSG = "An invalid upload message has been detected"

OFFLINE_EXPERIMENT_INVALID_PARAMETER_MSG = (
    "An invalid parameter message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_OS_PACKAGES_MSG = (
    "An invalid OS packages message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_GRAPH_MSG = "An invalid graph message has been detected"

OFFLINE_EXPERIMENT_INVALID_SYSTEM_INFO_MSG = (
    "An invalid system_info message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_SYSTEM_DETAILS_MSG = (
    "An invalid system details message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_CLOUD_DETAILS_MSG = (
    "An invalid cloud details message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_FILE_NAME_MSG = (
    "An invalid file name message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_LOG_OTHER_MSG = (
    "An invalid log other message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_INSTALLED_PACKAGES_MSG = (
    "An invalid installed packages message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_HTML_MSG = "An invalid html message has been detected"

OFFLINE_EXPERIMENT_INVALID_HTML_OVERRIDE_MSG = (
    "An invalid html override message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_GPU_STATIC_INFO_MSG = (
    "An invalid gpu static info message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_GIT_METADATA_MSG = (
    "An invalid git metadata message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_STANDARD_OUTPUT_MESSAGE = (
    "An invalid standard output message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_LOG_DEPENDENCY_MESSAGE = (
    "An invalid log dependency message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_REMOTE_MODEL_MESSAGE = (
    "An invalid remote model message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_3D_CLOUD_MESSAGE = (
    "An invalid 3D cloud message has been detected"
)

OFFLINE_EXPERIMENT_INVALID_REGISTER_MODEL_MESSAGE = (
    "An invalid register model message has been detected"
)


OFFLINE_EXPERIMENT_ALREADY_UPLOADED = "Upload failed for %r; experiment was already uploaded, you can re-upload it by using the --force-reupload flag"

OFFLINE_EXPERIMENT_ALREADY_EXISTS_CREATE_MODE = "Upload failed for %r; An experiment with the key %r already exists in Comet, you can upload it as a new experiment by using the --force-upload flag"

OFFLINE_EXPERIMENT_NOT_FOUND_GET_MODE = "Upload failed for %r; experiment was not found, you can upload it as a new experiment by using the --force-upload flag"

OFFLINE_AT_LEAST_ONE_EXPERIMENT_UPLOAD_FAILED = "At least one experiment failed to upload, check COMET ERROR messages above for more information."

OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_PARAMETER = "Using workspace '%(creation_workspace)s', set by parameter rather than workspace '%(metadata_workspace)s' set in the Offline Experiment Archive"

OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_PARAMETER = "Using project_name '%(creation_project_name)s', set by parameter rather than project_name '%(metadata_project_name)s' set in the Offline Experiment Archive"

OFFLINE_EXPERIMENT_CREATION_WORKSPACE_OVERRIDDEN_CONFIG = "Using workspace '%(creation_workspace)s', set in configuration rather than workspace '%(metadata_workspace)s' set in the Offline Experiment Archive"

OFFLINE_EXPERIMENT_CREATION_PROJECT_NAME_OVERRIDDEN_CONFIG = "Using project_name '%(creation_project_name)s', set in configuration rather than project_name '%(metadata_project_name)s' set in the Offline Experiment Archive"

EXPERIMENT_START_MODE_UNSUPPORTED = (
    "Unsupported experiment start mode: %r, you should use any of %r"
)

EXPERIMENT_INVALID_STEP = "Passed step value %r is not a number, ignoring it"

EXPERIMENT_INVALID_EPOCH = "Passed epoch value %r is not a number, ignoring it"

EXPERIMENT_INITIAL_DATA_LOGGER_INCOMPLETE = (
    "Not all initial data has been logged for experiment %s, call Experiment.end() "
    "to ensure that all data to have been logged"
)

EXPERIMENT_INITIAL_DATA_LOGGER_WAITING_TO_COMPLETE = (
    "The process of logging environment details (conda environment, git patch) is underway. "
    "Please be patient as this may take some time."
)

EXPERIMENT_INITIAL_DATA_LOGGER_FLUSH_FAILED = "Failed to complete logging of all environment details (conda environment, git patch)"

EXPERIMENT_GET_PARAMETER_SHORT_NAME_DEPRECATION = (
    "Getting a parameter without its full name is deprecated and will be removed in the future, "
    "potential full-names for the '%s' parameter are %s"
)

STREAMER_WAIT_FOR_FINISH_FAILED = "Comet failed to send all the data back (%d messages and %d uploads), experiment ID: %s"

STREAMER_CLOSED_PUT_MESSAGE_FAILED = (
    "Attempting to put message into closed streamer: message ignored! "
    "You need to call Experiment.end() to have everything logged properly."
)

STREAMER_FAILED_TO_PROCESS_ALL_MESSAGES = (
    "Failed to send all messages, metrics and output will likely be incomplete."
)

REGISTER_RPC_FAILED = "Failed to register callback named %r"

SEND_NOTIFICATION_FAILED = "Error sending a notification, make sure you have opted-in for notifications. Failure: %s"

EXPERIMENT_MARK_AS_STARTED_FAILED = (
    "Failed to report experiment status (experiment started)"
)
EXPERIMENT_MARK_AS_ENDED_FAILED = (
    "Failed to report experiment status (experiment ended)"
)

GET_CALLBACK_FAILURE = "Failure to add Comet SDK callback to %s, the run will likely miss parameters and/out metrics"

COMET_DISABLED_AUTO_LOGGING_MSG = "COMET_DISABLE_AUTO_LOGGING is 1; ignoring '%s'"

OS_PACKAGE_MSG_SENDING_ERROR = "Error sending os_packages message, got %d %r"

METRICS_BATCH_MSG_SENDING_ERROR = "Failed to send metrics batch message, got %d %r"

PARAMETERS_BATCH_MSG_SENDING_ERROR = (
    "Failed to send parameters batch message, got %d %r"
)

MODEL_GRAPH_MSG_SENDING_ERROR = "Error sending model graph message, got %d %r"

SYSTEM_DETAILS_MSG_SENDING_ERROR = (
    "Error sending model system details message, got %d %r"
)

CLOUD_DETAILS_MSG_SENDING_ERROR = "Error sending cloud details messages, got %d %r"

HTML_MSG_SENDING_ERROR = "Error sending html message, got %d %r"

HTML_OVERRIDE_MSG_SENDING_ERROR = "Error sending html override message, got %d %r"

GPU_STATIC_INFO_MSG_SENDING_ERROR = "Error sending gpu static info message, got %d %r"

GIT_METADATA_MSG_SENDING_ERROR = "Error sending git metadata message, got %d %r"

SYSTEM_INFO_MESSAGE_SENDING_ERROR = "Error sending system_info message, got %d %r"

LOG_OTHER_MSG_SENDING_ERROR = "Error sending log other messages, got %d %r"

LOG_DEPENDENCY_MESSAGE_SENDING_ERROR = (
    "Error sending log dependency messages, got %d %r"
)

REMOTE_MODEL_MESSAGE_SENDING_ERROR = (
    "Error sending log remote model messages, got %d %r"
)

REGISTER_MODEL_MESSAGE_SENDING_ERROR = (
    "Failed to send register model message, got %d %r"
)

STANDARD_OUTPUT_SENDING_ERROR = "Failed to send stdout/stderr message batch, got %d %r"

ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_REST_ERROR = (
    "Failed to send artifact remote assets batch, got %d %r"
)

ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_UNKNOWN_ERROR = (
    "Failed to send artifact remote assets batch (online experiment)"
)

ARTIFACT_NAME_TRUNCATED_WARNING = (
    "Artifact name is too long. It was truncated to be %d characters long."
)

FILENAME_DETAILS_MSG_SENDING_ERROR = (
    "Error sending file name details messages, got %d %r"
)

INSTALLED_PACKAGES_MSG_SENDING_ERROR = (
    "Error sending installed packages messages, got %d %r"
)

OPTIMIZER_KWARGS_CONSTRUCTOR_DEPRECATED = "Passing Experiment through Optimizer constructor is deprecated; pass them to Optimizer.get_experiments or Optimizer.next"

BACKEND_VERSION_CHECK_ERROR = "Failed to check backend version at URL: %r, reason: %s"

INVALID_CONFIG_MINIMAL_BACKEND_VERSION = "Invalid configured `comet.rest_v2_minimal_backend_version` value %r, skipping backend version check"

MLFLOW_NESTED_RUN_UNSUPPORTED = "MLFlow Nested Runs are not tracked in Comet.ml SDK."

MLFLOW_OFFLINE_EXPERIMENT_FALLBACK = "No Comet API Key was found, creating an offline experiment. Set up your API Key to get the full Comet experience https://www.comet.com/docs/python-sdk/advanced/#python-configuration"

MLFLOW_RESUMED_RUN = "Resumed MLFlow run are tracked as new Comet.ml Experiments"

EXPERIMENT_THROTTLED = "Experiment has been throttled. Some data (like experiment metrics) might be missing. Mission-critical data, like Artifacts, are never throttled.\n%s"

TF_KERAS_FALLBACK_FAILED = (
    "Keras is not available; attempted to fallback to tensorflow.keras but still failed"
)

TF_KERAS_CALLBACK_WARNING_CLOSED_EXPERIMENT = "A Keras callback is trying to report to a un-alive experiment object (it was either ended or the creation failed) from the %s callback method, data will not be logged."

CONFUSION_MATRIX_ERROR_WRONG_LENGTH = (
    "y_true and y_predicted should have the same lengths (%d != %d)"
)

CONFUSION_MATRIX_ERROR_RESULTING_LENGTH = (
    "Resulting len(y_true) != len(y_predicted) (%s != %s)"
)

CONFUSION_MATRIX_ERROR_MUST_GIVE_BOTH = (
    "if you give y_true OR y_predicted you must give both"
)
CODECARBON_NOT_INSTALLED = "codecarbon package not installed; skipping co2 tracking"

CODECARBON_START_FAILED = "Failed to set-up CO2 logger"

CODECARBON_STOP_FAILED = "Failed to shutdown the CO2Tracker properly"

CODECARBON_DIR_CREATION_FAILED = "Failed to create the temporary CO2 tracker directory %r; skipping CO2 tracking; error %r"

CONFUSION_MATRIX_INDEX_TO_EXAMPLE_ERROR = (
    "%r failed for index %s; example not generated"
)
CONFUSION_MATRIX_EXAMPLE_NONE = "%r returned None for index %s; example not generated"

CONFUSION_MATRIX_EXAMPLE_DICT_INVALID_FORMAT = "%r returned an invalid dict for index %s, must match {'sample': ..., 'assetId': ...}"

CONFUSION_MATRIX_EXAMPLE_INVALID_TYPE = "%r returned invalid type %r at index %s, must return an int, string, URL, or {'sample': string, 'assetId': string}"

MISSING_PANDAS_OR_YDATA_PROFILING = (
    "ydata_profiling or pandas_profiling is required to log profile; ignoring"
)

SET_CODE_FILENAME_DEPRECATED = "Experiment.set_code(filename=...) is deprecated, use Experiment.log_code(file_name=...) instead"

SET_CODE_CODE_DEPRECATED = "Experiment.set_code(code=...) is deprecated, use Experiment.log_code(code=..., code_name=...) instead"

LOG_PARAMS_EMPTY_MAPPING = (
    "Empty mapping given to Experiment.log_parameters(%r); ignoring"
)

LOG_PARAMS_EMPTY_CONVERTED_MAPPING = (
    "%r passed to Experiment.log_parameters converted to an empty mapping; ignoring"
)

LOG_PARAMS_EMPTY_CONVERTED_MAPPING_ERROR = (
    "%r passed to Experiment.log_parameters converted to an empty parameters mapping. "
    "Make sure to pass parameters mapping with at least one key/value pair."
)

LOG_PARAMS_MAX_DEPTH_REACHED = "%r passed to Experiment.log_parameters exceeded the nested limit of %d, Comet will convert the sub layers into strings."

LOG_METRICS_MAX_DEPTH_REACHED = "%r passed to Experiment.log_metrics exceeded the nested limit of %d, Comet will convert the sub layers into strings."

CONFIG_API_KEY_IS_NOT_SET = "API key is not set. Please provide as the argument to init() or as an environment variable named COMET_API_KEY."

CONFIG_API_KEY_IS_INVALID = (
    "Invalid Comet API key %r for %s\n"
    + "Please see https://www.comet.com/docs/command-line/#comet-check for more information.\n"
    + "Use: comet_ml.login() to try again"
)

CONFIG_API_KEY_CHECK_FAILED = "Unable to verify Comet API key at this time"

JUPYTER_NEEDS_END = "As you are running in a Jupyter environment, you will need to call `experiment.end()` when finished to ensure all metrics and code are logged before exiting."

PYDEV_NEEDS_END = "As you are running in a Pydev Console, you will need to call `experiment.end()` when finished to ensure all metrics and code are logged before exiting."

AWS_LAMBDA_NEEDS_END = "As you are running in a Lambda function, you will need to call `experiment.end()` when finished to ensure all metrics and code are logged before exiting."

ARTIFACT_VERSION_CREATED_WITH_PREVIOUS = (
    "Artifact %r version %s created (previous was: %s)"
)

ARTIFACT_VERSION_CREATED_WITHOUT_PREVIOUS = "Artifact %r version %s created"

GET_ARTIFACT_WORKSPACE_GIVEN_TWICE = "Workspace was given both explicitly %r and as part of the fully-qualified artifact name %r, using the explicit value"

GET_ARTIFACT_VERSION_OR_ALIAS_GIVEN_TWICE = "Version_or_alias was given both explicitly %r and as part of the fully-qualified artifact name %r, using the explicit value"

ARTIFACT_DOWNLOAD_FILE_OVERWRITTEN = (
    "File %r has been overwritten by asset %s of artifact version %s"
)

ARTIFACT_DOWNLOAD_FILE_PRESERVED = 'File %r is different from asset %s of artifact version %s, but was kept because overwrite_strategy="PRESERVE" was set'

ARTIFACT_UPLOAD_STARTED = "Artifact '%s/%s:%s' has started uploading asynchronously"

ARTIFACT_UPLOAD_FINISHED = "Artifact '%s/%s:%s' has been fully uploaded successfully"

ARTIFACT_ASSET_UPLOAD_FAILED = "Asset %r of artifact '%s/%s:%s' upload failed"

LOG_ARTIFACT_IN_PROGRESS_MESSAGE = "Still scheduling the upload of {} local assets, remaining size {}; and linking of {} remote assets."

UPLOADING_DATA_BEFORE_TERMINATION = "Uploading metrics, params, and assets to Comet before program termination (may take several seconds)"

WAITING_FOR_FILE_UPLOADS_COMPLETION = (
    "Waiting for completion of the file uploads (may take several seconds)"
)

FILE_UPLOADS_PROMPT = (
    "Please wait for assets to finish uploading (timeout is %s seconds)"
)

FILE_UPLOAD_MANAGER_MONITOR_FIRST_MESSAGE = (
    "Still uploading %d file(s), remaining %s/%s"
)

FILE_UPLOAD_MANAGER_MONITOR_WAITING_BACKEND_ANSWER = (
    "All assets have been sent, waiting for delivery confirmation"
)

FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION = (
    "Still uploading %d asset(s), remaining %s/%s, Throughput %s/s, ETA ~%ss"
)

FILE_UPLOAD_MANAGER_MONITOR_PROGRESSION_UNKOWN_ETA = (
    "Still uploading %d asset(s), remaining %s/%s, Throughput %s/s, ETA unknown"
)

FILE_UPLOAD_MANAGER_FAILED_TO_SUBMIT_EXECUTOR_CLOSED = "Failed to upload data because uploader thread already closed. You may need to call Experiment.end() explicitly to resolve this issue."

FILE_UPLOAD_MANAGER_FAILED_TO_SUBMIT_ALREADY_CLOSED = "Will not submit data to be uploaded because uploader already closed.\nUpload data: %r"

ARTIFACT_DOWNLOAD_START_MESSAGE = (
    "Artifact '%s/%s:%s' download has been started asynchronously"
)

ARTIFACT_DOWNLOAD_FINISHED = "Artifact '%s/%s:%s' has been successfully downloaded"

ARTIFACT_DOWNLOAD_READ_FILE_EXCEPTION = (
    "Cannot read file %r to compare content, check logs for details."
)

ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_EXCEPTION = (
    "Cannot write Asset %r on disk path %r, check logs for details."
)

ARTIFACT_DOWNLOAD_WRITE_ASSET_FILE_ALREADY_EXISTS_EXCEPTION = (
    "Cannot write Asset %r on path %r, a file already exists."
)

ARTIFACT_ASSET_DOWNLOAD_FAILED = "Asset %r of the artifact '%s/%s:%s' download failed"

ARTIFACT_ASSET_DOWNLOAD_FAILED_REPR = "Cannot download Asset %r for Artifact %s"

ARTIFACT_FAILED_TO_MARK_VERSION_AS_FAILED_ERROR = (
    "Failed to mark the artifact version as error."
)

ARTIFACT_FAILED_TO_MARK_VERSION_AS_CLOSED_ERROR = (
    "Failed to mark the artifact version as closed"
)

ARTIFACT_ASSET_TYPE_DEPRECATED_WARNING = "The asset type parameter is deprecated."

ARTIFACT_FAILED_TO_PARSE_REMOTE_ASSET_URI_WARNING = (
    "Failed to parse artifact's remote asset's URI: %r"
)

ARTIFACT_GET_NOT_FINAL_EXCEPTION = (
    "Artifact %r is not in a finalized state and cannot be accessed."
)

ARTIFACT_GET_NOT_FOUND_EXCEPTION = "Artifact not found with %r."

ARTIFACT_GET_GENERAL_EXCEPTION = (
    "Get artifact failed with an error, check the logs for details."
)

ARTIFACT_GET_UNAUTHORIZED_EXCEPTION = "Unauthorized to get artifact. Please check that you have access to the artifact: %r at workspace: %r"

ARTIFACT_GET_FAILED_TO_DECODE_METADATA_WARNING = (
    "Couldn't decode metadata for artifact %r:%r"
)

ARTIFACT_DOWNLOAD_CANNOT_GET_ASSETS_LIST_EXCEPTION = (
    "Cannot get asset list for Artifact %r"
)

ARTIFACT_DOWNLOAD_CANNOT_DOWNLOAD_ASSET_EXCEPTION = (
    "Cannot download Asset %s for Artifact %s"
)

ARTIFACT_LOG_EMPTY_ASSETS_WARNING = (
    "Artifact %s created without adding any assets, was this the intent?"
)

ARTIFACT_LOG_ASSETS_SCHEDULING_INFO = (
    "Scheduling the upload of {} assets: {} local assets for a size of {}, "
    "and {} remote assets (will be linked, not uploaded). "
    "This can take some time."
)

ARTIFACT_OVERWRITE_INVALID_STRATEGY_EXCEPTION = "Invalid user_overwrite value %r"

ASSET_DOWNLOAD_FAILED_WITH_ERROR = (
    "Cannot download synced Remote Asset with logical path %r, original error: '%s'"
)

ASSET_WRITE_ERROR = "Cannot write Asset %r on disk path %r, check logs for details"

REMOTE_MODEL_DOWNLOAD_START_MESSAGE = (
    "Remote Model '%s/%s:%s' download has been started asynchronously."
)

REMOTE_MODEL_DOWNLOAD_FAILED = (
    "Asset %r of the remote model '%s/%s:%s' download failed."
)

REMOTE_MODEL_DOWNLOAD_FINISHED = (
    "Remote Model '%s/%s:%s' has been successfully downloaded."
)

REMOTE_MODEL_DOWNLOAD_FILE_OVERWRITTEN = (
    "File %r has been overwritten by asset %r of remote model"
)

REMOTE_MODEL_ASSET_DOWNLOAD_FAILED_REPR = "Cannot download Asset %r for Remote Model %s"

FILE_DOWNLOAD_MANAGER_COMPLETED = "All files downloaded"

FILE_DOWNLOAD_MANAGER_MONITOR_FIRST_MESSAGE = (
    "Still downloading %d file(s), remaining %s/%s"
)

FILE_DOWNLOAD_MANAGER_MONITOR_PROGRESSION_UNKNOWN_ETA = (
    "Still downloading %d file(s), remaining %s/%s, Throughput %s/s, ETA unknown"
)

FILE_DOWNLOAD_MANAGER_MONITOR_PROGRESSION = (
    "Still downloading %d file(s), remaining %s/%s, Throughput %s/s, ETA ~%ss"
)

INITIAL_DATA_LOGGER_FLUSHING_EXPERIMENT_INTERRUPTED_BY_USER = "Experiment was interrupted by user while waiting for the initial data logger to be flushed."

INITIAL_DATA_LOGGER_FLUSHING_FAILED = (
    "Unexpected failure while waiting for the initial data logger to be flushed."
)

LOG_GIT_METADATA_ERROR = "Failed to log git metadata"

LOG_GIT_PATCH_ERROR = "Failed to log git patch"

GIT_REPO_NOT_FOUND = "Couldn't find a Git repository in %r nor in any parent directory. Set `COMET_GIT_DIRECTORY` if your Git Repository is elsewhere."

GIT_LOGGING_ERROR = "Error logging git-related information"

REPORTING_ERROR = "Failing to report %s"

OFFLINE_EXPERIMENT_NAME_ACCESS = "Experiment Name is generated at upload time for Offline Experiments unless set explicitly with Experiment.set_name"

START_PARAMETERS_DONT_MATCH_EXISTING_EXPERIMENT = "An experiment already exists %r but all its parameters don't match comet_ml.start, closing it and recreating one"

START_RUNNING_EXPERIMENT_IGNORED_PARAMETER = "Parameter '%s' with value '%s' was ignored while checking if Experiment instance %r matched the start parameters"

LOG_COLAB_NOTEBOOK_ERROR = (
    "Couldn't retrieve and log Google Colab notebook content, reason: %s"
)

CONVERT_TABLE_INVALID_FORMAT = "Tabular filename must end with '.tsv' or '.csv', not %r"

CONVERT_DATAFRAME_INVALID_FORMAT = (
    "Tabular dataframe filename must end with 'json', 'csv', 'md', or 'html', not %r"
)

LOG_TABLE_NONE_VALUES = "Either tabular_data or a valid file-path are required when calling log_table; ignoring"

LOG_TABLE_FILENAME_AND_HEADERS = (
    "Headers are ignored when calling log_table with a filename"
)

DATAFRAME_CONVERSION_ERROR = "dataframe conversion to %r failed; ignored"

METRIC_NONE_WARNING = "The given value of the metric %s was None; ignoring"

LOG_METRIC_AUTOMATIC_IGNORED = "Ignoring automatic log_metric(%r) because '%s:%s' is in COMET_LOGGING_METRICS_IGNORE"

LATE_IMPORT_DEGRADED_MODE = "To get all data logged automatically, import comet_ml before the following modules: %s."

LOG_CONDA_ENV_ERROR = "Unknown error exporting current conda environment"
LOG_CONDA_ENV_RETURN_CODE = "Error exporting current conda environment"

LOG_CONDA_PACKAGES_ERRORS = "Unknown error retrieving Conda package as an explicit file"
LOG_CONDA_PACKAGES_RETURN_CODE = "Error retrieving Conda package as an explicit file"

LOG_CONDA_INFO_ERRORS = "Unknown error retrieving Conda information"
LOG_CONDA_INFO_RETURN_CODE = "Error retrieving Conda information"

OPTIMIZER_COULDNT_END = "Optimizer couldn't end"

FAILED_LOG_EMBEDDING_GROUPS = "Couldn't log embedding groups"

FAILED_LOG_IPYTHON_NOTEBOOK = "Couldn't log ipython notebook"

COULDNT_COMPLETE_ATEXIT_UNREGISTER = "Couldn't complete atexit unregister"

KUBEFLOW_LOGGER_ERROR = "Unkown error logging Kubeflow data in comet_ml.integration.kubeflow.initialize_comet_logger, experiment will be tagged with %r"

KUBEFLOW_LOGGER_IMPORT_ERROR = "python package 'kfp' is required to use comet_ml.integration.kubeflow.initialize_comet_logger, make sure to install a version compatible with your environment"

FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET = (
    "Failed to add remote synced asset with URI '%s'. Fallback to not synced mode."
)

FAILED_TO_ADD_REMOTE_MODEL_SYNC_ASSET = "Failed to add remote model synced asset with URI '%s'. Fallback to not synced mode."

SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_ARTIFACT = "Synced mode is not supported for string remote artifact assets (URI '%s'). Fallback to not synced mode."

SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_MODEL = "Synced mode is not supported for string remote model assets (URI '%s'). Fallback to not synced mode."

FAILED_TO_ADD_S3_ASSETS_TOO_MANY_KEYS = "Failed to add remote assets from S3 bucket '%s', the number of files in the bucket exceeds max_synced_objects: %d"
FAILED_TO_ADD_GS_ASSETS_TOO_MANY_KEYS = "Failed to add remote assets from GS bucket '%s', the number of files in the bucket exceeds max_synced_objects: %d"

BOTO3_IMPORT_FAILED = "boto3 is not installed or cannot be imported and is needed to sync s3:// remote asset urls."

GCP_CLIENT_IMPORT_FAILED = "GCP client is not installed or cannot be imported and is needed to sync gs:// remote asset urls."

UNSUPPORTED_URI_SYNCED_REMOTE_ASSET = "Unsupported URI '%s' for synced remote asset."

MODEL_REMOTE_ASSET_DOWNLOAD_UNSUPPORTED_URI = "Unsupported remote asset's link: %r for automatic asset download. Use model.get_assets() to download asset manually."

UNEXPECTED_STREAMING_ERROR = "Unexpected streaming error detected. Reason: %s."

UNEXPECTED_ERROR_WHEN_LOGGING_FIGURE = (
    "Unexpected error detected while logging a figure. Check logs for details."
)

UNEXPECTED_LOGGING_ERROR = (
    "Unexpected error detected while logging %s data. Check logs for details."
)

FAILED_TO_FLUSH_PARAMETERS_BATCH = "Failed to flush parameters batch."

FAILED_TO_FLUSH_METRICS_BATCH = "Failed to flush metrics batch."

FAILED_TO_FLUSH_STDOUT_BATCH = "Failed to flush stdout/stderr batch."

FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH = (
    "Failed to flush artifact remote assets batch."
)

FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH = (
    "Failed to add message to the parameters batch: %r"
)

FAILED_TO_SEND_OS_PACKAGES_MESSAGE_ERROR = "Error sending os_packages message"

FAILED_TO_SEND_MODEL_GRAPH_MESSAGE_ERROR = "Error sending model_graph message"

FAILED_TO_SEND_SYSTEM_DETAILS_MESSAGE_ERROR = "Error sending system details message"

FAILED_TO_SEND_LOG_OTHER_MESSAGE_ERROR = "Error sending log other message"

FAILED_TO_SEND_CLOUD_DETAILS_MESSAGE_ERROR = "Error sending cloud details message"

FAILED_TO_SEND_FILE_NAME_MESSAGE_ERROR = "Error sending file name message"

FAILED_TO_SEND_HTML_MESSAGE_ERROR = "Error sending html message"

FAILED_TO_SEND_INSTALLED_PACKAGES_MESSAGE_ERROR = (
    "Error sending installed packages message"
)

FAILED_TO_SEND_HTML_OVERRIDE_MESSAGE_ERROR = "Error sending html override message"

FAILED_TO_SEND_GPU_STATIC_MESSAGE_ERROR = "Error sending gpu static info message"

FAILED_TO_SEND_GIT_METADATA_MESSAGE_ERROR = "Error sending git metadata message"

FAILED_TO_SEND_SYSTEM_INFO_MESSAGE_ERROR = "Error sending system_info message"

FAILED_TO_SEND_LOG_DEPENDENCY_MESSAGE_ERROR = "Error sending log dependency message"

FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR = "Error sending log remote model message"

STREAMER_FLUSH_DIDNT_COMPLETE_SUCCESSFULLY_INFO = "Experiment flushing did not complete successfully - some data was not uploaded to Comet:\n\t messages queue flushed [%r]\n\t batches flushed [%r]\n\t file uploads flushed [%r]"

STREAMER_FAILED_TO_REMOVE_TMP_FILE_COPY_WARNING = (
    "Failed to remove temporary copy of the uploaded file: %r"
)

STREAMER_PROGRESS_MESSAGE_INFO = "Uploading %d metrics, params and output messages"

STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION = "Unknown message type %r"

UNEXPECTED_CUSTOMER_ERROR = "Unexpected error detected in the user script: %r"

UNEXPECTED_OFFLINE_STREAMER_ERROR = (
    "Unknown offline streaming error: %r. Check logs for details."
)

UNEXPECTED_OFFLINE_PROCESS_MESSAGE_ERROR = "Unexpected error detected while processing messages by offline experiment. Check logs for details."

FAILED_TO_REGISTER_MODEL = "Failed to register model."

FALLBACK_STREAMER_ONLINE_FAILED_ARCHIVE_UPLOAD_MESSAGE = "Could not send live data to Comet during experiment runtime. An offline experiment will be available for upload:\n    comet upload --force-reupload %s"

FALLBACK_STREAMER_ARCHIVE_UPLOAD_MESSAGE_KEEP_ENABLED = "Live data was sent to Comet during experiment runtime.\nThe backup archive with experiment's data is available for upload:\n    comet upload --force-reupload %s"

FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE = "Failed to create offline archive while try to fallback to offline experiment. Check logs for details."

FALLBACK_STREAMER_FAILED_NO_CONNECTION_NO_OFFLINE = "Experiment failed to send some/all data to Comet. Failed to write some/all data to the file system. Check logs for details."

FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_STREAMER_WARNING = (
    "Failed to create fallback offline streamer"
)

FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE_ERROR = "Can not create offline archive. Offline streamer disabled or failed. Check logs for details."

FALLBACK_STREAMER_FAILED_TO_WRITE_OFFLINE_DATA_ERROR = "Failed to write all experiment's data during fallback to offline mode, some data may be missing."

REGISTER_FAILED_DUE_TO_UPLOADS_FAILED = (
    "Registering %s failed because all corresponding model uploads have failed"
)

PYTORCH_INTEGRATION_LOG_MODEL_ANNOUNCEMENT = """---------------------------------------------------------------------------------------
Log your models to better track, deploy, share, and reproduce your work using: 'comet_ml.integration.pytorch.log_model'.
Learn more at: https://comet.com/docs/v2/pytorch_log_model
---------------------------------------------------------------------------------------------------
Hide this message by setting environment variable "COMET_DISABLE_ANNOUNCEMENT=1"
"""

NEW_FEATURE_PANEL_TITLE = "[dark_cyan bold]New Comet feature!"

PYTORCH_INTEGRATION_LOG_MODEL_RICH_ANNOUNCEMENT = """Log your models to better track, deploy, share, and reproduce your work using: 'comet_ml.integration.pytorch.log_model'.
Learn more at: https://comet.com/docs/v2/pytorch_log_model

Hide this message by setting environment variable "COMET_DISABLE_ANNOUNCEMENT=1" """

HEARTBEAT_PROCESSING_ERROR = (
    "Due to connectivity issues, there's an error in processing the heartbeat. "
    "The experiment's status updates might be inaccurate until the connection issues are resolved."
)

UNSUPPORTED_SECRET_MANAGER = "Current SDK version does not support retrieving encrypted API Key from: %s Secret Manager. Please update your SDK Version to %s. Your experiment will not be logged"

FAILED_TO_GET_API_KEY_FROM_SECRET_MANAGER = "We could not retrieve your API Key from Secret Manager. Your experiment will not be logged. See the error below:"

BAD_ENCODED_SECRET_API_KEY = "The encrypted API Key is invalid. Your experiment will not be logged. See the error below:"

OPTIMIZER_FAILED_TO_RETRIVE_DATA = "Optimizer failed to retrieve data from the optimizer service, for more information look at: https://comet.com/docs/v2/api-and-sdk/python-sdk/warnings-errors/"

SKLEARN_INTEGRATION_SKLEARN_VERSION_MISMATCH_WARNING = "Scikit-learn packages mismatch during model loading. Model was uploaded with version %s. Installed version: %s"

SKLEARN_INTEGRATION_PIPELINE_ERROR = (
    "Couldn't extract parameters from Pipeline step {}: {}. This step will be skipped."
)

DIRECT_S3_UPLOAD_ENABLED = "Direct S3 upload enabled."
DIRECT_S3_UPLOAD_DISABLED = "Direct S3 upload disabled."

AWS_SECRET_MANAGER_PROFILE_NAME_WARNING = "If you are using multiple AWS profiles when retrieving the Comet API Key, you will need to call get_api_key_from_secret_manager every time with the appropriate profile name"

SAGEMAKER_FAILED_TO_IMPORT_METRICS = (
    "Failed to import Sagemaker metrics, see below for more details"
)

SAGEMAKER_FAILED_TO_IMPORT_TAGS = (
    "Failed to import Sagemaker tags, see below for mode details"
)

SAGEMAKER_FAILED_TO_IMPORT_METADATA_FIELDS = (
    "Failed to import Sagemaker metadata fields, see below for more details"
)

SAGEMAKER_FAILED_TO_IMPORT_HYPERPARAMETERS = (
    "Failed to import Sagemaker HyperParameters, see below for mode details"
)

INSECURE_KEY_SHARING_VERTEX = """
Using VertexPipelineLogger(share_api_key_to_workers=True) is insecure.
Please refer to https://www.comet.com/docs/v2/guides/tracking-ml-training/distributed-training/ for more secure alternatives.
"""

EXTENSION_NOT_SUPPORTED = "%s is not supported, supported extensions are: %s"

EXTENSION_NOT_FOUND = "No extension was found for the provided input"

DEPRECATED_COPY_TO_TEMP_ARGUMENT = "copy_to_tmp argument from experiment.%s method is DEPRECATED. \n It doesn't have influence on method's behaviour."

ENV_DETAILS_MISMATCHED_WARNING = "The value for the parameter %r was %s but 'log_env_details' is set to %s. The parameter for 'log_env_details' was ignored."

OFFLINE_DISPLAY_METHOD_ERROR = (
    "The display method is not available for offline Experiments."
)

OFFLINE_DISPLAY_PROJECT_METHOD_ERROR = (
    "The display_project method is not available for offline Experiments."
)

OFFLINE_SEND_NOTIFICATION_METHOD_ERROR = (
    "Notifications are only supported for Online Experiments."
)

OFFLINE_LOG_EMBEDDING_METHOD_ERROR = (
    "Logging embedding is only supported for Online Experiment."
)

OFFLINE_LOG_ARTIFACT_METHOD_ERROR = (
    "Logging an artifact is only supported for Online Experiment."
)

OFFLINE_GET_ARTIFACT_METHOD_ERROR = (
    "Getting an artifact is only supported for Online Experiment."
)

CREATE_SYMLINK_ONLINE_ONLY_ERROR = (
    "Create symlink is only supported for online experiment"
)

REGISTER_CALLBACK_ONLINE_ONLY_ERROR = (
    "Register callback is only supported for online experiment."
)

CONFIG_USING_ANONYMOUS_MODE = "Comet is now logging experiments in anonymous mode, they will be publicly accessing by using the experiment url"

CONFIG_CLAIMED_ANONYMOUS_USER = "Could not start anonymous mode. If you would like to log data to your Comet account, call `comet_ml.login()`. If you would like to log data to an anonymous account, delete the Comet config file located at %s and call `comet_ml.init(anonymous=True)."

CONFIG_ANONYMOUS_UNALLOWED_ON_PREM = "Anonymous mode is only available on Comet Cloud, please create an account on your Comet instance and call: comet_ml.init(api_key='...')"

PYNVML_FAILED_TO_GET_GPU_DATA = "Failed to get GPU data via pynvml API"

DEPRECATED_EXPAND_MODEL_DOWNLOAD = "expand argument from Model.download() is DEPRECATED.\n It doesn't have influence on method's behaviour."

RESILIENCE_NOT_INITIALIZED_ERROR = "Due to an internal error, some network resiliency features were disabled which could lead to data loss. Contact us at support@comet.com. Error details: %r"

PARSE_API_KEY_EMPTY_KEY = "Can not parse empty Comet API key"

PARSE_API_KEY_EMPTY_EXPECTED_ATTRIBUTES = (
    "Expected attributes not found in the Comet API key: %r"
)

PARSE_API_KEY_TOO_MANY_PARTS = "Too many parts (%d) found in the Comet API key: %r"

BASE_URL_MISMATCH_CONFIG_API_KEY = "Comet URL conflict detected between config (%r) and API Key (%r). SDK will use config URL. Resolve by either removing config URL or set it to the same value."

DEPRECATED_WORKSPACE_MODEL_REGISTRY_ARGUMENT = (
    "The workspace parameter is deprecated and no longer supported. "
    "To register the model to a new workspace, first log the model to the "
    "model registry and then use the move option in the Comet UI."
)

API_EXPERIMENT_DELETE_TAGS_UNSUPPORTED_BACKEND_VERSION_ERROR = "This feature requires Comet backend version %r or newer. Please contact your Comet Admin."
API_EXPERIMENT_DELETE_PARAMETERS_UNSUPPORTED_BACKEND_VERSION_ERROR = (
    API_EXPERIMENT_DELETE_TAGS_UNSUPPORTED_BACKEND_VERSION_ERROR
)

API_EXPERIMENT_MISSING_API_EXCEPTION = "need APIExperiment(api=API) for this usage"

API_EXPERIMENT_WORKSPACE_AND_PROJECT_MISSING_EXCEPTION = (
    "workspace and project_name must be provided for new APIExperiment()"
)

API_EXPERIMENT_GET_ENV_DETAILS_DEPRECATED_WARNING = "APIExperiment.get_environment_details() has been deprecated; please use APIExperiment.get_os_packages() instead."

API_EXPERIMENT_MISSING_TF_FOLDER_EXCEPTION = (
    "No Tensorflow folder were logged in Experiment %r"
)

API_EXPERIMENT_NOT_FOUND_MSG = (
    "Could not find an experiment with the specified parameters"
)

API_EXPERIMENT_TF_DOWNLOAD_FILE_ALREADY_EXISTS_WARNING = (
    "Could not download %r as a file already exists at that path"
)

API_EXPERIMENT_TF_DOWNLOAD_VIEW_IN_TB_INFO = (
    "You can now view with `tensorboard serve --logdir=%r`"
)

API_EXPERIMENT_DOWNLOAD_MODEL_START_INFO = "Downloading experiment model %r ..."

API_EXPERIMENT_DOWNLOAD_MODEL_UNZIP_INFO = "Unzipping model to %r ..."

API_EXPERIMENT_DOWNLOAD_MODEL_COPY_INFO = "Copying model to %r ..."

API_EXPERIMENT_DOWNLOAD_MODEL_DONE_INFO = "done!"

API_EXPERIMENT_DOWNLOAD_MODEL_FAILED_INFO = "Unable to download model"

API_EXPERIMENT_REGISTER_MODEL_INVALID_VERSION_EXCEPTION = "APIExperiment.register_model() requires a proper semantic version string, such as '1.0.0'"

API_EXPERIMENT_EXTRA_KWARGS_IGNORED_INFO = "Extra kwargs are ignored"

API_EXPERIMENT_EXTRA_WORKSPACE_KWARGS_IGNORED_INFO = "Extra workspace kwarg is ignored"

API_EXPERIMENT_EXTRA_PROJECT_NAME_KWARGS_IGNORED_INFO = (
    "Extra project_name kwarg is ignored"
)

API_EXPERIMENT_SET_CODE_FILE_AND_CODE_NOT_ALLOWED_WARNING = (
    "can't set code from string and filename; ignoring filename"
)

API_EXPERIMENT_SET_CODE_FROM_FILENAME_FAILED_WARNING = (
    "unable to set code from filename; ignoring"
)

API_EXPERIMENT_GPU_STATIC_LIST_EXPECTED_EXCEPTION = "gpu_static_info must be a list"

API_EXPERIMENT_GPU_STATIC_LIST_OF_DICTS_EXPECTED_EXCEPTION = "gpu_static_info must be a list of dicts: {gpuIndex, name, uuid, totalMemory, powerLimit}"

API_EXPERIMENT_LOG_GPU_METRICS_LIST_OF_DICTS_EXPECTED_EXCEPTION = (
    "log_gpu_metrics requires a gpu_metrics list of dicts"
)

API_EXPERIMENT_LOG_GPU_METRICS_MISSING_PARAMETERS_EXCEPTION = 'log_gpu_metrics requires a gpu_metrics list of dicts containing "gpuId", "freeMemory", "usedMemory", "gpuUtilization", and "totalMemory"'

API_EXPERIMENT_LOG_CPU_METRICS_LIST_EXPECTED_EXCEPTION = (
    "log_cpu_metrics requires a list"
)

API_EXPERIMENT_LOG_CPU_METRICS_LIST_OF_NUMBERS_EXPECTED_EXCEPTION = (
    "log_cpu_metrics requires a list of numbers"
)

API_EXPERIMENT_LOG_TABLE_HEADERS_IGNORED_INFO = (
    "Headers are ignored when calling log_table with a filename"
)

API_EXPERIMENT_LOG_TABLE_MISSING_TABULAR_DATA_EXCEPTION = (
    "Either tabular_data or a valid file-path are required when calling log_table"
)

API_EXPERIMENT_LOG_TABLE_WRONG_FILENAME_EXCEPTION = (
    "filename argument of log_table must be a string"
)

API_EXPERIMENT_WRONG_BACKEND_VERSION_FOR_METHOD_EXCEPTION = (
    "Backend is too old. %r requires at least %r version of backend"
)

API_EXPERIMENT_GET_ARTIFACT_LINEAGE_WRONG_DIRECTION = (
    "Unsupported direction value: %r. Please use one of the following: %r."
)

API_EXPERIMENT_GET_ASSET_BY_NAME_DEPRECATION_WARNING = "Deprecated: please use get_assets_by_name() to get list of experiment assets with specific asset name."

API_UPDATE_CACHE_DEPRECATED_WARNING = (
    "API.update_cache() has been deprecated; please use API.clear_cache() instead."
)

API_GET_SLASH_WORKSPACE_AND_PROJECT_EXCEPTION = (
    "Can't use slash format in workspace name and provide project name"
)

API_GET_SLASH_PROJECT_AND_KEY_EXCEPTION = (
    "Can't use slash format in project name and provide experiment key/name"
)

API_QUERY_INVALID_QUERY_EXPRESSION_EXCEPTION = "invalid query expression: you must use an operator, such as '==' or QueryVariable.contains('substring')"

API_QUERY_MISSING_QUERY_EXPRESSION_EXCEPTION = (
    "invalid query expression: do not use 'and', 'or', 'not', 'is', or 'in'"
)

API_QUERY_ERROR_INFO = "%s; ignoring query, returning no matches"

API_EXPERIMENT_BY_ID_DEPRECATED_WARNING = "API.get_experiment_by_id() has been deprecated; please use API.get_experiment_by_key() instead."

API_MISSING_PROJECT_IN_PATTERN_EXCEPTION = (
    "Must provide project_name when providing pattern"
)

API_INVALID_WORKSPACE_PROJECT_EXCEPTION = "invalid workspace/project: %s/%s"

API_GET_PROJECT_NOTES_UNKNOWN_PROJECT_EXCEPTION = "unknown project %r in workspace %r"

API_GET_METRICS_FOR_CHART_REQUIRES_LIST_EXPERIMENTS_EXCEPTION = (
    "get_metrics_for_chart requires a list of experiments"
)

API_GET_METRICS_FOR_CHART_REQUIRES_LIST_METRICS_EXCEPTION = (
    "get_metrics_for_chart requires a list of metrics"
)

API_GET_METRICS_FOR_CHART_REQUIRES_LIST_PARAM_NAMES_EXCEPTION = (
    "get_metrics_for_chart requires a list of parameter names, or None"
)

API_USE_CACHE_NOT_SUPPORTED_EXCEPTION = "client does not support cache"

API_GET_LATEST_REGISTRY_MODEL_VERSION_DETAILS_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.find_versions(...)"

API_GET_MODEL_REGISTRY_VERSION_ASSETS_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.get_assets(...)"

API_DOWNLOAD_REGISTRY_MODEL_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.download(...)"

API_DOWNLOAD_REGISTRY_MODEL_START_INFO = (
    "Downloading registry model %r, version %r, stage %r from workspace %r..."
)

API_DOWNLOAD_REGISTRY_MODEL_UNZIP_INFO = "Unzipping model to %r ..."

API_DOWNLOAD_REGISTRY_MODEL_COPY_INFO = "Copying model to %r ..."

API_DOWNLOAD_REGISTRY_MODEL_COMPLETED_INFO = "done!"

API_DOWNLOAD_REGISTRY_MODEL_FAILED_INFO = "Unable to download model"

API_UPDATE_REGISTRY_MODEL_VERSION_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.set_status(...) or model.add_tag(...)"

API_ADD_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.set_status(version='<version>', status='<status>')"

API_DELETE_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING = "This method has been deprecated, we recommend using the api.get_model(...) method to get the Model Object and then using model.set_status(version='<version>', status=None)"

API_GET_PANEL_METRICS_NAMES_EXCEPTION = (
    "Please use API.get_panel_metrics_names() from inside a Comet Panel"
)

API_GET_PANEL_EXPERIMENTS_EXCEPTION = (
    "Please use API.get_panel_experiments() from inside a Comet Panel"
)

API_GET_PANEL_EXPERIMENT_KEYS_EXCEPTION = (
    "Please use API.get_panel_experiment_keys() from inside a Comet Panel"
)

API_GET_PANEL_PROJECT_ID_EXCEPTION = (
    "Please use API.get_panel_project_id() from inside a Comet Panel"
)

MAX_NUMBER_OF_EXPERIMENTS_PER_WORKSPACE_REACHED = "Maximum number of experiments reached for project '%s' of workspace '%s'. Archive experiments or use a different project. Data will not be logged."

BATCH_UTILS_CALLBACK_MISSING_EXCEPTION = "Callback is None"

CLI_ARGS_PARSE_FAILED_EXCEPTION = "Failed to parse argv arguments"

CONFIG_GET_DISPLAY_SUMMARY_LEVEL_INVALID_WARNING = (
    "invalid display_summary_level %r; ignoring"
)

CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING = "Unable to make a backup of config file"

CONFIG_API_KEY_SAVED_INFO = (
    "Valid Comet API Key saved in %s (set COMET_CONFIG to change where it is saved)."
)

CONFIG_API_KEY_SAVE_FAILED_WARNING = "Unable to save Comet API key to disk"

CONFIG_INIT_ONPREM_START_INFO = (
    "For help on these settings, please see: https://www.comet.com/docs/onprem/"
)

CONFIG_INIT_ONPREM_USER_INPUT_INFO = "Please enter your onprem COMET_URL_OVERRIDE: "

CONFIG_INIT_ONPREM_FAILED_EXCEPTION = "invalid url or api key; use comet_ml.init_onprem(force=True) or `comet init --onprem --force` to reset"

CONFIG_COMET_INI_DEPRECATED_WARNING = "COMET_INI is deprecated; use COMET_CONFIG"

CONFIG_DEPRECATED_USER_CONFIG_KEY_WARNING = (
    "Deprecated config key %r was set, but ignored as new config key %r is set"
)

CONFIG_DEPRECATED_CONFIG_KEY_WARNING = (
    "Deprecated config key %r was set in %r, but ignored as new config key %r is set"
)

CONFIG_KEY_IS_DEPRECATED_WARNING = "Config key %r is deprecated, please use %r instead"

CONFIG_KEY_IS_DEPRECATED_WAS_SET_IN_WARNING = (
    "Config key %r is deprecated (was set in %r), please use %r instead"
)

CONFIG_SETTING_FAILED_WARNING = (
    "Comet configuration option %r failed with value %r, reason: %s."
)

CONFIG_SETTING_INCORRECT_KEY_SUGGESTION_WARNING = (
    "The configuration option %r is not supported by Comet config. "
    "You might want to use one of the following options instead: %r."
)

CONFIG_SETTING_NOT_FOUND_NO_SUGGESTIONS = (
    "The configuration option %r is not supported by Comet config. "
    "Please check Comet documentation or contact your Comet admin."
)

CONFIG_SETTINGS_FILE_EXISTS_ERROR = (
    "Failed to save Comet configuration to the '%s' file. The file already exists. "
    "To overwrite it please set `force=True`."
)

CONFIG_SETTINGS_USER_REFUSED_OVERWRITE_ERROR = (
    "User refused to overwrite config file %r, aborting"
)

CONFIG_SETTINGS_NOT_VALID_CONFIG_KEY_EXCEPTION = "'%s' is not a valid config key"

CONFIG_SETTINGS_SAVE_STARTED = 'Saving config to "%s"...'

CONFIG_SETTINGS_SAVE_COMPLETED = " done!"

CONFUSION_MATRIX_SHOULD_BE_TWO_DIMENSIONAL_EXCEPTION = (
    "matrix should be two dimensional"
)

CONFUSION_MATRIX_TRUE_PREDICTED_OR_MATRIX_EXCEPTION = (
    "you need to give either (y_true and y_predicted) or matrix, NOT both"
)

CONFUSION_MATRIX_TRUE_AND_PREDICTED_REQUIRED_EXCEPTION = (
    "if you give y_true OR y_predicted you must give both"
)

CONFUSION_MATRIX_CONVERT_TO_MATRIX_FAILED_ERROR = (
    "convert_to_matrix failed; confusion matrix not generated"
)

CONFUSION_MATRIX_IMAGE_INDEX_IMAGES_NOT_SET_INFO = (
    "images were not set; ignoring examples"
)

CONFUSION_MATRIX_IMAGE_INDEX_EXPERIMENT_NOT_SET_INFO = (
    "experiment is not set; use experiment.create_confusion_matrix(); ignoring examples"
)

CONFUSION_MATRIX_IMAGE_INDEX_FAILED_TO_GENERATE_IMAGE_INFO = (
    "unable to generate image from images; ignoring example"
)

CONFUSION_MATRIX_IMAGE_HASH_FAILED_INFO = "unable to generate hash from image"

CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_TRUE_WARNING = (
    "Invalid y_true value %r, ignoring it"
)

CONFUSION_MATRIX_COMPUTE_MATRIX_INVALID_Y_PREDICTOR_WARNING = (
    "Invalid y_predictor value %r, ignoring it"
)

CONFUSION_MATRIX_NUM_LABELS_NOT_EQUAL_CATEGORIES_EXCEPTION = (
    "The length of labels does not match number of categories"
)

CONNECTION_MISSING_HEARTBEAT_DURATION_EXCEPTION = "Missing heart-beat duration"

CONNECTION_MISSING_GPU_MONITOR_INTERVAL_EXCEPTION = "Missing gpu-monitor interval"

CONNECTION_MISSING_CPU_MONITOR_INTERVAL_EXCEPTION = "Missing cpu-monitor interval"

CONNECTION_MISSING_PARAMETERS_UPDATE_INTERVAL_EXCEPTION = (
    "Missing parameters update interval"
)

CONNECTION_NAME_TOO_LONG_DEFAULT_ERROR = (
    "Project name it too long, it should should be < 100 characters"
)

CONNECTION_VIEW_ONLY_CREATE_EXPERIMENT_EXCEPTION = (
    "Users with view-only access can not create experiments"
)

CONNECTION_GET_REST_API_CLIENT_NO_API_KEY_EXCEPTION = (
    "get_rest_api_client requires an api_key"
)

CONNECTION_UPDATE_PROJECT_MISSING_REQUIRED_EXCEPTION = (
    "update_project requires workspace and project_name"
)

CONNECTION_UPDATE_PROJECT_BY_ID_MISSING_REQUIRED_EXCEPTION = (
    "update_project_by_id requires project_id"
)

CONNECTION_LOG_EXPERIMENT_SYSTEM_INFO_EXCEPTION = (
    "system_info must be a list of {key:..., value:...} dicts"
)

CONNECTION_LOG_EXPERIMENT_ASSET_NO_NAME_WARNING = (
    "logging file-like asset with no name; using `unnamed`"
)

CONNECTION_SET_EXPERIMENT_STATE_UNSUPPORTED_EXCEPTION = "Unsupported experiment state value: %r. Use one from the: 'running', 'finished', 'crashed'"

CONNECTION_DOWNLOAD_REGISTRY_MODEL_VERSION_OR_STAGE_EXCEPTION = (
    "Please specify version OR stage (not both) to download model"
)

CONNECTION_REGISTER_MODEL_TAGS_IGNORED_WARNING = "register_model tags={} field is ignored due to old backend, use stages field instead"

CONNECTION_REGISTER_MODEL_STATUS_IGNORED_WARNING = (
    "register_model status={} field is ignored due to old backend"
)

CONNECTION_REGISTER_MODEL_STAGES_IGNORED_WARNING = (
    "register_model stages={} field is ignored, use the new tags field instead"
)

CONNECTION_REGISTER_MODEL_SUCCESS_INFO = (
    "Successfully registered %r, version %r in workspace %r"
)

CONNECTION_REGISTER_MODEL_NO_MODEL_EXCEPTION = "There are no models for experiment %r"

CONNECTION_REGISTER_MODEL_INVALID_STAGES_LIST_EXCEPTION = (
    "Invalid stages list: should be a list of strings"
)

CONNECTION_REGISTER_MODEL_DESCRIPTION_WARNING = "The argument 'description' was given, but ignored when adding a new registry model version"

CONNECTION_REGISTER_MODEL_PUBLIC_WARNING = "The argument 'public' was given, but ignored when adding a new registry model version"

CONNECTION_REGISTER_MODEL_INVALID_MODEL_NAME_EXCEPTION = (
    "Invalid experiment model name: %r; should be one of %r"
)

CONNECTION_ONLY_S3_DIRECT_UPLOAD_TYPES_SUPPORTED_EXCEPTION = (
    "Only %r S3 direct uploads supported!"
)

CONNECTION_FAILED_TO_CALL_ON_ASSET_UPLOAD_WARNING = "Failed to call on_asset_upload"

CONNECTION_FAILED_TO_CALL_ON_FAILED_ASSET_UPLOAD_WARNING = (
    "Failed to call on_failed_asset_upload"
)

CONNECTION_S3_DIRECT_FILE_UPLOAD_FAILED_ERROR = (
    "S3 direct file upload failed: %r, file: %s, upload type: %r"
)

CONNECTION_S3_DIRECT_FILE_LIKE_UPLOAD_FAILED_ERROR = (
    "S3 direct file-like upload failed: %r, file name: %s, upload type: %r"
)

CONNECTION_FILE_UPLOAD_FAILED_EXCEPTION = (
    "Uploading file failed with max retries: %d on url %r"
)

CONNECTION_FILE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION = (
    "Uploading file failed (%s) with max retries: %d on url %r: %r"
)

CONNECTION_FILE_UPLOAD_FAILED_ERROR = (
    "File upload failed: %r, file: %s, url: %r, max retries: %d"
)

CONNECTION_FILE_LIKE_UPLOAD_FAILED_EXCEPTION = (
    "Uploading file-like failed with max retries: %d on url %r"
)

CONNECTION_FILE_LIKE_UPLOAD_FAILED_WITH_STATUS_EXCEPTION = (
    "Uploading file-like failed (%s) with max retries: %d on url %r: %r"
)

CONNECTION_FILE_LIKE_UPLOAD_FAILED_ERROR = (
    "File-like upload failed: %r, url: %r, max retries: %d"
)

CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_WITH_STATUS_EXCEPTION = (
    "Remote asset upload failed (%s) on url %r: %r"
)

CONNECTION_UPLOAD_REMOTE_ASSET_FAILED_ERROR = "Remote Asset could not be uploaded: %r"

CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_WITH_STATUS_EXCEPTION = (
    "Asset thumbnail upload failed (%s) on url %r: %r"
)

CONNECTION_UPLOAD_ASSET_THUMBNAIL_FAILED_ERROR = (
    "Asset thumbnail could not be uploaded: %r, url: %r, reason: %r"
)

CONNECTION_UPLOAD_ASSET_ITEM_FAILED_WITH_STATUS_EXCEPTION = (
    "Asset item upload failed (%s) on url %r: %r"
)

CONNECTION_UPLOAD_ASSET_ITEM_FAILED_ERROR = (
    "Asset item could not be uploaded: %r, url: %r, reason: %r"
)

CONSOLE_CONSUMER_THREAD_DID_NOT_FINISH_WARNING = "Consumer thread did not finish"

CONVERT_UTILS_NUMPY_NOT_FOUND_WARNING = (
    "numpy not installed; some functionality will be unavailable"
)

CONVERT_UTILS_WRONG_TYPE_EXCEPTION = "%r is not of type %r"

CONVERT_UTILS_CONVERT_TO_LIST_SHOULD_BE_ONE_DIMENSIONAL_EXCEPTION = (
    "list should be one dimensional"
)

CONVERT_UTILS_NO_NUMPY_ERROR = "The Python library numpy is required for this operation"

CONVERT_UTILS_INVALID_IMAGE_FILE_TYPE_ERROR = "invalid image file_type: %s"

CONVERT_UTILS_INSTALL_PIL_ERROR = (
    "The Python library PIL is required for this operation"
)

CONVERT_UTILS_PIL_CONVERT_ERROR = "Couldn't convert input:\n{}\ninto an image with shape {} using PIL.Image.fromarray: {}"

CONVERT_UTILS_INSTALL_PIL_NUMPY_MATPLOTLIB_ERROR = "The Python libraries PIL, numpy, and matplotlib are required for converting a numpy array into an image"

CONVERT_UTILS_INSTALL_PIL_NUMPY_ERROR = (
    "The Python libraries PIL and numpy are required for this operation"
)

CONVERT_UTILS_INSTALL_NUMPY_SCIPY_ERROR = (
    "The Python libraries numpy, and scipy are required for this operation"
)

CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_EXCEPTION = (
    "invalid image_shape: %s; should be 2D or 3D"
)

CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_SHAPE_INFO = (
    "WARNING: invalid image_shape; ignored"
)

CONVERT_UTILS_ARRAY_TO_IMAGE_INVALID_IMAGE_COLORMAP_INFO = (
    "WARNING: invalid image_colormap; ignored"
)

CONVERT_UTILS_IMAGE_BACKGROUND_COLOR_NO_ALPHA_CHANNEL_EXCEPTION = (
    "image must have an alpha channel in order to set a background color"
)

CONVERT_UTILS_DATA_TO_FP_ERROR = "Failed to log asset data as JSON"

CONVERT_UTILS_FAST_FLATTEN_NO_NUMPY_INFO = "numpy not installed; using a slower flatten"

CONVERT_UTILS_CONVERT_TO_STRING_WARNING = (
    "Converting %s '%s' into a string using str(), resulting string might be invalid"
)

CONVERT_UTILS_CONVERT_TO_STRING_TRUNCATED_WARNING = "String value length exceeds %d characters and will be truncated. Provided value: '%s'"

CONVERT_UTILS_CONVERT_MODEL_TO_STRING_TF_TO_JSON_FAILED_WARNING = (
    "Failed to convert Tensorflow graph to JSON"
)

CONVERT_UTILS_CONVERT_MODEL_TO_STRING_FAILED_RETURN = (
    "Unable to convert model to a string"
)

CONVERT_UTILS_PREPARE_DATAFRAME_WARNING = (
    "invalid asset_format %r; should be 'json', 'csv', 'md', or 'html'; ignoring"
)

CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_STR_VALUE_WARNING = (
    "You are trying to log string value as a metric. This is not recommended."
)

CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_VALUE_DICTIONARY_VALUE_WARNING = (
    "You are trying to log dictionary value as a metric. This is not recommended."
)

CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_VALUE = "You are trying to log list with wrong data type as metric %r. Only lists of int and float data types are supported."

CONVERT_UTILS_CONVERT_USER_INPUT_TO_METRIC_WRONG_LIST_SIZE = "You are trying to log list with size not equal to 1 as metric %r. Only lists with size equal to 1 are supported."

EXPERIMENT_INIT_DISPLAY_SUMMARY_WARNING = "display_summary is set in Comet config but has been deprecated; use display_summary_level instead"

EXPERIMENT_COMET_EXPERIMENT_KEY_INVALID_EXCEPTION = "COMET_EXPERIMENT_KEY is invalid: '%s' must be alphanumeric and between 32 and 50 characters"

EXPERIMENT_ON_END_CLEAN_FAILED_WARNING = (
    "Failed to properly clean the experiment, reason: %r"
)

EXPERIMENT_ON_END_STREAMER_FLUSH_FAILED_ERROR = (
    "Failed to flush experiment streamer: %r, reason: %r"
)

EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_WITH_EXCEPTION_ERROR = (
    "Run will not be logged: %s"
)

EXPERIMENT_START_RUN_WILL_NOT_BE_LOGGED_ERROR = "Run will not be logged"

EXPERIMENT_START_FAILED_SETUP_STD_LOGGER_ERROR = "Failed to setup the std logger"

EXPERIMENT_START_FAILED_SET_RUN_FILE_NAME_ERROR = "Failed to set run file name"

EXPERIMENT_START_FAILED_SET_SOURCE_CODE_ERROR = "Failed to set run source code"

EXPERIMENT_START_FAILED_LOG_COLAB_NOTEBOOK_URL_ERROR = (
    "Failed to log Google Colab notebook URL, reason: %s"
)

EXPERIMENT_START_FAILED_SET_PIP_PACKAGES_ERROR = "Failed to set run pip packages"

EXPERIMENT_START_FAILED_SET_OS_PACKAGES_ERROR = "Failed to set run os packages"

EXPERIMENT_START_FAILED_LOG_CLOUD_DETAILS_ERROR = "Failed to log cloud details"

EXPERIMENT_START_FAILED_LOG_ENV_DETAILS_ERROR = "Failed to log environment details"

EXPERIMENT_START_FAILED_SET_RUN_CMD_ARGS_ERROR = "Failed to set run cmd args"

EXPERIMENT_START_FAILED_CREATE_GPU_LOGER_ERROR = (
    "Failed to create the GPU metrics data logger"
)

EXPERIMENT_START_FAILED_CREATE_CPU_LOGER_ERROR = (
    "Failed to create the CPU metrics data logger"
)

EXPERIMENT_START_FAILED_CREATE_NETWORK_LOGER_ERROR = (
    "Failed to create the network metrics data logger"
)

EXPERIMENT_START_FAILED_CREATE_DISK_LOGER_ERROR = (
    "Failed to create the disk utilization metrics data logger"
)

EXPERIMENT_LOG_OTHER_EXCEPTION_ERROR = (
    "Unknown exception happened in Experiment.log_other; ignoring"
)

EXPERIMENT_LOG_OTHER_KEY_VALUE_TRUNCATED = (
    "The key value length exceeds %d characters and will be truncated. Original key: %s"
)

EXPERIMENT_LOG_OTHER_IGNORE_AUTOMATIC_INFO = (
    "Ignoring automatic log_other(%r) because '%s:%s' is in COMET_LOGGING_OTHERS_IGNORE"
)

EXPERIMENT_LOG_OTHERS_DICT_OR_KEY_VALUE_ERROR = (
    "log_other requires a dict or key/value but not both"
)

EXPERIMENT_LOG_LLM_PROJECT_ERROR_MSG = 'You are trying to log an experiment to an LLM project: {}. You should log to an "Experiment Tracking" project instead.'

EXPERIMENT_LOG_METRIC_EXCEPTION_ERROR = (
    "Unknown exception happened in Experiment.log_metric; ignoring"
)

EXPERIMENT_LOG_METRIC_NAME_TRUNCATED = "The metric name length exceeds %d characters and will be truncated. Provided name: %s"

EXPERIMENT_LOG_PARAMETER_EXCEPTION_ERROR = (
    "Unknown exception happened in Experiment.log_parameter; ignoring"
)

EXPERIMENT_LOG_PARAMETER_IGNORE_AUTOMATIC_INFO = "Ignoring automatic log_parameter(%r) because '%s:%s' is in COMET_LOGGING_PARAMETERS_IGNORE"

EXPERIMENT_LOG_PARAMETER_EMPTY_NAME_ERROR = "You are trying to log parameter with empty or None name. Parameter will not be logged. Please provide parameter name."

EXPERIMENT_LOG_PARAMETER_NAME_TRUNCATED = "The parameter name length exceeds %d characters and will be truncated. Provided name: %s"

EXPERIMENT_LOG_MODEL_NO_SUCH_FILE_OR_DIR_ERROR = (
    "Error logging the model %r, no such file or directory: %r"
)

EXPERIMENT_REGISTER_MODEL_NO_LOG_MODEL_CALL_EXCEPTION = (
    "There must be log_model call for {} before register attempt"
)

EXPERIMENT_REGISTER_MODEL_TIMEOUT_WARNING = (
    "The register_model synchronized call timeout of %s seconds was exceeded."
)

EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_STRING_EXCEPTION = "filename should be a string"

EXPERIMENT_LOG_NOTEBOOK_FILENAME_NOT_IPYNB_EXCEPTION = (
    "filename should be a notebook file ending in '.ipynb'"
)

EXPERIMENT_LOG_CURVE_VALIDATION_ERROR = (
    "Invalid 'curve' data, ignoring. Validation failures: %r"
)

EXPERIMENT_LOG_ASSET_DATA_DEPRECATED_FILE_NAME_WARNING = "log_asset_data(..., file_name=...) is deprecated; use log_asset_data(..., name=...)"

EXPERIMENT_LOG_ASSET_DATA_STEP_MANDATORY_EXCEPTION = (
    "Step is mandatory.\n It can either be passed on "
    "most log methods, set manually with set_step method or "
    "set automatically through auto-logging integrations"
)

EXPERIMENT_LOG_ASSET_FOLDER_LOG_FILE_NAME_WARNING = (
    "The default value for the log_file_name parameter will change from False to True in a future version. "
    "Explicitly pass log_file_name=True or log_file_name=False to disable this warning."
)

EXPERIMENT_LOG_ASSET_FILE_DATA_NONE_EXCEPTION = "file_data cannot be None"

EXPERIMENT_LOG_ASSET_UNSUPPORTED_UPLOAD_TYPE_EXCEPTION = "Unsupported upload input %r"

EXPERIMENT_LOG_PREPROCESSED_ASSET_ON_FAILED_ASSET_UPLOAD_CALL_FAILED_WARNING = (
    "Failed to call on_failed_asset_upload"
)

EXPERIMENT_LOG_PREPROCESSED_ASSET_STEP_MANDATORY_ERROR = (
    "Step is mandatory.\n It can either be passed on "
    "most log methods, set manually with set_step method or "
    "set automatically through auto-logging integrations"
)

EXPERIMENT_LOG_PREPROCESSED_ASSET_NAME_TRUNCATED_WARNING = (
    "Asset name is too long. It was truncated to be %d characters long."
)

EXPERIMENT_LOG_PREPROCESSED_REMOTE_ASSET_NAME_TRUNCATED_WARNING = (
    "Remote asset name is too long. It was truncated to be %d characters long."
)

EXPERIMENT_LOG_AUDIO_NO_DATA_EXCEPTION = "audio_data cannot be None"

EXPERIMENT_LOG_AUDIO_NAME_TRUNCATED_WARNING = (
    "Audio file name is too long. It was truncated to be %d characters long."
)

EXPERIMENT_LOG_AUDIO_UNSUPPORTED_NORMALIZATION_ERROR = "Unsupported normalize_data parameter value: %r. Please use one of the following: %r"

EXPERIMENT_LOG_CONFUSION_MATRIX_EMPTY_MATRIX_ERROR = (
    "Attempt to log empty confusion matrix; ignoring"
)

EXPERIMENT_LOG_CONFUSION_MATRIX_CREATE_FAILED_ERROR = (
    "Error creating confusion matrix: %s; ignoring"
)

EXPERIMENT_LOG_CONFUSION_MATRIX_GENERAL_ERROR = (
    "Error logging confusion matrix; ignoring"
)

EXPERIMENT_LOG_IMAGE_GPU_TENSOR_NOT_SUPPORTED = "GPU tensor logging is not supported. Move tensor to CPU with Tensor.cpu() before calling log_image()."

EXPERIMENT_LOG_IMAGE_NO_DATA_EXCEPTION = "image_data cannot be None"

EXPERIMENT_LOG_IMAGE_NAME_TRUNCATED_WARNING = (
    "Image name is too long. It was truncated to be %d characters long."
)

EXPERIMENT_LOG_IMAGE_ANNOTATION_VALIDATION_ERROR = (
    "Failed to validate image annotations:\n%r"
)

EXPERIMENT_LOG_IMAGE_MATPLOTLIB_FIGURE_WARNING = "Please use the log_figure() method for logging a matplotlib.Figure. The log_image() method is not suitable for this purpose and will not record the image data."

EXPERIMENT_LOG_VIDEO_NO_DATA_EXCEPTION = "video_data cannot be None"

EXPERIMENT_LOG_VIDEO_NAME_TRUNCATED_WARNING = (
    "Video file name is too long. It was truncated to be %d characters long."
)

EXPERIMENT_LOG_DATASET_INFO_NAME_VERSION_PATH_EMPTY_WARNING = (
    "log_dataset_info: name, version, and path can't all be None"
)

EXPERIMENT_LOG_DATASET_HASH_WARNING = "Failed to create dataset hash"

EXPERIMENT_LOG_TABLE_FILENAME_NOT_STRING_EXCEPTION = (
    "filename argument of log_table must be a string"
)

EXPERIMENT_LOG_TAG_VALIDATION_ERROR = "Failed to validate tag value(s)"

EXPERIMENT_CREATE_EMBEDDING_IMAGE_FAILED_CREATING_IMAGE_ERROR = (
    "create_embedding_image: error creating images; continuing without images"
)

EXPERIMENT_CREATE_EMBEDDING_EMPTY_VECTOR_ERROR = (
    "create_embedding: empty vector; ignoring"
)

EXPERIMENT_CREATE_EMBEDDING_VECTOR_SHAPE_ERROR = (
    "create_embedding: error getting vector shape; ignoring"
)

EXPERIMENT_CREATE_EMBEDDING_EMPTY_METADATA_ERROR = (
    "create_embedding: empty metadata; ignoring"
)

EXPERIMENT_CREATE_EMBEDDING_NO_IMAGE_SIZE_ERROR = (
    "create_embedding: no image_size given; ignoring images"
)

EXPERIMENT_LOG_EMBEDDING_NOT_IMPLEMENTED_EXCEPTION = (
    "Logging embedding is only supported for Online Experiment at the moment"
)

EXPERIMENT_LOG_DATAFRAME_PROFILE_DATAFRAME_PROFILE_ERROR = (
    "Unable to profile dataframe; ignoring"
)

EXPERIMENT_LOG_DATAFRAME_PROFILE_MISSING_PANDAS_LOG_DATAFRAME = (
    "pandas is required to log a dataframe; ignoring"
)

EXPERIMENT_LOG_DATAFRAME_PROFILE_NOT_PANDAS_DATAFRAME = (
    "dataframe must be a pandas DataFrame; ignoring"
)

PANDAS_DATAFRAME_IS_REQUIRED = "The Python library Pandas is required for this method. You can install it with `pip install pandas`."

EXPERIMENT_LOG_CODE_CALLER_NOT_FOUND = "unable to find caller source code; ignoring"

EXPERIMENT_LOG_CODE_CALLER_JUPYTER = (
    "unable to find caller source code in a jupyter notebook; ignoring"
)

EXPERIMENT_LOG_CODE_FILE_NAME_FOLDER_MUTUALLY_EXCLUSIVE = "Experiment.log_code either only one of code, file_name or folder, not several; ignoring"

EXPERIMENT_LOG_CODE_NOT_A_FILE_WARNING = "filename %r is not a file; ignoring"

EXPERIMENT_LOG_CODE_MISSING_CODE_NAME = (
    "code_name is mandatory when passing in `code`; ignoring"
)

EXPERIMENT_LOG_ARTIFACT_NOT_SUPPORTED_EXCEPTION = (
    "Logging an artifact is only supported for Online Experiment at the moment"
)

EXPERIMENT_LOG_NON_TRAINABLE_LAYER_INFO = (
    "Skipping weight and bias logging for non-trainable layers: {}"
)

EXPERIMENT_LOG_HOOK_REGISTRATION_WARNING = "Unable to log weights and biases for layers due to errors, check debug logs for details: {}"

EXPERIMENT_PYTORCH_WATCH_DISTRIBUTED_TENSOR_NOT_SUPPORTED = "Currently, comet_ml.integration.pytorch.watch() can't be used for logging distributed tensors. Ignoring."

EXPERIMENT_GET_ARTIFACT_NOT_SUPPORTED_EXCEPTION = (
    "Getting an artifact is only supported for Online Experiment at the moment"
)

EXPERIMENT_GET_HF_DATASET_NOT_SUPPORTED_EXCEPTION = "Getting a HuggingFace dataset is only supported for Online Experiment at the moment"

EXPERIMENT_SET_CODE_IGNORE_FILENAME_WARNING = (
    "can't set code from string and filename; ignoring filename"
)

EXPERIMENT_SET_CODE_ALREADY_CALLED_WARNING = (
    "Set code ignored; already called. Call with overwrite=True to replace code"
)

EXPERIMENT_SET_MODEL_GRAPH_ALREADY_CALLED_WARNING = "Set model graph ignored; already called. Call with overwrite=True to replace graph definition"

EXPERIMENT_SET_OS_PACKAGES_FAILED_WARNING = (
    "Failing to collect the installed os packages"
)

EXPERIMENT_SET_PIP_PACKAGES_FAILED_WARNING = (
    "Failing to collect the installed pip packages"
)

EXPERIMENT_GET_KERAS_CALLBACK_DEPRECATED_WARNING = "Experiment.get_keras_callback() is deprecated; use Experiment.get_callback('keras')"

EXPERIMENT_LOG_OTHERS_FROM_ENVIRONMENT_FAILED_ERROR = (
    "Failed to log comet_log_other_* environment variables"
)

ONLINE_INIT_API_KEY_MISSING_EXCEPTION = (
    "Comet.ml requires an API key. Please provide as the "
    "first argument to Experiment(api_key) or as an environment"
    " variable named COMET_API_KEY "
)

ONLINE_ON_END_FAILED_TO_LOG_RUN_WARNING = "Failed to log run in comet.com"

ONLINE_ON_END_STREAMER_UPLOAD_FAILED_EXCEPTION = "Failed to successfully send all of the experiment's artifact or model assets. See logs for details."

ONLINE_GET_EXPERIMENT_URL_TAB_INVALID_INFO = "tab must be one of: %r"

LOG_EMBEDDING_EXPERIMENTAL_WARNING = (
    "Logging embedding is experimental - the API and logged data are subject to change"
)

ONLINE_LOG_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION = (
    "Experiment %r is disabled, cannot log artifact"
)

ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION = (
    "Experiment %r is not alive, cannot log artifact"
)

ONLINE_LOG_ARTIFACT_EXPERIMENT_IS_NOT_ARTIFACT_EXCEPTION = (
    "%r is not an Artifact and cannot be logged"
)

ONLINE_GET_ARTIFACT_EXPERIMENT_DISABLED_EXCEPTION = (
    "Experiment %r is disabled, cannot get artifact"
)

ONLINE_GET_ARTIFACT_EXPERIMENT_IS_NOT_ALIVE_EXCEPTION = (
    "Experiment %r is not alive, cannot get artifact"
)

ONLINE_EXISTING_INIT_INVALID_KEY_EXCEPTION = "Invalid experiment key: %s"

FILE_UPLOADER_NUMPY_MISSING_WARNING = (
    "numpy not installed; some functionality will be unavailable"
)

FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_EMPTY_EXCEPTION = (
    "The figure is empty, please call log_figure() before calling show()."
)

FILE_UPLOADER_SAVE_MATPLOTLIB_FIGURE_UNSUPPORTED = "Unsupported figure type: %r provided. Please use the following figure types: Plotly Figures, Matplotlib Figures, or Seaborn plots."

FILE_UPLOADER_TOTAL_LEN_EXCEPTION = "Don't know how to compute total_len for %r[%r]"

FILE_UPLOADER_UNSUPPORTED_UPLOAD_INPUT = "Unsupported upload input %r"

FILE_UPLOADER_PROCESS_UPLOAD_BY_FILEPATH_COPY_WARNING = (
    "File %s should have been copied to a temporary location but was not"
)

FILE_UPLOADER_SAVE_PLOTLY_FAILED_WARNING = "Failing to save the plotly figure; requires dependencies: see https://plotly.com/python/static-image-export/"

FILE_UPLOADER_SAVE_MATPLOTLIB_FAILED_WARNING = (
    "Failing to save the matplotlib figure, reason: %s"
)

FILE_UPLOADER_IMAGE_COVERT_EXCEPTION = (
    "Could not convert image_data into an image due to: %r"
)

FILE_UPLOADER_IMAGE_COVERT_ERROR = "Could not convert image_data into an image"

FILE_UPLOADER_AUDIO_UNSUPPORTED_TYPE_EXCEPTION = "Unsupported audio_data type %r"

FILE_UPLOADER_AUDIO_NUMPY_REQUIRED_EXCEPTION = (
    "Numpy is needed when passing a numpy array to log_audio"
)

FILE_UPLOADER_AUDIO_SAMPLE_RATE_NONE_EXCEPTION = (
    "sample_rate cannot be None when logging a numpy array"
)

FILE_UPLOADER_AUDIO_SAMPLE_RATE_ZERO_EXCEPTION = (
    "sample_rate cannot be 0 when logging a numpy array"
)

FILE_UPLOADER_ASSET_JSON_FAILED_ERROR = "Failed to log asset data as JSON"

FILE_UPLOADER_PREPROCESS_REMOTE_ASSET_FILE_NAME_ERROR = (
    "Couldn't parse a file_name from URI %r, defaulting to 'remote'"
)

FILE_UPLOADER_PREPROCESS_ASSET_MEMORY_FILE_NAME_REQUIRED_EXCEPTION = (
    "file_name shouldn't be None"
)

SYSTEM_GPU_CUDA_INVALID_DEVICE_ID_ERROR = (
    "Invalid device identifier. Next identifiers won't be parsed."
)

SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_FORMAT_ERROR = (
    "CUDA_VISIBLE_DEVICES contains value of invalid or unsupported format: {}"
)

SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_UUID_ERROR = (
    "CUDA_VISIBLE_DEVICES contains invalid uuid abbreviation: {}"
)

SYSTEM_GPU_CUDA_VISIBLE_DEVICES_AMBIGUOUS_UUID_ERROR = (
    "CUDA_VISIBLE_DEVICES contains ambiguous uuid abbreviation: {}"
)

SYSTEM_GPU_CUDA_VISIBLE_DEVICES_WRONG_DEVICE_NUM_ERROR = "CUDA_VISIBLE_DEVICES contains device number which exceeds actual devices amount: {}"

SYSTEM_GPU_CUDA_VISIBLE_DEVICES_PARSE_FAILED_PYNVML_ERROR = (
    "Failed to parse CUDA_VISIBLE_DEVICES due to internal pynvml error."
)

SYSTEM_GPU_MANUAL_OVERRIDE_FAILED_ERROR = "Failed to manually override devices to log"

SYSTEM_BASE_FAILED_LOG_METRICS_WARNING = "Failed to log system metrics: %s"

SYSTEM_BASE_SMALL_SAMPLE_INTERVAL_WARNING = (
    "Provided interval is too low, falling back to the minimum interval (%.1f Seconds)"
)

COMET_START_FAILED_TO_CREATE_EXPERIMENT_ERROR = (
    "Failed to create Comet experiment, reason: %r"
)

COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_ERROR = (
    "Experiment with key: %r doesn't exists at Comet. Start arguments: %s, %s"
)

COMET_START_EXPERIMENT_WITH_KEY_NOT_FOUND_EXCEPTION = (
    "Experiment with key: %r doesn't exists at Comet."
)

COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ERROR = "Can not create new experiment with key: %r - it is already exists at Comet. Start arguments: %s, %s"

COMET_START_EXPERIMENT_CAN_NOT_BE_CREATED_ALREADY_EXISTS_EXCEPTION = "Can not create new experiment with key: %r - it is already exists at Comet. Please use %r or %r experiment creation modes."

COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_ERROR = (
    "Unsupported experiment creation mode: %r. Start arguments: %s, %s"
)

COMET_START_UNSUPPORTED_EXPERIMENT_CREATION_MODE_EXCEPTION = (
    "Unsupported experiment creation mode: %r"
)

COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR = "Can not use online experiment creation mode 'get' without 'experiment_key' value provided. Start arguments: %s, %s"

COMET_START_ONLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION = "Can not use online experiment creation mode 'get' without 'experiment_key' value provided."

COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_ERROR = "Can not use offline experiment creation mode 'get' without 'experiment_key' value provided. Start arguments: %s, %s"

COMET_START_OFFLINE_GET_WITHOUT_EXPERIMENT_KEY_EXCEPTION = "Can not use offline experiment creation mode 'get' without 'experiment_key' value provided."

COMET_START_REUSING_EXPERIMENT_PROMPT = "An experiment with the same configuration options is already running and will be reused."

COMET_START_UNSUPPORTED_RUNNING_EXPERIMENT = "The current running experiment is neither an online nor an offline experiment. A new experiment will be created."

COMET_START_STARTING_NEW_EXPERIMENT = (
    "Ending the running experiment and creating a new Experiment because:"
)

COMET_START_STARTING_NEW_EXPERIMENT_WITH_SOURCE = (
    "Ending the running experiment and creating a new Experiment from the '%s' because:"
)

LOG_FIGURE_KALEIDO_NOT_INSTALLED = "Kaleido is not installed, please run `pip install kaleido` to use log_figure for plotly figures."

SENTRY_FILTER_CHAIN_EXCEPTION = "Failed chain filters: %r"

COMET_INIT_DEPRECATED = "comet_ml.init() is deprecated and will be removed soon. Please use comet_ml.login()"

COMET_INIT_API_KEY_DEPRECATED = (
    "'comet init --api-key' is deprecated. Please use 'comet login' instead."
)

METRICS_SAMPLER_MIN_OPERATION_STRING_VALUE_WARNING = "The value '%s' for the metric '%s' is not a valid number (float or int), so the min sampling operation has been skipped."

METRICS_SAMPLER_MAX_OPERATION_STRING_VALUE_WARNING = "The value '%s' for the metric '%s' is not a valid number (float or int), so the max sampling operation has been skipped."

HF_DATASET_USING_CACHED = "Using cached dataset from [%s] folder"

HF_DATASET_DOWNLOADING_DATASET = "Downloading dataset from [%s] to [%s] folder"
