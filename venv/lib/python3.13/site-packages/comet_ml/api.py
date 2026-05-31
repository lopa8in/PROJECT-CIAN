# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import functools
import io
import logging
import numbers
import os.path
import re
import shutil
import zipfile
from typing import IO

from comet_ml.api_helpers.metric_dataframes import (
    get_dataframe_from_multi_metrics,
    interpolate_metric_dataframe,
    metrics_to_total_fidelity_dataframe,
)
from comet_ml.validation.method_parameters_validator import (
    MethodParametersTypeValidator,
)

import requests

from . import messages
from ._typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    PanelColorMap,
    Sequence,
    Tuple,
    Union,
)
from .api_objects import model
from .common_experiment import CommonExperiment
from .config import get_api_key, get_config
from .connection.connection_factory import get_rest_api_client
from .connection.connection_helpers import write_stream_response_to_file
from .constants import (
    ASSET_TYPE_CURVE,
    ASSET_TYPE_TENSORFLOW_FILE,
    DEFAULT_PROJECT_NAME,
)
from .convert_utils import convert_log_table_input_to_io, convert_model_to_string
from .exceptions import (
    INVALID_VERSION_STRING,
    CometException,
    CometRestApiException,
    NotFound,
    QueryException,
    ValidationError,
)
from .file_utils import expand_user_home_path
from .flatten_dict import exclusions, flattener
from .flatten_dict.flattener import (
    METRICS_DELIMITER,
    METRICS_MAX_DEPTH,
    PARAMETERS_DELIMITER,
    PARAMETERS_MAX_DEPTH,
    flatten_dict,
)
from .logging_messages import (
    API_ADD_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING,
    API_DELETE_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING,
    API_DOWNLOAD_REGISTRY_MODEL_COMPLETED_INFO,
    API_DOWNLOAD_REGISTRY_MODEL_COPY_INFO,
    API_DOWNLOAD_REGISTRY_MODEL_DEPRECATED_WARNING,
    API_DOWNLOAD_REGISTRY_MODEL_FAILED_INFO,
    API_DOWNLOAD_REGISTRY_MODEL_START_INFO,
    API_DOWNLOAD_REGISTRY_MODEL_UNZIP_INFO,
    API_EXPERIMENT_BY_ID_DEPRECATED_WARNING,
    API_EXPERIMENT_DELETE_PARAMETERS_UNSUPPORTED_BACKEND_VERSION_ERROR,
    API_EXPERIMENT_DELETE_TAGS_UNSUPPORTED_BACKEND_VERSION_ERROR,
    API_EXPERIMENT_DOWNLOAD_MODEL_COPY_INFO,
    API_EXPERIMENT_DOWNLOAD_MODEL_DONE_INFO,
    API_EXPERIMENT_DOWNLOAD_MODEL_FAILED_INFO,
    API_EXPERIMENT_DOWNLOAD_MODEL_START_INFO,
    API_EXPERIMENT_DOWNLOAD_MODEL_UNZIP_INFO,
    API_EXPERIMENT_EXTRA_KWARGS_IGNORED_INFO,
    API_EXPERIMENT_EXTRA_PROJECT_NAME_KWARGS_IGNORED_INFO,
    API_EXPERIMENT_EXTRA_WORKSPACE_KWARGS_IGNORED_INFO,
    API_EXPERIMENT_GET_ARTIFACT_LINEAGE_WRONG_DIRECTION,
    API_EXPERIMENT_GET_ASSET_BY_NAME_DEPRECATION_WARNING,
    API_EXPERIMENT_GET_ENV_DETAILS_DEPRECATED_WARNING,
    API_EXPERIMENT_GPU_STATIC_LIST_EXPECTED_EXCEPTION,
    API_EXPERIMENT_GPU_STATIC_LIST_OF_DICTS_EXPECTED_EXCEPTION,
    API_EXPERIMENT_LOG_CPU_METRICS_LIST_EXPECTED_EXCEPTION,
    API_EXPERIMENT_LOG_CPU_METRICS_LIST_OF_NUMBERS_EXPECTED_EXCEPTION,
    API_EXPERIMENT_LOG_GPU_METRICS_LIST_OF_DICTS_EXPECTED_EXCEPTION,
    API_EXPERIMENT_LOG_GPU_METRICS_MISSING_PARAMETERS_EXCEPTION,
    API_EXPERIMENT_LOG_TABLE_HEADERS_IGNORED_INFO,
    API_EXPERIMENT_LOG_TABLE_MISSING_TABULAR_DATA_EXCEPTION,
    API_EXPERIMENT_LOG_TABLE_WRONG_FILENAME_EXCEPTION,
    API_EXPERIMENT_MISSING_API_EXCEPTION,
    API_EXPERIMENT_MISSING_TF_FOLDER_EXCEPTION,
    API_EXPERIMENT_NOT_FOUND_MSG,
    API_EXPERIMENT_REGISTER_MODEL_INVALID_VERSION_EXCEPTION,
    API_EXPERIMENT_SET_CODE_FILE_AND_CODE_NOT_ALLOWED_WARNING,
    API_EXPERIMENT_SET_CODE_FROM_FILENAME_FAILED_WARNING,
    API_EXPERIMENT_TF_DOWNLOAD_FILE_ALREADY_EXISTS_WARNING,
    API_EXPERIMENT_TF_DOWNLOAD_VIEW_IN_TB_INFO,
    API_EXPERIMENT_WORKSPACE_AND_PROJECT_MISSING_EXCEPTION,
    API_EXPERIMENT_WRONG_BACKEND_VERSION_FOR_METHOD_EXCEPTION,
    API_GET_LATEST_REGISTRY_MODEL_VERSION_DETAILS_DEPRECATED_WARNING,
    API_GET_METRICS_FOR_CHART_REQUIRES_LIST_EXPERIMENTS_EXCEPTION,
    API_GET_METRICS_FOR_CHART_REQUIRES_LIST_METRICS_EXCEPTION,
    API_GET_METRICS_FOR_CHART_REQUIRES_LIST_PARAM_NAMES_EXCEPTION,
    API_GET_MODEL_REGISTRY_VERSION_ASSETS_DEPRECATED_WARNING,
    API_GET_PANEL_EXPERIMENT_KEYS_EXCEPTION,
    API_GET_PANEL_EXPERIMENTS_EXCEPTION,
    API_GET_PANEL_METRICS_NAMES_EXCEPTION,
    API_GET_PANEL_PROJECT_ID_EXCEPTION,
    API_GET_PROJECT_NOTES_UNKNOWN_PROJECT_EXCEPTION,
    API_GET_SLASH_PROJECT_AND_KEY_EXCEPTION,
    API_GET_SLASH_WORKSPACE_AND_PROJECT_EXCEPTION,
    API_INVALID_WORKSPACE_PROJECT_EXCEPTION,
    API_MISSING_PROJECT_IN_PATTERN_EXCEPTION,
    API_QUERY_ERROR_INFO,
    API_QUERY_INVALID_QUERY_EXPRESSION_EXCEPTION,
    API_QUERY_MISSING_QUERY_EXPRESSION_EXCEPTION,
    API_UPDATE_CACHE_DEPRECATED_WARNING,
    API_UPDATE_REGISTRY_MODEL_VERSION_DEPRECATED_WARNING,
    API_USE_CACHE_NOT_SUPPORTED_EXCEPTION,
    DEPRECATED_WORKSPACE_MODEL_REGISTRY_ARGUMENT,
    EXPERIMENT_LOG_CODE_NOT_A_FILE_WARNING,
    EXPERIMENT_LOG_CURVE_VALIDATION_ERROR,
    EXPERIMENT_LOG_TAG_VALIDATION_ERROR,
    LOG_METRICS_MAX_DEPTH_REACHED,
    LOG_PARAMS_MAX_DEPTH_REACHED,
    PANDAS_DATAFRAME_IS_REQUIRED,
)
from .query import (  # noqa
    Environment,
    Metadata,
    Metric,
    Other,
    Parameter,
    QueryExpression,
    QueryVariable,
    Tag,
)
from .semantic_version import SemanticVersion
from .utils import compress_git_patch, merge_url, read_git_patch_zip, valid_ui_tabs
from .validation.curve_data_validator import CurveDataValidator
from .validation.tag_validator import TagsValidator, TagValidator

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "API",
    "APIExperiment",
    "Environment",
    "Metadata",
    "Metric",
    "Other",
    "Parameter",
    "Tag",
]


class APIExperiment(CommonExperiment):
    """
    The APIExperiment class is used to access data from the
    Comet.ml Python API.

    You can use an instance of the APIExperiment() class to easily
    access all of your logged experiment information
    at [Comet](https://www.comet.com), including metrics, parameters,
    tags, and assets.

    Example:
        The following examples assume your `COMET_API_KEY` is configured as per
        [Python Configuration](/docs/v2/guides/experiment-management/configure-sdk/).

        This example shows looking up an experiment by its URL:

        ```python
        >>> from comet_ml.api import API, APIExperiment

        ## (assumes api keys are configured):
        >>> api = API() # can also: API(api_key="...")

        ## Get an APIExperiment from the API:
        >>> experiment = api.get("cometpublic/comet-notebooks/example 001")
        ```

        You can also make a new experiment using the API:

        ```python
        ## Make a new APIExperiment (assumes api keys are configured):
        >>> experiment = APIExperiment(workspace="my-username",
                                    project_name="general")
        ```

        Here is an end-to-end snippet to rename a metric. You can use this
        basic structure for logging new metrics (after the experiment has completed)
        such as averages or scaling to a baseline.

        ```python
        from comet_ml import API

        WORKSPACE = "your-comet-id"
        PROJECT_NAME = "general"
        EXP_KEY = "your-experiment-key"
        OLD_METRIC_NAME = "loss"
        NEW_METRIC_NAME = "train_loss"

        api = API() # can also: API(api_key="...")

        experiment = api.get_experiment(WORKSPACE, PROJECT_NAME, EXP_KEY)
        old_metrics = experiment.get_metrics(OLD_METRIC_NAME)

        for old_metric in old_metrics:
            experiment.log_metric(
                NEW_METRIC_NAME,
                old_metric["metricValue"],
                step=old_metric["step"],
                timestamp=old_metric["timestamp"],
            )
        ```

        For more usage examples, see [Comet Python API examples](../Comet-Python-API/).
    """

    _ATTR_FIELD_MAP = [
        # Standard in json:
        # (attr, jsonKey)
        ("id", "experimentKey"),
        ("duration_millis", "durationMillis"),
        ("start_server_timestamp", "startTimeMillis"),
        ("end_server_timestamp", "endTimeMillis"),
        ("archived", "archived"),
        ("project_name", "projectName"),
        ("project_id", "projectId"),
        ("workspace", "workspaceName"),
        ("name", "experimentName"),
        ("file_path", "filePath"),
        ("file_name", "fileName"),
        # Optional in json:
        ("optimization_id", "optimizationId"),
    ]

    def __init__(self, *args, **kwargs):
        """
        Create a new APIExperiment, or use a previous experiment key
        to access an existing experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()

            # Python API to create a new experiment:
            experiment = comet_ml.APIExperiment(workspace=WORKSPACE,
                                        project_name=PROJECT)

            # Python API to access an existing experiment:
            # (assumes api keys are configured):
            experiment = comet_ml.APIExperiment(previous_experiment=EXPERIMENT_KEY)
            ```

        Note:
            api_key may be defined in environment (COMET_API_KEY)
            or in a .comet.config file. Additional arguments will be
            given to API().
        """
        self.id = None  # type: Union[str, None]
        self._name = None  # type: Union[str, None]
        self.workspace = None  # type: Union[str, None]
        self.project_name = None  # type: Union[str, None]
        self.project_id = None  # type: Union[str, None]
        self.archived = False  # type: bool
        self.duration_millis = 0
        self.start_server_timestamp = 0
        self.end_server_timestamp = 0
        self.optimization_id = None

        if "metadata" in kwargs:
            # System usage: APIExperiment(api=API, metadata=METADATA)
            if "api" not in kwargs:
                raise ValueError(API_EXPERIMENT_MISSING_API_EXCEPTION)
            else:
                self._api = kwargs["api"]  # type: API
            if (
                ("workspace" in kwargs)
                or ("project_name" in kwargs)
                or ("previous_experiment" in kwargs)
            ):
                LOGGER.info(API_EXPERIMENT_EXTRA_KWARGS_IGNORED_INFO)
            self._set_from_metadata(kwargs["metadata"])
        elif "previous_experiment" in kwargs:
            # Parallel to ExistingExperiment(); APIExperiment(previous_experiment=KEY)
            # api may be provided
            previous_experiment = kwargs.pop("previous_experiment")
            # Not needed anymore:
            if "workspace" in kwargs:
                LOGGER.info(API_EXPERIMENT_EXTRA_WORKSPACE_KWARGS_IGNORED_INFO)
                kwargs.pop("workspace")
            if "project_name" in kwargs:
                LOGGER.info(API_EXPERIMENT_EXTRA_PROJECT_NAME_KWARGS_IGNORED_INFO)
                kwargs.pop("project_name")
            if "api" in kwargs:
                self._api = kwargs["api"]  # type: API
            else:
                self._api = API(**kwargs)
            metadata = self._api._get_experiment_metadata(previous_experiment)
            # Could check to see if workspace and project_name (if given) match metadata items
            self._set_from_metadata(metadata)
        else:
            # Parallel to Experiment(); APIExperiment(api_key=KEY, workspace=WS, project_name=PJ)
            # api may be provided
            workspace = kwargs.pop("workspace", None) or get_config("comet.workspace")
            project_name = (
                kwargs.pop("project_name", None)
                or get_config("comet.project_name")
                or DEFAULT_PROJECT_NAME
            )
            if (workspace is None) or (project_name is None):
                raise ValueError(API_EXPERIMENT_WORKSPACE_AND_PROJECT_MISSING_EXCEPTION)
            experiment_name = kwargs.pop("experiment_name", None)
            if "api" in kwargs:
                self._api = kwargs["api"]  # type: API
            else:
                self._api = API(**kwargs)
            results = self._api._client.create_experiment(
                workspace, project_name, experiment_name
            )
            if self._check_results(results):
                result_json = results.json()
                metadata = self._api._get_experiment_metadata(
                    result_json["experimentKey"]
                )
                self._set_from_metadata(metadata)

    def _check_results(self, results):
        return self._api._check_results(results)

    @property
    def name(self):
        """
        Get the experiment name.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(workspace=WORKSPACE)
            print(api_experiment.name)
            ```
        """
        if self._name is None:
            metadata = self._api._get_experiment_metadata(self.id)
            self._name = metadata["experimentName"]

        return self._name

    @name.setter
    def name(self, value):
        """
        Set the experiment name in this APIExperiment instance.

        Example:

            ```python linenums="1"
            >>> api_experiment.name = "my-preferred-name"
            ```

        Note:
            Setting the name here does not changed the logged
            experiment name. To change that, use:

            ```python linenums="1"
            api_experiment.log_other("name", "my-preferred-name")
            ```
        """
        self._name = value

    @property
    def key(self):
        """
        Get the experiment key (the unique id).

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(workspace=WORKSPACE)

            print(api_experiment.key)
            ```
        """
        return self.id

    @property
    def url(self):
        """
        Get the url of the experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(workspace=WORKSPACE)

            print(api_experiment.url)
            ```
        """
        return self._get_experiment_url()

    def end(self):
        """
        Method called at end of experiment.
        """
        # for compatibility with other Experiments
        pass

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return "<APIExperiment '%s/%s/%s'>" % (
            self.workspace,
            self.project_name,
            self.id,
        )

    def _set_from_metadata(self, metadata):
        # Set to the given value, or None
        for attr, item in self._ATTR_FIELD_MAP:
            setattr(self, attr, metadata.get(item, None))

    def _update_from_metadata(self, metadata):
        """
        Args:
            metadata: dictionary of data as shown below

        Example metadata:

        ```python
        {
         "experimentKey": "someExperimentKey",
         "experimentName": "someExperimentName",
         "optimizationId": "someOptimizationId",
         "projectId": "someProjectId",
         "projectName": "someProjectName",
         "workspaceName": "someWorkspaceName",
         "durationMillis": someDurationMillis,
         "startTimeMillis": someStartTimeMillis,
         "endTimeMillis": someEndTimeMillis
        }
        ```
        """
        # Set to the given value if given:
        for attr, item in self._ATTR_FIELD_MAP:
            if item in metadata:
                setattr(self, attr, metadata[item])

    def _get_experiment_url(self, tab=None):
        # type: (Optional[str]) -> str
        if self.archived:
            url = "/".join(
                [
                    self._api._get_url_server(),
                    self.workspace or "UNKNOWN",
                    self.project_name or DEFAULT_PROJECT_NAME,
                    "archive",
                    self.id or "UNKNOWN",
                ]
            )
        else:
            url = "/".join(
                [
                    self._api._get_url_server(),
                    self.workspace or "UNKNOWN",
                    self.project_name or DEFAULT_PROJECT_NAME,
                    self.id or "UNKNOWN",
                ]
            )

        if tab:
            if tab in valid_ui_tabs():
                return merge_url(
                    url,
                    {"experiment-tab": valid_ui_tabs(tab)},
                )
            else:
                LOGGER.info("tab must be one of: %r", valid_ui_tabs(preferred=True))
                return url
        else:
            return url

    def to_json(self, full=False):
        """
        The experiment data in JSON-like format.

        Args:
            full (bool): if True, get all experiment information.

        Example:
            ```python
            experiment.to_json()
            ```

        Returns:
            json: Experiment data in json format, follows the format:

                ```json
                {'id': '073e272581ac48c283910a05e5495381',
                'name': None,
                'workspace': 'testuser',
                'project_name': 'test-project-7515',
                'archived': False,
                'url': 'https://www.comet.com/testuser/test-project-7515/073e272581ac48c283910a05e54953801',
                'duration_millis': 4785,
                'start_server_timestamp': 1571318652586,
                'end_server_timestamp': 7437457,
                'optimization_id': None,
                }
                ```
        """
        # Without further net access, full=False:
        retval = {
            "id": self.id,
            "name": self.name,
            "workspace": self.workspace,
            "project_name": self.project_name,
            "archived": self.archived,
            "url": self.url,
            "duration_millis": self.duration_millis,
            "start_server_timestamp": self.start_server_timestamp,
            "end_server_timestamp": self.end_server_timestamp,
            "optimization_id": self.optimization_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
        }
        # Everything else, except individual assets, full=True:
        if full:
            git_patch = self.get_git_patch()
            if git_patch is not None:
                git_patch = read_git_patch_zip(git_patch)

            retval.update(
                {
                    "asset_list": self.get_asset_list(),
                    "code": self.get_code(),
                    "html": self.get_html(),
                    "metrics": self.get_metrics(),
                    "metrics_summary": self.get_metrics_summary(),
                    "model_graph": self.get_model_graph(),
                    "others_summary": self.get_others_summary(),
                    "output": self.get_output(),
                    "parameters_summary": self.get_parameters_summary(),
                    "system_details": self.get_system_details(),
                    "tags": self.get_tags(),
                    "git_patch": git_patch,
                    "git_metadata": self.get_git_metadata(),
                }
            )
        return retval

    # Read methods:

    def get_state(self) -> str:
        """
        Get current state of experiment.

        Returns:
            str: 'running', 'finished' or 'crashed'
        """

        MINIMAL_SUPPORTED_BACKEND = "3.7.91"
        self._raise_if_old_backend("get_state", MINIMAL_SUPPORTED_BACKEND)
        metadata = self.get_metadata()

        if metadata["running"]:
            return "running"

        if metadata["hasCrashed"]:
            return "crashed"

        return "finished"

    def set_state(self, state: str):
        """
        Set current state of the experiment.

        Args:
            state (str): String representing new experiment state. Must be either: 'running', 'finished' or 'crashed'
        """
        MINIMAL_SUPPORTED_BACKEND = "3.7.91"
        self._raise_if_old_backend("set_state", MINIMAL_SUPPORTED_BACKEND)

        self._api._client.set_experiment_state(experiment_key=self.key, state=state)

    def get_artifact_lineage(self, direction: str = "all") -> Dict[str, Any]:
        """
        Get the artifact lineage filtered using `direction` parameter.

        Args:
            direction (str): String representing filtering criteria to be applied to the artifact's lineage before
                returning. Allowed values: 'all', 'output', or 'input'.
        Returns:
            the artifact's lineage data obtained from server as dictionary.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_artifact_lineage())
            ```
        """
        allowed_directions = ["all", "output", "input"]
        if direction not in allowed_directions:
            raise ValueError(
                API_EXPERIMENT_GET_ARTIFACT_LINEAGE_WRONG_DIRECTION
                % (direction, allowed_directions)
            )

        return self._api._client.get_artifact_lineage(
            experiment_key=self.id, direction=direction
        )

    def get_name(self):
        """
        Get the name of the experiment, if one.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(workspace=WORKSPACE)
            api_experiment.set_name("My Name")

            print(api_experiment.get_name())
            ```
        """
        others = self.get_others_summary()
        query = [s["valueCurrent"] for s in others if s["name"] == "Name"]
        if len(query) == 1:
            return query[0]

    def get_html(self):
        """
        Get the HTML associated with this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(workspace=WORKSPACE)
            api_experiment.log_html("<b>Hello, world!</b>")

            print(api_experiment.get_html())
            ```
        """
        results = self._api._client.get_experiment_html(self.id)
        if results:
            return results["html"]

    def get_metadata(self):
        """
        Get the metadata associated with this experiment.

        Example:
            Running the following code sample:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            metadata = api_experiment.get_metadata()
            ```

            will print the following dictionary:
            ```json
            {
                'archived': False,
                'durationMillis': 7,
                'endTimeMillis': 1586174765277,
                'experimentKey': 'EXPERIMENT-KEY',
                'experimentName': None,
                'fileName': None,
                'filePath': None,
                'optimizationId': None,
                'projectId': 'PROJECT-ID',
                'projectName': 'PROJECT-NAME',
                'running': False,
                'startTimeMillis': 1586174757596,
                'throttle': False,
                'workspaceName': 'WORKSPACE-NAME',
            }
            ```
        """
        results = self._api._get_experiment_metadata(self.id)
        return results

    def get_code(self):
        """
        Get the associated source code for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_code())
            ```
        """
        results = self._api._client.get_experiment_code(self.id)
        if results:
            return results["code"]

    def get_output(self):
        """
        Get the associated standard output for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')
            print(api_experiment.get_output())
            ```
        """
        results = self._api._client.get_experiment_output(self.id)
        if results:
            return results["output"]

    def get_installed_packages(self):
        """
        Get the associated installed packages for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_installed_packages())
            ```
        """
        results = self._api._client.get_experiment_installed_packages(self.id)
        return results

    def get_environment_details(self):
        """
        Deprecated. Use [comet_ml.APIExperiment.get_os_packages][] instead.
        """
        LOGGER.warning(API_EXPERIMENT_GET_ENV_DETAILS_DEPRECATED_WARNING)
        return self.get_os_packages()

    def get_os_packages(self):
        """
        Get the OS packages for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_os_packages())
            ```
        """
        results = self._api._client.get_experiment_os_packages(self.id)
        return results

    def get_user(self):
        """
        Get the associated user for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_user())
            ```
        """
        results = self._api._client.get_experiment_user(self.id)
        return results

    def get_python_version(self):
        """
        Get the Python version for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_python_version())
            ```
        """
        results = self._api._client.get_experiment_python_version(self.id)
        return results

    def get_python_version_verbose(self):
        """
        Get the Python version verbose for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_python_version_verbose())
            ```
        """
        results = self._api._client.get_experiment_python_version_verbose(self.id)
        return results

    def get_pid(self):
        """
        Get the pid for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_pid())
            ```
        """
        results = self._api._client.get_experiment_pid(self.id)
        return results

    def get_os_type(self):
        """
        Get the associated os type for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_os_type())
            ```
        """
        results = self._api._client.get_experiment_os_type(self.id)
        return results

    def get_os(self):
        """
        Get the associated OS for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_os())
            ```
        """
        results = self._api._client.get_experiment_os(self.id)
        return results

    def get_os_release(self):
        """
        Get the associated OS release for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_os_release())
            ```
        """
        results = self._api._client.get_experiment_os_release(self.id)
        return results

    def get_ip(self):
        """
        Get the associated IP for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_ip())
            ```
        """
        results = self._api._client.get_experiment_ip(self.id)
        return results

    def get_hostname(self):
        """
        Get the associated hostname for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_hostname())
            ```
        """
        results = self._api._client.get_experiment_hostname(self.id)
        return results

    def get_gpu_static_info(self):
        """
        Get the associated GPU static info for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_gpu_statis_info())
            ```
        """
        results = self._api._client.get_experiment_gpu_static_info(self.id)
        return results

    def get_additional_system_info(self):
        """
        Get the associated additional system info for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_additional_system_info())
            ```
        """
        results = self._api._client.get_experiment_additional_system_info(self.id)
        return results

    def get_system_metric_names(self):
        """
        Get the associated system metric names for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_system_metric_names())
            ```
        """
        results = self._api._client.get_experiment_system_metric_names(self.id)
        return results

    def get_max_memory(self):
        """
        Get the associated max total memory for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_max_memory())
            ```
        """
        results = self._api._client.get_experiment_max_memory(self.id)
        return results

    def get_network_interface_ips(self):
        """
        Get the associated network interface IPs for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_netword_interface_ips())
            ```
        """
        results = self._api._client.get_experiment_network_interface_ips(self.id)
        return results

    def get_command(self):
        """
        Get the associated command-line script and args for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_command())
            ```
        """
        results = self._api._client.get_experiment_command(self.id)
        return results

    def get_executable(self):
        """
        Get the associated executable for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_executable())
            ```
        """
        results = self._api._client.get_experiment_executable(self.id)
        return results

    def get_total_memory(self):
        """
        Get the associated total RAM for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_total_memory())
            ```
        """
        results = self._api._client.get_experiment_total_memory(self.id)
        return results

    def get_machine(self):
        """
        Get the associated total RAM for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_machine())
            ```
        """
        results = self._api._client.get_experiment_machine(self.id)
        return results

    def get_processor(self):
        """
        Get the associated total RAM for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_processor())
            ```
        """
        results = self._api._client.get_experiment_processor(self.id)
        return results

    # APIExperiment model methods:

    def get_model_data(self, name):
        """
        Deprecated. Use [comet_ml.APIExperiment.get_model_asset_list][] instead.
        """
        LOGGER.warning(
            "APIExperiment.get_model_data() has been deprecated; please use APIExperiment.get_model_asset_list() instead."
        )
        return self.get_model_asset_list(name)

    def get_model_asset_list(self, model_name):
        """
        Get an experiment model's asset list by model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: A list of asset dictionaries with these fields: fileName, fileSize,
                  runContext, step, link, createdAt, dir, canView, audio, histogram,
                  image, type, metadata, assetId

        Example:
            Running the following code:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            api_exp = api.get("workspace/project/765643463546345364536453436")

            res = api_exp.get_model_asset_list("Model Name")
            print(res)
            ```

            will display the dictionary:
            ```json
            [
                {
                    "assetId": 74374637463476,
                    "audio": False,
                    "canView": False,
                    "createdAt": 7337347634,
                    "dir": "trained-models",
                    "fileName": "model.h5",
                    "fileSize": 254654,
                    "histogram": False,
                    "image": False,
                    "link": "https://link-to-download-asset-file",
                    "metadata": None,
                    "remote": False,
                    "runContext": "train",
                    "step": 54,
                    "type": "asset",
                }
            ]
            ```
        """
        return self._api._client.get_experiment_model_asset_list(self.id, model_name)

    def download_tensorflow_folder(self, output_path="./", overwrite=False):
        """
        Download all files logged with [comet_ml.Experiment.log_tensorflow_folder][].

        Args:
            output_path (str): Where to download the files
            overwrite (bool): If True, then overwrite any file that exists

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()

            experiment = comet_ml.Experiment()
            experiment.log_tensorboard_folder("logs")

            api = comet_ml.API()
            api_experiment = api.get_experiment_by_key(experiment.get_key())
            api_experiment.download_tensorflow_folder()
            ```
        """

        asset_list = self.get_asset_list(asset_type=ASSET_TYPE_TENSORFLOW_FILE)

        if len(asset_list) == 0:
            raise ValueError(
                API_EXPERIMENT_MISSING_TF_FOLDER_EXCEPTION % self._get_experiment_url()
            )

        output_path = expand_user_home_path(output_path)
        for asset_json in asset_list:
            asset_filename = asset_json["fileName"]

            filename = os.path.join(output_path, asset_filename)

            if os.path.exists(filename) and not overwrite:
                LOGGER.warning(
                    API_EXPERIMENT_TF_DOWNLOAD_FILE_ALREADY_EXISTS_WARNING,
                    filename,
                )
                continue

            LOGGER.debug("Downloading %r to %r", asset_filename, filename)
            response = self.get_asset(
                asset_json["assetId"], return_type="response", stream=True
            )
            path, basename = os.path.split(filename)
            os.makedirs(path, exist_ok=True)
            with io.open(filename, "wb") as output_file:
                write_stream_response_to_file(response, output_file, None)

        LOGGER.info(
            API_EXPERIMENT_TF_DOWNLOAD_VIEW_IN_TB_INFO % os.path.join(output_path)
        )

    def download_model(self, name, output_path="./", expand=True):
        """
        Download and save all files from the model.

        Args:
            name (str): The name of the model.
            output_path (str): The output directory; defaults to current directory.
            expand (bool): If True, the downloaded zipfile is unzipped; if False, then the zipfile
                is copied to the output_path.
        """
        LOGGER.info(API_EXPERIMENT_DOWNLOAD_MODEL_START_INFO, name)
        zip_file = self._api._client.get_experiment_model_zipfile(self.id, name)
        if zip_file is not None:
            output_path = expand_user_home_path(output_path)
            with io.BytesIO(zip_file) as fp:
                if expand:
                    LOGGER.info(API_EXPERIMENT_DOWNLOAD_MODEL_UNZIP_INFO, output_path)
                    with zipfile.ZipFile(fp) as zp:
                        zp.extractall(output_path)
                else:
                    output_file = os.path.join(output_path, "%s.zip" % (name,))
                    LOGGER.info(API_EXPERIMENT_DOWNLOAD_MODEL_COPY_INFO, output_file)
                    with open(output_file, "wb") as op:
                        shutil.copyfileobj(fp, op)
                LOGGER.info(API_EXPERIMENT_DOWNLOAD_MODEL_DONE_INFO)
        else:
            LOGGER.info(API_EXPERIMENT_DOWNLOAD_MODEL_FAILED_INFO)

    def get_model_names(self):
        """
        Get a list of model names associated with this experiment.

        Returns:
            list: List of model names
        """
        model_names = [
            model["modelName"]
            for model in self._api._client.get_experiment_models(self.id)
        ]
        return list(set(model_names))

    def get_model_graph(self):
        """
        Get the associated graph/model description for this
        experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_model_graph())
            ```
        """
        results = self._api._client.get_experiment_model_graph(self.id)
        if results:
            return results["graph"]

    def get_tags(self):
        """
        Get the associated tags for this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_tags())
            ```
        """
        results = self._api._client.get_experiment_tags(self.id)
        if results:
            return results["tags"]

    def get_parameters_summary(self, parameter=None):
        """
        Return the experiment parameters summary.  Optionally, also if you
        provide a parameter name, the method will only return the
        summary of the given parameter.

        Args:
            parameter (str):Name of a parameter

        Example:
            Running the following code sample:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_parameters_summary())
            ```

            will print the list:
            ```json
            [{
                'name': 'batch_size',
                'valueMax': '120',
                'valueMin': '120',
                'valueCurrent': '120',
                'timestampMax': 1558962363411,
                'timestampMin': 1558962363411,
                'timestampCurrent': 1558962363411
            },
            ...]
            ```


            Specifying a parameter name:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            print(api_experiment.get_parameters_summary("batch_size"))
            ```

            will return the dictionary:
            ```json
            {
                'name': 'batch_size',
                'valueMax': '120',
                'valueMin': '120',
                'valueCurrent': '120',
                'timestampMax': 1558962363411,
                'timestampMin': 1558962363411,
                'timestampCurrent': 1558962363411
            }
            ```
        """
        results = self._api._client.get_experiment_parameters_summaries(self.id)
        if results:
            if parameter is not None:
                retval = [p for p in results["values"] if p["name"] == parameter]
                if retval:
                    return retval[0]
                else:
                    return []
            else:
                return results["values"]
        else:
            return []

    def get_metrics_summary(self, metric=None):
        """
        Return the experiment metrics summary.  Optionally, also if you
        provide the metric name, the function will only return the
        summary of the metric.

        Args:
            metric (str): Name of a metric.

        Example:
            Getting all metrics for an experiment:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_metrics_summary()
            print(res)
            ```

            will print the following list:
            ```json
            [{
                'name': 'val_loss',
                'valueMax': '0.24951280827820302',
                'valueMin': '0.13101346811652184',
                'valueCurrent': '0.13101346811652184',
                'timestampMax': 1558962367938,
                'timestampMin': 1558962367938,
                'timestampCurrent': 1558962376383,
                'stepMax': 500,
                'stepMin': 1500,
                'stepCurrent': 1500
            },
            ...]
            ```

            Specifying the metric name:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_metrics_summary("val_loss")
            print(res)
            ```

            will print the following dictionary:
            ```json
            {
                'name': 'val_loss',
                'valueMax': '0.24951280827820302',
                'valueMin': '0.13101346811652184',
                'valueCurrent': '0.13101346811652184',
                'timestampMax': 1558962367938,
                'timestampMin': 1558962367938,
                'timestampCurrent': 1558962376383,
                'stepMax': 500,
                'stepMin': 1500,
                'stepCurrent': 1500
            }
            ```
        """
        results = self._api._client.get_experiment_metrics_summaries(self.id)
        if results:
            if metric is not None:
                retval = [m for m in results["values"] if m["name"] == metric]
                if retval:
                    return retval[0]
                else:
                    return []
            else:
                return results["values"]
        else:
            return []

    def get_others_summary(self, other=None):
        """
        Get the other items logged in summary form.

        Args:
            other (str):The name of the other item
                logged. If given, return the valueCurrent of
                the other item. Otherwise, return all other
                items logged.

        Example:
            Getting all metrics for an experiment:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_others_summary()
            print(res)
            ```

            will print the following list:
            ```json
            [{
                'name': 'trainable_params',
                'valueMax': '712723',
                'valueMin': '712723',
                'valueCurrent': '712723',
                'timestampMax': 1558962363411,
                'timestampMin': 1558962363411,
                'timestampCurrent': 1558962363411
            },
            ...]
            ```

            Specifying the metric name:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_others_summary("trainable_params")
            print(res)
            ```

            will print the following dictionary:
            ```json
            ['712723']
            ```
        """
        results = self._api._client.get_experiment_others_summaries(self.id)
        if results:
            if other is not None:
                retval = [
                    m["valueCurrent"] for m in results["values"] if m["name"] == other
                ]
                return retval
            else:
                return results["values"]
        else:
            return []

    def _get_metric_asset_df(self, asset_id: str) -> Optional["pd.DataFrame"]:
        """
        Given an asset id of a zipped CSV file, return its DataFrame.
        """
        try:
            import pandas
        except ImportError:
            LOGGER.error(PANDAS_DATAFRAME_IS_REQUIRED)

            return None

        df = None
        data = self.get_asset(asset_id, return_type="binary")
        with io.BytesIO(data) as fp:
            with zipfile.ZipFile(fp, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    with zip_ref.open(file_info.filename) as file:
                        df = pandas.read_csv(file)
        return df

    def get_metric_total_df(
        self, metric_name: str, fallback_to_sampled_data: bool = True
    ) -> Optional["pd.DataFrame"]:
        """
        Given the name of a total-fidelity metric, return a Pandas DataFrame
        with all the logged metric data.

        Args:
            metric_name (str): Name of the total-fidelity metric
            fallback_to_sampled_data (bool): If set to `True`, when the API returns no data,
                it would fetch regular sampled data instead.

        The returned DataFrame contains the following columns:

        * value - the value of the metric
        * timestep - the time of the metric
        * step - the step that the metric was logged at
        * epoch - the epoch that the metric was logged at
        * datetime - the timestamp as a datetime
        * duration - the duration time between this row and the previous
        """
        try:
            import pandas
        except ImportError:
            LOGGER.error(PANDAS_DATAFRAME_IS_REQUIRED)

            return None

        metric_name = re.sub("[^a-zA-Z0-9-+]+", "_", metric_name)
        asset_list = self.get_asset_list("ASSET_TYPE_FULL_METRIC")
        metric_list = sorted(
            [
                metric
                for metric in asset_list
                if re.match(metric_name + "_\\d+.csv.zip$", metric["fileName"])
            ],
            key=lambda item: item["fileName"],
        )

        df = None
        for metric in metric_list:
            df_part = self._get_metric_asset_df(metric["assetId"])
            if df_part is not None and not df_part.empty:
                if df is None:
                    df = df_part
                else:
                    df.append(df_part, ignore_index=True)

        if df is None and fallback_to_sampled_data:
            metrics = self.get_metrics(metric=metric_name)
            if metrics is not None and len(metrics) > 0:
                df = metrics_to_total_fidelity_dataframe(metrics)

        if df is not None:
            df["datetime"] = pandas.to_datetime(df["timestamp"], unit="s")
            df["duration"] = df["timestamp"].diff()

        return df

    def get_metrics(self, metric=None):
        """
        Get all of the logged metrics. Optionally, just get the given metric name.

        Args:
            metric (str): If given, filter the metrics by name.


        Example:
            Getting all metrics for an experiment:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_metrics()
            print(res)
            ```

            will print the following list:
            ```json
            [{
                'metricName': 'val_loss',
                'metricValue': '0.13101346811652184',
                'timestamp': 1558962376383,
                'step': 1500,
                'epoch': None,
                'runContext': None
            },
            {
                'metricName': 'acc',
                'metricValue': '0.876',
                'timestamp': 1564536453647,
                'step': 100,
                'epoch': None,
                'runContext': None
            },
            ...]
            ```

            Specifying the metric name:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_metrics("acc")
            print(res)
            ```

            will print the following dictionary:
            ```json
            [{
                'metricName': 'acc',
                'metricValue': '0.876',
                'timestamp': 1564536453647,
                'step': 100,
                'epoch': None,
                'runContext': None
            },
            ...]
            ```
        """
        if metric is None:
            # first try to get all metrics at once if supported by backend
            results = self._get_all_experiment_metrics()
            if results is not None:
                return results

            # fallback to one-by-one supported by older backend versions
            metric_names = self._api._get_metrics_name(
                workspace=self.workspace, project_name=self.project_name
            )
            results = self._get_metrics(metric_names)
        else:
            results = self._get_metric(metric)

        return results

    def _get_all_experiment_metrics(self) -> Optional[List[Dict[str, Any]]]:
        current_backend_version = self._api._client.get_api_backend_version()
        check_result = get_config().has_api_experiment_get_all_metrics_enabled(
            current_backend_version
        )
        if check_result.feature_supported is False:
            LOGGER.debug(
                "get_all_experiment_metrics is not supported by backend, minimal required backend version: %s",
                check_result.min_backend_version_supported,
            )
            return None

        results = self._api._client.get_all_experiment_metrics(self.id)
        if results:
            return results["metrics"]
        else:
            return []

    def _get_metrics(self, metric_names: List[str]) -> List[Dict[str, Any]]:
        retval = []
        for metric_name in metric_names:
            metric = self._get_metric(metric_name)
            retval.extend(metric)
        return retval

    def _get_metric(self, metric_name: str) -> List[Dict[str, Any]]:
        results = self._api._client.get_experiment_metric(self.id, metric_name)
        if results:
            return results["metrics"]
        else:
            return []

    def get_asset_list(
        self, asset_type: str = "all", timeout: int = 600
    ) -> List[Dict[str, Any]]:
        """
        Get a list of assets associated with the experiment.

        Args:
            asset_type (str):Type of asset to return. Can be
                "all", "image", "histogram_combined_3d", "video", or "audio".
            timeout (int): Timeout in seconds.

        Returns:
            dict: A list of dictionaries of asset properties

        Example:
            Getting all metrics for an experiment:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            res = api_experiment.get_asset_list()
            print(res)
            ```

            will print the following list:
            ```json
            [{
                'fileName': 'My Filename.png',
                'fileSize': 21113,
                'runContext': None,
                'step': None,
                'link': 'https://www.comet.com/api/asset/download?experimentKey=KEY&assetId=ASSET_ID',
                'createdAt': 1565898755830,
                'dir': 'assets',
                'canView': False,
                'audio': False,
                'video': False,
                'histogram': False,
                'image': True,
                'type': 'image',
                'metadata': None,
                'assetId': ASSET_ID
            },
            ...]
            ```
        """
        results = self._api._client.get_experiment_asset_list(
            self.id, asset_type, timeout=timeout
        )
        # results is the list directly
        return results

    def get_asset_by_name(
        self,
        asset_filename: str,
        asset_type: str = "all",
        return_type: str = "binary",
        stream: bool = False,
        timeout: int = 600,
    ) -> Optional[Union[bytes, Dict[str, Any], requests.Response]]:
        """
        Get an asset, given the asset filename.

        Args:
            asset_filename (str): The asset filename.
            asset_type (str): Type of asset to return. Can be
                "all", "image", "histogram_combined_3d", "video", or "audio".
            return_type (str): The type of the object returned. The default is
                "binary". Options: "binary", "json", or "response"
            stream (bool): When return_type is "response", you can also
                use stream=True to use the response as a stream
            timeout (int): Timeout in seconds.

        Note: Will return the first asset found, if there are more than one with
        the same name. If you give the asset_type, the function will run faster.
        If no asset is found, then the method returns None.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.get_asset_by_name("features.json", return_type="json")
            ```
        """
        version = self._api._client.get_api_backend_version()
        if version.compare(SemanticVersion.parse("4.3.385")) < 0:
            results = self._api._client.get_experiment_asset_list(
                experiment_key=self.id, asset_type=asset_type, timeout=timeout
            )
            for asset in results:
                if asset["fileName"] == asset_filename:
                    return self.get_asset(
                        asset["assetId"], return_type=return_type, stream=stream
                    )
        else:
            LOGGER.warning(API_EXPERIMENT_GET_ASSET_BY_NAME_DEPRECATION_WARNING)
            if asset_type == "all":
                # with the new endpoint None is all
                asset_type = None

            results = self._api._client.get_experiment_assets_list_by_name(
                experiment_key=self.id,
                asset_name=asset_filename,
                asset_type=asset_type,
                timeout=timeout,
            )
            if results:
                return self.get_asset(
                    results[0]["assetId"], return_type=return_type, stream=stream
                )

        return None

    def get_assets_by_name(
        self,
        asset_filename: str,
        asset_type: Optional[str] = None,
        return_type: str = "binary",
        timeout: int = 600,
    ) -> Optional[List[Union[bytes, Dict[str, Any]]]]:
        """
        Retrieves assets by their name from the current experiment.

        Searches for assets in the current experiment with the specified filename and
        optionally filters by asset type, return data type, and a timeout duration.

        Args:
            asset_filename:
                The name of the asset file to search for.
            asset_type:
                The type of the asset to retrieve (default is None). Must match valid
                asset type strings ("image", "histogram_combined_3d", "video", "audio", etc.).
                If None, then the method will match all asset types.
            return_type:
                The desired return type of the asset's content. The default is
                "binary". Options: "binary", "json" or "text".
            timeout:
                The maximum time (in seconds) to allow for retrieving a response
                (default is 600).

        Returns:
            A list of assets matching the given criteria if found, with each asset
            represented in the specified return type, or None if no matching assets
            are found.
        """
        results = self._api._client.get_experiment_assets_list_by_name(
            experiment_key=self.id,
            asset_name=asset_filename,
            asset_type=asset_type,
            timeout=timeout,
        )
        if results is not None and len(results) > 0:
            return [
                self.get_asset(asset["assetId"], return_type=return_type)
                for asset in results
            ]

        return None

    def get_asset(
        self, asset_id: str, return_type: str = "binary", stream: bool = False
    ) -> Union[bytes, Dict[str, Any], requests.Response]:
        """
        Get an asset, given the asset_id.

        Args:
            asset_id (str): The asset ID
            return_type (str): The type of object returned. Default is
                "binary". Options: "binary", "json", or "response"
            stream (bool): When return_type is "response", you can also
                use stream=True to use the response as a stream

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.get_asset("298378237283728", return_type="json")
            ```

            To use with the streaming option:

            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            asset_response = api_experiment.get_asset(
                "298378237283728",
                return_type="response",
                stream=True,
            )

            with open(filename, 'wb') as fd:
                for chunk in asset_response.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)
            ```
        """
        results = self._api._client.get_experiment_asset(
            asset_id=asset_id,
            experiment_key=self.id,
            return_type=return_type,
            stream=stream,
        )
        # Return directly
        return results

    def get_curves(self):
        """
        Get all curves logged with experiment.

        Example:
            Running:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.get_curves()
            ```

            Will return the dictionary:
            ```json
            [{
                "name": "curve1",
                "x": [1, 2, 3],
                "y": [4, 5, 6], "step": 0
            }]
            ```
        """
        curves = []
        for asset_curve in self.get_asset_list(ASSET_TYPE_CURVE):
            asset = self.get_asset(asset_curve["assetId"], return_type="json")
            asset["step"] = asset_curve["step"]
            asset["assetId"] = asset_curve["assetId"]
            curves.append(asset)
        return curves

    def get_curve(self, asset_id):
        """
        Get curve logged with experiment by asset id.

        Args:
            asset_id (str): The asset id of the curve to download

        Example:
            Running the code sample:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.get_curve("57457745745745774")
            ```

            Will return the dictionary:
            ```json
            {
                "name": "curve1",
                "x": [1, 2, 3],
                "y": [4, 5, 6], "step": 0
            }
            ```
        """
        asset = self.get_asset(asset_id, return_type="json")
        # TODO: Replace the get_asset_list call + filtering which is O(n)
        for asset_curve in self.get_asset_list(ASSET_TYPE_CURVE):
            if asset_curve["assetId"] == asset_id:
                asset["step"] = asset_curve["step"]
                asset["assetId"] = asset_curve["assetId"]
                break
        return asset

    def get_system_details(self):
        """
        Get the system details associated with this experiment.

        Returns:
            dict: A dictionary that follows the format:
                ```python
                {
                    "experimentKey": "someExperimentKey",
                    "user": "system username"
                    "pythonVersion": "python version"
                    "pythonVersionVerbose": "python version with verbose flag"
                    "pid": <Integer, pid>,
                    "osType": "os experiment ran on",
                    "os": "os with version info",
                    "ip": "ip address",
                    "hostname": "hostname",
                    "gpuStaticInfoList": [
                        {
                        "gpuIndex": <Integer, index>,
                        "name": "name",
                        "uuid": "someUniqueId",
                        "totalMemory": <Integer, total memory>,
                        "powerLimit": <Integer, max power>
                        }
                    ],
                    "logAdditionalSystemInfoList": [
                        {
                        "key": "someKey",
                        "value": "someValue"
                        }
                    ],
                    "systemMetricNames": ["name", "anotherName"],
                    "maxTotalMemory": <double, max memory used>,
                    "networkInterfaceIps": ["ip", "anotherIp"]
                    "command": ["part1", "part2"],
                    "executable": "The python Exe, if any (in future could be non python executables)",
                    "osPackages": ["package", "anotherPackage"],
                    "installedPackages": ["package", "anotherPackage"]
                }
                ```
        """
        results = self._api._client.get_experiment_system_details(self.id)
        # Return directly
        return results

    def get_git_patch(self):
        """
        Get the git-patch associated with this experiment as a zipfile containing a unique file
        named `zip_file.patch`.

        Example:
            ```python linenums="1"
            import io, zipfile
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            zip_patch = io.BytesIO(api_experiment.get_git_patch())
            archive = zipfile.ZipFile(zip_patch)
            patch = archive.read("git_diff.patch")
            ```
        """
        results = self._api._client.get_experiment_git_patch(self.id)
        # Return directly
        return results

    def get_git_metadata(self):
        """
        Get the git-metadata associated with this experiment.

        Example:
            Running the code sample"
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.get_git_metadata()
            ```

            will return the json:
            ```json
            {
                "branch": 'refs/heads/master',
                "origin": 'git@github.com:comet-ml/comet-examples.git',
                "parent": '96ff529b4c02e4e0bb92992a7c4ce81275985764',
                "root": 'eec2d16daa057d0cf4c2c49974e6ea51e732a7b2',
                "user": 'user',
            }
            ```
        """
        results = self._api._client.get_experiment_git_metadata(self.id)
        # Return directly
        return results

    # Write methods:

    def register_model(
        self,
        model_name,
        version=None,
        workspace=None,
        registry_name=None,
        public=None,
        description=None,
        comment=None,
        status=None,
        tags=None,
        stages=None,
    ):
        """
        Register an experiment model in the workspace registry.

        Args:
            model_name (str): The name of the experiment model.
            workspace (str, optional): This argument is deprecated and ignored.
            version (str, optional): A proper semantic version string; defaults to "1.0.0".
            registry_name (str, optional): The name of the registered workspace model, if not provided the
                model_name will be used instead.
            public (bool, optional): If True, then the model will be publically viewable.
            description (str, optional): A textual description of the model.
            comment (str, optional): A textual comment about the model.
            tags (Any, optional): A list of textual tags such as ["tag1", "tag2"], etc.
            stages (Any, optional): Equivalent to tags, DEPRECATED with newer backend versions.
            status (str, optional): Allowed values are configured at the organization level.

        Returns:
            dict: if successful, the dict will looks like this:
                ```python
                {"registryModelId": "ath6ho4eijaexeShahJ9sohQu", "registryModelItemId": "yoi5saes7ea2vooG2ush1uuwi"}
                ```
        """
        try:
            if registry_name is None:
                registry_name = model_name

            if workspace:
                LOGGER.warning(DEPRECATED_WORKSPACE_MODEL_REGISTRY_ARGUMENT)

            response = self._api._client.register_model_v2(
                self.id,
                model_name,
                version,
                self.workspace,
                registry_name,
                public,
                description,
                comment,
                tags,
                status,
                stages,
            )
        except CometRestApiException as exc:
            if (
                exc.safe_json_response
                and exc.safe_json_response.get("sdk_error_code", None)
                == INVALID_VERSION_STRING
            ):
                raise ValueError(
                    API_EXPERIMENT_REGISTER_MODEL_INVALID_VERSION_EXCEPTION
                )
            else:
                raise

        if response:
            return response.json()

    def create_symlink(self, project_name):
        """
        Create a copy of this experiment in another project
        in the workspace.

        Args:
            project_name (str): the name of the project with which to create
                a symlink to this experiment in.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.create_symlink("my-other-project")
            ```
        """
        response_content = self._api._client.create_experiment_symlink(
            self.id, project_name
        )
        return response_content["link"]

    def archive(self):
        """
        Archive this experiment.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.archive()
            ```
        """
        results = self._api._client.archive_experiment(self.id)
        if results:
            self.archived = True
            return results

    def set_git_metadata(self, user, root, branch, parent, origin):
        """
        Set the git metadata for this experiment.

        Args:
            user (str): The name of the git user.
            root (str): The name of the git root.
            branch (str): The name of the git branch.
            parent (str): The name of the git parent.
            origin (str): The name of the git origin.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_git_metadata("user", "root", "branch", "parent", "origin")
            ```
        """
        results = self._api._client.set_experiment_git_metadata(
            self.id, user, root, branch, parent, origin
        )
        if self._check_results(results):
            return results.json()

    def set_git_patch(self, file_data):
        """
        Set the git patch for this experiment.

        Args:
            file_data (str): the contents or filename of the git patch file

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_git_patch("git.patch")
            ```
        """
        if file_data is None:
            return
        if os.path.isfile(file_data):
            with open(file_data, "rb") as fp:
                git_patch_data = fp.read()
        else:
            git_patch_data = file_data

        # compress and send
        _, zip_path = compress_git_patch(git_patch_data)
        with open(zip_path, "rb") as fp:
            results = self._api._client.set_experiment_git_patch(self.id, fp)

        if self._check_results(results):
            return results.json()

    def set_code(self, code=None, filename=None):
        """
        Set the code for this experiment. Pass in either
        the code as a string, or provide filename.

        Args:
            code (str): The source code for this experiment
            filename (str): The filename for this experiment

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_code("import comet_ml\\nexperiment = comet_ml.Experiment()")
            api_experiment.set_code(filename="script.py")
            ```
        """
        if filename:
            if code is not None:
                LOGGER.warning(
                    API_EXPERIMENT_SET_CODE_FILE_AND_CODE_NOT_ALLOWED_WARNING
                )
            elif os.path.isfile(filename):
                try:
                    with open(filename) as source_file:
                        code = source_file.read()
                except Exception:
                    LOGGER.warning(
                        API_EXPERIMENT_SET_CODE_FROM_FILENAME_FAILED_WARNING,
                        exc_info=True,
                    )
                    return
            else:
                LOGGER.warning(EXPERIMENT_LOG_CODE_NOT_A_FILE_WARNING, filename)
                return

        results = self._api._client.set_experiment_code(self.id, code)
        if self._check_results(results):
            return results.json()

    def set_model_graph(self, graph):
        """
        Set the model graph for this experiment.

        Args:
            graph (Any): A representation of the model graph

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_model_graph(model)
            ```
        """
        graph_str = convert_model_to_string(graph)
        results = self._api._client.set_experiment_model_graph(self.id, graph_str)
        if self._check_results(results):
            return results.json()

    def set_os_packages(self, os_packages):
        """
        Set the OS packages for this experiment.

        Args:
            os_packages (List[str]): The OS package list

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_os_packages(['accountsservice=0.6.45-1ubuntu1', ...])
            ```
        """
        results = self._api._client.set_experiment_os_packages(self.id, os_packages)
        if self._check_results(results):
            return results.json()

    def set_user(self, user):
        """
        Set the user for this experiment.

        Args:
            user (str): The OS username.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_user("os-user-name")
            ```
        """
        results = self._api._client.set_experiment_user(self.id, user)
        if self._check_results(results):
            return results.json()

    def set_python_version(self, python_version):
        """
        Set the Python version for this experiment.

        Args:
            python_version (str): The verbose Python version

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_python_version("3.9.7")
            ```
        """
        results = self._api._client.set_experiment_python_version(
            self.id, python_version
        )
        if self._check_results(results):
            return results.json()

    def set_python_version_verbose(self, python_version_verbose):
        """
        Set the Python version verbose for this experiment.

        Args:
            python_version_verbose (str): The verbose Python version.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_python_version_verbose("3.9.7, by Anaconda")
            ```
        """
        results = self._api._client.set_experiment_python_version_verbose(
            self.id, python_version_verbose
        )
        if self._check_results(results):
            return results.json()

    def set_pid(self, pid):
        """
        Set the process ID for this experiment.

        Args:
            pid (str): The OS process ID

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_pid(54238)
            ```
        """
        results = self._api._client.set_experiment_pid(self.id, pid)
        if self._check_results(results):
            return results.json()

    def set_os_type(self, os_type):
        """
        Set the OS type for this experiment.

        Args:
            os_type (str): The OS type.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_os_type("Linux 2.0.1, Ubuntu 16.10")
            ```
        """
        results = self._api._client.set_experiment_os_type(self.id, os_type)
        if self._check_results(results):
            return results.json()

    def set_os(self, os):
        """
        Set the OS for this experiment.

        Args:
            os (str): The OS platform identifier.

        Example:
            ```python linenums="1"
            import comet_ml
            import platform

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_os(platform.platform(aliased=True))
            ```
        """
        results = self._api._client.set_experiment_os(self.id, os)
        if self._check_results(results):
            return results.json()

    def set_os_release(self, os_release):
        """
        Set the OS release for this experiment.

        Args:
            os_release (str): The OS release.

        Example:
            ```python linenums="1"
            import comet_ml
            import platform

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_os_release(platform.uname()[2])
            ```
        """
        results = self._api._client.set_experiment_os_release(self.id, os_release)
        if self._check_results(results):
            return results.json()

    def set_ip(self, ip):
        """
        Set the internet protocol (IP) address for this experiment.

        Args:
            ip (str): The internet protocol address.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_ip("10.0.0.7")
            ```
        """
        results = self._api._client.set_experiment_ip(self.id, ip)
        if self._check_results(results):
            return results.json()

    def set_hostname(self, hostname):
        """
        Set the hostname for this experiment.

        Args:
            hostname (str): The hostname of the computer the experiment ran on.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_hostname("machine.company.com")
            ```
        """
        results = self._api._client.set_experiment_hostname(self.id, hostname)
        if self._check_results(results):
            return results.json()

    def set_gpu_static_info(self, gpu_static_info):
        """
        Set the GPU static info for this experiment.

        Args:
            gpu_static_info (list): list of dicts containing keys
                `gpuIndex`, `name`, `uuid`, `totalMemory`, and `powerLimit`
                and their values.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_gpu_static_info([{
                "gpuIndex": 0,
                "name": "GeForce GTX 950",
                "uuid": "GPU-cb6c1b39-5a56-6d79-8899-3796f23c6425",
                "totalMemory": 2090074112,
                "powerLimit": 110000,
            }, ...])
            ```
        """
        if not isinstance(gpu_static_info, (tuple, list)):
            raise TypeError(API_EXPERIMENT_GPU_STATIC_LIST_EXPECTED_EXCEPTION)
        for items in gpu_static_info:
            if not isinstance(items, dict):
                raise ValueError(
                    API_EXPERIMENT_GPU_STATIC_LIST_OF_DICTS_EXPECTED_EXCEPTION
                )
            if (
                ("gpuIndex" not in items)
                or ("name" not in items)
                or ("uuid" not in items)
                or ("totalMemory" not in items)
                or ("powerLimit" not in items)
            ):
                raise ValueError(
                    API_EXPERIMENT_GPU_STATIC_LIST_OF_DICTS_EXPECTED_EXCEPTION
                )
            if not isinstance(items["gpuIndex"], int):
                raise TypeError("gpuIndex must be an int")
            if not isinstance(items["totalMemory"], int):
                raise TypeError("totalMemory must be an int")
            if not isinstance(items["powerLimit"], int):
                raise TypeError("powerLimit must be an int")
            if not isinstance(items["name"], str):
                raise TypeError("name must be a str")
            if not isinstance(items["uuid"], str):
                raise TypeError("uuid must be a str")

        results = self._api._client.set_experiment_gpu_static_info(
            self.id, gpu_static_info
        )
        if self._check_results(results):
            return results.json()

    def log_additional_system_info(self, key, value):
        """
        Log additional system information for this experiment.

        Args:
            key (str): The name for this system information.
            value (Any): The value of the system information.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            experiment.log_additional_system_info("some name": 42)
            ```
        """
        dict_info = [{"key": key, "value": value}]
        results = self._api._client.log_experiment_system_info(self.id, dict_info)
        if self._check_results(results):
            return results.json()

    def set_network_interface_ips(self, network_interface_ips):
        """
        Set the network interface ips for this experiment.

        Args:
            network_interface_ips (List[str]): Local
                network interfaces

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_network_interface_ips(["127.0.0.1", "192.168.1.100"])
            ```
        """
        results = self._api._client.set_experiment_network_interface_ips(
            self.id, network_interface_ips
        )
        if self._check_results(results):
            return results.json()

    def set_command(self, command_args_list):
        """
        Set the command-line (script and args) for this experiment.

        Args:
            command_args_list (List[str]): Starting with name of script,
                and followed by arguments.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_command(["script.py", "arg1", "arg2", "--flag", "arg3"])
            ```
        """
        results = self._api._client.set_experiment_command(self.id, command_args_list)
        if self._check_results(results):
            return results.json()

    def set_executable(self, executable):
        """
        Set the executable for this experiment.

        Args:
            executable (str): The python executable.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_executable("/usr/bin/python3")
            ```
        """
        results = self._api._client.set_experiment_executable(self.id, executable)
        if self._check_results(results):
            return results.json()

    def set_filename(self, filename):
        """
        Set the path and filename for this experiment.

        Args:
            filename (str): The python path and filename.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_filename("../src/script.py")
            ```
        """
        results = self._api._client.set_experiment_filename(self.id, filename)
        if self._check_results(results):
            # Update local copy too:
            self.file_path = filename
            return results.json()

    def set_installed_packages(self, installed_packages):
        """
        Set the installed Python packages for this experiment.

        Args:
            installed_packages (List[str]): A list of the installed Python packages.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_installed_packages(["comet_ml", "matplotlib"])
            ```
        """
        results = self._api._client.set_experiment_installed_packages(
            self.id, installed_packages
        )
        if self._check_results(results):
            return results.json()

    def set_processor(self, processor):
        """
        Set the processor for this experiment.

        Args:
            processor (str): The processor name.

        Example:
            ```python linenums="1"
            import comet_ml
            import platform

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_processor(platform.processor())
            ```
        """
        results = self._api._client.set_experiment_processor(self.id, processor)
        if self._check_results(results):
            return results.json()

    def set_machine(self, machine):
        """
        Set the machine for this experiment.

        Args:
            machine (str): The machine type.

        Example:
            ```python linenums="1"
            import comet_ml
            import platform

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_machine(platform.machine())
            ```
        """
        results = self._api._client.set_experiment_machine(self.id, machine)
        if self._check_results(results):
            return results.json()

    def log_ram_metrics(
        self, total_ram, used_ram, context=None, step=None, epoch=None, timestamp=None
    ):
        """
        Log an instance of RAM metrics.

        Args:
            total_ram (float): Total RAM available.
            used_ram (float): RAM used.
            context (str): The run context.
            step (int): The current step.
            epoch (int): The current epoch.
            timestamp (int): The current timestamp in millisconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_ram_metrics(1024, 865, "train", 100, 1, 3645346534)
            ```
        """
        results = self._api._client.add_experiment_ram_metrics(
            self.id, total_ram, used_ram, context, step, epoch, timestamp
        )
        if self._check_results(results):
            return results.json()

    def log_gpu_metrics(self, gpu_metrics):
        """
        Log an instance of gpu_metrics.

        Args:
            gpu_metrics (list): A list of dicts with keys:

                - gpuId: required, Int identifier
                - freeMemory: required, Long
                - usedMemory: required, Long
                - gpuUtilization: required, Int percentage utilization
                - totalMemory: required, Long

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_gpu_metrics([{
                "gpuId": 1,
                "freeMemory": 1024,
                "usedMemory": 856,
                "gpuUtilization": 25,
                "totalMemory": 2056,
            }])
            ```
        """
        if not isinstance(gpu_metrics, (list, tuple)):
            raise ValueError(
                API_EXPERIMENT_LOG_GPU_METRICS_LIST_OF_DICTS_EXPECTED_EXCEPTION
            )
        for metric in gpu_metrics:
            if not isinstance(metric, dict):
                raise ValueError(
                    API_EXPERIMENT_LOG_GPU_METRICS_LIST_OF_DICTS_EXPECTED_EXCEPTION
                )
            if (
                ("gpuId" not in metric)
                or ("freeMemory" not in metric)
                or ("usedMemory" not in metric)
                or ("gpuUtilization" not in metric)
                or ("totalMemory" not in metric)
            ):
                raise ValueError(
                    API_EXPERIMENT_LOG_GPU_METRICS_MISSING_PARAMETERS_EXCEPTION
                )
        results = self._api._client.add_experiment_gpu_metrics(self.id, gpu_metrics)
        if self._check_results(results):
            return results.json()

    def log_cpu_metrics(
        self, cpu_metrics, context=None, step=None, epoch=None, timestamp=None
    ):
        """
        Log an instance of cpu_metrics.

        Args:
            cpu_metrics (list): A list of integer percentages, ordered by cpu.
            context (str): A run context.
            step (int): The current step.
            epoch (int): The current epoch.
            timestamp (int): Current time, in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_cpu_metrics([25, 50, 10, 45])
            ```
        """
        if not isinstance(cpu_metrics, (list, tuple)):
            raise ValueError(API_EXPERIMENT_LOG_CPU_METRICS_LIST_EXPECTED_EXCEPTION)
        for metric in cpu_metrics:
            if not isinstance(metric, numbers.Number):
                raise ValueError(
                    API_EXPERIMENT_LOG_CPU_METRICS_LIST_OF_NUMBERS_EXPECTED_EXCEPTION
                )
        results = self._api._client.add_experiment_cpu_metrics(
            self.id, cpu_metrics, context, step, epoch, timestamp
        )
        if self._check_results(results):
            return results.json()

    def log_load_metrics(
        self, load_avg, context=None, step=None, epoch=None, timestamp=None
    ):
        """
        Log an instance of system load metrics.

        Args:
            load_avg (float): The load average.
            context (str): The run context.
            step (int): The current step.
            epoch (int): The current epoch.
            timestamp (int): The current timestamp in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_load_metrics(1.5, "validate", 100, 25, 65364346)
            ```
        """
        results = self._api._client.add_experiment_load_metrics(
            self.id, load_avg, context, step, epoch, timestamp
        )
        if self._check_results(results):
            return results.json()

    def update_status(self):
        """
        Update the status for this experiment. Sends the keep-alive
        status for it in the UI. The return JSON dictionary contains
        the recommended interval to send subsequent `update_status()`
        messages.

        Returns:
            dict: Returns the following dictionary object:

                ```json
                {
                    'isAliveBeatDurationMillis': 10000,
                    'gpuMonitorIntervalMillis': 60000,
                    'cpuMonitorIntervalMillis': 68000
                }
                ```

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.update_status()
            ```
        """
        results = self._api._client.update_experiment_status(self.id)
        if self._check_results(results):
            return results

    def set_start_time(self, start_server_timestamp):
        """
        Set the start time of an experiment.

        Args:
            start_server_timestamp (int): A timestamp in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_start_time(2652656352)
            ```

        Note:
            Time is in milliseconds. If the end time has not been set
            it will automatically be set for 1 second after the start
            time.
        """
        results = self._api._client.set_experiment_start_end(
            self.id, start_server_timestamp, None
        )
        if self._check_results(results):
            metadata = self._api._client.get_experiment_metadata(self.id)
            self._update_from_metadata(metadata)
            return results.json()

    def set_end_time(self, end_server_timestamp):
        """
        Set the end time of an experiment.

        Args:
            end_server_timestamp (int): A timestamp in milliseconds

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.set_end_time(2652656352)
            ```

        Note:
            Time is in milliseconds. If the start time has not
            been set, it will be set to 1 second before the end
            time.
        """
        results = self._api._client.set_experiment_start_end(
            self.id, None, end_server_timestamp
        )
        if self._check_results(results):
            metadata = self._api._client.get_experiment_metadata(self.id)
            self._update_from_metadata(metadata)
            return results.json()

    def log_output(self, output, context=None, stderr=False, timestamp=None):
        """
        Log output line(s).

        Args:
            output (str): String representing standard output or error.
            context (str): The run context.
            stderr (bool): If True, the lines are standard errors
            timestamp (int): The current timestamp in milliseconds

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_output("output line 1\\noutput line 2")
            ```
        """
        results = self._api._client.log_experiment_output(
            self.id, output, context, stderr, timestamp
        )
        if self._check_results(results):
            return results.json()

    def set_name(self, name):
        """
        Set a name for the experiment. Useful for filtering and searching on Comet.ml.
        Will shown by default under the `Other` tab.

        Args:
            name (str): A name for the experiment.
        """
        self.log_other("Name", name)

    def log_other(self, key, value, timestamp=None):
        """
        Set another key/value pair for an experiment.

        Args:
            key (str): The name of the other information.
            value (Any): The value of the other information.
            timestamp (int): optional, the current timestamp in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_other("key", value)
            ```
        """
        results = self._api._client.log_experiment_other(self.id, key, value, timestamp)
        if self._check_results(results):
            if key == "Name":
                self._name = value
            return results.json()

    def log_parameter(
        self,
        parameter: str,
        value: Any,
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Set a parameter name/value pair for an experiment.

        Args:
            parameter (str): The name of the parameter.
            value (Any): The value of the parameter.
            step (int): The current step.
            timestamp (int): The current timestamp in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_parameter("hidden_layer_size", 64)
            ```
        """
        results = self._api._client.log_experiment_parameter(
            self.id, parameter, value, step, timestamp
        )
        if self._check_results(results):
            return results.json()

    def log_parameters(
        self,
        param_dict: Dict[str, Any],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
        nested_support: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Set a dictionary of parameter name/value pairs for an experiment.

        Args:
            param_dict (dict): Dict in the form of {"param_name": value, ...}.
            step (int): The current step.
            timestamp (int): The current timestamp in milliseconds.
            nested_support (bool): Support nested parameters.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_parameters({"learning_rate": 0.12, "layers": 3})
            ```
        """
        results = []
        if nested_support and exclusions.can_parameters_be_flattened(
            param_dict, source=messages.ParameterMessage.source_manual
        ):
            flatten_op_result = flattener.flatten_dict(
                d=param_dict,
                separator=PARAMETERS_DELIMITER,
                max_depth=PARAMETERS_MAX_DEPTH,
            )
            param_dict = flatten_op_result.flattened
            if flatten_op_result.max_depth_limit_reached:
                LOGGER.warning(
                    LOG_PARAMS_MAX_DEPTH_REACHED, param_dict, PARAMETERS_MAX_DEPTH
                )
            if flatten_op_result.has_nested_dictionary():
                self.log_other("hasNestedParams", True)

        for key in param_dict:
            value = param_dict[key]
            results.append(
                self.log_parameter(key, value, step=step, timestamp=timestamp)
            )
        return results

    def delete_parameter(self, parameter: str) -> bool:
        """
        Delete parameter from an experiment.

        Args:
            parameter: string, parameter name

        Example:

        ```python
        >>> api_experiment.delete_parameter("learning_rate")
        ```
        """
        current_backend_version = self._api._client.get_api_backend_version()
        check_result = get_config().has_api_experiment_delete_parameters_enabled(
            current_backend_version
        )
        if not check_result.feature_supported:
            raise CometException(
                API_EXPERIMENT_DELETE_PARAMETERS_UNSUPPORTED_BACKEND_VERSION_ERROR
                % check_result.min_backend_version_supported
            )

        results = self._api._client.delete_experiment_parameter(self.id, parameter)
        if self._check_results(results):
            return True
        return False

    def delete_parameters(self, parameters: List[str]) -> bool:
        """
        Delete parameter from an experiment.

        Args:
            parameters: list of strings, parameter names

        Example:

        ```python
        >>> api_experiment.delete_parameters(["learning_rate", "layers"])
        ```
        """
        current_backend_version = self._api._client.get_api_backend_version()
        check_result = get_config().has_api_experiment_delete_parameters_enabled(
            current_backend_version
        )
        if not check_result.feature_supported:
            raise CometException(
                API_EXPERIMENT_DELETE_PARAMETERS_UNSUPPORTED_BACKEND_VERSION_ERROR
                % check_result.min_backend_version_supported
            )

        results = self._api._client.delete_experiment_parameters(self.id, parameters)
        if self._check_results(results):
            return True
        return False

    def log_metric(
        self,
        metric: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        timestamp: Optional[int] = None,
    ):
        """
        Set a metric name/value pair for an experiment.

        Args:
            metric (str): The name of the metric.
            value (Any): The value of the metric.
            step (int, optional): The current step.
            epoch (int, optional): The current epoch.
            timestamp (int, optional): The current timestamp in seconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_metric("loss", 0.698)
            ```
        """
        results = self._api._client.log_experiment_metric(
            experiment_key=self.id,
            metric=metric,
            value=value,
            step=step,
            epoch=epoch,
            timestamp=timestamp,
        )
        if self._check_results(results):
            return results.json()

    def log_metrics(
        self,
        metric_dict: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        timestamp: Optional[int] = None,
    ):
        """
        Set a dictionary of metric name/value pairs for an experiment.

        Args:
            metric_dict (dict): A dict in the form of {"metric_name": value, ...}.
            step (int): The current step.
            epoch (int): The current epoch.
            timestamp (int): The current timestamp in milliseconds.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_metrics({"loss": 0.698, "accuracy": 0.12})
            ```
        """
        # flatten dictionary if appropriate
        flatten_op_result = flatten_dict(
            d=metric_dict, separator=METRICS_DELIMITER, max_depth=METRICS_MAX_DEPTH
        )
        dic = flatten_op_result.flattened
        if flatten_op_result.max_depth_limit_reached:
            LOGGER.warning(
                logging.WARNING,
                LOG_METRICS_MAX_DEPTH_REACHED,
                metric_dict,
                METRICS_MAX_DEPTH,
            )

        results = []
        for key in dic:
            value = dic[key]
            results.append(
                self.log_metric(
                    metric=key, value=value, step=step, epoch=epoch, timestamp=timestamp
                )
            )

        return results

    def log_html(self, html, clear=False, timestamp=None):
        """
        Set, or append onto, an experiment's HTML.

        Args:
            html (str): The HTML text to associate with this experiment.
            clear (bool): If True, clear any previously logged HTML.
            timestamp (int): The current time (in milliseconds).

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_html("<b>Hello!</b>")
            ```
        """
        results = self._api._client.log_experiment_html(self.id, html, clear, timestamp)
        if self._check_results(results):
            return results.json()

    def add_tags(self, tags: List[str]) -> Dict[str, Any]:
        """
        Append onto an experiment's list of tags.

        Args:
            tags (List[str]): A list of tags (strings).

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.add_tags(["successful", "best"])
            ```
        """
        validator = TagsValidator(
            tags, method_name=self.add_tags.__name__, class_name=self.__class__.__name__
        )
        result = validator.validate()

        if not result:
            validator.print_result(logger=LOGGER)
            raise ValidationError(EXPERIMENT_LOG_TAG_VALIDATION_ERROR)

        results = self._api._client.add_experiment_tags(self.id, tags)
        return results.json()

    def delete_tags(self, tags: List[str]) -> Dict[str, Any]:
        """
        Delete from an experiment the list of tags.

        Args:
            tags (List[str]): A list of tags (strings).

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.delete_tags(["successful", "best"])
            ```
        """
        current_backend_version = self._api._client.get_api_backend_version()
        check_result = get_config().has_api_experiment_delete_tags_enabled(
            current_backend_version
        )
        if not check_result.feature_supported:
            raise CometException(
                API_EXPERIMENT_DELETE_TAGS_UNSUPPORTED_BACKEND_VERSION_ERROR
                % check_result.min_backend_version_supported
            )

        response = self._api._client.delete_experiment_tags(self.id, tags)
        return response.json()

    def add_tag(self, tag: str) -> Dict[str, Any]:
        """
        Append onto an experiment's list of tags.

        Args:
            tag (str): A tag.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.add_tag("baseline")
            ```
        """
        validator = TagValidator(
            tag, method_name=self.add_tag.__name__, class_name=self.__class__.__name__
        )
        result = validator.validate()

        if not result:
            validator.print_result(logger=LOGGER)
            raise ValidationError(EXPERIMENT_LOG_TAG_VALIDATION_ERROR)

        results = self._api._client.add_experiment_tags(self.id, [tag])
        return results.json()

    def log_asset(
        self,
        filename: str,
        step: Optional[int] = None,
        name: Optional[str] = None,
        overwrite: Optional[bool] = None,
        context: Optional[str] = None,
        ftype: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Upload an asset to an experiment.

        Args:
            filename (str): The name of the asset file to upload.
            step (int): The current step.
            name (str): A custom name can be provided to be displayed on the assets
                tab. If not provided, the filename from the file argument will be used if it is a path.
            overwrite (bool): If True, overwrite any previous upload.
            context (str): The current context (e.g., "train" or "test").
            ftype (str): the type of asset (e.g., "image", "histogram_combined_3d",
                "image", "audio", or "video").
            metadata (dict): a JSON object to attach to image.

        Note: Don't delete the file until upload is complete

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_asset("histogram.json", ftype="histogram_compbined_3d")
            ```
        """
        filename = expand_user_home_path(filename)
        results = self._api._client.log_experiment_asset(
            self.id,
            file_data=filename,
            step=step,
            overwrite=overwrite,
            context=context,
            ftype=ftype,
            metadata=metadata,
            file_name=name,
        )

        if self._check_results(results):
            return results.json()

    def delete_asset(self, asset_id):
        """
        Delete an experiment's asset.

        Args:
            asset_id (str): The asset id of the asset to delete.
        """
        results = self._api._client.delete_experiment_asset(self.id, asset_id)
        return results

    def log_curve(self, name, x, y, overwrite=False, step=None):
        """
        Log timeseries data.

        Args:
            name (str): Name of data.
            x (list): List of x-axis values.
            y (list): List of y-axis values.
            overwrite (bool): If True, overwrite previous log.
            step (int): The step value

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            experiment.log_curve("my curve", x=[1, 2, 3, 4, 5],
                                             y=[10, 20, 30, 40, 50])
            ```
        """
        validator = CurveDataValidator(name=name, x=x, y=y)
        result = validator.validate()
        if result.failed():
            raise ValueError(
                EXPERIMENT_LOG_CURVE_VALIDATION_ERROR % result.failure_reasons
            )

        data = {"x": list(x), "y": list(y), "name": name}
        results = self._api._client.log_experiment_asset(
            self.id,
            file_data=name,
            step=step,
            overwrite=overwrite,
            context=None,
            ftype="curve",
            metadata=None,
            file_content=data,
        )
        if self._check_results(results):
            return results.json()

    def log_video(
        self,
        filename: Union[str, IO],
        name: Optional[str] = None,
        overwrite: bool = False,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Logs the video to Comet. Videos are displayed on the assets tab in Comet and support the
        following formats: MP4, MOV, WMV, and GIF.

        Args:
            filename (str): The path to the video file.
            name (str): A custom name can be provided to be displayed on the assets
                tab. If not provided, the filename from the file argument will be used if it is a path.
            overwrite (bool): If another video with the same name exists, it will be
                overwritten if overwrite is set to True.
            step (int): This is used to associate the video asset with a specific step.
            epoch (int): Used to associate the asset to a specific epoch.
            context (str): The current context (e.g., "train" or "test")
            metadata (dict): Additional custom metadata can be associated with the logged video.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_video("video.mp4")
            ```
        """
        filename = expand_user_home_path(filename)
        results = self._api._client.log_experiment_video(
            experiment_key=self.id,
            filename=filename,
            video_name=name,
            overwrite=overwrite,
            step=step,
            epoch=epoch,
            context=context,
            metadata=metadata,
        )
        if self._check_results(results):
            return results.json()

    def log_image(
        self,
        filename: str,
        image_name: Optional[str] = None,
        step: Optional[int] = None,
        overwrite: Optional[bool] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Upload an image asset to an experiment.

        Args:
            filename (str): The name of the image file to upload.
            image_name (str): The name of the image.
            step (int): The current step.
            overwrite (bool): If True, overwrite any previous upload.
            context (str): The current context (e.g., "train" or "test").
            metadata (dict): Some additional data to attach to the image.
                Must be a JSON-compatible dict.

        Note: don't delete the file until upload is complete

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_image("image.png", "Weights")
            ```
        """
        filename = expand_user_home_path(filename)
        results = self._api._client.log_experiment_image(
            experiment_key=self.id,
            filename=filename,
            image_name=image_name,
            step=step,
            overwrite=overwrite,
            context=context,
            metadata=metadata,
        )
        if self._check_results(results):
            return results.json()

    def log_table(
        self,
        filename: str,
        tabular_data: Optional[Any] = None,
        headers: Union[Sequence[str], bool] = False,
        **format_kwargs: Any,
    ) -> Optional[Dict[str, str]]:
        """
        Log tabular data, including data, csv files, tsv files, and Pandas dataframes.

        Args:
            filename (str): A filename ending in ".csv", or ".tsv" (for tablular
                data) or ".json", ".csv", ".md", or ".html" (for Pandas dataframe data).
            tabular_data (Any): Data that can be interpreted as 2D tabular data
                or a Pandas dataframe.
            headers (bool | list): If True, will add column headers automatically
                if tabular_data is given; if False, no headers will be added; if list
                then it will be used as headers. Only useful with tabular data (csv, or tsv).
            format_kwargs (Any): When passed a Pandas dataframe
                these keyword arguments are used in the conversion to "json", "csv",
                "md", or "html". See Pandas Dataframe conversion methods (like `to_json()`)
                for more information.

        See also:

        * [pandas.DataFrame.to_json documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)
        * [pandas.DataFrame.to_csv documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
        * [pandas.DataFrame.to_html documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html)
        * [pandas.DataFrame.to_markdown documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html)

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api_experiment = comet_ml.APIExperiment(previous_experiment='EXPERIMENT-KEY')

            api_experiment.log_table("vectors.tsv",
                                     [["one", "two", "three"],
                                     [1, 2, 3],
                                     [4, 5, 6]])

            api_experiment.log_table("dataframe.json", pandas_dataframe)
            ```
        """
        filename = expand_user_home_path(filename)
        if isinstance(filename, str) and os.path.isfile(filename):
            if headers is not False:
                LOGGER.info(API_EXPERIMENT_LOG_TABLE_HEADERS_IGNORED_INFO)

            results = self._api._client.log_experiment_asset(
                self.id, file_data=filename, ftype="asset"
            )

            if self._check_results(results):
                return results.json()

        # Filename is not a file
        if tabular_data is None:
            raise TypeError(API_EXPERIMENT_LOG_TABLE_MISSING_TABULAR_DATA_EXCEPTION)

        # Tabular-data is not None
        if not isinstance(filename, str):
            raise ValueError(API_EXPERIMENT_LOG_TABLE_WRONG_FILENAME_EXCEPTION)

        converted = convert_log_table_input_to_io(
            filename=filename,
            tabular_data=tabular_data,
            headers=headers,
            format_kwargs=format_kwargs,
        )

        if not converted:
            # TODO: Raise error instead?
            return None

        fp, asset_type = converted

        results = self._api._client.log_experiment_asset(
            self.id,
            file_data=fp,
            ftype=asset_type,
            file_name=filename,
        )

        if self._check_results(results):
            return results.json()

    def _raise_if_old_backend(self, method_name: str, minimal_version: str) -> None:
        version = self._api._client.get_api_backend_version()
        if version.compare(minimal_version) < 0:
            raise CometException(
                API_EXPERIMENT_WRONG_BACKEND_VERSION_FOR_METHOD_EXCEPTION
                % (
                    method_name,
                    minimal_version,
                )
            )


class API(object):
    """
    The API class is used as a Python interface to the Comet.ml Python
    API.

    You can use an instance of the API() class to quickly and easily
    access all of your logged information at [comet](https://www.comet.com),
    including metrics, parameters, tags, and assets.

    Example calls to get workspace, project, and experiment data:

    * API.get(): gets all of your personal workspaces
    * API.get(WORKSPACE): gets all of your projects from WORKSPACE
    * API.get(WORKSPACE, PROJECT_NAME): get all APIExperiments in WORKSPACE/PROJECT
    * API.get_experiment(WORKSPACE, PROJECT_NAME, EXPERIMENT_KEY): get an APIExperiment
    * API.get_experiment("WORKSPACE/PROJECT_NAME/EXPERIMENT_KEY"): get an APIExperiment
    * API.get_experiments(WORKSPACE): get all APIExperiments in WORKSPACE
    * API.get_experiments(WORKSPACE, PROJECT_NAME): get all APIExperiments in WORKSPACE/PROJECT
    * API.get_experiments(WORKSPACE, PROJECT_NAME, PATTERN): get all APIExperiments in WORKSPACE/PROJECT/PATTERN

    Example:
        ```python linenums="1"
        import comet_ml

        comet_ml.login()
        api = comet_ml.API()

        ## Return all of my workspace names in a list:
        api.get()

        ## Get an APIExperiment:
        experiment = api.get("cometpublic/comet-notebooks/example 001")

        ## Get metrics:
        experiment.get_metrics("train_accuracy")
        ```

        The API instance also gives you access to the low-level Python API function
        calls:

        ```python linenums="1"
        api.delete_experiment(experiment_key)
        ```

    For more usage examples, see [Comet Python API examples](../Comet-Python-API/).
    """

    def __init__(self, api_key=None, cache=True, version="v2"):
        """
        Application Programming Interface to the Comet Python interface.

        Args:
            api_key (str): Your private COMET_API_KEY.
            cache (bool): Whether to cache on values or not.
            version (str): The version of the REST API to use.

        Note:
            api_key may be defined in environment (COMET_API_KEY)
            or in a .comet.config file.

        Example:
            ```python
            import comet_ml
            from comet_ml.api import API

            comet_ml.login()
            api = API()
            api.get("my-workspace")
            ```
        """
        self.config = get_config()
        self.api_key = get_api_key(api_key, self.config)
        self._client = get_rest_api_client(
            version,
            api_key=self.api_key,
            use_cache=cache,
            headers={"X-COMET-SDK-SOURCE": "API"},
        )

    @property
    def server_url(self):
        return self._client.server_url

    def _check_results(self, results):
        return results is not None

    def update_cache(self):
        """
        Deprecated: Use API.clear_cache()
        """
        LOGGER.warning(API_UPDATE_CACHE_DEPRECATED_WARNING)
        self.clear_cache()

    def clear_cache(self):
        """
        Used when cache is on, but you have added/changed
        data outside of this API instance.

        Note:
            You could also just start with no cache.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API(cache=False)
            ```

            Or, if you had started with cache, turn it off:

            ```python
            >>> api = API(cache=True)
            >>> api.do_cache(False)
            ```
        """
        self._client.reset()

    def get(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment: Optional[str] = None,
    ):
        """
        Get the following items:

        * list of workspace names, given no arguments
        * list of project names, given a workspace name
        * list of experiment names/keys, given workspace and project names
        * an experiment, given workspace, project, and experiment name/key

        Args:
            workspace (str): Workspace name.
            project_name (str): Project name.
            experiment (str): Experiment key.

        Note:
            `workspace`, `project_name`, and `experiment` can also be given as a single
            string, delimited with a slash.
        """
        # First, we check for delimiters:
        if workspace is not None and "/" in workspace:
            if project_name is not None:
                raise SyntaxError(API_GET_SLASH_WORKSPACE_AND_PROJECT_EXCEPTION)

            workspace, project_name = workspace.split("/", 1)
        if project_name is not None and "/" in project_name:
            if experiment is not None:
                raise SyntaxError(API_GET_SLASH_PROJECT_AND_KEY_EXCEPTION)

            project_name, experiment = project_name.split("/", 1)
        # Now, return the appropriate item:
        if workspace is None:
            return self.get_workspaces()
        elif project_name is None:
            return self.get_projects(workspace)
        elif experiment is None:
            return self.get_experiments(workspace, project_name)
        else:
            api_experiment = self.get_experiment(workspace, project_name, experiment)
            if api_experiment is None:
                LOGGER.warning(API_EXPERIMENT_NOT_FOUND_MSG)
            return api_experiment

    def query(self, workspace, project_name, query, archived=False):
        """
        Perform a query on a workspace/project to find matching
        APIExperiment. Queries are composed of

        ```python
        ((QUERY-VARIABLE OPERATOR VALUE) & ...)

        # or:

        (QUERY-VARIABLE.METHOD(VALUE) & ...)
        ```

        where:

        `QUERY-VARIABLE` is `Environment(NAME)`, `Metric(NAME)`, `Parameter(NAME)`,
        `Other(NAME)`, `Metadata(NAME)`, or `Tag(VALUE)`.

        `OPERATOR` is any of the standard mathematical operators
        `==`, `<=`, `>=`, `!=`, `<`, `>`.

        `METHOD` is `between()`, `contains()`, `startswith()`, or `endswith()`.

        You may also place the bitwise `~` not operator in front of an expression
        which means to invert the expression. Use `&` to combine additional
        criteria. Currently, `|` (bitwise or) is not supported.

        `VALUE` can be any query type, includeing `string`, `boolean`, `double`,
        `datetime`, or `timenumber` (number of seconds). `None` and `""` are special
        values that mean `NULL` and `EMPTY`, respectively. Use
        `API.get_query_variables(WORKSPACE, PROJECT_NAME)` to see query variables
        and types for a project.

        When using `datetime`, be aware that the backend is using UTC datetimes. If you
        do not receive the correct experiments via a datetime query, please check with
        the web UI query builder to verify timezone of the server.

        `query()` returns a list of matching `APIExperiments()`.

        Args:
            workspace (str): The name of the workspace
            project_name (str): The name of the project
            query (Any): A query expression (see below)
            archived (bool): Query the archived experiments if True

        Example:
            ```python linenums="1"
            import comet_ml
            from comet_ml.query import (
                Environment,
                Metric,
                Parameter,
                Other,
                Metadata,
                Tag
            )

            comet_ml.login()
            api = comet_ml.API()

            # Find all experiments that have an acc metric value > .98:
            api.query("workspace", "project", Metric("acc") > .98)

            # Find all experiments that have a loss metric < .1 and
            # a learning_rate parameter value >= 0.3:
            loss = Metric("loss")
            lr = Parameter("learning_rate")
            query = ((loss < .1) & (lr >= 0.3))
            api.query("workspace", "project", query)

            # Find all of the experiments tagged "My simple tag":
            tagged = Tag("My simple tag")
            api.query("workspace", "project", tagged)

            # Find all experiments started before Sept 24, 2019 at 5:00am:
            q = Metadata("start_server_timestamp") < datetime(2019, 9, 24, 5)
            api.query("workspace", "project", q)

            # Find all experiments lasting more that 2 minutes (in seconds):
            q = Metadata("duration") > (2 * 60)
            api.query("workspace", "project", q)
            ```

        Note:
            * Use `~` for `not` on any expression
            * Use `~QUERY-VARIABLE.between(2,3)` for values not between 2 and 3
            * Use `(QUERY-VARIABLE == True)` for truth
            * Use `(QUERY-VARIABLE == False)` for not true
            * Use `(QUERY-VARIABLE == None)` for testing null
            * Use `(QUERY-VARIABLE != None)` or `~(QUERY-VARIABLE == None)` for testing not null
            * Use `(QUERY-VARIABLE == "")` for testing empty
            * Use `(QUERY-VARIABLE != "")` or `~(QUERY-VARIABLE == "")` for testing not empty
            * Use Python's datetime(YEAR, MONTH, DAY, HOUR, MINUTE, SECONDS) for comparing datetimes, like
                `Metadata("start_server_timestamp")` or `Metadata("end_server_timestamp")`
            * Use seconds for comparing timenumbers, like `Metadata("duration")`
            * Use `API.get_query_variables(WORKSPACE, PROJECT_NAME)` to see query variables
                and types.

            Do not use 'and', 'or', 'not', 'is', or 'in'. These
            are logical operators and you must use mathematical
            operators for queries. For example, always use '=='
            where you might usually use 'is'.
        """
        columns = self._client.get_project_columns(workspace, project_name)
        if isinstance(query, QueryVariable):
            raise Exception(API_QUERY_INVALID_QUERY_EXPRESSION_EXCEPTION)
        if not isinstance(query, QueryExpression):
            raise Exception(API_QUERY_MISSING_QUERY_EXPRESSION_EXCEPTION)

        try:
            predicates = query.get_predicates(columns)
        except QueryException as exc:
            LOGGER.info(API_QUERY_ERROR_INFO, exc)
            return []
        results = self._client.query_project(
            workspace, project_name, predicates, archived
        )
        if results:
            results_json = results.json()
            return [
                self._get_experiment(workspace, project_name, key)
                for key in results_json["experimentKeys"]
            ]

    def get_archived_experiment(self, workspace, project_name, experiment):
        """
        Get a single archived APIExperiment by workspace, project, experiment.

        Args:
            workspace (str): Workspace name
            project_name (str): Project name
            experiment (str): Experiment key
        """
        return self._get_experiment(workspace, project_name, experiment)

    def get_experiment(self, workspace, project_name, experiment):
        """
        Get a single APIExperiment by workspace, project, experiment.


        Args:
            workspace (str): Workspace name
            project_name (str): Project name
            experiment (str): Experiment key
        """
        return self._get_experiment(workspace, project_name, experiment)

    def get_experiment_by_id(self, experiment):
        LOGGER.warning(API_EXPERIMENT_BY_ID_DEPRECATED_WARNING)
        return self.get_experiment_by_key(experiment)

    def get_experiment_by_key(self, experiment_key):
        """
        Get an APIExperiment by experiment key.

        Args:
            experiment_key (str): Experiment key
        """
        try:
            metadata = self._get_experiment_metadata(experiment_key)
        except CometRestApiException as exc:
            # It doesn't exist; older backends return 400, newer 404
            if exc.response.status_code in [400, 404]:
                return None
            raise exc

        return self._get_experiment(
            metadata["workspaceName"], metadata["projectName"], experiment_key
        )

    def get_archived_experiments(self, workspace, project_name=None, pattern=None):
        """
        Get archived APIExperiments by workspace, workspace + project, or
        workspace + project + regular expression pattern.

        Args:
            workspace (str): Workspace name
            project_name (str): The project name, if ommitted all projects will
                be searched
            pattern (str): Regex pattern to apply on the experiment name or the experiment key
        """
        return list(
            self._gen_experiments(workspace, project_name, pattern, archived=True)
        )

    def get_experiments(
        self,
        workspace,
        project_name=None,
        pattern=None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """
        Get APIExperiments by workspace, workspace + project, or
        workspace + project + regular expression pattern.

        Args:
            workspace (str): Workspace name
            project_name (str): The project name, if omitted all projects will
                be searched
            pattern (str): Regex pattern to apply on the experiment name or the experiment key
            page (int, optional): Page number for pagination (1-indexed). If provided, page_size is required.
            page_size (int, optional): Number of experiments per page. Required when page is specified.
            sort_by (str, optional): Field to sort by. Must be "startTime" or "endTime" if provided.
            sort_order (str, optional): Sort direction. Must be "asc" or "desc" if provided.
                                      Required when page, page_size, and sort_by are all specified.

        Returns:
            List of APIExperiment objects

        Raises:
            ValueError: If pagination or sorting parameters are invalid
        """
        return list(
            self._gen_experiments(
                workspace,
                project_name,
                pattern,
                archived=False,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        )

    def gen_experiments(
        self,
        workspace,
        project_name=None,
        pattern=None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """
        Get APIExperiments by workspace, workspace + project, or
        workspace + project + regular expression pattern.

        Args:
            workspace (str): Workspace name
            project_name (str): The project name, if omitted all projects will
                be searched
            pattern (str): Regex pattern to apply on the experiment name or the experiment key
            page (int, optional): Page number for pagination (1-indexed). If provided, page_size is required.
            page_size (int, optional): Number of experiments per page. Required when page is specified.
            sort_by (str, optional): Field to sort by. Must be "startTime" or "endTime" if provided.
            sort_order (str, optional): Sort direction. Must be "asc" or "desc" if provided.
                                      Required when page, page_size, and sort_by are all specified.

        Yields:
            APIExperiment objects

        Raises:
            ValueError: If pagination or sorting parameters are invalid
        """
        return self._gen_experiments(
            workspace,
            project_name,
            pattern,
            archived=False,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_metrics_df(
        self,
        experiment_keys: List[str],
        metrics: List[str],
        x_axis: str = "step",
        interpolate: bool = True,
    ) -> Optional["pd.DataFrame"]:
        """
        Get a DataFrame of multiple metrics from a set of experiments across a specified x-axis.

        Args:
            experiment_keys (List[str]): A list of experiment keys.
            metrics (List[str]): a list of metric names.
            x_axis ("step" | "epoch" | "duration"): a specified field that metric values will be based on.
            interpolate (bool): whether to apply linear interpolation to numerical columns or not.

        Returns:
            pd.DataFrame: A Pandas DataFrame of metrics from a set of experiments across a specified x-axis

        Note: You should have Pandas installed to get this method to work.
        """
        try:
            import pandas
        except ImportError:
            LOGGER.error(PANDAS_DATAFRAME_IS_REQUIRED)

            return None

        validator = MethodParametersTypeValidator(
            method_name=self.get_metrics_df.__name__, class_name=self.__class__.__name__
        )

        validator.add_list_parameter(
            experiment_keys, name="experiment_keys", allow_empty=True
        )
        validator.add_list_parameter(metrics, name="metrics", allow_empty=True)
        validator.add_str_parameter(
            x_axis,
            name="x_axis",
            possible_values=["step", "epoch", "duration"],
            allow_empty=False,
        )
        validator.add_bool_parameter(interpolate, name="interpolate", allow_empty=False)

        if not validator.validate():
            validator.throw_validation_error()

        multi_metrics = self.get_metrics_for_chart(
            experiment_keys=experiment_keys, metrics=metrics
        )

        if not experiment_keys or not metrics or not multi_metrics:
            return None

        column_order = ["experiment_key", "experiment_name", x_axis, *metrics]

        df = get_dataframe_from_multi_metrics(
            multi_metrics=multi_metrics, x_axis=x_axis, columns=column_order
        )

        if not interpolate or df.empty:
            return df

        return interpolate_metric_dataframe(
            df=df, x_axis=x_axis, metrics=metrics, columns=column_order
        )

    # Private methods:

    def _get_experiment(self, workspace, project_name, experiment):
        # type: (str, str, str) -> Optional[APIExperiment]
        try:
            metadata = self._get_experiment_metadata(experiment)
        except CometRestApiException as exc:
            # It doesn't exist; older backends return 400, newer 404
            if exc.response.status_code in [400, 404]:
                # try via name
                metadata = self._get_experiment_metadata_by_name(
                    workspace, project_name, experiment
                )
            else:
                raise
        if metadata is None:
            return None
        return APIExperiment(api=self, metadata=metadata)

    def _gen_experiments(
        self,
        workspace,
        project_name=None,
        pattern=None,
        archived=False,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """
        Private method to generate APIExperiments by workspace, workspace + project, or
        workspace + project + regular expression pattern.

        Args:
            workspace (str): Workspace name
            project_name (str, optional): The project name, if omitted all projects will
                be searched
            pattern (str, optional): Regex pattern to apply on the experiment name or the experiment key
            archived (bool): Whether to return archived experiments (default: False)
            page (int, optional): Page number for pagination (1-indexed). If provided, page_size is required.
            page_size (int, optional): Number of experiments per page. Required when page is specified.
            sort_by (str, optional): Field to sort by. Must be "startTime" or "endTime" if provided.
            sort_order (str, optional): Sort direction. Must be "asc" or "desc" if provided.
                                      Required when page, page_size, and sort_by are all specified.

        Yields:
            APIExperiment objects

        Raises:
            ValueError: If pagination or sorting parameters are invalid
        """
        if project_name is None:
            if pattern is not None:
                raise ValueError(API_MISSING_PROJECT_IN_PATTERN_EXCEPTION)
            # Return all experiments in a workspace:
            for project_name in self.get_projects(workspace):
                for exp in self._gen_experiments(
                    workspace, project_name, archived=archived
                ):
                    yield exp
            return
        elif pattern is None:
            experiments = self._get_project_experiments(
                workspace,
                project_name,
                archived=archived,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            for metadatum in experiments.values():
                yield APIExperiment(api=self, metadata=metadatum)
            return
        else:
            experiments = self._get_project_experiments(
                workspace,
                project_name,
                use_names=False,
                archived=archived,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            if experiments is None:
                raise ValueError(
                    API_INVALID_WORKSPACE_PROJECT_EXCEPTION % (workspace, project_name)
                )
            for metadata in experiments.values():
                if re.match(pattern, metadata["experimentKey"]) or (
                    "experimentName" in metadata
                    and metadata["experimentName"] is not None
                    and re.match(pattern, metadata["experimentName"])
                ):
                    yield APIExperiment(api=self, metadata=metadata)

            return

    def _get_experiment_metadata(self, experiment_key):
        return self._client.get_experiment_metadata(experiment_key)

    def _get_experiment_metadata_by_name(self, workspace, project_name, experiment):
        # type: (str, str, str) -> Any
        try:
            experiments = self._get_project_experiments(
                workspace,
                project_name,
                use_names=True,
            )
        except NotFound:
            return None
        if experiment in experiments:
            return experiments[experiment]
        else:
            return None

    def _get_project_experiments(
        self,
        workspace,
        project_name,
        use_names=False,
        archived=False,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
        """
        Private method to get project experiments with pagination and sorting support.

        Args:
            workspace (str): Workspace name
            project_name (str): The project name
            use_names (bool): Whether to use experiment names as keys (default: False)
            archived (bool): Whether to return archived experiments (default: False)
            page (int, optional): Page number for pagination (1-indexed). If provided, page_size is required.
            page_size (int, optional): Number of experiments per page. Required when page is specified.
            sort_by (str, optional): Field to sort by. Must be "startTime" or "endTime" if provided.
            sort_order (str, optional): Sort direction. Must be "asc" or "desc" if provided.
                                      Required when page, page_size, and sort_by are all specified.

        Returns:
            dict: A dictionary mapping experiment names or keys to experiment metadata dictionaries.

        Raises:
            ValueError: If pagination or sorting parameters are invalid
        """
        # Get the project details:
        project_json = self._client.get_project_experiments(
            workspace,
            project_name,
            archived=archived,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        if project_json is None:
            return
        if use_names:
            experiments = {
                metadata["experimentName"]: metadata
                for metadata in project_json["experiments"]
                if metadata["experimentName"] is not None
            }
        else:
            experiments = {
                metadata["experimentKey"]: metadata
                for metadata in project_json["experiments"]
            }
        return experiments

    def _get_url_server(self, version=None):
        """
        Returns the URL server for this version of the API.
        """
        return self._client.server_url

    def _create_experiment(
        self, workspace, project_name="general", experiment_name=None
    ):
        """
        Create an experiment and return its associated APIExperiment.
        """
        return APIExperiment(
            api=self,
            workspace=workspace,
            project_name=project_name,
            experiment_name=experiment_name,
        )

    def _get_metrics_name(self, workspace, project_name):
        metric_names = []
        query_vars = self.get_query_variables(workspace, project_name)
        for var in query_vars:
            if isinstance(var, Metric):
                metric_names.append(var.name)
        return metric_names

    ## ---------------------------------------------------------
    # Public Read Methods
    ## ---------------------------------------------------------

    def get_account_details(self) -> Dict[str, str]:
        """
        Return the username and the default workspace name for the
        authorized user.

        Returns:
            dict: Returns dictionary object of the format:

                ```python
                {
                'userName': 'USERNAME',
                'defaultWorkspaceName': 'WORKSPACE',
                }
                ```
        """
        return self._client.get_account_details()

    def get_workspaces(self) -> Optional[List[str]]:
        """
        Return a list of names of the workspaces for this user.
        """
        results = self._client.get_workspaces()
        if self._check_results(results):
            return results["workspaceNames"]

    def get_projects(self, workspace):
        """
        Return the details of the projects in a workspace.

        Args:
            workspace (str): The name of the workspace

        Returns:
            list: List of project details in workspace.
        """
        return self._client.get_projects(workspace)

    def get_project(self, workspace, project_name):
        # type: (str, str) -> Any
        """
        Return the details of a project in a workspace.

        Args:
            workspace (str): The name of the workspace
            project_name (str): The name of the project

        Returns:
            dict: Dict of project details if the workspace/project
                exists, otherwise None.

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            print(api.get_project("workspace", "project-name"))
            ```
        """
        try:
            retval = self._client.get_project(workspace, project_name)
        except NotFound:
            retval = None
        return retval

    def get_project_by_id(self, project_id):
        """
        Return the details of a project given its project id.

        Args:
            project_id (str): The ID of the project

        Returns:
            dict: Dict of project details if the project_id exists,
                otherwise None.

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            print(api.get_project_by_id("2727432637263"))
            ```
        """
        try:
            retval = self._client.get_project_by_id(project_id)
        except NotFound:
            retval = None
        return retval

    def get_project_notes(self, workspace, project_name):
        """
        Get the notes of a project.

        Args:
            workspace (str): The name of the workspace
            project_name (str): The name of the project

        Returns:
            str: Project notes

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            notes = api.get_project_notes("my-workspace", "my-project")

            print(notes)
            ```
        """
        project_json = self.get_project(workspace, project_name)
        if project_json:
            project_id = project_json["projectId"]
            return self._client.get_project_notes_by_id(project_id)
        else:
            raise ValueError(
                API_GET_PROJECT_NOTES_UNKNOWN_PROJECT_EXCEPTION
                % (project_name, workspace)
            )

    def get_query_variables(self, workspace, project_name):
        # type: (str, str) -> List[Union[QueryVariable, Tag]]
        """
        Return the query variables of a project in a workspace. Used
        with `API.query()`.

        Args:
            workspace (str): The name of the workspace
            project_name (str): The name of the project

        Returns:
            list: Objects used in forming queries, like:

                ```python
                [Metadata('user_name'),
                Metadata('start_server_timestamp'),
                Tag('my_tag'),
                ...]
                ```
        """
        columns = self._client.get_project_columns(workspace, project_name)
        if columns:
            return make_query_vars(columns)
        else:
            return []

    def get_artifact_list(self, workspace, artifact_type=None):
        # type: (str, Optional[str]) -> Dict[str, Any]
        """
        Return the list of artifacts in a given workspace. Could be optionally filtered by a
        specific type.

        Args:
            workspace (str): The name of the workspace
            artifact_type (str): If provided only returns Artifacts with the given type

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            artifacts = api.get_artifact_list("demo")

            print(artifacts)
            ```
        """
        return self._client.get_artifact_list(workspace, artifact_type)

    def get_artifact_details(
        self, workspace=None, artifact_name=None, artifact_id=None
    ):
        # type: (Optional[str], Optional[str], Optional[str]) -> Dict[str, Any]
        """
        Returns the details of a single artifact identified either by the workspace name + the artifact name or by its unique artifact ID.

        Args:
            workspace (str): The name of the workspace
            artifact_name (str): The name of the artifact
            artifact_id (str): The unique ID of the artifact, for example `6194e719-f596-48e7-8cca-8530c16dd007`

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            artifact_details = api.get_artifact_details("demo", "demo-artifact")
            print(artifact_details)
            ```

            The print statement above prints the following output:
            ```json
            {'artifactId': '6194e719-f596-48e7-8cca-8530c16dd007',
                'project': 'demo-artifacts',
                'type': 'dataset',
                'name': 'demo-artifact',
                'description': None,
                'latestVersion': '2.0.0',
                'tags': [],
                'isPublic': False,
                'emoji': None,
                'sizeInBytes': 21113,
                'versions': [{'artifactVersionId': 'a8286090-c637-4270-99ab-25b18676a035',
                        'version': '1.0.0',
                        'owner': 'lothiraldan',
                        'metadata': None,
                        'createdFrom': None,
                        'sizeInBytes': 0,
                        'state': None,
                        'added': 1621948911721,
                        'alias': ['current-production'],
                        'tags': ['production']},
                    {'artifactVersionId': 'bf778c64-a97c-4bff-9752-7fa6bfebbe2e',
                        'version': '2.0.0',
                        'owner': 'lothiraldan',
                        'metadata': None,
                        'createdFrom': None,
                        'sizeInBytes': 21113,
                        'state': None,
                        'added': 1621948972987,
                        'alias': ['Latest'],
                        'tags': ['staging']}]}
            ```
        """
        return self._client.get_artifact_details(artifact_id, workspace, artifact_name)

    def get_artifact_files(
        self,
        workspace=None,
        artifact_name=None,
        artifact_id=None,
        version=None,
        alias=None,
    ):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]) -> Dict[str, Any]
        """
        Returns the files of a single artifact version. The artifact is identified either by the
        workspace name + the artifact name or by its unique artifact ID. The artifact version is
        identified either by an explicit version or by an explicit alias.

        Args:
            workspace (str):The name of the workspace
            artifact_name (str): The name of the artifact
            artifact_id (str): The unique ID of the artifact, for example `6194e719-f596-48e7-8cca-8530c16dd007`
            version (str): The version number of the artifact version you want
            alias (str): The alias of the artifact version you want

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            artifact_files = api.get_artifact_files("demo", artifact_name="demo-artifact", version="2.0.0")
            print(artifact_files)
            ```

            The print statement above return the following:
            ```json
            {'files': [{'artifactId': '6194e719-f596-48e7-8cca-8530c16dd007',
                'artifactVersionId': 'bf778c64-a97c-4bff-9752-7fa6bfebbe2e',
                'assetId': '6aa914ffbee94e11b69445383d7732f4',
                'fileName': 'logo.png',
                'fileSize': 21113,
                'link': None,
                'dir': None,
                'type': 'unknown',
                'metadata': None}]}
            ```

            To query by alias you can run:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            artifact_files = api.get_artifact_files("demo", artifact_name="demo-artifact", alias="current-production")
            print(artifact_files)
            ```

            which will print the following:
            ```json
            {'files': [{'artifactId': '6194e719-f596-48e7-8cca-8530c16dd007',
                'artifactVersionId': 'a8286090-c637-4270-99ab-25b18676a035',
                'assetId': 'dea243de41714a48961f725dbbe4d214',
                'fileName': 'file',
                'fileSize': 0,
                'link': 's3://bucket/dir/file',
                'dir': None,
                'type': 'unknown',
                'metadata': None}]}
            ```
        """
        return self._client.get_artifact_files(
            artifact_id=artifact_id,
            workspace=workspace,
            name=artifact_name,
            version=version,
            alias=alias,
        )

    ## ---------------------------------------------------------
    # Public Write Methods
    ## ---------------------------------------------------------

    def move_experiments(
        self, experiment_keys, target_workspace, target_project_name, symlink=False
    ):
        """
        Move or symlink a list of experiments to another project_name.

        Args:
            experiment_keys (list): List of experiment keys
            target_workspace (str): Workspace name to move experiments to
            target_project_name (str): The project name to move experiments to
            symlink (bool): If True, then create a symlink
                in target_workspace/target_project_name.

        Note: you cannot move experiments from one workspace to another.

        Example:
            ```python linenums="1"
            import comet_ml
            from comet_ml.query import Tag

            comet_ml.login()
            api = comet_ml.API()

            # Move all experiments with a particular tag:
            experiments = api.query("workspace", "project", Tag("My tag"))
            api.move_experiments([e.id for e in experiments],
                                    "workspace",
                                    "other-project")
            ```
        """
        result = None
        for chunk in range(0, len(experiment_keys), 100):
            result = self._client.move_experiments(
                experiment_keys[chunk : chunk + 100],
                target_workspace,
                target_project_name,
                symlink,
            )
            if result and result.status_code != 200:
                break
        return result

    def delete_experiment(self, experiment_key):
        """
        Delete one experiment.

        Args:
            experiment_key (str): The experiment key for the experiment to delete.
        """
        results = self._client.delete_experiment(experiment_key)
        return results

    def create_project(
        self, workspace, project_name, project_description=None, public=False
    ):
        """
        Create a project.

        Args:
            project_name (str): The project name.
            project_description (str): The description for this project.
            public (bool): If True the project will be public.
        """
        results = self._client.create_project(
            workspace, project_name, project_description, public
        )
        return results

    def update_project(
        self,
        workspace,
        project_name,
        new_project_name=None,
        description=None,
        public=None,
    ):
        """
        Update the metadata of a project by project_name and workspace.

        Args:
            workspace (str): name of workspace
            project_name (str): name of project
            new_project_name (str): new name of project
            description (str): new description of project
            public (bool): new setting of visibility

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            api.update_project("mywork", "oldproj",
                new_project_name="newproj", description="desc",
                public=True)
            ```
        """
        # error checking in the client method:
        results = self._client.update_project(
            workspace, project_name, new_project_name, description, public
        )
        if self._check_results(results):
            return results.json()

    def update_project_by_id(
        self, project_id, new_project_name=None, description=None, public=None
    ):
        """
        Update the metadata of a project by project_id.

        Args:
            project_id (str): project id
            new_project_name (str): new name of project
            description (str): new description of project)
            public (bool): new setting of visibility

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            api.update_project_by_id("2627523253623",
                new_project_name="newproj", description="desc",
                public=True)
            ```
        """
        # error checking in the client method:
        results = self._client.update_project_by_id(
            project_id, new_project_name, description, public
        )
        if self._check_results(results):
            return results.json()

    def delete_project(
        self,
        workspace=None,
        project_name=None,
        project_id=None,
        delete_experiments=False,
    ):
        """
        Delete a project.

        Args:
            workspace (str): the name of the workspace (required if project_id not given)
            project_name (str): the name of the project (required if project_id not given)
            project_id (str): the project id (required, if workspace and project name not given)
            delete_experiments (bool): if True, delete all of the experiments, too
        """
        results = self._client.delete_project(
            workspace, project_name, project_id, delete_experiments
        )
        return results

    def set_project_notes(self, workspace, project_name, notes):
        """
        Set the notes of a project. Overwrites any previous
        notes.

        Args:
            workspace (str): The name of the workspace
            project_name (str): The name of the project
            notes (str): The full notes

        Returns: a JSON message

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            api.set_project_notes("my-workspace", "my-project",
                                  "These are my project-level notes")
            ```
        """
        project_json = self.get_project(workspace, project_name)
        if project_json:
            project_id = project_json["projectId"]
            return self._client.set_project_notes_by_id(project_id, notes)
        else:
            raise ValueError(
                "unknown project %r in workspace %r" % (project_name, workspace)
            )

    def get_default_workspace(self):
        # type: () -> str
        """
        Get the default workspace name.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            workspace = api.get_default_workspace()
            ```
        """
        details = self.get_account_details()
        return details["defaultWorkspaceName"]

    def get_project_share_keys(self, project_id):
        """
        Get the share keys for a private project ID.

        Args:
            project_id (str): The ID of the project

        Example:
        ```python
        import comet_ml

        comet_ml.login()
        api = comet_ml.API()
        SHARE_KEYS = api.get_project_share_keys(PROJECT_ID)
        ```

        See also: API.create_project_share_key(), and API.delete_project_share_key().
        """
        results = self._client.get_project_share_keys(project_id)
        if results:
            return results["shareCodes"]

    def create_project_share_key(self, project_id):
        """
        Get the share keys for a private project ID.

        Args:
            project_id (str): The ID of the project

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_mlAPI()
            SHARE_KEY = api.create_project_share_key(PROJECT_ID)
            ```

        See also: API.get_project_share_keys(), and API.delete_project_share_key().
        """
        results = self._client.create_project_share_key(project_id)
        if results:
            return results["shareCode"]

    def delete_project_share_key(self, project_id, share_key):
        """
        Delete a share key for a private project ID.

        Args:
            project_id (str): The ID of the project
            share_key (str): The share key to delete

        Example:
            ```python
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            SHARE_KEYS = api.get_project_share_keys(PROJECT_ID)

            api.delete_project_share_key(PROJECT_ID, SHARE_KEYS[0])
            ```

        See also: API.get_project_share_keys(), and API.create_project_share_key().
        """
        results = self._client.delete_project_share_key(project_id, share_key)
        return results

    def stop_experiment(self, experiment_key):
        """
        Stop a running experiment.

        Args:
            experiment_key (str): the experiment key

        Example:
            ```python
            import comet_ml

            # Start an online experiment:
            experiment = comet_ml.Experiment()

            # Perhaps somewhere else, while experiment
            # is running:
            api = comet_ml.API()
            api.stop_experiment(experiment.get_key())
            ```
        """
        results = self._client.stop_experiment(experiment_key)
        return results

    def delete_experiments(self, experiment_keys):
        """
        Delete list of experiments.

        Args:
            experiment_keys (list): a list of experiment keys to delete.
        """
        results = self._client.delete_experiments(experiment_keys)
        return results

    def restore_experiment(self, experiment_key):
        """
        Restore one experiment.

        Args:
            experiment_key (str): the experiment ID to restore.
        """
        results = self._client.restore_experiment(experiment_key)
        return results

    def archive_experiment(self, experiment_key):
        """
        Archive one experiment.

        Args:
            experiment_key (str): the experiment key to archive
        """
        results = self._client.archive_experiment(experiment_key)
        return results

    def archive_experiments(self, experiment_keys):
        """
        Archive list of experiments.

        Args:
            experiment_keys (list): the experiment keys to archive
        """
        results = self._client.archive_experiments(experiment_keys)
        return results

    def get_metrics_for_chart(
        self,
        experiment_keys,
        metrics=None,
        parameters=None,
        independent=True,
        full=False,
    ):
        """
        Get multiple metrics and parameters from a set of
        experiments. This method is designed to make custom charting
        easier.

        Args:
            experiment_keys (list): a list of experiment keys
            metrics (list): List of metric names (e.g., "loss")
            parameters (list): List of parameter names (e.g., "learning-rate")
            independent (bool):get independent results?
            full (bool): Fetch the full result?

        Returns:
            dict: A dictionary of experiment keys with the following
                structure. `{EXPERIMENT_KEY: {'params'}` will be `None` if there are no
                parameters passed in.

        Note:
            You should pass in a list of metric names, or a list of
            parameter names, or both.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            res = api.get_metrics_for_chart([experiment_key1, experiment_key2, ...],
                                                ["loss"], ["hidden_layer_size"])
            print(res)
            ```

            The print statement above would return an object like:
            ```json
            {EXPERIMENT_KEY: {
            'experiment_key': EXPERIMENT_KEY,
            'steps': STEPS,
            'epochs': None,
            'metrics': [
                {'metricName': 'loss',
                'values': [VALUE, ...],
                'steps': [STEP, ...],
                'epochs': [EPOCH, ...],
                'timestamps': [TIMESTAMP, ...],
                'durations': [DURATION, ...],
                }],
            'params': {'hidden_layer_size': VALUE, ...},
            }, ...}
            ```
        """
        if not isinstance(experiment_keys, (list, tuple)):
            raise TypeError(
                API_GET_METRICS_FOR_CHART_REQUIRES_LIST_EXPERIMENTS_EXCEPTION
            )
        if (metrics is not None) and not isinstance(metrics, (list, tuple)):
            raise TypeError(API_GET_METRICS_FOR_CHART_REQUIRES_LIST_METRICS_EXCEPTION)
        if (parameters is not None) and not isinstance(parameters, (list, tuple)):
            raise TypeError(
                API_GET_METRICS_FOR_CHART_REQUIRES_LIST_PARAM_NAMES_EXCEPTION
            )

        # Make sure that the default Python Panel code doesn't crash
        # if there are no selected metrics:
        metrics = [metric for metric in metrics if metric]

        if len(metrics) == 0:
            return []

        results = self._client.get_experiment_multi_metrics(
            experiment_keys, metrics, parameters, independent, full
        )
        if self._check_results(results):
            results_json = results.json()
            # Also: results_json["empty"] indicates results_json["experiments"] or not
            return results_json["experiments"]
        else:
            return []

    def use_cache(self, cache=None):
        """
        Turn cache on/off or return cache.

        Args:
            cache (bool): Whether or not to use the cache

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            api.use_cache(False)
            print(api.use_cache()) # Prints False

            api.use_cache(True)
            print(api.use_cache()) # Prints True

            ```
        """
        if cache is None:
            if hasattr(self._client, "use_cache"):
                return self._client.use_cache
            else:
                return False
        else:
            if hasattr(self._client, "use_cache"):
                self._client.use_cache = cache
            else:
                if cache:
                    raise Exception(API_USE_CACHE_NOT_SUPPORTED_EXCEPTION)
                # else ignore

    def do_cache(self, *endpoints):
        """
        Cache the given endpoints.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            api.do_cache("experiments", "projects")
            ```
        """
        self._client.do_cache(*endpoints)

    def do_not_cache(self, *endpoints):
        """
        Do not cache the given endpoints.

        Example:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            api.do_not_cache("experiments", "projects")
            ```
        """
        self._client.do_not_cache(*endpoints)

    # API Registry model read methods:

    def model_registry_allowed_status_values(self, workspace):
        """
        Get a list of the allowed values for the status of a model version in a given workspace.

        Args:
            workspace (str): The name of workspace

        Returns: list of allowed values
        """
        response = self._client.get_from_endpoint(
            "registry-model/allowed_status_values", {"workspaceName": workspace}
        )
        return response["allowedStatus"]

    def get_registry_model_names(self, workspace):
        """
        Get a list of model names associated with this workspace.

        Args:
            workspace (str): The name of workspace.

        Returns: list of model names
        """
        return [
            model["modelName"] for model in self._client.get_registry_models(workspace)
        ]

    def get_registry_model_count(self, workspace):
        """
        Get a count of the number of registered models in this workspace.

        Args:
            workspace (str): The name of workspace.
        """
        return self._client.get_registry_model_count(workspace)

    def get_model(self, workspace: str, model_name: str) -> model.Model:
        """
        Get a Model API object corresponding to a given model

        Args:
            workspace (str): The name of workspace.
            model_name (str): The name of registered model.
        """
        return model.Model.from_registry(workspace, model_name, api_key=self.api_key)

    def get_registry_model_details(self, workspace, registry_name, version=None):
        """
        Get the details of a registered model in a workspace. If version is given
        then it will return the details of the workspace/registry-name/version.
        Otherwise, it will return the details of the workspace/registry-name.

        Args:
            workspace (str): The name of workspace.
            registry_name (str): The name of the model.
            version (str): The version string of the model.

        Example:
            Running the following code:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()

            api_exp = api.get("workspace/project/765643463546345364536453436")
            res = api_exp.get_registry_model_details("myworkspace", "model-name")

            print(res)
            ```

            will print out the following dictionary:
            ```json linenums="1"
            {
                "registryModelId": "someRegistryModelId",
                "modelName": "someModelName",
                "description": "someDescription",
                "isPublic": "[Boolean]",
                "createdAt": "[long, when this model was created in the DB]",
                "lastUpdated": "[long, last time this model was updated in the DB]",
                "userName": "someUserName",
                "versions": [
                    {
                        "registryModelItemId": "someRegistryModelItemId",
                        "experimentModel": {
                        "experimentModelId": "someExperimentModelId",
                        "experimentModelName": "someExperimentModelName",
                        "experimentKey": "someExperimentKey"
                        },
                        "version": "someVersion",
                        "comment": "someComment",
                        "stages": ["production", "staging"],
                        "userName": "someUserName",
                        "createdAt": "[long, when this model item was created in the DB]",
                        "lastUpdated": "[long, last time this model item was updated in the DB]",
                        "assets": [
                            {
                                "fileName": "someFileName",
                                "fileSize": "[Long, file size]",
                                "runContext": "someRunContext",
                                "step": "[Integer, step asset was logged during]",
                                "link": "link to download asset file",
                                "createdAt": "[Long, timestamp asset was created in DB]",
                                "dir": "someDirectory",
                                "canView": "[Boolean, whether the asset is viewable as an image]",
                                "audio": "[Boolean, whether the asset is an audio file]",
                                "histogram": "[Boolean, whether the asset is a histogram file]",
                                "image": "[Boolean, whether the asset was stored as an image]",
                                "type": "the type of asset",
                                "metadata": "Metadata associated with the asset",
                                "assetId": "someAssetId",
                            }
                        ],
                    }
                ]
            }
            ```
        """
        return self._client.get_registry_model_details(
            workspace, registry_name, version
        )

    def get_latest_registry_model_version_details(
        self,
        workspace: str,
        registry_name: str,
        stage: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return details about the latest model registry version, including its asset list.

        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [comet_ml.Model.find_versions][]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version_major (int): The major part of version string of the model.
            version_minor (int): The minor part of version string of the model.
            stage (str): A textual tag such as "production" or "staging".

        Example:
            Running the following code:
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = comet_ml.API()
            res = api.get_latest_registry_model_version_details("myworkspace", "model-name")

            print(res)
            ```

            will print the following dictionary:
            ```json linenums="1"
            {
                "registryModelId": "someRegistryModelId",
                "modelName": "someModelName",
                "description": "someDescription",
                "isPublic": "[Boolean]",
                "createdAt": "[long, when this model was created in the DB]",
                "lastUpdated": "[long, last time this model was updated in the DB]",
                "userName": "someUserName",
                "versions": [
                    {
                        "registryModelItemId": "someRegistryModelItemId",
                        "experimentModel": {
                            "experimentModelId": "someExperimentModelId",
                            "experimentModelName": "someExperimentModelName",
                            "experimentKey": "someExperimentKey"
                        },
                        "version": "someVersion",
                        "comment": "someComment",
                        "stages": [
                            "production",
                            "staging"
                        ],
                        "userName": "someUserName",
                        "createdAt": "[long, when this model item was created in the DB]",
                        "lastUpdated": "[long, last time this model item was updated in the DB]",
                        "assets": [
                            {
                                "fileName": "someFileName",
                                "fileSize": "[Long, file size]",
                                "runContext": "someRunContext",
                                "step": "[Integer, step asset was logged during]",
                                "link": "link to download asset file",
                                "createdAt": "[Long, timestamp asset was created in DB]",
                                "dir": "someDirectory",
                                "canView": "[Boolean, whether the asset is viewable as an image]",
                                "audio": "[Boolean, whether the asset is an audio file]",
                                "histogram": "[Boolean, whether the asset is a histogram file]",
                                "image": "[Boolean, whether the asset was stored as an image]",
                                "type": "the type of asset",
                                "metadata": "Metadata associated with the asset",
                                "assetId": "someAssetId"
                            }
                        ]
                    }
                ]
            }
            ```
        """
        LOGGER.warning(API_GET_LATEST_REGISTRY_MODEL_VERSION_DETAILS_DEPRECATED_WARNING)
        return self._client.get_latest_registry_model_details(
            workspace=workspace,
            registry_name=registry_name,
            stage=stage,
            version_major=version_major,
            version_minor=version_minor,
        )

    def get_model_registry_version_assets(
        self,
        workspace,
        registry_name,
        version=None,
        stage=None,
    ):
        # type: (str, str, Optional[str], Optional[str]) -> Any
        """
        Return details about a single model registry version, including its asset list.

        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [comet_ml.Model.get_assets][]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version (str): The version string of the model.
            stage (str): A textual tag such as "production" or "staging".

        Example:
            Running the following code
            ```python linenums="1"
            import comet_ml

            comet_ml.login()
            api = API()
            res = api.get_model_registry_version_assets("myworkspace", "model-name")

            print(res)
            ```

            will print the dictionary:
            ```json linenums="1"
            {
                "registryModelItemId": "someRegistryModelItemId",
                "experimentModel": {
                    "experimentModelId": "someExperimentModelId",
                    "experimentModelName": "someExperimentModelName",
                    "experimentKey": "someExperimentKey"
                },
                "version": "someVersion",
                "comment": "someComment",
                "stages": ["production", "staging"],
                "userName": "someUserName",
                "createdAt": "[long, when this model item was created in the DB]",
                "lastUpdated": "[long, last time this model item was updated in the DB]",
                "assets": [
                    {
                        "fileName": "someFileName",
                        "fileSize": "[Long, file size]",
                        "runContext": "someRunContext",
                        "step": "[Integer, step asset was logged during]",
                        "link": "link to download asset file",
                        "createdAt": "[Long, timestamp asset was created in DB]",
                        "dir": "someDirectory",
                        "canView": "[Boolean, whether the asset is viewable as an image]",
                        "audio": "[Boolean, whether the asset is an audio file]",
                        "histogram": "[Boolean, whether the asset is a histogram file]",
                        "image": "[Boolean, whether the asset was stored as an image]",
                        "type": "the type of asset",
                        "metadata": "Metadata associated with the asset",
                        "assetId": "someAssetId",
                    }
                ],
            }
            ```
        """
        LOGGER.warning(API_GET_MODEL_REGISTRY_VERSION_ASSETS_DEPRECATED_WARNING)

        details = self._client.get_registry_model_items_download_links(
            workspace, registry_name, version, stage
        )
        assert len(details["versions"]) == 1
        return details["versions"][0]

    def get_registry_model_versions(self, workspace, registry_name):
        """
        Get a list of the version strings of a registered model in a workspace.

        Args:
            workspace (str): The name of workspace.
            registry_name (str): The name of the model.
        """
        return self._client.get_registry_model_versions(workspace, registry_name)

    def get_registry_model_notes(self, workspace, registry_name):
        """
        Get the notes of a registered model in a workspace.

        Args:
            workspace (str): The name of workspace.
            registry_name (str): The name of the model.
        """
        return self._client.get_registry_model_notes(workspace, registry_name)

    def download_registry_model(
        self,
        workspace: str,
        registry_name: str,
        version: Optional[str] = None,
        output_path: str = "./",
        expand: bool = True,
        stage: Optional[str] = None,
    ) -> None:
        """
        Download and save all files from the registered model.

        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [comet_ml.Model.download][]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version (str): The version string of the model.
            output_path (str): The output directory
            expand (bool): If True, the downloaded zipfile is unzipped; if False, then the zipfile
                is copied to the output_path.
            stage (str): A textual tag such as "production" or "staging".
        """
        LOGGER.warning(API_DOWNLOAD_REGISTRY_MODEL_DEPRECATED_WARNING)

        LOGGER.info(
            API_DOWNLOAD_REGISTRY_MODEL_START_INFO,
            registry_name,
            version,
            stage,
            workspace,
        )
        zip_file = self._client.get_registry_model_zipfile(
            workspace, registry_name, version, stage
        )
        if zip_file is not None:
            output_path = expand_user_home_path(output_path)
            with io.BytesIO(zip_file) as fp:
                if expand:
                    LOGGER.info(
                        API_DOWNLOAD_REGISTRY_MODEL_UNZIP_INFO,
                        os.path.abspath(output_path),
                    )
                    with zipfile.ZipFile(fp) as zp:
                        zp.extractall(output_path)
                else:
                    # The case where both version and stage is set if handled by get_registry_model_zipfile
                    if version is not None:
                        suffix = version
                    elif stage is not None:
                        suffix = stage
                    else:
                        suffix = "latest"

                    output_file = os.path.join(
                        output_path, "%s_%s.zip" % (registry_name, suffix)
                    )

                    LOGGER.info(
                        API_DOWNLOAD_REGISTRY_MODEL_COPY_INFO,
                        os.path.abspath(output_file),
                    )
                    with open(output_file, "wb") as op:
                        shutil.copyfileobj(fp, op)
                LOGGER.info(API_DOWNLOAD_REGISTRY_MODEL_COMPLETED_INFO)
        else:
            LOGGER.info(API_DOWNLOAD_REGISTRY_MODEL_FAILED_INFO)

    # API Registry model write methods:

    def update_registry_model(
        self, workspace, registry_name, new_name=None, description=None, public=None
    ):
        """
        Updates a registered model's name, description, and/or visibility.

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            new_name (str): The new name of model.
            description (str): The new description of model.
            public (bool): The new visibility of model.
        """
        return self._client.update_registry_model(
            workspace, registry_name, new_name, description, public
        )

    def update_registry_model_version(
        self, workspace, registry_name, version, comment=None, stages=None
    ):
        """
        Update a registered model version's comments and stages.


        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [comet_ml.Model.set_status][] or [comet_ml.Model.add_tag][]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version (str): The version of model to update.
            comment (str): Comments of model version.
            stages (List[str]): A new list of stages, e.g. ["production", "staging"].
        """
        LOGGER.warning(API_UPDATE_REGISTRY_MODEL_VERSION_DEPRECATED_WARNING)
        return self._client.update_registry_model_version(
            workspace, registry_name, version, comment, stages
        )

    def delete_registry_model(self, workspace, registry_name):
        """
        Deletes a registered model.

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
        """
        return self._client.delete_registry_model(workspace, registry_name)

    def delete_registry_model_version(self, workspace, registry_name, version):
        """
        Deletes a registered model version.

        Args:
            workspace (str): The name of the workspace
            registry_name (str): The name of the model
            version (str): The version of model to update
        """
        return self._client.delete_registry_model_version(
            workspace, registry_name, version
        )

    def update_registry_model_notes(self, workspace, registry_name, notes):
        """
        Updates a registered model's notes.

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            notes (str): Notes of model.
        """
        return self._client.update_registry_model_notes(workspace, registry_name, notes)

    def add_registry_model_version_stage(
        self, workspace, registry_name, version, stage
    ):
        """
        Adds a stage to a registered model version.

        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [`model.set_status(version='<version>', status='<status>')`][comet_ml.Model.set_status]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version (str): The version of model to update.
            stage (str): "production", or "staging", etc.
        """
        LOGGER.warning(API_ADD_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING)

        return self._client.add_registry_model_version_stage(
            workspace, registry_name, version, stage
        )

    def delete_registry_model_version_stage(
        self, workspace, registry_name, version, stage
    ):
        """
        Removes a stage from a registered model version.

        This method has been deprecated as `stage` has been replaced by `status` and a new
        [comet_ml.Model][] object was introduced. We recommend
        using the [comet_ml.API.get_model][] method to get
        the Model object and then using [`model.set_status(version='<version>', status='<status>')`][comet_ml.Model.set_status]

        Args:
            workspace (str): The name of the workspace.
            registry_name (str): The name of the model.
            version (str): The version of model to update.
            stage (str): "production", or "staging", etc.
        """
        LOGGER.warning(API_DELETE_REGISTRY_MODEL_VERSION_STAGE_DEPRECATED_WARNING)

        return self._client.delete_registry_model_version_stage(
            workspace, registry_name, version, stage
        )

    # Experiment assets methods
    def download_experiment_asset(
        self, experiment_key: str, asset_id: str, output_path: str
    ) -> None:
        """
        Download an experiment (or a model registry) asset to the specified output_path.

        Args:
            experiment_key (str): The experiment unique key to download from.
            asset_id (str): The asset ID.
            output_path (str): Where to download the asset.

        Raises:
            comet_ml.exceptions.CometRestApiException: if the asset or experiment_key is not found.
            OSError: if the asset cannot be written to the output_path.
        """
        response: requests.Response = self._client.get_experiment_asset(
            asset_id=asset_id,
            experiment_key=experiment_key,
            return_type="response",
            stream=True,
            allow301=True,
        )

        output_path = expand_user_home_path(output_path)
        with io.open(output_path, "wb") as output_file:
            write_stream_response_to_file(response, output_file, None)

    # Compatibility methods with Python Panels comet_ml
    def get_panel_metrics_names(self) -> List[str]:
        """
        Returns the metric names for all experiments in the
        workspace/project_name associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        workspace = get_config("comet.workspace")
        project_name = get_config("comet.project_name")
        if workspace and project_name:
            return sorted(
                [
                    name
                    for name in self._get_metrics_name(
                        workspace,
                        project_name,
                    )
                    if not name.startswith("sys.")
                ]
            )
        else:
            raise AttributeError(API_GET_PANEL_METRICS_NAMES_EXCEPTION)

    def get_panel_options(self) -> Dict[str, Any]:
        """
        Returns the panel options as a dictionary.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        return streamlit_config.override.get("COMET_PANEL_OPTIONS", {})

    def get_panel_experiments(self) -> List[APIExperiment]:
        """
        Returns the experiments associated with the workspace/
        project_name associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()

        experiment_keys = streamlit_config.override.get("COMET_EXPERIMENT_KEYS", None)
        if experiment_keys is not None:
            return [
                self.get_experiment_by_key(experiment_key)
                for experiment_key in experiment_keys
            ]
        else:
            workspace = streamlit_config["comet.workspace"]
            project_name = streamlit_config["comet.project_name"]
            if workspace and project_name:
                return self.get_experiments(workspace, project_name)

        raise AttributeError(API_GET_PANEL_EXPERIMENTS_EXCEPTION)

    def get_panel_experiment_keys(self) -> List[str]:
        """
        Returns the experiment keys associated with the workspace/
        project_name associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        experiment_keys = streamlit_config.override.get("COMET_EXPERIMENT_KEYS", None)
        if experiment_keys is not None:
            return experiment_keys
        else:
            workspace = get_config("comet.workspace")
            project_name = get_config("comet.project_name")
            if workspace and project_name:
                return [
                    experiment.id
                    for experiment in self.get_experiments(workspace, project_name)
                ]

        raise AttributeError(API_GET_PANEL_EXPERIMENT_KEYS_EXCEPTION)

    def get_panel_project_id(self) -> str:
        """
        Returns the project_id associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        workspace = get_config("comet.workspace")
        project_name = get_config("comet.project_name")
        if workspace and project_name:
            project_json = self.get_project(
                workspace,
                project_name,
            )
            if project_json:
                project_id = project_json["projectId"]
                return project_id
        else:
            raise AttributeError(API_GET_PANEL_PROJECT_ID_EXCEPTION)

    def get_panel_project_name(self) -> Union[str, None]:
        """
        Returns the project name associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        project_name = get_config("comet.project_name")
        return project_name

    def get_panel_workspace(self) -> Union[str, None]:
        """
        Returns the project name associated with this panel.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        return get_config("comet.workspace")

    def get_panel_experiment_colors(self) -> PanelColorMap:
        """
        Returns a dictionary of experiment keys and their associated colors.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        return streamlit_config.override.get("COMET_PANEL_EXPERIMENT_COLORS", {})

    def get_panel_metric_colors(self) -> PanelColorMap:
        """
        Returns a dictionary of metrics and their associated colors.

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        return streamlit_config.override.get("COMET_PANEL_METRIC_COLORS", {})

    def get_panel_width(self) -> int:
        """
        Returns a width of the panel (in pixels).

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        return streamlit_config.override.get("COMET_PANEL_WIDTH", 0)

    def get_panel_height(self) -> int:
        """
        Returns a height of the panel (in pixels).

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        streamlit_config = get_config()
        return streamlit_config.override.get("COMET_PANEL_HEIGHT", 0)

    def get_panel_size(self) -> Tuple[int, int]:
        """
        Returns a tuple of the width and height of the panel (in pixels). (width, height)

        This method is designed for use inside a Comet Panel.
        For more information, please see:
        [Python Panels](/docs/v2/guides/comet-ui/experiment-management/visualizations/python-panel/)
        """
        panel_width = self.get_panel_width()
        panel_height = self.get_panel_height()

        return (panel_width, panel_height)


def make_query_vars(columns) -> List[Union[QueryVariable, Tag]]:
    """Parse the results of `/api/rest/v2/project/column-names` endpoint and
    returns queriable objects.
    """
    result = []
    for column in columns["columns"]:
        query_var = make_single_query_var(column)
        if query_var is not None:
            result.append(query_var)
    return result


def make_single_query_var(column) -> Optional[Union[QueryVariable, Tag]]:
    if column["source"] == "metadata":
        return Metadata(column["name"], qtype=column["type"])
    elif column["source"] == "metrics":
        return Metric(column["name"], qtype=column["type"])
    elif column["source"] == "log_other":
        return Other(column["name"], qtype=column["type"])
    elif column["source"] == "params":
        return Parameter(column["name"], qtype=column["type"])
    elif column["source"] == "tag":
        return Tag(column["name"])
    elif column["source"] == "env_details":
        return Environment(column["name"], qtype=column["type"])
    else:
        LOGGER.debug("Unknown query variable type: %r" % column["source"])
        return None


@functools.lru_cache(maxsize=10)
def get_instance(api_key=None, cache=True) -> API:
    api = API(api_key=api_key, cache=cache)
    return api
