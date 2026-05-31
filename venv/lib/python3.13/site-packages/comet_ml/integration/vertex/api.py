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

import json
import logging
import time

import comet_ml.integration.sanitation.vertex
from comet_ml import CometExperiment, Experiment
from comet_ml.config import get_api_key, get_config
from comet_ml.integration import sanitation

from ..._typing import Any, Dict, List, Optional, Sequence
from ...constants import OTHER_KEY_CREATED_FROM
from .. import KEY_PIPELINE_TYPE
from . import version_utils

LOGGER = logging.getLogger(__name__)


def _comet_logger_experiment_name(
    pipeline_task_name: str, pipeline_job_name: str
) -> str:
    return "{} - {}".format(pipeline_task_name, pipeline_job_name)


def _comet_logger_sidecar_experiment_name(pipeline_run_name: str) -> str:
    return "comet-pipeline-logger-{}".format(pipeline_run_name)


def initialize_comet_logger(
    experiment: CometExperiment,
    pipeline_job_name: str,
    pipeline_task_name: str,
    pipeline_task_uuid: str,
) -> CometExperiment:
    """Logs the Vertex task identifiers needed to track your pipeline status in Comet.ml. You need
    to call this function from every component you want to track with Comet.ml.


    Deprecated: If you are using
    [comet_ml.integration.vertex.CometVertexPipelineLogger.track_task][],
    you don't need to use this function anymore.


    Args:
        experiment: An already created Experiment object.
        pipeline_job_name: The Vertex pipeline job name, see below how to get it automatically from Vertex.
        pipeline_task_name: The Vertex task name, see below how to get it automatically from Vertex.
        pipeline_task_uuid: The Vertex task unique id, see below how to get it automatically from Vertex.

    Example:
        ```python linenums="1"
        import comet_ml.integration.vertex

        experiment = comet_ml.start()
        pipeline_run_name = "{{$.pipeline_job_name}}"
        pipeline_task_name = "{{$.pipeline_task_name}}"
        pipeline_task_id = "{{$.pipeline_task_uuid}}"

        comet_ml.integration.Vertex.initialize_comet_logger(experiment, pipeline_run_name,
        pipeline_task_name, pipeline_task_id)
        ```
    """
    experiment.set_name(
        _comet_logger_experiment_name(pipeline_task_name, pipeline_job_name)
    )
    experiment._log_other("vertex_run_name", pipeline_job_name, include_context=False)
    experiment._log_other("vertex_task_id", pipeline_task_uuid, include_context=False)
    experiment._log_other("vertex_task_name", pipeline_task_name, include_context=False)
    experiment._log_other("vertex_task_type", "task", include_context=False)

    return experiment


COMET_LOGGER_TIMEOUT = 30


def _count_running_tasks(task_details: Sequence[Dict[str, Any]]) -> int:
    """Return the number of running tasks at a single point of time for a Vertex pipeline"""
    running_tasks = 0

    for task in task_details:
        if task["state"].lower() == "running":
            running_tasks += 1

    return running_tasks


def _comet_logger_implementation(
    experiment: CometExperiment,
    pipeline_run_name: str,
    pipeline_task_id: str,
    resource_name: str,
    timeout: int,
) -> None:
    """Extracted comet logger implementation to ease testing.

    Collect and logs the current Vertex pipeline status every second. When this component is the
    last component to run for TIMEOUT second, logs the status one last time and exit.
    """

    import google.cloud.aiplatform as aip

    experiment.set_name(_comet_logger_sidecar_experiment_name(pipeline_run_name))
    experiment._log_other("vertex_run_name", pipeline_run_name, include_context=False)
    experiment._log_other("vertex_run_id", pipeline_task_id, include_context=False)
    experiment._log_other("vertex_task_type", "pipeline", include_context=False)
    experiment._log_other(KEY_PIPELINE_TYPE, "vertex", include_context=False)
    experiment._log_other(OTHER_KEY_CREATED_FROM, "vertex", include_context=False)

    # Need to add a while condition based on the state of the pipeline so that it auto-terminates
    # Check there are
    iterator_nb = 0
    step = 0

    while True:
        pipeline_job = aip.PipelineJob.get(resource_name).to_dict()
        experiment._log_asset_data(
            json.dumps(
                sanitation.vertex.sanitize_environment_variables(pipeline_job),
                default=str,
            ),
            overwrite=True,
            file_name="vertex-pipeline",
            asset_type="vertex-pipeline",
        )

        task_details = pipeline_job["jobDetail"]["taskDetails"]

        # Simple check that looks at the number of running tasks and if there is only one running
        # for more than X seconds stops monitoring the pipeline
        nb_tasks_running = _count_running_tasks(task_details)

        # We always have at least 2 tasks running, this component and the pipeline itself
        if nb_tasks_running > 2:
            iterator_nb = 0
        else:
            iterator_nb += 1

        if iterator_nb > timeout:
            break

        step += 1
        time.sleep(1)

    LOGGER.info(
        "Pipeline has finished running - number tasks running = %d",
        nb_tasks_running,
    )


def _comet_logger_component(
    timeout: int,
    experiment_key: str,
):
    """The actual top-level code Vertex component"""

    # This function code run is copied and ran by Vertex. We cannot access anything from outside,
    # so we need to re-import everything

    from comet_ml import ExistingExperiment
    from comet_ml.integration.vertex import _comet_logger_implementation

    # Create an experiment with your api key
    experiment = ExistingExperiment(
        experiment_key=experiment_key,
        log_git_metadata=False,
        log_git_patch=False,
    )

    # These variables are injected at run time by Vertex
    pipeline_run_name = "{{$.pipeline_job_name}}"
    pipeline_task_id = "{{$.pipeline_task_uuid}}"
    resource_name = "{{$.pipeline_job_resource_name}}"

    _comet_logger_implementation(
        experiment, pipeline_run_name, pipeline_task_id, resource_name, timeout
    )

    return None


def comet_logger_component(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    workspace: Optional[str] = None,
    packages_to_install: Optional[List[str]] = None,
    base_image: Optional[str] = None,
    custom_experiment: Optional[CometExperiment] = None,
) -> Any:
    """
    Inject the Comet Logger component which continuously track and report the current pipeline
    status to Comet.ml.

    Deprecated: Use [comet_ml.integration.vertex.CometVertexPipelineLogger][] instead.

    Args:
        api_key: string, optional. Your Comet API Key, if not provided, the value set in the
            configuration system will be used.

        project_name: string, optional. The project name where all pipeline tasks are logged.
            If not provided, the value set in the configuration system will be used.

        workspace: string, optional. The workspace name where all pipeline tasks are logged.
            If not provided, the value set in the configuration system will be used.

        packages_to_install: List of string, optional. Which packages to install, given directly to
            `kfp.components.create_component_from_func`. Default is ["google-cloud-aiplatform", "comet_ml"].

        base_image: string, optional. Which docker image to use. If not provided, the default
            Kubeflow base image will be used.

        custom_experiment: Experiment, optional. The Comet Experiment with custom configuration which you can provide
            to be used instead of Experiment which would be implicitly created with default options.

    Example:

    ```python
    @dsl.pipeline(name='ML training pipeline')
    def ml_training_pipeline():
        import comet_ml.integration.vertex

        comet_ml.integration.vertex.comet_logger_component()
    ```
    """
    # Inject type hints as kfp use them
    _comet_logger_component.__annotations__ = {
        "timeout": int,
        "experiment_key": str,
    }

    if packages_to_install is None:
        packages_to_install = ["comet_ml", "google-cloud-aiplatform"]

    component = version_utils.create_component(
        _comet_logger_component, packages_to_install, base_image
    )

    config = get_config()
    final_project_name = config.get_string(project_name, "comet.project_name")
    final_workspace = config.get_string(workspace, "comet.workspace")
    final_api_key = get_api_key(api_key, config)

    # create implicit Experiment which log the pipeline source code and transfer its key to the Vertex component
    # to be used to continue as pipeline Experiment.
    if custom_experiment is None:
        exp = Experiment(
            api_key=final_api_key,
            workspace=final_workspace,
            project_name=final_project_name,
            display_summary_level=0,  # turn off summary for this experiment, we will have summary later
        )
    else:
        exp = custom_experiment

    kwargs = {
        "timeout": COMET_LOGGER_TIMEOUT,
        "experiment_key": exp.get_key(),
    }  # type: Dict[str, Any]

    task = component(**kwargs)

    # make sure to close implicit Experiment
    if custom_experiment is None:
        exp.end()

    # Inject api key through environment variable to not log it as a component input
    if final_api_key is not None:
        version_utils.set_environment_variable(task, "COMET_API_KEY", final_api_key)

    return task
