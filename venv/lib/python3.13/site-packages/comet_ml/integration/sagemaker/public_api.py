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

import calendar
import logging

import comet_ml._jupyter
import comet_ml.env_logging
from comet_ml._typing import Optional
from comet_ml.api import APIExperiment

import boto3
import sagemaker

from . import log_to_experiment
from .types import SageMakerClient

LOGGER = logging.getLogger(__name__)


def _get_boto_client() -> SageMakerClient:
    return boto3.client("sagemaker")


def log_last_sagemaker_training_job_v1(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    experiment: Optional[APIExperiment] = None,
):
    """
    This function retrieves the last training job and logs its data as a Comet Experiment. The training job must be in completed status.

    This API is in BETA.

    Args:
        api_key: The Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        workspace: Attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        project_name: Send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/)
        experiment: Pass an existing APIExperiment to be used for logging.

    Returns: an instance of [comet_ml.APIExperiment][] for the created Experiment
    """
    client = _get_boto_client()
    last_name = client.list_training_jobs()["TrainingJobSummaries"][0][
        "TrainingJobName"
    ]
    return log_sagemaker_training_job_by_name_v1(
        last_name,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment=experiment,
    )


def log_sagemaker_training_job_v1(
    estimator: sagemaker.estimator.Estimator,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    experiment: Optional[APIExperiment] = None,
):
    """
    This function retrieves the last training job from an
    [`sagemaker.estimator.Estimator`](https://sagemaker.readthedocs.io/en/v2.16.0/api/training/estimators.html#sagemaker.estimator.Estimator)
    object and log its data as a Comet Experiment. The training job must be in completed status.

    This API is in BETA.

    Args:
        estimator: The estimator object that was used to start the training job.
        api_key: The Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        workspace: Attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        project_name: Send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        experiment: Pass an existing APIExperiment to be used for logging.

    Returns: an instance of [comet_ml.APIExperiment][] for the created Experiment

    Example:
        ```python linenums="1"
        import sagemaker

        from comet_ml.integration.sagemaker import log_sagemaker_training_job_v1

        estimator = sagemaker.estimator.Estimator(
            training_image,
            role,
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=s3_output_location,
        )

        estimator.fit(s3_input_location)

        api_experiment = log_sagemaker_training_job_v1(
            estimator, api_key=API_KEY, workspace=WORKSPACE, project_name=PROJECT_NAME
        )
        ```
    """
    # Retrieve the training job name from the estimator
    if not hasattr(estimator, "latest_training_job"):
        raise ValueError("log_sagemaker_job expects a sagemaker Estimator object")

    if estimator.latest_training_job is None:
        raise ValueError(
            "The given Estimator object doesn't seem to have trained a model, call log_sagemaker_job after calling the fit method"
        )

    return log_sagemaker_training_job_by_name_v1(
        estimator.latest_training_job.job_name,
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment=experiment,
    )


def log_sagemaker_training_job_by_name_v1(
    sagemaker_job_name: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    experiment: Optional[APIExperiment] = None,
):
    """
    This function logs the training job identified by the `sagemaker_job_name` as a Comet Experiment.
    The training job must be in completed status.

    This API is in BETA.

    Args:
        sagemaker_job_name: The name of the Sagemaker Training Job.
        api_key: The Comet API Key. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        workspace: Attach the experiment to a project that belongs to this workspace. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        project_name: Send the experiment to a specific project. If not provided must be [configured in another way](/docs/v2/guides/tracking-ml-training/configuring-comet/).
        experiment: Pass an existing APIExperiment to be used for logging.

    Returns: an instance of [comet_ml.APIExperiment][] for the created Experiment
    """
    client = _get_boto_client()
    metadata = client.describe_training_job(TrainingJobName=sagemaker_job_name)

    if metadata["TrainingJobStatus"] != "Completed":
        raise ValueError(
            "Not importing %r as it's not completed, status %r"
            % (sagemaker_job_name, metadata["TrainingJobStatus"])
        )

    if not experiment:
        experiment = APIExperiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            experiment_name=sagemaker_job_name,
        )

    if not isinstance(experiment, comet_ml.APIExperiment):
        raise ValueError("'experiment' parameter must be an APIExperiment instance")

    start_time = metadata["TrainingStartTime"]
    start_time_timestamp = calendar.timegm(start_time.utctimetuple())
    experiment.set_start_time(start_time_timestamp * 1000)
    end_time = metadata.get("TrainingEndTime")
    if end_time:
        experiment.set_end_time(calendar.timegm(end_time.utctimetuple()) * 1000)

    log_to_experiment.hyperparameters(metadata, experiment)

    log_to_experiment.others(metadata, experiment)

    log_to_experiment.other_metadata_fields(
        "TrainingImage", "AlgorithmSpecification", metadata, experiment
    )
    log_to_experiment.other_metadata_fields(
        "TrainingInputMode", "AlgorithmSpecification", metadata, experiment
    )
    log_to_experiment.other_metadata_fields(
        "ModelArtifacts", "ModelArtifacts", metadata, experiment
    )
    log_to_experiment.other_metadata_fields(
        "OutputDataConfig", "OutputDataConfig", metadata, experiment
    )
    log_to_experiment.other_metadata_fields(
        "ResourceConfig", "ResourceConfig", metadata, experiment
    )

    log_to_experiment.other_input_data_config(metadata, experiment)

    log_to_experiment.tags(client, metadata, experiment)

    log_to_experiment.metrics(sagemaker_job_name, start_time_timestamp, experiment)

    experiment.set_installed_packages(comet_ml.env_logging.get_pip_packages())

    if comet_ml._jupyter._in_ipython_environment():
        source_code = comet_ml.env_logging.get_ipython_source_code()

        if source_code != "":
            experiment.set_code(source_code)

    return experiment
