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

import logging

import comet_ml
from comet_ml.flatten_dict import flattener

from sagemaker.analytics import TrainingJobAnalytics

from .types import DescribeTrainingJobResponseTypeDef, SageMakerClient

LOGGER = logging.getLogger(__name__)


def others(
    metadata: DescribeTrainingJobResponseTypeDef, experiment: comet_ml.APIExperiment
):
    other_list = [
        "BillableTimeInSeconds",
        "EnableInterContainerTrafficEncryption",
        "EnableManagedSpotTraining",
        "EnableNetworkIsolation",
        "RoleArn",
        "TrainingJobArn",
        "TrainingJobName",
        "TrainingJobStatus",
        "TrainingTimeInSeconds",
    ]

    for other_name in other_list:
        try:
            other_value = metadata.get(other_name)
            if other_value:
                experiment.log_other(other_name, other_value)
        except Exception:
            LOGGER.error(
                comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_METADATA_FIELDS,
                exc_info=True,
            )


def other_metadata_fields(
    field: str,
    parent_key: str,
    metadata: DescribeTrainingJobResponseTypeDef,
    experiment: comet_ml.APIExperiment,
):
    FIELDS_TO_FLATTEN = ["ModelArtifacts", "OutputDataConfig", "ResourceConfig"]

    try:
        if field in FIELDS_TO_FLATTEN:

            for other_key, other_value in flattener.flatten_dict(
                metadata[field], separator=".", parent_key=parent_key
            ).flattened.items():
                experiment.log_other(other_key, other_value)
        else:
            experiment.log_other(field, metadata[parent_key][field])
    except Exception:
        LOGGER.error(
            comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_METADATA_FIELDS,
            exc_info=True,
        )


def other_input_data_config(
    metadata: DescribeTrainingJobResponseTypeDef, experiment: comet_ml.APIExperiment
):
    try:
        for i, _input in enumerate(metadata["InputDataConfig"]):
            for other_key, other_value in flattener.flatten_dict(
                _input, separator=".", parent_key="InputDataConfig.%d" % i
            ).flattened.items():
                experiment.log_other(other_key, other_value)
    except Exception:
        LOGGER.error(
            comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_METADATA_FIELDS,
            exc_info=True,
        )


def hyperparameters(
    metadata: DescribeTrainingJobResponseTypeDef, experiment: comet_ml.APIExperiment
):
    try:
        for param_name, param_value in metadata["HyperParameters"].items():
            experiment.log_parameter(param_name, param_value)
    except Exception:
        LOGGER.error(
            comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_HYPERPARAMETERS,
            exc_info=True,
        )


def tags(
    client: SageMakerClient,
    metadata: DescribeTrainingJobResponseTypeDef,
    experiment: comet_ml.APIExperiment,
):
    try:
        response = client.list_tags(ResourceArn=metadata["TrainingJobArn"])
        for tag_name, tag_value in response["Tags"]:
            experiment.add_tags(["%s:%s" % (tag_name, tag_value)])
    except Exception:
        LOGGER.error(
            comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_TAGS, exc_info=True
        )


def metrics(job_name: str, start_time: int, experiment: comet_ml.APIExperiment):
    try:
        metrics_dataframe = TrainingJobAnalytics(training_job_name=job_name).dataframe()

        for iloc, (timestamp, metric_name, value) in metrics_dataframe.iterrows():
            experiment.log_metric(
                metric=metric_name,
                value=value,
                timestamp=start_time + timestamp,
            )
    except Exception:
        LOGGER.error(
            comet_ml.logging_messages.SAGEMAKER_FAILED_TO_IMPORT_METRICS, exc_info=True
        )
