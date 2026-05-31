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

from types import ModuleType
from typing import Any, Dict, Optional

import comet_ml
from comet_ml import _reporting, connection
from comet_ml.model_downloader.uri import parse, scheme

from . import comet_metadata, load_from, log_to_experiment, model_processor
from .persistence import dispatcher


def log_model(
    experiment: "comet_ml.CometExperiment",
    model_name: str,
    model: Any,
    metadata: Optional[Dict] = None,
    persistence_module: Optional[ModuleType] = None,
    **scikit_learn_dump_kwargs: Optional[Dict[str, Any]]
):
    """
    Logs a scikit-learn model to an experiment. This will save the model using
    provided persistence module and save it as an Experiment Model.

    The model argument can be any object that is serializable by provided
    persistence module.

    If a model has best_estimator_ attribute (as different model_selection.*SearchCV classes)
    then only this value will be logged to experiment.

    Here is an example of logging a model:

    ```python
    import comet_ml.integration.sklearn
    experiment = comet_ml.start()

    model = svm.SVC()
    model.fit(iris.data, iris.target)

    comet_ml.integration.sklearn.log_model(
        experiment,
        "my-model",
        model,
        persistence_module=pickle,
    )
    ```

    Args:
        experiment: Instance of experiment to log model.
        model: Model to log.
        model_name: The name of the model.
        metadata: Some additional data to attach to the data. Must be a JSON-encodable dict.
        persistence_module: Module for model serialization. If not specified - joblib is used.
            Currently supported modules: pickle, cloudpickle, joblib.
        scikit_learn_dump_kwargs: Key-value arguments for passing to persistence_module.dump method.

    Returns: None
    """
    if persistence_module is None:
        persistence_module = _get_default_persistence_module()

    model = model_processor.process(model)
    dumper = dispatcher.get_dumper(persistence_module, **scikit_learn_dump_kwargs)
    comet_model_metadata = comet_metadata.collect(dumper)

    log_to_experiment.comet_model_metadata(experiment, model_name, comet_model_metadata)

    model_path = comet_model_metadata["model_metadata"]["scikit-learn"]["model_path"]
    log_to_experiment.model(
        experiment,
        model=model,
        model_name=model_name,
        destination_filename=model_path,
        metadata=metadata,
        dumper_=dumper,
    )
    _track_log_model_usage(experiment, model)


def _track_log_model_usage(experiment: "comet_ml.CometExperiment", model: Any) -> None:
    experiment.__internal_api__report__(
        event_name=_reporting.SKLEARN_MODEL_SAVING_EXPLICIT_CALL,
        err_msg=str(type(model)),
    )


def _get_default_persistence_module() -> ModuleType:
    import joblib

    return joblib


def load_model(model_uri: str, **sklearn_load_args) -> Any:
    """
    Load model from experiment, registry or from disk by uri.
    This will load the model using the persistence module used for saving this
    model via log_model.

    Here is an example of loading a model from the Model Registry for inference:

    ```python
    model = comet_ml.integration.sklearn.load_model("registry://WORKSPACE/my-model")

    X, y = datasets.load_iris(return_X_y=True)
    model.predict(X)
    ```

    Args:
        model_uri (str): A uri string defining model location. Possible options are:

            * file://data/my-model
            * file:///path/to/my-model
            * registry://workspace/registry_name (takes the last version)
            * registry://workspace/registry_name:version
            * experiment://experiment_key/model_name
            * experiment://workspace/project_name/experiment_name/model_name
        sklearn_load_args (Any): Passed to persistence_module.load

    Returns: model
    """
    if parse.request_type(model_uri) == parse.RequestTypes.UNDEFINED:
        raise ValueError("Invalid MODEL_URI: '{}'".format(model_uri))

    if scheme.is_file(model_uri):
        model = load_from.disk(model_uri, **sklearn_load_args)
    else:
        model = load_from.remote(model_uri, **sklearn_load_args)

    _track_load_model_usage(model_uri)
    return model


def _track_load_model_usage(model_uri: str) -> None:
    config = comet_ml.get_config()

    connection.Reporting.report(
        config=config,
        api_key=comet_ml.get_api_key(None, config),
        event_name=_reporting.SKLEARN_MODEL_LOADING_EXPLICIT_CALL,
        err_msg=model_uri,
    )
