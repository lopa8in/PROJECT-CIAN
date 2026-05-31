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

import logging
from types import ModuleType
from typing import Any, Dict, Optional

import comet_ml
import comet_ml._reporting
import comet_ml.event_tracker
from comet_ml.model_downloader.uri import parse, scheme

from . import model_loading_helpers, model_logging_helpers, model_metadata
from .types import ModelStateDict

LOGGER = logging.getLogger(__name__)


def log_model(
    experiment: comet_ml.CometExperiment,
    model: Any,
    model_name: str,
    metadata: Dict[str, Any] = None,
    pickle_module: Optional[ModuleType] = None,
    **torch_save_args
):
    """
    Logs a Pytorch model to an experiment. This will save the model using
    [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) and save it as an
    Experiment Model.

    The model parameter can either be an instance of `torch.nn.Module` or any input supported by
    torch.save, see the [tutorial about saving and loading Pytorch
    models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for more details.


    Args:
        experiment (Experiment): Instance of experiment to log model
        model (dict | torch.nn.Module): Model to log
        model_name: The name of the model
        metadata: Some additional data to attach to the the data. Must be a JSON-encodable dict
        pickle_module: Passed to torch.save (see [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) documentation)
        torch_save_args: Passed to torch.save (see [torch.save](https://pytorch.org/docs/stable/generated/torch.save.html) documentation)

    Returns: None

    Example:
        Here is an example of logging a model for inference:

        ```python
        class TheModelClass(nn.Module):
            def __init__(self):
                super(TheModelClass, self).__init__()
                ...

            def forward(self, x):
                ...
                return x

        # Initialize model
        model = TheModelClass()

        # Train model
        train(model)

        # Save the model for inference
        log_model(experiment, model, model_name="TheModel")
        ```

        Here is an example of logging a checkpoint for resume training:

        ```python
        model_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            ...
        }
        log_model(experiment, model_checkpoint, model_name="TheModel")
        ```
    """
    model_logging_helpers.track_usage(experiment, model)

    state_dict = model_logging_helpers.get_state_dict(model)

    if pickle_module is None:
        pickle_module = model_metadata.get_torch_pickle_module()

    model_logging_helpers.log_comet_model_metadata(
        experiment, model_name, pickle_module
    )
    model_logging_helpers.log_state_dict(
        experiment, model_name, state_dict, metadata, pickle_module, **torch_save_args
    )


def load_model(
    model_uri: str,
    map_location: Any = None,
    pickle_module: Optional[ModuleType] = None,
    **torch_load_args: Optional[Dict[str, Any]]
) -> ModelStateDict:
    """
    Load model's state_dict from experiment, registry or from disk by uri. This will returns a
    Pytorch state_dict that you will need to load into your model. This will load the model using
    [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html).

    Args:
        model_uri: string (required), a uri string defining model location. Possible options are:

            - file://data/my-model
            - file:///path/to/my-model
            - registry://workspace/registry_name (takes the last version)
            - registry://workspace/registry_name:version
            - experiment://experiment_key/model_name
            - experiment://workspace/project_name/experiment_name/model_name
        map_location: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        pickle_module: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))
        torch_load_args: Passed to torch.load (see [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html))

    Returns: model's state dict

    Example:
        Here is an example of loading a model from the Model Registry for inference:

        ```python
        from comet_ml.integration.pytorch import load_model

        class TheModelClass(nn.Module):
            def __init__(self):
                super(TheModelClass, self).__init__()
                ...

            def forward(self, x):
                ...
                return x

        # Initialize model
        model = TheModelClass()

        # Load the model state dict from Comet Registry
        model.load_state_dict(load_model("registry://WORKSPACE/TheModel:1.2.4"))

        model.eval()

        prediction = model(...)
        ```

        Here is an example of loading a model from an Experiment for Resume Training:

        ```python
        from comet_ml.integration.pytorch import load_model

        # Initialize model
        model = TheModelClass()

        # Load the model state dict from a Comet Experiment
        checkpoint = load_model("experiment://e1098c4e1e764ff89881b868e4c70f5/TheModel")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.train()
        ```
    """
    if pickle_module is None:
        pickle_module = model_metadata.get_torch_pickle_module()

    if parse.request_type(model_uri) == parse.RequestTypes.UNDEFINED:
        raise ValueError("Invalid model_uri: '{}'".format(model_uri))

    if scheme.is_file(model_uri):
        model = model_loading_helpers.from_disk(
            model_uri,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )
    else:
        model = model_loading_helpers.from_remote(
            model_uri,
            map_location=map_location,
            pickle_module=pickle_module,
            **torch_load_args
        )
        model_loading_helpers.track_usage(model_uri)

    return model
