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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from comet_ml import config

from . import api, environment_creator, key_sharing, version_utils

if TYPE_CHECKING:  # pragma: no cover
    import comet_ml


class CometVertexPipelineLogger:
    """
    Creates a local experiment for tracking vertex pipeline work
    and provides an API to track vertex tasks with their own
    Comet experiments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        packages_to_install: Optional[List[str]] = None,
        base_image: Optional[str] = None,
        custom_experiment: Optional["comet_ml.CometExperiment"] = None,
        share_api_key_to_workers: bool = False,
    ) -> None:
        """
        Args:
            api_key: Your Comet API Key, if not provided, the value set in the
                configuration system will be used.
            project_name: The project name where all pipeline tasks are logged.
                If not provided, the value set in the configuration system will be used.
            workspace: The workspace name where all pipeline tasks are logged.
                If not provided, the value set in the configuration system will be used.
            packages_to_install: Which packages to install, given directly to
                `kfp.components.create_component_from_func`. Default is `["google-cloud-aiplatform", "comet_ml"]`.
            base_image: Which docker image to use. If not provided, the default
                Kubeflow base image will be used.
            custom_experiment: The Comet Experiment with custom configuration which you can provide
                to be used instead of Experiment which would be implicitly created with default options.
            share_api_key_to_workers: If `True`, Comet API key will be shared
                with workers by setting COMET_API_KEY environment variable. This is an unsafe solution and we recommend you to
                use a [more secure way to set up your API Key in your cluster](/docs/v2/guides/tracking-ml-training/distributed-training/).


        Example:
            ```python linenums="1"

            @dsl.pipeline(name='ML training pipeline')
            def ml_training_pipeline():
                import comet_ml.integration.vertex

                logger = comet_ml.integration.vertex.CometVertexPipelineLogger()

                data_preprocessing_op = components.load_component_from_file("data_preprocessing.yaml")
            ```
        """
        experiment_info_ = config.collect_experiment_info(
            api_key=api_key, project_name=project_name, workspace=workspace
        )

        if share_api_key_to_workers:
            key_sharing.perform_checks(experiment_info_.api_key)

        self._environment = environment_creator.create(
            experiment_info_, share_api_key=share_api_key_to_workers
        )

        self._comet_logger_task = api.comet_logger_component(
            api_key=experiment_info_.api_key,
            workspace=experiment_info_.workspace,
            project_name=experiment_info_.project_name,
            packages_to_install=packages_to_install,
            base_image=base_image,
            custom_experiment=custom_experiment,
        )

    @property
    def comet_logger_task(self) -> Any:
        return self._comet_logger_task

    def track_task(
        self,
        task: Any,
        additional_environment: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Inject all required information to track the given Vertex task with Comet. You still need to
        create an experiment inside that task.

        Args:
            task: The Vertex task to be tracked with Comet.
            additional_environment: A dictionary of additional environment
                variables to be set up in the tracked task.

        Example:
            ```python linenums="1"

            def data_preprocessing(input: str) -> str:
                from comet_ml import start

                # The `track_task` method automatically injects the workspace name and project name.
                # If `share_api_key_to_workers` is set to True, the Comet API Key can also be injected
                # by `CometVertexPipelineLogger`.
                # All Vertex information is automatically logged to the Experiment when the task is
                # wrapped with the `track_task` method.
                experiment = start()

                for i in range(60):
                    experiment.log_metric("accuracy", math.log(i + random.random())) time.sleep(0.1)
                experiment.end()

                return input

            @dsl.pipeline(name='ML training pipeline')
            def ml_training_pipeline():
                import comet_ml.integration.vertex

                logger = comet_ml.integration.vertex.CometVertexPipelineLogger()

                data_preprocessing_op = kfp.components.create_component_from_func(
                    func=data_preprocessing, packages_to_install=["comet_ml"]
                )

                task_1 = logger.track_task(data_preprocessing_op("test"))
            ```

        """
        _set_task_environment(task, self._environment)

        if additional_environment is not None:
            _set_task_environment(task, additional_environment)

        return task


def _set_task_environment(task: Any, environment: Dict[str, str]) -> None:
    for key, value in environment.items():
        version_utils.set_environment_variable(task, key, value)
