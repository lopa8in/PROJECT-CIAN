# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Optional

from ..config import (
    get_api_key,
    get_config,
    get_previous_experiment,
    get_project_name,
    get_workspace,
)
from ..exceptions import InvalidExperimentModeUnsupported
from .start_modes import (
    RESUME_STRATEGY_CREATE,
    RESUME_STRATEGY_GET,
    RESUME_STRATEGY_GET_OR_CREATE,
    SUPPORTED_START_MODES,
)


class InitParameters:
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
        source: Optional[str] = None,
    ):
        config = get_config()
        self.api_key = get_api_key(api_key, config=config)
        self.workspace = get_workspace(workspace, config=config)
        self.project = get_project_name(project_name, config=config)
        self.experiment_key = get_previous_experiment(experiment_key, config=config)

        self.mode = config.get_string(
            mode,
            "comet.start.mode",
            default=RESUME_STRATEGY_GET_OR_CREATE,
            not_set_value=None,
        )
        self._online = config.get_bool(
            online, "comet.start.online", default=None, not_set_value=None
        )
        self.source = source

    @property
    def online(self) -> bool:
        if self._online is None:
            # the default is True if not set by user explicitly
            return True

        return self._online

    @property
    def offline(self) -> bool:
        return not self.online

    def validate(self):
        if self.mode is None or self.mode not in SUPPORTED_START_MODES:
            raise InvalidExperimentModeUnsupported(
                mode=str(self.mode), supported_modes=SUPPORTED_START_MODES
            )

    def is_create(self) -> bool:
        return self.mode == RESUME_STRATEGY_CREATE

    def is_get_or_create(self) -> bool:
        return self.mode == RESUME_STRATEGY_GET_OR_CREATE

    def is_get(self) -> bool:
        return self.mode == RESUME_STRATEGY_GET

    def is_get_or_create_with_experiment_key(self) -> bool:
        return (
            self.mode == RESUME_STRATEGY_GET_OR_CREATE
            and self.experiment_key is not None
        )

    def __str__(self) -> str:
        return "InitParameters: %r" % self.__dict__

    def __repr__(self) -> str:
        return self.__str__()
