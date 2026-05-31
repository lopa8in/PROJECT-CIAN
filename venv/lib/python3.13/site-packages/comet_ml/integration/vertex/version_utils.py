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

from ..._typing import Any, Callable, List, Optional


def create_component(
    func: Callable, packages_to_install: List[str], base_image: Optional[str] = None
):
    import kfp
    import packaging.version

    if packaging.version.Version(kfp.__version__) >= packaging.version.Version("2.0.0"):
        return kfp.dsl.component(
            func=func, base_image=base_image, packages_to_install=packages_to_install
        )

    return kfp.components.create_component_from_func(
        func=func,
        packages_to_install=packages_to_install,
        base_image=base_image,
    )


def set_environment_variable(task, key: str, value: Any):
    if hasattr(task, "container"):
        task.container.set_env_variable(key, value)
    else:
        task.set_env_variable(key, value)
