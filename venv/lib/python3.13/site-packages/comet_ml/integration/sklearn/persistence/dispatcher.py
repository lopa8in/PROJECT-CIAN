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

import importlib
from types import ModuleType

import comet_ml
from comet_ml import exceptions

from . import dumper, loader, module_metadata


def get_dumper(persistence_module: ModuleType, **dump_kwargs) -> dumper.Dumper:
    module_name = persistence_module.__name__
    supported_modules = module_metadata.supported_modules()
    if module_name not in supported_modules:
        message = "Persistence module '{}' is not supported. Supported modules for the installed SDK version {} are: {}".format(
            module_name, comet_ml.__version__, supported_modules
        )
        raise exceptions.PersistenceModuleNotSupportedException(message)

    return dumper.Dumper(persistence_module, **dump_kwargs)


def get_loader(*, persistence_module_name: str, **load_kwargs) -> loader.Loader:
    supported_modules = module_metadata.supported_modules()
    if persistence_module_name not in supported_modules:
        message = "Cannot load model saved with persistence module '{}' in Comet SDK version {}. Update your Comet SDK version to load that model".format(
            persistence_module_name, comet_ml.__version__
        )
        raise exceptions.PersistenceModuleNotSupportedException(message)

    try:
        persistence_module = importlib.import_module(persistence_module_name)
    except Exception:
        message = "Could not find persistence module '{}'. You are likely missing that dependency".format(
            persistence_module_name
        )
        raise exceptions.PersistenceModuleNotFoundException(message)

    return loader.Loader(persistence_module, **load_kwargs)
