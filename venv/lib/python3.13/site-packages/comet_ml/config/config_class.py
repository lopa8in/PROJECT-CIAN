# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import io
import logging
import os
import shutil
from typing import Any, Dict, List, NamedTuple, Optional, Set

from everett.ext.inifile import ConfigIniEnv
from everett.manager import (
    NO_VALUE,
    ConfigDictEnv,
    ConfigEnvFileEnv,
    ConfigManager,
    ConfigOSEnv,
    ListOf,
    listify,
    parse_bool,
)

from .._typing import BackendFeatureSupportByVersion
from ..file_utils import expand_user_home_path
from ..logging_messages import (
    CONFIG_COMET_INI_DEPRECATED_WARNING,
    CONFIG_DEPRECATED_CONFIG_KEY_WARNING,
    CONFIG_DEPRECATED_USER_CONFIG_KEY_WARNING,
    CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING,
    CONFIG_KEY_IS_DEPRECATED_WARNING,
    CONFIG_KEY_IS_DEPRECATED_WAS_SET_IN_WARNING,
    CONFIG_SETTING_FAILED_WARNING,
    CONFIG_SETTING_INCORRECT_KEY_SUGGESTION_WARNING,
    CONFIG_SETTING_NOT_FOUND_NO_SUGGESTIONS,
    CONFIG_SETTINGS_FILE_EXISTS_ERROR,
    CONFIG_SETTINGS_NOT_VALID_CONFIG_KEY_EXCEPTION,
    CONFIG_SETTINGS_SAVE_COMPLETED,
    CONFIG_SETTINGS_SAVE_STARTED,
    CONFIG_SETTINGS_USER_REFUSED_OVERWRITE_ERROR,
)
from ..semantic_version import SemanticVersion
from ..utils import clean_string, log_once_at_level
from .config_helper import (
    clean_config_path,
    config_path_from_directory,
    disable_everett_debug_logger,
    get_default_config_path,
)

LOGGER = logging.getLogger(__name__)


class OptionValidationResult(NamedTuple):
    option_found: bool
    value_valid: bool
    failure_reason: Optional[str]


def _input_user(prompt):
    """Independent function to apply clean_string to all responses + make mocking easier"""
    return clean_string(input(prompt))


def _confirm_user_config_file_overwriting(filename: str) -> bool:
    prompt = "Are you sure you want to overwrite your %r file? [y/n] " % filename
    if _input_user(prompt).lower().startswith("y"):
        return True
    else:
        return False


# Vendor generate_uppercase_key for Python 2
def generate_uppercase_key(key, namespace=None):
    """Given a key and a namespace, generates a final uppercase key."""
    if namespace:
        namespace = [part for part in listify(namespace) if part]
        key = "_".join(namespace + [key])

    key = key.upper()
    return key


def parse_str_or_identity(_type):
    def parse(value):
        if not isinstance(value, str):
            return value

        return _type(value.strip())

    return parse


class ParseListOf(ListOf):
    """
    Superclass to apply subparser to list items.
    """

    def __init__(self, _type, _parser):
        super(ParseListOf, self).__init__(_type)
        self._type = _type
        self._parser = _parser

    def __call__(self, value):
        f = self._parser(self._type)
        if not isinstance(value, list):
            value = super(ParseListOf, self).__call__(value)
        return [f(v) for v in value]


PARSER_MAP = {
    str: parse_str_or_identity(str),
    int: parse_str_or_identity(int),
    float: parse_str_or_identity(float),
    bool: parse_str_or_identity(parse_bool),
    list: ParseListOf(str, parse_str_or_identity),
    "int_list": ParseListOf(int, parse_str_or_identity(int)),
}


class Config:
    def __init__(self, config_map: Dict[str, Any]) -> None:
        self.config_map = config_map
        self.override: Dict[str, Any] = self._get_override()
        self.backend_override = ConfigDictEnv({})

        config_override = os.environ.get("COMET_INI")
        if config_override is not None:
            log_once_at_level(logging.WARNING, CONFIG_COMET_INI_DEPRECATED_WARNING)
        else:
            config_override = os.environ.get("COMET_CONFIG")

        if config_override is not None and os.path.isdir(config_override):
            config_override = config_path_from_directory(config_override)

        # disable everett debug logger before reading configuration
        disable_everett_debug_logger()

        self.manager = ConfigManager(
            [  # User-defined overrides
                ConfigOSEnv(),
                ConfigEnvFileEnv(".env"),
                ConfigIniEnv(config_override),
                ConfigIniEnv("./.comet.config"),
                ConfigIniEnv("/content/drive/MyDrive/.comet.config"),
                ConfigIniEnv("~/.comet.config"),
                # Comet-defined overrides
                self.backend_override,
            ],
            doc=(
                "See https://comet.com/docs/python-sdk/getting-started/ for more "
                + "information on configuration."
            ),
        )

    def _get_override(self) -> Dict[str, Any]:
        return {}

    def __setitem__(self, name: str, value: Any):
        self.override[name] = value

    def _set_backend_override(self, cfg, namespace):
        # Reset the existing overrides
        self.backend_override.cfg = {}

        for key, value in cfg.items():
            namespaced_key = "_".join(namespace.split("_") + [key])
            full_key = generate_uppercase_key(namespaced_key)
            self.backend_override.cfg[full_key] = value

    def keys(self) -> Set[str]:
        return self.config_map.keys()

    def similar_settings_keys(self, candidate_key: str) -> List[str]:
        keys = [k.replace(".", "_").replace("comet_", "") for k in self.keys()]
        config_keys = []

        for key in keys:
            if key.find(candidate_key) >= 0:
                config_keys.append(key)

        return config_keys

    def get_raw(
        self,
        user_value: Any,
        config_name: str,
        default: Any = None,
        not_set_value: Any = None,
    ) -> Any:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default
        """

        # 1. User value
        if user_value is not not_set_value:
            return user_value

        # 2. Override
        if config_name in self.override:
            override_value = self.override[config_name]

            if override_value is not None:
                return override_value

        # 3. Configured value
        config_type = self.config_map[config_name].get("type", str)
        parser = PARSER_MAP[config_type]

        # Value
        splitted = config_name.split(".")

        config_value = self.manager(
            splitted[-1], namespace=splitted[:-1], parser=parser, raise_error=False
        )

        if config_value != NO_VALUE:
            return config_value

        else:
            # 4. Provided default
            if default is not None:
                return default

            # 5. Config default
            config_default = parser(self.config_map[config_name].get("default", None))
            return config_default

    def get_string(
        self,
        user_value: Optional[str],
        config_name: str,
        default: Optional[str] = None,
        not_set_value: Optional[str] = None,
    ) -> str:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition, make sure the returned value is a string
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_bool(
        self,
        user_value: Optional[bool],
        config_name: str,
        default: Optional[bool] = None,
        not_set_value: Optional[bool] = None,
    ) -> bool:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition, make sure the returned value is a bool
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_int(
        self,
        user_value: Optional[int],
        config_name: str,
        default: Optional[int] = None,
        not_set_value: Optional[int] = None,
    ) -> int:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition make sure the returned value is an int
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_int_list(
        self,
        user_value: Optional[List[int]],
        config_name: str,
        default: Optional[List[int]] = None,
        not_set_value: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition, make sure the returned value is a list of int
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_string_list(
        self,
        user_value: Optional[List[str]],
        config_name: str,
        default: Optional[List[str]] = None,
        not_set_value: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Returns the correct config value based on the following priority list:
        * User_value if set and not None
        * The override value from the Backend
        * The configured value
        * The default value passed in argument if not None
        * The configured value default

        In addition, make sure the returned value is a list of str
        """

        value = self.get_raw(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )

        return value

    def get_deprecated_raw(
        self,
        old_user_value: Any,
        old_config_name: str,
        new_user_value: Any,
        new_config_name: str,
        new_not_set_value: Any = None,
    ) -> Any:
        """
        Returns the correct value for deprecated config values:
        * New user value
        * Old user value
        * New config value
        * Old config value
        * New config default

        Note: The old config default is not used and should be set to None
        """
        old_config_value = self.get_raw(None, old_config_name, default=NO_VALUE)

        if new_user_value is not new_not_set_value:
            if old_user_value:
                LOGGER.warning(
                    CONFIG_DEPRECATED_USER_CONFIG_KEY_WARNING,
                    old_config_name,
                    new_config_name,
                )
            elif old_config_value:
                LOGGER.warning(
                    CONFIG_DEPRECATED_CONFIG_KEY_WARNING,
                    old_config_name,
                    self.get_config_origin(old_config_name),
                    new_config_name,
                )
            return new_user_value

        # Deprecated parameter default value must be None
        if old_user_value is not None:
            LOGGER.warning(
                CONFIG_KEY_IS_DEPRECATED_WARNING,
                old_config_name,
                new_config_name,
            )
            return old_user_value

        new_config_value = self.get_raw(None, new_config_name, default=NO_VALUE)
        if new_config_value is not NO_VALUE:
            return new_config_value

        old_config_value = self.get_raw(None, old_config_name, default=NO_VALUE)
        if old_config_value is not NO_VALUE:
            LOGGER.warning(
                CONFIG_KEY_IS_DEPRECATED_WAS_SET_IN_WARNING,
                old_config_name,
                self.get_config_origin(old_config_name),
                new_config_name,
            )
            return old_config_value

        config_type = self.config_map[new_config_name].get("type", str)
        parser = PARSER_MAP[config_type]
        return parser(self.config_map[new_config_name].get("default", None))

    def get_deprecated_bool(
        self,
        old_user_value: Any,
        old_config_name: str,
        new_user_value: Any,
        new_config_name: str,
        new_not_set_value: Optional[bool] = None,
    ) -> bool:
        """
        Returns the correct value for deprecated config values:
        * New user value
        * Old user value
        * New config value
        * Old config value
        * New config default

        Note: The old config default is not used and should be set to None
        """
        value = self.get_deprecated_raw(
            old_user_value,
            old_config_name,
            new_user_value,
            new_config_name,
            new_not_set_value=new_not_set_value,
        )

        return value

    def get_subsections(self) -> Set[str]:
        """
        Return the subsection config names.
        """
        sections = set()
        for key in self.keys():
            parts = key.split(".", 2)
            if len(parts) == 3:
                sections.add(parts[1])
        return sections

    def __getitem__(self, name: str) -> Any:
        # Config
        config_type = self.config_map[name].get("type", str)
        parser = PARSER_MAP[config_type]
        config_default = self.config_map[name].get("default", None)

        if name in self.override:
            return self.override[name]

        # Value
        splitted = name.split(".")

        value = self.manager(
            splitted[-1], namespace=splitted[:-1], parser=parser, raise_error=False
        )

        if value == NO_VALUE:
            return parser(config_default)

        return value

    def display(self, display_all: bool = False) -> None:
        """
        Show the Comet config variables and values.
        """
        n = 1
        print("=" * 65)
        print("Comet config variables and values, in order of preference:")
        print("    %d) Operating System Variable" % n)
        n += 1
        for path in ["./.env", "~/.comet.config", "./.comet.config"]:
            path = clean_config_path(path)
            if os.path.exists(path):
                print("    %d) %s" % (n, path))
                n += 1
        print("=" * 65)
        print("Settings:\n")
        last_section = None
        for section, setting in sorted(
            [key.rsplit(".", 1) for key in self.config_map.keys()]
        ):
            key = "%s.%s" % (section, setting)
            value = self[key]
            if "." in section:
                section = section.replace(".", "_")
            if value is None:
                value = "..."
            default_value = self.config_map[key].get("default", None)
            if value == default_value or value == "...":
                if display_all:
                    if section != last_section:
                        if last_section is not None:
                            print()  # break between sections
                        print("[%s]" % section)
                        last_section = section
                    print("%s = %s" % (setting, value))
            else:
                if section != last_section:
                    if last_section is not None:
                        print("")  # break between sections
                    print("[%s]" % section)
                    last_section = section
                print("%s = %s" % (setting, value))
        print("=" * 65)

    def get_setting_key(self, setting: str) -> str:
        # Given a setting short-name, return proper ".comet.config" name
        # eg, given "api_key" return "comet.api_key"
        # eg, given "logging_console" return "comet.logging.console"
        subsections = self.get_subsections()
        key = None
        for prefix in subsections:
            if setting.startswith(prefix + "_"):
                key = ("comet.%s." % prefix) + setting[len(prefix) + 1 :]
                break
        if key is None:
            key = "comet." + setting
        return key

    def get_setting_name(self, setting: str) -> str:
        # Given a setting short-name, return proper env NAME
        # eg, given "api_key" return "COMET_API_KEY"
        # eg, given "logging_console" return "COMET_LOGGING_CONSOLE"
        subsections = self.get_subsections()
        name = None
        for prefix in subsections:
            if setting.startswith(prefix + "_"):
                name = ("COMET_%s_" % prefix.upper()) + (
                    setting[len(prefix) + 1 :].upper()
                )
                break
        if name is None:
            name = "COMET_" + setting.upper()
        return name

    def validate_value(self, key: str, value: Any) -> OptionValidationResult:
        """
        Validates and converts value to proper type, or
        fails.

        Returns a tuple (valid, reason_if_failed)
        """
        if key in self.config_map:
            if value in [None, ""]:
                return OptionValidationResult(
                    option_found=True, value_valid=False, failure_reason="invalid value"
                )

            stype = self.config_map[key]["type"]
            if stype == "int_list":
                if not isinstance(value, list) or not all(
                    [isinstance(v, int) for v in value]
                ):
                    return OptionValidationResult(
                        option_found=True,
                        value_valid=False,
                        failure_reason="not all values in list are integers",
                    )

            elif not isinstance(value, stype):  # specific type, like bool, int, str
                return OptionValidationResult(
                    option_found=True,
                    value_valid=False,
                    failure_reason="value is wrong type for setting; type `%s` given but type `%s` expected"
                    % (type(value).__name__, stype.__name__),
                )

            return OptionValidationResult(
                option_found=True, value_valid=True, failure_reason=None
            )

        else:
            return OptionValidationResult(
                option_found=False,
                value_valid=False,
                failure_reason="unsupported configuration option",
            )

    def _set_settings(self, settings, environ=False):
        for setting in settings:
            key = self.get_setting_key(setting)
            value = settings[setting]
            validation_result = self.validate_value(key, value)
            if validation_result.value_valid:
                if environ:
                    name = self.get_setting_name(setting)
                    os.environ[name] = str(value)
                else:
                    self[key] = value
            elif not validation_result.option_found:
                # find similar Comet keys for suggestion
                comet_keys = self.similar_settings_keys(setting)
                if len(comet_keys) > 0:
                    LOGGER.warning(
                        CONFIG_SETTING_INCORRECT_KEY_SUGGESTION_WARNING,
                        setting,
                        comet_keys,
                    )
                else:
                    LOGGER.warning(CONFIG_SETTING_NOT_FOUND_NO_SUGGESTIONS, setting)
            else:
                LOGGER.warning(
                    CONFIG_SETTING_FAILED_WARNING,
                    setting,
                    value,
                    validation_result.failure_reason,
                )

    def save(
        self,
        directory: Optional[str] = None,
        filename: Optional[str] = None,
        save_all: bool = False,
        force: bool = False,
        _prompt_user_confirmation: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Save the settings to .comet.config (default) or
        other path/filename. Defaults are commented out.

        Args:
            directory: the path to save the .comet.config config settings.
            filename: the file name to use to save the config settings.
            save_all: save unset variables with defaults too
            force: force the file to save if it exists; else don't overwrite
            kwargs: key=value pairs to save
        """
        if directory is not None:
            filename = config_path_from_directory(directory)

        if filename is None:
            filename = get_default_config_path()

        if os.path.isfile(filename):
            if not force:
                LOGGER.error(CONFIG_SETTINGS_FILE_EXISTS_ERROR, filename)
                return
            else:
                # ASK the user and try to make a backup copy
                if _prompt_user_confirmation:
                    overwrite = _confirm_user_config_file_overwriting(filename)
                else:
                    # Assume user consent with only the force flag
                    overwrite = True

                if overwrite:
                    try:
                        shutil.copyfile(filename, filename + ".bak")
                    except Exception:
                        LOGGER.warning(
                            CONFIG_FAILED_BACKUP_CONFIG_KEY_WARNING, exc_info=True
                        )
                else:
                    LOGGER.warning(
                        CONFIG_SETTINGS_USER_REFUSED_OVERWRITE_ERROR, filename
                    )
                    return

        print(CONFIG_SETTINGS_SAVE_STARTED % filename, end="")
        with io.open(filename, "w", encoding="utf-8") as ini_file:
            ini_file.write("# Config file for Comet.com\n")
            ini_file.write(
                "# For help see https://www.comet.com/docs/python-sdk/getting-started/\n"
            )
            last_section = None
            for section, setting in sorted(
                [key.rsplit(".", 1) for key in self.config_map.keys()]
            ):
                key = "%s.%s" % (section, setting)
                key_arg = "%s_%s" % (section, setting)
                if key_arg in kwargs:
                    value = kwargs[key_arg]
                    del kwargs[key_arg]
                elif key_arg.upper() in kwargs:
                    value = kwargs[key_arg.upper()]
                    del kwargs[key_arg.upper()]
                else:
                    value = self[key]
                if len(kwargs) != 0:
                    raise ValueError(
                        CONFIG_SETTINGS_NOT_VALID_CONFIG_KEY_EXCEPTION
                        % list(kwargs.keys())[0]
                    )
                if "." in section:
                    section = section.replace(".", "_")
                if value is None or value == []:
                    value = "..."
                default_value = self.config_map[key].get("default", None)
                LOGGER.debug("default value for %s is %s", key, default_value)
                if value == default_value or value == "...":
                    # It is a default value
                    # Only save it, if save_all is True:
                    if save_all:
                        if section != last_section:
                            if section is not None:
                                ini_file.write("\n")  # break between sections
                            ini_file.write("[%s]\n" % section)
                            last_section = section
                        if isinstance(value, list):
                            value = ",".join(value)
                        ini_file.write("# %s = %s\n" % (setting, value))
                else:
                    # Not a default value; write it out:
                    if section != last_section:
                        if section is not None:
                            ini_file.write("\n")  # break between sections
                        ini_file.write("[%s]\n" % section)
                        last_section = section
                    if isinstance(value, list):
                        value = ",".join([str(v) for v in value])
                    ini_file.write("%s = %s\n" % (setting, value))
        print(CONFIG_SETTINGS_SAVE_COMPLETED)

    def get_config_origin(self, name: str) -> Optional[str]:
        splitted = name.split(".")

        for env in self.manager.envs:
            value = env.get(splitted[-1], namespace=splitted[:-1])

            if value != NO_VALUE:
                return env

        return None

    def has_enabled_by_minimal_backend_version(
        self,
        minimal_backend_version_key: str,
        current_backend_version: Optional[SemanticVersion],
    ) -> BackendFeatureSupportByVersion:
        minimal_backend_version = self[minimal_backend_version_key]
        min_version = SemanticVersion.parse(minimal_backend_version)

        enabled = (
            current_backend_version is not None
            and current_backend_version >= min_version
        )
        return BackendFeatureSupportByVersion(enabled, minimal_backend_version)

    def has_direct_s3_file_upload_enabled(self) -> bool:
        return self.get_bool(None, "comet.s3_direct_multipart.upload_enabled")

    def set_direct_s3_file_upload_enabled(self, enabled: bool) -> None:
        self["comet.s3_direct_multipart.upload_enabled"] = enabled

    def has_fallback_to_offline_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.fallback_streamer.fallback_to_offline_min_backend_version",
            current_backend_version=current_backend_version,
        )

    def has_api_experiment_delete_tags_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.api_experiment.delete_tags_minimum_backend_version",
            current_backend_version=current_backend_version,
        )

    def has_api_experiment_delete_parameters_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.api_experiment.delete_parameters_minimum_backend_version",
            current_backend_version=current_backend_version,
        )

    def has_artifact_remote_assets_batching_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.artifact.remote_assets_batch_minimum_backend_version",
            current_backend_version=current_backend_version,
        )

    def has_api_experiment_get_all_metrics_enabled(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.api_experiment.get_all_experiment_metrics_minimum_backend_version",
            current_backend_version=current_backend_version,
        )

    def should_use_raw_throttling_messages(
        self, current_backend_version: Optional[SemanticVersion]
    ) -> BackendFeatureSupportByVersion:
        return self.has_enabled_by_minimal_backend_version(
            minimal_backend_version_key="comet.internal.raw_throttling_messages_minimum_backend_version",
            current_backend_version=current_backend_version,
        )

    def _get_user_expanded_path(
        self,
        user_value: Optional[str],
        config_name: str,
        default: Optional[str] = None,
        not_set_value: Optional[str] = None,
    ):
        config_path_value = self.get_string(
            user_value=user_value,
            config_name=config_name,
            default=default,
            not_set_value=not_set_value,
        )
        if config_path_value is not None:
            config_path_value = expand_user_home_path(config_path_value)

        return config_path_value

    def get_comet_offline_directory(
        self,
        user_value: Optional[str] = None,
        default: Optional[str] = None,
        not_set_value: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_user_expanded_path(
            user_value=user_value,
            config_name="comet.offline_directory",
            default=default,
            not_set_value=not_set_value,
        )

    def get_comet_git_directory(
        self,
        user_value: Optional[str] = None,
        default: Optional[str] = None,
        not_set_value: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_user_expanded_path(
            user_value=user_value,
            config_name="comet.git_directory",
            default=default,
            not_set_value=not_set_value,
        )

    def get_log_file_path(
        self,
        user_value: Optional[str] = None,
        default: Optional[str] = None,
        not_set_value: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_user_expanded_path(
            user_value=user_value,
            config_name="comet.logging.file",
            default=default,
            not_set_value=not_set_value,
        )
