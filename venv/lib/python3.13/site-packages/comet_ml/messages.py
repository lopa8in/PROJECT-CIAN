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
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

from ._typing import MemoryUploadable
from .assets import asset_item
from .constants import ASSET_TYPE_3D_POINTS
from .convert_utils import fix_special_floats
from .json_encoder import NestedEncoder
from .utils import encode_metadata, local_timestamp

HttpMessageType = Union[
    "LogOtherMessage",
    "LogDependencyMessage",
    "SystemInfoMessage",
    "InstalledPackagesMessage",
    "HtmlMessage",
    "HtmlOverrideMessage",
    "StandardOutputMessage",
    "FileNameMessage",
    "GpuStaticInfoMessage",
    "GitMetadataMessage",
    "ModelGraphMessage",
    "OsPackagesMessage",
    "SystemDetailsMessage",
]


class RegisterModelMessageCallbacks(NamedTuple):
    upload_status_observer_callback: Optional[Callable[..., Any]]
    on_model_register: Optional[Callable[..., Any]]
    on_failed_model_register: Optional[Callable[..., Any]]


class UploadAssetMessageCallbacks(NamedTuple):
    on_asset_upload: Optional[Callable[..., Any]]
    on_failed_asset_upload: Optional[Callable[..., Any]]


class UploadModelMessageCallbacks(NamedTuple):
    on_model_upload: Optional[Callable[..., Any]]
    on_failed_model_upload: Optional[Callable[..., Any]]


MessageCallbacks = Union[
    UploadAssetMessageCallbacks,
    UploadModelMessageCallbacks,
    RegisterModelMessageCallbacks,
]


class BaseMessage(metaclass=ABCMeta):
    source_manual = "manual"

    def __init__(self, message_id: Optional[int] = None):
        self.message_id = message_id

    @property
    @abstractmethod
    def type(self):
        pass

    def get_user_friendly_identifier(self) -> str:
        """
        Returns a user-friendly identifier for the instance.

        This identifier is intended to be used for logging, user-facing
        messages, or any context where a descriptive and easily understandable
        representation of the object is required.

        The subclasses should override this method to return a string which
        is the best suitable for specific message, such as: name, file_name, etc.
        """
        return self.type

    @staticmethod
    def translate_message_class__to_handler_name(class_name):
        """
        This function turns the message class name to the equivalent function in the Message class.
        """
        class_name_splitted = re.findall("[A-Z][^A-Z]*", class_name)
        class_name_splitted_lowered = [word.lower() for word in class_name_splitted]
        method_name = "_".join(class_name_splitted_lowered[:-1])
        return "set_" + method_name

    @classmethod
    def create(cls, context: Optional[str], **kwargs: Any) -> "BaseMessage":
        return cls(**kwargs)

    def to_batch_message_dict(self):
        """To be defined by subclasses to return JSON dictionary representation to be used for batching"""
        return None

    @abstractmethod
    def to_message_dict(self):
        pass

    def to_db_message_dict(self):
        return self.to_message_dict()

    def to_ws_message_dict(self):
        return self.filtered_message_dict()

    def filtered_message_dict(self, keys_to_filter: List[str] = None):
        if keys_to_filter is None:
            keys_to_filter = ["message_id"]
        message_dict = self.to_message_dict()
        return {
            key: value
            for key, value in message_dict.items()
            if key not in keys_to_filter
        }

    def has_invalid_message_id(self):
        return self.message_id is None or self.message_id <= 0

    def get_message_callbacks(self) -> Optional[MessageCallbacks]:
        return None

    def set_message_callbacks(self, callbacks: MessageCallbacks):
        pass


class CloseMessage(BaseMessage):
    """A special message indicating Streamer to ends and exit"""

    type = "close"

    def to_message_dict(self):
        pass


class UploadFileMessage(BaseMessage):
    type = "file_upload"

    def __init__(
        self,
        file_path: str,
        upload_type: str,
        additional_params: Optional[Dict[str, Optional[Any]]],
        metadata: Optional[Dict[str, Any]],
        size: int,
        clean: bool = True,
        critical: bool = False,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.local_timestamp = local_timestamp()

        self.file_path = file_path
        self.upload_type = upload_type
        self.additional_params = additional_params
        self.metadata = metadata
        self.clean = clean
        self._size = size
        self._critical = critical
        self._on_asset_upload = on_asset_upload
        self._on_failed_asset_upload = on_failed_asset_upload

        if (
            additional_params
            and "fileName" in additional_params
            and additional_params["fileName"] is None
        ):
            raise TypeError("fileName value must not be None")

    def get_user_friendly_identifier(self) -> str:
        identifier = None
        if self.additional_params is not None:
            identifier = self.additional_params.get("fileName")

        if identifier is None:
            identifier = super().get_user_friendly_identifier()

        return identifier

    def get_message_callbacks(self) -> UploadAssetMessageCallbacks:
        return UploadAssetMessageCallbacks(
            on_asset_upload=self._on_asset_upload,
            on_failed_asset_upload=self._on_failed_asset_upload,
        )

    def set_message_callbacks(self, callbacks: UploadAssetMessageCallbacks):
        self._on_asset_upload = callbacks.on_asset_upload
        self._on_failed_asset_upload = callbacks.on_failed_asset_upload

    def to_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if (not key.startswith("_"))
        }

    def to_db_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["_on_asset_upload", "_on_failed_asset_upload"]
        }

    @classmethod
    def from_db_message_dict(
        cls: Type["UploadFileMessage"], message_dict: Dict[str, Any]
    ) -> "UploadFileMessage":
        message = cls(
            file_path=message_dict["file_path"],
            upload_type=message_dict["upload_type"],
            additional_params=message_dict.get("additional_params", None),
            metadata=message_dict.get("metadata", None),
            size=message_dict.get("_size", 0),
            clean=message_dict.get("clean", True),
            critical=message_dict.get("_critical", False),
            message_id=message_dict.get("message_id", None),
        )
        message.local_timestamp = message_dict["local_timestamp"]
        return message


class UploadInMemoryMessage(BaseMessage):
    type = "file_upload"

    def __init__(
        self,
        file_like: MemoryUploadable,
        upload_type: str,
        additional_params: Dict[str, Optional[Any]],
        metadata: Dict[Any, Any],
        size: int,
        critical: bool = False,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.local_timestamp = local_timestamp()

        self.file_like = file_like
        self.upload_type = upload_type
        self.additional_params = additional_params
        self.metadata = metadata
        self._size = size
        self._critical = critical
        self._on_asset_upload = on_asset_upload
        self._on_failed_asset_upload = on_failed_asset_upload

        if (
            additional_params
            and "fileName" in additional_params
            and additional_params["fileName"] is None
        ):
            raise TypeError("fileName value must not be None")

    def get_user_friendly_identifier(self) -> str:
        identifier = None
        if self.additional_params is not None:
            identifier = self.additional_params.get("fileName")

        if identifier is None:
            identifier = super().get_user_friendly_identifier()

        return identifier

    def get_message_callbacks(self) -> UploadAssetMessageCallbacks:
        return UploadAssetMessageCallbacks(
            on_asset_upload=self._on_asset_upload,
            on_failed_asset_upload=self._on_failed_asset_upload,
        )

    def set_message_callbacks(self, callbacks: UploadAssetMessageCallbacks):
        self._on_asset_upload = callbacks.on_asset_upload
        self._on_failed_asset_upload = callbacks.on_failed_asset_upload

    def to_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if (not key.startswith("_"))
        }


class RemoteAssetMessage(BaseMessage):
    type = "remote_file"

    def __init__(
        self,
        remote_uri: Any,
        upload_type: str,
        additional_params: Dict[str, Optional[Any]],
        metadata: Optional[Dict[str, str]],
        size: int,
        critical: bool = False,
        on_asset_upload: Optional[Callable] = None,
        on_failed_asset_upload: Optional[Callable] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.remote_uri = remote_uri
        self.upload_type = upload_type
        self.additional_params = additional_params
        self.metadata = metadata
        self._size = size
        self._critical = critical
        self._on_asset_upload = on_asset_upload
        self._on_failed_asset_upload = on_failed_asset_upload

    def get_user_friendly_identifier(self) -> str:
        return self.remote_uri

    @property
    def asset_id(self) -> Optional[str]:
        if self.additional_params is None:
            return None
        return self.additional_params.get("assetId", None)

    @property
    def artifact_version_id(self) -> Optional[str]:
        if self.additional_params is None:
            return None
        return self.additional_params.get("artifactVersionId", None)

    def is_artifact_asset(self) -> bool:
        return self.artifact_version_id is not None

    def get_message_callbacks(self) -> UploadAssetMessageCallbacks:
        return UploadAssetMessageCallbacks(
            on_asset_upload=self._on_asset_upload,
            on_failed_asset_upload=self._on_failed_asset_upload,
        )

    def set_message_callbacks(self, callbacks: UploadAssetMessageCallbacks):
        self._on_asset_upload = callbacks.on_asset_upload
        self._on_failed_asset_upload = callbacks.on_failed_asset_upload

    def to_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if (not key.startswith("_"))
        }

    def to_batch_message_dict(self):
        message_dict = {
            "link": self.remote_uri,
            "type": self.upload_type,
        }
        if self.metadata is not None:
            message_dict["metadata"] = encode_metadata(self.metadata)
        message_dict.update(self.additional_params)
        return message_dict

    @classmethod
    def from_db_message_dict(
        cls: Type["RemoteAssetMessage"], message_dict: Dict[str, Any]
    ) -> "RemoteAssetMessage":
        return cls(
            remote_uri=message_dict["remote_uri"],
            upload_type=message_dict["upload_type"],
            additional_params=message_dict["additional_params"],
            metadata=message_dict["metadata"],
            size=0,
            message_id=message_dict.get("message_id", None),
        )


class OsPackagesMessage(BaseMessage):
    type = "os_packages"

    def __init__(
        self, os_packages: List[str], message_id: Optional[int] = None
    ) -> None:
        super().__init__(message_id)
        self.os_packages = os_packages

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["OsPackagesMessage"], message_dict: Dict[str, Any]
    ) -> "OsPackagesMessage":
        return cls(
            os_packages=message_dict["os_packages"],
            message_id=message_dict.get("message_id", None),
        )


class ModelGraphMessage(BaseMessage):
    type = "graph"

    def __init__(self, graph: str, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)
        self.graph = graph

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["ModelGraphMessage"], message_dict: Dict[str, Any]
    ) -> "ModelGraphMessage":
        return cls(
            graph=message_dict["graph"], message_id=message_dict.get("message_id", None)
        )


class SystemDetailsMessage(BaseMessage):
    type = "system_details"

    def __init__(
        self,
        command: Optional[Union[str, List[str]]],
        env: Optional[Dict[str, str]],
        hostname: Optional[str],
        ip: Optional[str],
        machine: Optional[str],
        os_release: Optional[str],
        os_type: Optional[str],
        os: Optional[str],
        pid: Optional[int],
        processor: Optional[str],
        python_exe: Optional[str],
        python_version_verbose: Optional[str],
        python_version: Optional[str],
        user: Optional[str],
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)

        self.command = command
        self.env = env
        self.hostname = hostname
        self.ip = ip
        self.machine = machine
        self.os = os
        self.os_release = os_release
        self.os_type = os_type
        self.pid = pid
        self.processor = processor
        self.python_exe = python_exe
        self.python_version = python_version
        self.python_version_verbose = python_version_verbose
        self.user = user

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["SystemDetailsMessage"], message_dict: Dict[str, Any]
    ) -> "SystemDetailsMessage":
        return cls(
            command=message_dict.get("command", None),
            env=message_dict.get("env", None),
            hostname=message_dict.get("hostname", None),
            ip=message_dict.get("ip", None),
            machine=message_dict.get("machine", None),
            os=message_dict.get("os", None),
            os_release=message_dict.get("os_release", None),
            os_type=message_dict.get("os_type", None),
            pid=message_dict.get("pid", None),
            processor=message_dict.get("processor", None),
            python_exe=message_dict.get("python_exe", None),
            python_version=message_dict.get("python_version", None),
            python_version_verbose=message_dict.get("python_version_verbose", None),
            user=message_dict.get("user", None),
            message_id=message_dict.get("message_id", None),
        )


class CloudDetailsMessage(BaseMessage):
    type = "cloud_details"

    def __init__(
        self,
        provider: str,
        cloud_metadata: Dict[str, Any],
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)

        self.provider = provider
        self.cloud_metadata = cloud_metadata

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["CloudDetailsMessage"], message_dict: Dict[str, Any]
    ) -> "CloudDetailsMessage":
        return cls(
            provider=message_dict["provider"],
            cloud_metadata=message_dict["cloud_metadata"],
            message_id=message_dict.get("message_id", None),
        )


class ParameterMessage(BaseMessage):
    """The Message type to encapsulate named parameter value.
    The parameter value can be either float or the list of floats."""

    type = "parameter_msg"
    source_autologger = "auto-logged"
    source_cli = "cli"

    def __init__(
        self,
        context: Optional[str] = None,
        timestamp: Optional[int] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        if timestamp is None:
            timestamp = local_timestamp()

        self.context = context
        self.local_timestamp = timestamp

        # The following attributes are optional
        self.param = None  # type: Optional[Dict[str, Any]]
        self.params = None  # type: Optional[Dict[str, Any]]

    def set_param(self, name, value, step=None, source=source_autologger):
        # type: (str, Any, Optional[int], str) -> None
        safe_value = fix_special_floats(value)
        self.param = {
            "paramName": name,
            "paramValue": safe_value,
            "step": step,
            "source": source,
        }

    def set_params(self, name, values, step=None, source=source_autologger):
        # type: (str, List[Any], Optional[int], str) -> None
        safe_values = list(map(fix_special_floats, values))
        self.params = {
            "paramName": name,
            "paramValue": safe_values,
            "step": step,
            "source": source,
        }

    def get_param_name(self):
        # type: () -> Optional[str]
        """Returns the name of the parameter associated with this message."""
        if self.param is not None:
            return self.param["paramName"]
        elif self.params is not None:
            return self.params["paramName"]
        else:
            return None

    def get_source(self):
        """Returns the source of the parameter value"""
        param_dict = self._get_param_dict()
        if param_dict is not None:
            return param_dict.get("source")
        return None

    def _get_param_dict(self):
        # type: () -> Optional[Dict[str, Any]]
        if self.param is not None:
            return self.param
        elif self.params is not None:
            return self.params
        else:
            return None

    def get_user_friendly_identifier(self) -> str:
        identifier = self.get_param_name()
        if identifier is None:
            identifier = super().get_user_friendly_identifier()

        return identifier

    @classmethod
    def from_message_dict(
        cls: Type["ParameterMessage"], message_dict: Dict[str, Any]
    ) -> "ParameterMessage":
        """Recreate a ParameterMessage from its Dict representation"""
        parameter_message = cls(
            context=message_dict.get("context", None),
            timestamp=message_dict.get("local_timestamp", None),
            message_id=message_dict.get("message_id", None),
        )

        if message_dict.get("param", None) is not None:
            parameter_message.set_param(
                message_dict["param"]["paramName"],
                message_dict["param"]["paramValue"],
                step=message_dict["param"]["step"],
                source=message_dict["param"].get(
                    "source", ParameterMessage.source_autologger
                ),
            )
        elif message_dict.get("params", None) is not None:
            parameter_message.set_params(
                message_dict["params"]["paramName"],
                message_dict["params"]["paramValue"],
                step=message_dict["params"]["step"],
                source=message_dict["params"].get(
                    "source", ParameterMessage.source_autologger
                ),
            )
        else:
            # to support an offline experiment generated with SDK version 2.0.12
            # (see test_offline_sender.py#test_backward_compatibility_2_0_12)
            pass

        return parameter_message

    @classmethod
    def from_db_message_dict(
        cls: Type["ParameterMessage"], message_dict: Dict[str, Any]
    ) -> "ParameterMessage":
        return cls.from_message_dict(message_dict)

    def to_batch_message_dict(self) -> Dict[str, Any]:
        repr_dict = dict()
        param_dict = self._get_param_dict()
        if param_dict is not None:
            repr_dict["parameterName"] = param_dict["paramName"]
            repr_dict["parameterValue"] = param_dict["paramValue"]
            repr_dict["step"] = param_dict["step"]
            repr_dict["source"] = param_dict["source"]

        repr_dict["context"] = self.context
        repr_dict["timestamp"] = self.local_timestamp
        return repr_dict

    def to_message_dict(self):
        return _keep_non_null_public_fields(self.__dict__)

    def __str__(self) -> str:
        return f"ParameterMessage: {str(self.__dict__)}"


class MetricMessage(BaseMessage):
    type = "metric_msg"
    metric_key = "metric"

    """The Message type to encapsulate named metric value."""

    def __init__(
        self,
        context: Optional[str] = None,
        timestamp: Optional[int] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.context = context
        self.metric = None  # type: Optional[Dict[str, Any]]
        if timestamp is None:
            timestamp = local_timestamp()

        self.local_timestamp = timestamp

    def set_metric(self, name, value, step=None, epoch=None):
        safe_value = fix_special_floats(value)
        self.metric = {
            "metricName": name,
            "metricValue": safe_value,
            "step": step,
            "epoch": epoch,
        }

    def get_user_friendly_identifier(self) -> str:
        if self.metric is not None:
            return self.metric.get("metricName")
        else:
            return super().get_user_friendly_identifier()

    @classmethod
    def from_message_dict(
        cls: Type["MetricMessage"], message_dict: Dict[str, Any]
    ) -> "MetricMessage":
        """Recreate a MetricMessage from its Dict representation"""
        metric_message = cls(
            context=message_dict.get("context", None),
            timestamp=message_dict.get("local_timestamp", None),
            message_id=message_dict.get("message_id", None),
        )

        if message_dict.get(cls.metric_key, None):
            metric_message.set_metric(
                message_dict[cls.metric_key]["metricName"],
                message_dict[cls.metric_key]["metricValue"],
                step=message_dict[cls.metric_key].get("step", None),
                epoch=message_dict[cls.metric_key].get("epoch", None),
            )
        else:
            raise ValueError("no metric message data found")

        return metric_message

    @classmethod
    def from_db_message_dict(
        cls: Type["MetricMessage"], message_dict: Dict[str, Any]
    ) -> "MetricMessage":
        return cls.from_message_dict(message_dict)

    def to_batch_message_dict(self) -> Dict[str, Any]:
        repr_dict = dict(self.metric)
        repr_dict["context"] = self.context
        repr_dict["timestamp"] = self.local_timestamp
        return repr_dict

    def __str__(self) -> str:
        return f"MetricMessage: {self.to_batch_message_dict()}"

    def to_message_dict(self):
        return _keep_non_null_public_fields(self.__dict__)


class InstalledPackagesMessage(BaseMessage):
    type = "installed_packages"

    def __init__(
        self, installed_packages: List[str], message_id: Optional[int] = None
    ) -> None:
        super().__init__(message_id)
        self.installed_packages = installed_packages

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["InstalledPackagesMessage"], message_dict: Dict[str, Any]
    ) -> "InstalledPackagesMessage":
        return cls(
            installed_packages=message_dict["installed_packages"],
            message_id=message_dict.get("message_id", None),
        )


class LogOtherMessage(BaseMessage):
    type = "log_other"

    def __init__(self, key: Any, value: Any, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)

        self.key = key
        self.value = value

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["LogOtherMessage"], message_dict: Dict[str, Any]
    ) -> "LogOtherMessage":
        return cls(
            key=message_dict["key"],
            value=message_dict["value"],
            message_id=message_dict.get("message_id", None),
        )


class RemoteModelMessage(BaseMessage):
    type = "remote_model"

    def __init__(
        self,
        model_name: str,
        remote_assets: List[Dict[str, Any]],
        on_model_upload: Optional[Callable] = None,
        on_failed_model_upload: Optional[Callable] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.model_name = model_name
        self.remote_assets = remote_assets
        self.on_model_upload = on_model_upload
        self.on_failed_model_upload = on_failed_model_upload

    def get_message_callbacks(self) -> UploadModelMessageCallbacks:
        return UploadModelMessageCallbacks(
            on_model_upload=self.on_model_upload,
            on_failed_model_upload=self.on_failed_model_upload,
        )

    def set_message_callbacks(self, callbacks: UploadModelMessageCallbacks):
        self.on_model_upload = callbacks.on_model_upload
        self.on_failed_model_upload = callbacks.on_failed_model_upload

    def to_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["on_model_upload", "on_failed_model_upload"]
        }

    def get_user_friendly_identifier(self) -> str:
        return self.model_name

    @classmethod
    def from_db_message_dict(
        cls: Type["RemoteModelMessage"], message_dict: Dict[str, Any]
    ) -> "RemoteModelMessage":
        return cls(
            model_name=message_dict["model_name"],
            remote_assets=message_dict["remote_assets"],
            message_id=message_dict.get("message_id", None),
        )


class RegisterModelMessage(BaseMessage):
    type = "register_model"

    def __init__(
        self,
        experiment_id: str,
        model_name: str,
        version: str,
        workspace: Optional[str],
        registry_name: str,
        public: bool,
        description: str,
        comment: str,
        tags: List[str],
        status: Optional[str],
        stages: Optional[List[str]],
        upload_status_observer_callback: Optional[Callable],
        on_model_register: Optional[Callable],
        on_failed_model_register: Optional[Callable],
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.version = version
        self.workspace = workspace
        self.registry_name = registry_name
        self.public = public
        self.description = description
        self.comment = comment
        self.tags = tags
        self.status = status
        self.stages = stages
        self.upload_status_observer_callback = upload_status_observer_callback
        self.on_model_register = on_model_register
        self.on_failed_model_register = on_failed_model_register

    def get_message_callbacks(self) -> RegisterModelMessageCallbacks:
        return RegisterModelMessageCallbacks(
            upload_status_observer_callback=self.upload_status_observer_callback,
            on_model_register=self.on_model_register,
            on_failed_model_register=self.on_failed_model_register,
        )

    def set_message_callbacks(self, callbacks: RegisterModelMessageCallbacks):
        self.upload_status_observer_callback = callbacks.upload_status_observer_callback
        self.on_model_register = callbacks.on_model_register
        self.on_failed_model_register = callbacks.on_failed_model_register

    def get_user_friendly_identifier(self) -> str:
        return self.model_name

    def to_message_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "upload_status_observer_callback",
                "on_model_register",
                "on_failed_model_register",
            ]
        }

    @classmethod
    def from_db_message_dict(
        cls: Type["RegisterModelMessage"], message_dict: Dict[str, Any]
    ) -> "RegisterModelMessage":
        return cls(
            experiment_id=message_dict["experiment_id"],
            model_name=message_dict["model_name"],
            version=message_dict["version"],
            workspace=message_dict["workspace"],
            registry_name=message_dict["registry_name"],
            public=message_dict["public"],
            description=message_dict["description"],
            comment=message_dict["comment"],
            tags=message_dict["tags"],
            status=message_dict["status"],
            stages=message_dict["stages"],
            message_id=message_dict.get("message_id", None),
            upload_status_observer_callback=None,
            on_model_register=None,
            on_failed_model_register=None,
        )


class FileNameMessage(BaseMessage):
    type = "file_name"

    def __init__(self, file_name: str, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)
        self.file_name = file_name

    def get_user_friendly_identifier(self) -> str:
        return self.file_name

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["FileNameMessage"], message_dict: Dict[str, Any]
    ) -> "FileNameMessage":
        return cls(
            file_name=message_dict["file_name"],
            message_id=message_dict.get("message_id", None),
        )


class HtmlMessage(BaseMessage):
    type = "html"

    def __init__(self, html: str, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)
        self.html = html

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["HtmlMessage"], message_dict: Dict[str, Any]
    ) -> "HtmlMessage":
        return cls(
            html=message_dict["html"], message_id=message_dict.get("message_id", None)
        )


class HtmlOverrideMessage(BaseMessage):
    type = "htmlOverride"

    def __init__(self, htmlOverride: str, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)
        self.htmlOverride = htmlOverride

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["HtmlOverrideMessage"], message_dict: Dict[str, Any]
    ) -> "HtmlOverrideMessage":
        return cls(
            htmlOverride=message_dict["htmlOverride"],
            message_id=message_dict.get("message_id", None),
        )


class GpuStaticInfoMessage(BaseMessage):
    type = "gpu_static_info"

    def __init__(
        self, gpu_static_info: List[Dict[str, Any]], message_id: Optional[int] = None
    ) -> None:
        super().__init__(message_id)
        self.gpu_static_info = gpu_static_info

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["GpuStaticInfoMessage"], message_dict: Dict[str, Any]
    ) -> "GpuStaticInfoMessage":
        return cls(
            gpu_static_info=message_dict["gpu_static_info"],
            message_id=message_dict.get("message_id", None),
        )


class GitMetadataMessage(BaseMessage):
    type = "git_metadata"

    def __init__(
        self, git_metadata: Dict[str, Any], message_id: Optional[int] = None
    ) -> None:
        super().__init__(message_id)
        self.git_metadata = git_metadata

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["GitMetadataMessage"], message_dict: Dict[str, Any]
    ) -> "GitMetadataMessage":
        return cls(
            git_metadata=message_dict["git_metadata"],
            message_id=message_dict.get("message_id", None),
        )


class SystemInfoMessage(BaseMessage):
    type = "system_info"

    def __init__(self, key: str, value: Any, message_id: Optional[int] = None) -> None:
        super().__init__(message_id)
        self.system_info = {
            "key": key,
            "value": value,
        }

    def to_message_dict(self):
        return self.__dict__

    @classmethod
    def from_db_message_dict(
        cls: Type["SystemInfoMessage"], message_dict: Dict[str, Any]
    ) -> "SystemInfoMessage":
        system_info = message_dict["system_info"]
        return cls(
            key=system_info["key"],
            value=system_info["value"],
            message_id=message_dict.get("message_id", None),
        )


class LogDependencyMessage(BaseMessage):
    type = "log_dependency"

    def __init__(
        self,
        name: Any,
        version: Any,
        timestamp: int = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.name = name
        self.version = version

        if timestamp is None:
            timestamp = local_timestamp()
        self.local_timestamp = timestamp

    def to_message_dict(self):
        return _keep_non_null_public_fields(self.__dict__)

    @classmethod
    def from_db_message_dict(
        cls: Type["LogDependencyMessage"], message_dict: Dict[str, Any]
    ) -> "LogDependencyMessage":
        return cls(
            name=message_dict["name"],
            version=message_dict["version"],
            timestamp=message_dict.get("local_timestamp", None),
            message_id=message_dict.get("message_id", None),
        )


class StandardOutputMessage(BaseMessage):
    type = "standard_output"

    def __init__(
        self,
        output: str,
        stderr: bool = False,
        context: Optional[str] = None,
        timestamp: Optional[int] = None,
        message_id: Optional[int] = None,
    ) -> None:
        super().__init__(message_id)
        self.output = output
        self.stderr = stderr
        self.context = context

        if timestamp is None:
            timestamp = local_timestamp()
        self.local_timestamp = timestamp

    @classmethod
    def create(cls, context: Optional[str], **kwargs: Any) -> BaseMessage:
        kwargs["context"] = context
        return StandardOutputMessage(**kwargs)

    @classmethod
    def from_message_dict(
        cls: Type["StandardOutputMessage"], message_dict: Dict[str, Any]
    ) -> "StandardOutputMessage":
        """Recreate a StandardOutputMessage from its Dict representation"""
        stdout_message = cls(
            output=message_dict.get("output", ""),
            stderr=message_dict.get("stderr", False),
            context=message_dict.get("context", None),
            timestamp=message_dict.get("local_timestamp", None),
            message_id=message_dict.get("message_id", None),
        )
        return stdout_message

    def to_message_dict(self):
        return _keep_non_null_public_fields(self.__dict__)

    @classmethod
    def from_db_message_dict(
        cls: Type["StandardOutputMessage"], message_dict: Dict[str, Any]
    ) -> "StandardOutputMessage":
        return cls.from_message_dict(message_dict)


class WebSocketMessage(BaseMessage):
    """
    A bean used to send messages to the server over websockets.
    """

    type = "ws_msg"

    def __init__(
        self, context: Optional[str] = None, message_id: Optional[int] = None
    ) -> None:
        super().__init__(message_id)
        self.local_timestamp = local_timestamp()

        # The following attributes are optional
        self.graph = None
        self.code = None
        self.stdout = None
        self.stderr = None
        self.fileName = None
        self.env_details = None
        self.html = None
        self.htmlOverride = None
        self.installed_packages = None
        self.os_packages = None
        self.log_other = None
        self.gpu_static_info = None
        self.git_meta = None
        self.log_dependency = None
        self.log_system_info = None
        self.context = context

    def set_log_other(self, key: Any, value: Any):
        self.log_other = {"key": key, "val": value}

    def set_log_dependency(self, name, version, timestamp=None):
        self.log_dependency = {"name": name, "version": version}
        if timestamp is not None:
            self.local_timestamp = timestamp

    def set_system_info(self, key: Any, value: Any):
        self.log_system_info = {"key": key, "value": value}

    def set_installed_packages(self, installed_packages: List[str]):
        self.installed_packages = installed_packages

    def set_html(self, html: str):
        self.html = html

    def set_html_override(self, htmlOverride: str):
        self.htmlOverride = htmlOverride

    def set_code(self, code: str):
        self.code = code

    def set_standard_output(
        self,
        output: str,
        stderr: bool = False,
        context: Optional[str] = None,
        timestamp: Optional[int] = None,
    ):
        self.stdout = output
        self.stderr = stderr
        if context is not None:
            self.context = context
        if timestamp is not None:
            self.local_timestamp = timestamp

    def set_file_name(self, file_name: str):
        self.fileName = file_name

    def set_gpu_static_info(self, gpu_static_info: List[Dict[str, Any]]):
        self.gpu_static_info = gpu_static_info

    def set_git_metadata(self, git_metadata: Dict[str, Any]):
        self.git_meta = git_metadata

    def set_model_graph(self, graph: str):
        self.graph = graph

    def set_os_packages(self, os_packages: List[str]):
        self.os_packages = os_packages

    def set_system_details(
        self,
        command: List[str],
        env: Dict[str, Any],
        hostname: str,
        ip: str,
        machine: str,
        os_release: str,
        os_type: str,
        os: str,
        pid: int,
        processor: str,
        python_exe: str,
        python_version_verbose: str,
        python_version: str,
        user: str,
    ):
        self.env_details = {
            "command": command,
            "env": env,
            "hostname": hostname,
            "ip": ip,
            "machine": machine,
            "os_release": os_release,
            "os_type": os_type,
            "os": os,
            "pid": pid,
            "processor": processor,
            "python_exe": python_exe,
            "python_version_verbose": python_version_verbose,
            "python_version": python_version,
            "user": user,
        }

    def to_http_message(self) -> HttpMessageType:
        """Converts this message into appropriate message supported by HTTP endpoint"""
        if self.log_other is not None:
            return LogOtherMessage(
                key=self.log_other["key"], value=self.log_other["val"]
            )
        elif self.log_dependency is not None:
            return LogDependencyMessage(
                name=self.log_dependency["name"],
                version=self.log_dependency["version"],
                timestamp=self.local_timestamp,
            )
        elif self.log_system_info is not None:
            return SystemInfoMessage(
                key=self.log_system_info["key"], value=self.log_system_info["value"]
            )
        elif self.installed_packages is not None:
            return InstalledPackagesMessage(installed_packages=self.installed_packages)
        elif self.html is not None:
            return HtmlMessage(html=self.html)
        elif self.htmlOverride is not None:
            return HtmlOverrideMessage(htmlOverride=self.htmlOverride)
        elif self.stdout is not None:
            return StandardOutputMessage(
                output=self.stdout,
                stderr=self.stderr,
                context=self.context,
                timestamp=self.local_timestamp,
            )
        elif self.fileName is not None:
            return FileNameMessage(file_name=self.fileName)
        elif self.gpu_static_info is not None:
            return GpuStaticInfoMessage(gpu_static_info=self.gpu_static_info)
        elif self.git_meta is not None:
            return GitMetadataMessage(git_metadata=self.git_meta)
        elif self.graph is not None:
            return ModelGraphMessage(graph=self.graph)
        elif self.os_packages is not None:
            return OsPackagesMessage(os_packages=self.os_packages)
        elif self.env_details is not None:
            return SystemDetailsMessage(
                command=self.env_details["command"],
                env=self.env_details["env"],
                hostname=self.env_details["hostname"],
                ip=self.env_details["ip"],
                machine=self.env_details["machine"],
                os_release=self.env_details["os_release"],
                os_type=self.env_details["os_type"],
                os=self.env_details["os"],
                pid=self.env_details["pid"],
                processor=self.env_details["processor"],
                python_exe=self.env_details["python_exe"],
                python_version_verbose=self.env_details["python_version_verbose"],
                python_version=self.env_details["python_version"],
                user=self.env_details["user"],
            )
        else:
            raise ValueError(
                "can not convert to HTTP message format, message: %r" % self
            )

    @classmethod
    def from_message_dict(
        cls: Type["WebSocketMessage"], message_dict: Dict[str, Any]
    ) -> "WebSocketMessage":
        ws_message = cls(context=message_dict.get("context", None))
        # copy values
        for k, v in message_dict.items():
            ws_message.__dict__[k] = v

        return ws_message

    @classmethod
    def from_message_dict_to_http_message(
        cls: Type["WebSocketMessage"], message_dict: Dict[str, Any]
    ) -> HttpMessageType:
        ws_message = WebSocketMessage.from_message_dict(message_dict)
        return ws_message.to_http_message()

    @classmethod
    def from_db_message_dict(
        cls: Type["WebSocketMessage"], message_dict: Dict[str, Any]
    ) -> "WebSocketMessage":
        return cls.from_message_dict(message_dict)

    def __repr__(self):
        filtered_dict = [(key, value) for key, value in self.__dict__.items() if value]
        string = ", ".join("%r=%r" % item for item in filtered_dict)
        return "Message(%s)" % string

    def to_json(self):
        json_re = json.dumps(
            self.to_message_dict(), sort_keys=True, indent=4, cls=NestedEncoder
        )
        return json_re

    def __str__(self):
        return self.to_json()

    def __len__(self):
        return len(self.to_json())

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.__dict__ == other.__dict__

    def to_message_dict(self):
        return _keep_non_null_public_fields(self.__dict__)


class Log3DCloudMessage(BaseMessage):
    type = ASSET_TYPE_3D_POINTS

    def __init__(
        self,
        name: str,
        upload_type: str,
        items: List[asset_item.AssetItem],
        thumbnail_path: Optional[str],
        metadata: Optional[Dict[str, Any]],
        step: Optional[int],
        message_id: Optional[int] = None,
        asset_id: Optional[str] = None,
    ) -> None:
        super().__init__(message_id)
        self.name = name
        self.upload_type = upload_type
        self.items = items
        self.metadata = metadata
        self.step = step
        self.thumbnail_path = thumbnail_path
        self.asset_id = asset_id

    def get_user_friendly_identifier(self) -> str:
        return self.name

    def to_message_dict(self):
        dictionary = {key: value for key, value in self.__dict__.items()}

        dictionary["items"] = [item.serialize() for item in self.items]

        return dictionary

    @classmethod
    def from_db_message_dict(
        cls: Type["Log3DCloudMessage"], message_dict: Dict[str, Any]
    ) -> "Log3DCloudMessage":
        return cls(
            name=message_dict["name"],
            upload_type=message_dict["upload_type"],
            items=asset_item.deserialize_items(message_dict["items"]),
            metadata=message_dict["metadata"],
            step=message_dict["step"],
            thumbnail_path=message_dict["thumbnail_path"],
            asset_id=message_dict.get("asset_id", None),
        )


def _keep_non_null_public_fields(message_dict: Dict[str, Any]):
    return {
        key: value
        for key, value in message_dict.items()
        if (value is not None and not key.startswith("_"))
    }
