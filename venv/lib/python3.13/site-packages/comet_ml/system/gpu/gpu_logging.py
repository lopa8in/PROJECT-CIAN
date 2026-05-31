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

"""
Author: Boris Feld

This module contains the main components of GPU information logging

"""
import logging

from comet_ml import _logging, logging_messages
from comet_ml.logging_messages import SYSTEM_GPU_MANUAL_OVERRIDE_FAILED_ERROR
from comet_ml.system.gpu import devices, identifiers
from comet_ml.system.system_metrics_types import NamedSystemMetricsList
from comet_ml.utils import metric_name, to_utf8
from comet_ml.vendor.nvidia_ml import pynvml

LOGGER = logging.getLogger(__name__)


class _DevicesToReport:
    def __init__(self):
        self._manual_override = None

    def manually_override(self, ids):
        if ids is not None:
            if len(ids) == 0:
                self._manual_override = []
                return

            try:
                pynvml.nvmlInit()
                uuids = [identifiers.get_by_index(index) for index in ids]
                self._manual_override = uuids
            except pynvml.NVMLError:
                LOGGER.error(SYSTEM_GPU_MANUAL_OVERRIDE_FAILED_ERROR, exc_info=True)
        else:
            self._manual_override = None

    def get(self):
        if self._manual_override is not None:
            result = self._manual_override
        else:
            result = devices.find_visible()
            if result is None:
                result = devices.all()

        LOGGER.debug("devices to report: {}".format(result))
        return result


_devices_to_report = _DevicesToReport()
set_devices_to_report = _devices_to_report.manually_override

_exceptions_filter = _logging.convert_exception_to_log_message(
    message=logging_messages.PYNVML_FAILED_TO_GET_GPU_DATA,
    logger=LOGGER,
    return_on_exception=None,
    exception_info=True,
    logging_level=logging.DEBUG,
)


@_exceptions_filter
def get_gpu_name(handle):
    """Returns the name of the GPU device

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1ga5361803e044c6fdf3b08523fb6d1481
    """
    name = pynvml.nvmlDeviceGetName(handle)
    return to_utf8(name)


@_exceptions_filter
def get_uuid(handle):
    """Returns the globally unique GPU device UUID

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g72710fb20f30f0c2725ce31579832654
    """
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    return to_utf8(uuid)


@_exceptions_filter
def get_memory_info(handle):
    """Returns memory info in bytes

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g2dfeb1db82aa1de91aa6edf941c85ca8
    """
    return pynvml.nvmlDeviceGetMemoryInfo(handle)


@_exceptions_filter
def get_temperature(handle):
    """Returns degrees in the Celsius scale

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g92d1c5182a14dd4be7090e3c1480b121
    """
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


@_exceptions_filter
def get_power_usage(handle):
    """Returns power usage in milliwatts

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g7ef7dff0ff14238d08a19ad7fb23fc87
    """
    return pynvml.nvmlDeviceGetPowerUsage(handle)


@_exceptions_filter
def get_power_limit(handle):
    """Returns max power usage in milliwatts

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g263b5bf552d5ec7fcd29a088264d10ad
    """
    return pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)


@_exceptions_filter
def get_gpu_utilization(handle):
    """Returns the % of utilization of the kernels during the last sample

    https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
    """
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


@_exceptions_filter
def get_memory_utilization_rate(handle):
    """Returns the % of time over the past sample period during which global (device) memory was being read or written

    https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
    """
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory


@_exceptions_filter
def get_compute_mode(handle):
    """Returns the compute mode of the GPU

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceEnumvs.html#group__nvmlDeviceEnumvs_1gbed1b88f2e3ba39070d31d1db4340233
    """
    return pynvml.nvmlDeviceGetComputeMode(handle)


@_exceptions_filter
def get_compute_processes(handle):
    """Returns the list of processes ids having a compute context on the
    device with the memory used

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g46ceaea624d5c96e098e03c453419d68
    """
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    return [{"pid": p.pid, "used_memory": p.usedGpuMemory} for p in processes]


@_exceptions_filter
def get_graphics_processes(handle):
    """Returns the list of processes ids having a graphics context on the
    device with the memory used

    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g7eacf7fa7ba4f4485d166736bf31195e
    """
    processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)

    return [{"pid": p.pid, "used_memory": p.usedGpuMemory} for p in processes]


@_exceptions_filter
def get_driver_version():
    return to_utf8(pynvml.nvmlSystemGetDriverVersion())


def get_gpu_static_info():
    try:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            device_details = {
                "name": get_gpu_name(handle),
                "uuid": get_uuid(handle),
                "power_limit": get_power_limit(handle),
                "gpu_index": i,
            }
            # Memory
            memory = get_memory_info(handle)
            if memory is not None:
                device_details["total_memory"] = memory.total

            devices.append(device_details)
        return devices

    except pynvml.NVMLError:
        LOGGER.debug("Failed to retrieve gpu static info", exc_info=True)
        return []


def get_initial_gpu_metric():
    try:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Memory
            memory = get_memory_info(handle)
            device_details = (
                {"total_memory": memory.total} if memory is not None else {}
            )
            devices.append(device_details)

        return devices

    except pynvml.NVMLError:
        LOGGER.debug("Failed to retrieve gpu initial metrics", exc_info=True)
        return []


def get_recurrent_gpu_metric():
    try:
        pynvml.nvmlInit()
        devices_details = []

        for uuid in _devices_to_report.get():
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)

            device_details = {
                "gpu_utilization": get_gpu_utilization(handle),
                "power_usage": get_power_usage(handle),
                "temperature": get_temperature(handle),
                "memory_utilization": get_memory_utilization_rate(handle),
            }

            # Memory
            memory = get_memory_info(handle)
            if memory is not None:
                device_memory_details = {
                    "free_memory": memory.free,
                    "used_memory": memory.used,
                    "percent.used_memory": (memory.used / memory.total) * 100,
                }
                device_details.update(device_memory_details)

            devices_details.append(device_details)
        return devices_details

    except pynvml.NVMLError:
        LOGGER.debug("Failed to retrieve gpu recurrent metrics", exc_info=True)
        return []


def get_gpu_details():
    try:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            device_details = {
                "name": get_gpu_name(handle),
                "uuid": get_uuid(handle),
                "temperature": get_temperature(handle),
                "power_usage": get_power_usage(handle),
                "power_limit": get_power_limit(handle),
                "gpu_utilization": get_gpu_utilization(handle),
                "compute_mode": get_compute_mode(handle),
                "compute_processes": get_compute_processes(handle),
                "graphics_processes": get_graphics_processes(handle),
            }

            # Memory
            memory = get_memory_info(handle)
            if memory is not None:
                device_memory_details = {
                    "free_memory": memory.free,
                    "total_memory": memory.total,
                    "used_memory": memory.used,
                }
                device_details.update(device_memory_details)

            devices.append(device_details)
        return devices

    except pynvml.NVMLError:
        LOGGER.debug("Failed to retrieve gpu information", exc_info=True)
        return []


def is_gpu_details_available():
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            return True

        except pynvml.NVMLError:
            return False
    else:
        return False


def convert_gpu_details_to_metrics(
    gpu_details: NamedSystemMetricsList, prefix: str = None
) -> NamedSystemMetricsList:
    metrics = []
    for i, gpu in enumerate(gpu_details):
        for key, value in gpu.items():
            if value is None:
                continue

            metric = {
                "name": metric_name("sys.gpu.%d.%s" % (i, key), prefix=prefix),
                "value": value,
            }
            metrics.append(metric)
    return metrics
