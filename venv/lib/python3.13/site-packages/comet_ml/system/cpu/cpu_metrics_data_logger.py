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
from ..base_metrics_data_logger import BaseMetricsDataLogger
from ..system_metrics_types import NamedSystemMetrics, SystemMetricsCallable
from .utilization import process_tree_cpu_utilization

try:
    import psutil
except Exception:
    psutil = None


class CPUMetricsDataLogger(BaseMetricsDataLogger):
    def __init__(
        self,
        initial_interval: float,
        callback: SystemMetricsCallable,
        include_cpu_per_core: bool,
        include_compute_metrics: bool,
        **kwargs
    ):
        super().__init__(initial_interval, callback, **kwargs)
        self.include_compute_metrics = include_compute_metrics
        self._include_cpu_per_core = include_cpu_per_core

    def get_metrics(self) -> NamedSystemMetrics:
        cpu_metrics = _cpu_percent_metrics(
            self.include_compute_metrics, self._include_cpu_per_core
        )
        ram_metrics = _ram_metrics()
        loadavg_metrics = _loadavg_metrics()

        metrics = {**cpu_metrics, **ram_metrics, **loadavg_metrics}

        return metrics

    def get_name(self) -> str:
        return "[sys.ram,sys.cpu,sys.load]"

    def available(self) -> bool:
        return psutil is not None


def _ram_metrics() -> NamedSystemMetrics:
    virtual_memory = psutil.virtual_memory()

    result = {
        "sys.ram.total": virtual_memory.total,
        "sys.ram.used": virtual_memory.total - virtual_memory.available,
        "sys.ram.available": virtual_memory.available,
        "sys.ram.percent.used": virtual_memory.percent,
    }

    return result


def _cpu_percent_metrics(include_compute, include_per_core) -> NamedSystemMetrics:
    percents = psutil.cpu_percent(interval=None, percpu=True)

    result = {}
    if len(percents) > 0:
        avg_percent = sum(percents) / len(percents)
        result["sys.cpu.percent.avg"] = avg_percent

        if include_compute:
            result["sys.compute.overall"] = round(avg_percent, 1)
            result["sys.compute.utilized"] = process_tree_cpu_utilization()

        if include_per_core:
            for i, percent in enumerate(percents):
                result["sys.cpu.percent.%02d" % (i + 1)] = percent

    return result


def _loadavg_metrics() -> NamedSystemMetrics:
    result = {}

    try:
        result["sys.load.avg"] = psutil.getloadavg()[0]
    except OSError:
        pass

    return result
