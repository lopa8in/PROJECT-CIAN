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
import platform

from ..base_metrics_data_logger import BaseMetricsDataLogger
from ..system_metrics_types import CompatibleSystemMetrics, SystemMetricsCallable
from .disk_io import DiskIOUtilizationProbe
from .disk_usage import DiskUsageProbe

try:
    import psutil
except Exception:
    psutil = None


class DiskMetricsDataLogger(BaseMetricsDataLogger):
    def __init__(
        self, initial_interval: float, callback: SystemMetricsCallable, **kwargs
    ):
        super().__init__(initial_interval, callback, **kwargs)
        if platform.system() != "Windows":
            self.disk_usage_probe = DiskUsageProbe()
        else:
            self.disk_usage_probe = None
        self.disk_io_probe = DiskIOUtilizationProbe()

    def get_metrics(self) -> CompatibleSystemMetrics:
        metrics = {}

        io_rates = self.disk_io_probe.sample()
        if io_rates is not None:
            metrics["sys.disk.read_bps"] = io_rates.disk_read_bps
            metrics["sys.disk.write_bps"] = io_rates.disk_write_bps

        if self.disk_usage_probe is not None:
            disk_usage = self.disk_usage_probe.sample()
            if disk_usage is not None:
                metrics["sys.disk.root.used"] = disk_usage.disk_used_bytes
                metrics["sys.disk.root.percent.used"] = disk_usage.disk_used_percent

        return metrics

    def get_name(self) -> str:
        return "[sys.disk]"

    def available(self) -> bool:
        return psutil is not None
