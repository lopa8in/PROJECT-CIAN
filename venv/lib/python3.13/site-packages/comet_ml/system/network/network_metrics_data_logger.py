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
from ..system_metrics_types import CompatibleSystemMetrics, SystemMetricsCallable
from .network_rates import NetworkRatesProbe

try:
    import psutil
except Exception:
    psutil = None


class NetworkMetricsDataLogger(BaseMetricsDataLogger):
    def __init__(
        self,
        initial_interval: float,
        callback: SystemMetricsCallable,
    ):
        super().__init__(initial_interval, callback)
        self.network_rates_probe = NetworkRatesProbe()

    def get_metrics(self) -> CompatibleSystemMetrics:
        result = self.network_rates_probe.current_rate()
        if result is None:
            return {}
        metrics = {
            "sys.network.send_bps": result.bytes_sent_rate,
            "sys.network.receive_bps": result.bytes_recv_rate,
        }
        return metrics

    def get_name(self) -> str:
        return "[sys.network]"

    def available(self) -> bool:
        return psutil is not None
