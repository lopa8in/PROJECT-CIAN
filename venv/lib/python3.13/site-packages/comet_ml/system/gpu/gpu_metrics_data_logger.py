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
from .gpu_logging import get_recurrent_gpu_metric, is_gpu_details_available


class GPUMetricsDataLogger(BaseMetricsDataLogger):
    def __init__(
        self, initial_interval: float, callback: SystemMetricsCallable, **kwargs
    ):
        super().__init__(initial_interval, callback, **kwargs)

        self.gpu_details_available = is_gpu_details_available()

    def get_metrics(self) -> CompatibleSystemMetrics:
        return get_recurrent_gpu_metric()

    def get_name(self) -> str:
        return "[sys.gpu]"

    def available(self) -> bool:
        return self.gpu_details_available
