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
import threading
import time
from typing import List

from .base_metrics_data_logger import BaseMetricsDataLogger


class SystemMetricsLoggingThread(threading.Thread):
    def __init__(
        self,
        metric_data_loggers: List[BaseMetricsDataLogger],
        probe_interval: float = 1,
    ):
        threading.Thread.__init__(self, daemon=True)
        self.metric_data_loggers = metric_data_loggers
        self.probe_interval = probe_interval
        self.closed = False

    def close(self):
        self.closed = True

    def run(self):
        while not self.closed:
            for logger in self.metric_data_loggers:
                if logger.should_log_data():
                    logger.log_metric_data()

                time.sleep(self.probe_interval / 10.0)

            time.sleep(self.probe_interval)
