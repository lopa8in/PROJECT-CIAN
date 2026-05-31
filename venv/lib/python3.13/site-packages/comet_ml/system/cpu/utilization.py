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

import time
from typing import List

try:
    import psutil
except Exception:
    psutil = None

from .. import processes

UTILIZATION_MEASURE_INTERVAL = 0.3


def _processes_family_utilization(processes_family: List[psutil.Process]) -> float:
    return sum(
        [
            process.cpu_percent()
            for process in processes_family
            if process.is_running() and process.status() == psutil.STATUS_RUNNING
        ]
    )


def process_tree_cpu_utilization():
    if psutil is None:
        return

    processes_family = [
        psutil.Process(pid=pid) for pid in processes.family() if psutil.pid_exists(pid)
    ]
    _processes_family_utilization(
        processes_family=processes_family,
    )

    # We need to sleep here in order to get CPU utilization from psutil
    time.sleep(UTILIZATION_MEASURE_INTERVAL)

    result = _processes_family_utilization(
        processes_family=processes_family,
    )

    return result
