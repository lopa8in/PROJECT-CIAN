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

import logging
import threading
import time

from comet_ml.logging_extensions.level_shifter import LevelShiftingLogger
from comet_ml.logging_messages import HEARTBEAT_PROCESSING_ERROR

LOGGER = logging.getLogger(__name__)

HEARTBEAT_PARAMETERS_BATCH_UPDATE_INTERVAL = "parameters_update_interval"
HEARTBEAT_GPU_MONITOR_INTERVAL = "gpu_monitor_interval"
HEARTBEAT_CPU_MONITOR_INTERVAL = "cpu_monitor_interval"


class HeartbeatThread(threading.Thread):
    def __init__(
        self, beat_duration, status_update_callback, pending_rpcs_callback=None
    ):
        threading.Thread.__init__(self)

        self.daemon = True

        self.queue_timeout = beat_duration
        self.last_beat = float("-inf")
        self.status_update_callback = status_update_callback
        self.on_parameters_update_interval_callback = None
        self.on_pending_rpcs_callback = pending_rpcs_callback
        self.closed = threading.Event()
        self._level_shifting_logger = LevelShiftingLogger(
            logger=LOGGER,
            initial_level=logging.ERROR,
            level=logging.DEBUG,
            shift_after=1,
        )

    def run(self):
        try:
            while self.closed.is_set() is False:
                self._loop()
        except Exception:
            LOGGER.debug("Unexpected heartbeat error", exc_info=True)

    def _loop(self):
        # Wait on an event, so we wake up as soon as we close the heartbeat thread
        self.closed.wait(self.queue_timeout)
        if self.closed.is_set():
            return

        try:
            self.do_heartbeat()
        except Exception:
            self._level_shifting_logger.log(HEARTBEAT_PROCESSING_ERROR, exc_info=True)

    def close(self):
        self.closed.set()

    def do_heartbeat(self):
        """
        Check if we should send a heartbeat
        """
        next_beat = self.last_beat + self.queue_timeout
        now = time.time()
        if next_beat < now:
            LOGGER.debug("Doing an heartbeat, interval: %f seconds", self.queue_timeout)
            # We need to update the last beat time before doing the actual
            # call as the call might fail and the last beat would not been
            # updated. That would trigger a heartbeat for each message.
            self.last_beat = time.time()

            response = self.status_update_callback()
            if response is None:
                # no connection to server
                return

            new_beat_duration, data, pending_rpcs = response
            if self.closed.is_set():
                LOGGER.debug(
                    "Heartbeat already closed, skip further response processing"
                )
                return

            LOGGER.debug(
                "Getting a new heartbeat duration: %d milliseconds", new_beat_duration
            )
            self.queue_timeout = new_beat_duration / 1000.0  # We've got milliseconds

            # get parameter_update_interval_millis parameter and update related callback
            parameters_update_interval_millis = data.get(
                HEARTBEAT_PARAMETERS_BATCH_UPDATE_INTERVAL
            )
            LOGGER.debug(
                "Getting a new parameters update interval %d milliseconds for %r",
                parameters_update_interval_millis,
                self.on_parameters_update_interval_callback,
            )

            if self.on_parameters_update_interval_callback is not None:
                try:
                    self.on_parameters_update_interval_callback(
                        parameters_update_interval_millis / 1000.0
                    )
                except Exception:
                    LOGGER.debug(
                        "Error calling the parameters update interval callback",
                        exc_info=True,
                    )

            # If there are some pending rpcs
            if pending_rpcs and self.on_pending_rpcs_callback is not None:
                try:
                    self.on_pending_rpcs_callback()
                except Exception:
                    LOGGER.debug("Error calling the rpc callback", exc_info=True)
