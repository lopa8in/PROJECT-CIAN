# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from typing import Optional

from comet_ml._typing import List
from comet_ml.console import StdLogger
from comet_ml.messages import BaseMessage, CloseMessage
from comet_ml.streamer import OnlineStreamer


class DummyStdLogger(StdLogger):
    def __init__(self, *args, **kwargs):
        super(DummyStdLogger, self).__init__(*args, **kwargs)


class DummyOnlineStreamer(OnlineStreamer):
    def __init__(self, api_key, run_id, project_id):
        self.messages: List[BaseMessage] = []
        self.closed = False
        self.api_key = api_key
        self.run_id = run_id
        self.project_id = project_id
        self.error_reported = False
        self.error_message = None
        self.has_crashed = False
        self._counter = 0
        self.use_http_messages = True

    def put_message_in_q(self, message: BaseMessage):
        message.message_id = self._counter
        self.messages.append(message)

    def _put_close_message_in_queue(self) -> None:
        self.messages.append(CloseMessage())

    def get_last_msg(self) -> BaseMessage:
        return self.messages[-1]

    def getn(self, n) -> List[BaseMessage]:
        return self.messages[-n:]

    def get_all(self) -> List[BaseMessage]:
        return self.messages

    def get_one_before_last(self) -> BaseMessage:
        return self.getn(2)[0]

    def has_connection_to_server(self) -> bool:
        return True

    def flush(self, timeout: Optional[int] = None) -> bool:
        return True

    def wait_for_finish(self, **kwargs) -> bool:
        return True

    def has_upload_failed(self) -> bool:
        return False

    def clean(self):
        self.messages = []

    def close(self) -> None:
        self.closed = True

    def _report_experiment_error(self, message: str, has_crashed: bool = False):
        self.error_reported = True
        self.error_message = message
        self.has_crashed = has_crashed

    def __str__(self) -> str:
        return "DummyStreamer()"

    def __repr__(self) -> str:
        return "DummyStreamer()"


class DummyHeartBeatThread(object):
    def close(self):
        return None

    def join(self, timeout):
        return True


def make_dummy_overrides(config_map):
    """
    Create an override dict of defaults
    """
    return {key: config_map[key].get("default", None) for key in config_map}
