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

from typing import List

import box

from . import _reporting, event_tracker


def report(experiment: "Experiment"):  # noqa: F821
    events = event_tracker.registered_events(experiment.id)

    torch_save_called_bi_events = _torch_save_called_bi_events(events)

    for bi_event in torch_save_called_bi_events:
        experiment.__internal_api__report__(bi_event.name, err_msg=bi_event.err_msg)


def _torch_save_called_bi_events(events: List[str]) -> List[str]:
    result = []

    for event in events:
        if not event.startswith("torch.save-called-by-"):
            continue

        caller = event.replace("torch.save-called-by-", "")
        bi_event = box.Box(name=_reporting.TORCH_SAVE_CALL, err_msg=caller)
        result.append(bi_event)

    return result
