# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
import threading
from typing import Dict, List, NamedTuple, Optional, Set

from comet_ml.messages import BaseMessage


class ActiveRetryIncident(NamedTuple):
    """
    Represents an active retry incident with its details.

    This class is used to define the structure of an active retry incident.
    It includes the type of the message related to the incident and the specific
    time when the incident should be reset.
    """

    message_type: str
    reset_at: float
    messages: Set[BaseMessage] = []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.update(messages)


class RetryIncidentsManager:
    """
    Manages retry incidents and their states to avoid excessive retries and manage
    resource utilization efficiently.

    The RetryManager class handles retry incidents by tracking their states, ensuring
    that retries do not occur too frequently, and allowing the release of outdated
    incidents when appropriate. It uses a thread-safe mechanism to manage incidents
    and can be integrated in systems requiring retry functionality for message or
    task handling.
    """

    def __init__(self):
        self.incidents: Dict[str, ActiveRetryIncident] = {}
        self.registered_incidents_count = 0
        self.__incident_lock__ = threading.RLock()

    def add_or_update_incident(
        self, message_type: str, reset_at: float, messages: List[BaseMessage]
    ):
        with self.__incident_lock__:
            self.registered_incidents_count += 1
            if message_type not in self.incidents:
                self.incidents[message_type] = ActiveRetryIncident(
                    message_type, reset_at, messages=set(messages)
                )
            else:
                self.incidents[message_type].add_messages(messages)

    def get_incident(self, message_type: str) -> Optional[ActiveRetryIncident]:
        with self.__incident_lock__:
            return self.incidents.get(message_type)

    def has_active_incidents(self) -> bool:
        """
        Checks whether there are any active incidents associated with the object.

        This method evaluates if the `incidents` list contains one or more entries, indicating
        the presence of active incidents waiting to be released.

        Returns:
            bool: True if there are active incidents, otherwise False.
        """
        return len(self.incidents) > 0

    def messages_to_retry(self) -> int:
        if len(self.incidents) == 0:
            return 0

        with self.__incident_lock__:
            count = 0
            for incident in self.incidents.values():
                count += len(incident.messages)
            return count

    def release_outdated_incidents(self, now: float) -> List[ActiveRetryIncident]:
        """
        Releases outdated incidents from the active incidents list if their reset time has passed.

        This method iterates through all active incidents to check whether their 'reset_at'
        time is less than or equal to the provided current time. If so, it removes the outdated
        incidents from the internal storage and returns whether any incidents have been released.
        The operation is thread-safe.

        Parameters:
            now (float): The current time used to determine whether an incident is outdated.

        Returns:
            The list of released incidents or empty if no incidents have been released.
        """
        result = []
        with self.__incident_lock__:
            for message_type, incident in list(self.incidents.items()):
                if incident.reset_at <= now:
                    del self.incidents[message_type]
                    result.append(incident)

            return result
