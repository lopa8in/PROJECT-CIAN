# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from abc import ABC, abstractmethod
from queue import Queue
from typing import List, Tuple


class Bucket(ABC):

    def __init__(self, max_size: int = 0):
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @abstractmethod
    def size(self) -> int:
        """Return the current size of the bucket, ie the number of all items in the bucket"""

    @abstractmethod
    def put(self, item: float) -> bool:
        """Put a current time in the bucket.
        Return True if successful.
        """

    @abstractmethod
    def pop(self, number: int) -> int:
        """Pop items from the bucket in the FIFO order.
        Return the number of items that has been removed.
        """

    @abstractmethod
    def all_items(self) -> List[float]:
        """Return a list as copies of all items in the bucket"""

    @abstractmethod
    def flush(self) -> None:
        """Flush/reset bucket"""

    def evaluate_expired_items(self, time: float) -> Tuple[int, float]:
        """Evaluates how many items in the bucket that are expired

        Returns:
            The number of unexpired items, and the time until the next item will expire
        """
        volume = self.size()
        item_count, remaining_time = 0, 0.0

        for idx, item_time in enumerate(self.all_items()):
            if item_time > time:
                item_count = volume - idx
                remaining_time = round(item_time - time, 3)
                break

        return item_count, remaining_time


class QueueBucket(Bucket):

    def __init__(self, max_size: int = 0):
        super().__init__()
        self._queue: Queue = Queue(maxsize=max_size)

    def size(self) -> int:
        return self._queue.qsize()

    def put(self, item: float):
        return self._queue.put(item)

    def pop(self, number: int) -> int:
        counter = 0
        for _ in range(number):
            self._queue.get()
            counter += 1

        return counter

    def all_items(self) -> List[float]:
        return list(self._queue.queue)

    def flush(self):
        while not self._queue.empty():
            self._queue.get()
