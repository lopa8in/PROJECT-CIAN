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
from time import monotonic
from typing import Dict, List, Tuple

from .bucket import Bucket, QueueBucket
from .exceptions import LimitRateExceededError
from .rate import OpRate


class OpRateLimiter:
    """Limits the rate of operations according to the Leaky Bucket algorithm
    https://en.wikipedia.org/wiki/Leaky_bucket
    Allows limiting rate of operations by applying several rates simultaneously.
    For example:
    - 10 operations per minute
    - 30 operations per hour
    - 100 operations per day
    """

    def __init__(
        self,
        rates: List[OpRate],
    ):
        self._rates = rates
        validate_rate_list(rates)

        self.bucket_group: Dict[str, Bucket] = {}
        self.time_function = monotonic

    def _init_buckets(self, identities: Tuple[str, ...]) -> None:
        """Lazy initialize a bucket for each identity, if needed.
        Each bucket's maxsize is equal to the max limit of operation rates.
        """
        max_size = self._rates[-1].op_limit
        for item_id in identities:
            if not self.bucket_group.get(item_id):
                self.bucket_group[item_id] = QueueBucket(max_size=max_size)

    def try_acquire(self, *identities: str) -> None:
        """Attempt to acquire an item(s) with given identity(ies), or raise an error if a rate limit has been exceeded.

        Args:
            identities: One or more identities to acquire. Typically, this is the name of an operation
                or resource that is rate-limited.

        Raises:
            LimitRateExceededError: If rate limit has been exceeded for any among provided identities
        """
        self._init_buckets(identities)
        now = round(self.time_function(), 3)

        for rate in self._rates:
            for item_id in identities:
                bucket = self.bucket_group[item_id]
                volume = bucket.size()

                if volume < rate.op_limit:
                    continue

                # Determine rate's starting point, and check requests made during its time window
                item_count, remaining_time = bucket.evaluate_expired_items(
                    now - rate.interval_seconds
                )
                if item_count >= rate.op_limit:
                    raise LimitRateExceededError(
                        identity=item_id, rate=rate, remaining_time=remaining_time
                    )

                # Remove expired bucket items beyond the last (maximum) rate limit
                if rate is self._rates[-1]:
                    bucket.pop(volume - item_count)

        # add another item to each bucket representing the next request
        for item_id in identities:
            self.bucket_group[item_id].put(now)

    def get_current_volume(self, identity: str) -> int:
        """Gets the current bucket's volume for provided identity"""
        bucket = self.bucket_group[identity]
        return bucket.size()

    def flush_all(self) -> int:
        """Flushes all buckets and returns number of bucket groups flushed.
        One bucket group per one unique known item identity."""
        count = 0
        for _, bucket in self.bucket_group.items():
            bucket.flush()
            count += 1

        return count


def validate_rate_list(rates: List[OpRate]):
    """Raise exception if rates are incorrectly ordered.
    The correct rate order is from smallest to biggest.
    """
    if len(rates) == 0:
        raise ValueError("Rate(s) must be provided")

    for idx, rate in enumerate(rates[1:]):
        prev_rate = rates[idx]
        invalid = (
            rate.op_limit <= prev_rate.op_limit
            or rate.interval_seconds <= prev_rate.interval_seconds
        )
        if invalid:
            msg = f"{prev_rate} cannot come before {rate}"
            raise ValueError(msg)
