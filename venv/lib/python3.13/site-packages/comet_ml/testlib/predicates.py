# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.com
#  Copyright (C) 2015-2025 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************


class AlwaysEquals(object):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class NotEquals(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value != other

    def __ne__(self, other):
        return self.value == other

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.value)

    def __repr__(self):
        return "%r(%r)" % (self.__class__.__name__, self.value)


class CompareFunction(object):
    def __init__(self, function):
        self.function = function

    def __eq__(self, other):
        try:
            return self.function(other)
        except Exception:
            return False

    def __ne__(self, other):
        try:
            return not self.function(other)
        except Exception:
            return True


class PartialEquals(object):
    def __init__(self, function, args=None, check_args=True):
        self.function = function
        self.args = args
        self.check_args = check_args

    def __eq__(self, other):
        try:
            # Kwargs is not supported in Python 2 so we cannot use it nor test
            # it
            result = other.func == self.function

            if self.check_args:
                result = result and other.args == self.args

            return result
        except Exception:
            return False

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __repr__(self):
        return "PartialEquals(%r, %r)" % (self.function, self.args)


class Within10Percent(object):
    def __init__(self, target):
        self.target = target
        self.delta = abs(0.1 * target)

    def __eq__(self, other):
        difference = abs(other - self.target)
        return difference <= self.delta

    def __repr__(self):
        return "Within10Percent(%s)" % self.target

    def __str__(self):
        return repr(self)


class TracebackEquals(object):
    def __eq__(self, other):
        return "Traceback" in other

    def __ne__(self, other):
        return "Traceback" not in other


class ListContentEquals(object):
    def __init__(self, expected_list):
        self.expected_list = list(sorted(expected_list))

    def __eq__(self, other):
        return self.expected_list == list(sorted(other))


class IsInstanceOf(object):
    def __init__(self, klass):
        self.klass = klass

    def __eq__(self, other):
        return isinstance(other, self.klass)

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __repr__(self):
        return "IsInstanceOf(%r)" % (self.klass)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class InRange:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __eq__(self, other):
        return self.low <= other <= self.high

    def __repr__(self):
        return "InRange[%0.2f:%0.2f]" % (self.low, self.high)

    def __str__(self):
        return repr(self)


class FloatClose(object):
    def __init__(self, number, precision):
        self.number = number
        self.precision = precision

    def __eq__(self, other):
        try:
            return isclose(self.number, other, self.precision)
        except Exception:
            return False

    def __ne__(self, other):
        try:
            return not isclose(self.number, other, self.precision)
        except Exception:
            return True

    def __str__(self):
        return "%s at %s precision" % (self.number, self.precision)

    def __repr__(self):
        return "%r at %s precision" % (self.number, self.precision)
