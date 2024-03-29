"""tzinfo implementations for ksycopg2

This module holds two different tzinfo implementations that can be used as
the 'tzinfo' argument to datetime constructors, directly passed to ksycopg
functions or used to set the .tzinfo_factory attribute in cursors.
"""
# ksycopg/tz.py - tzinfo implementation
#
# Copyright (C) 2003-2019 Federico Di Gregorio  <fog@debian.org>
# Copyright (C) 2020 The ksycopg Team
#
# ksycopg2 is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# In addition, as a special exception, the copyright holders give
# permission to link this program with the OpenSSL library (or with
# modified versions of OpenSSL that use the same license as OpenSSL),
# and distribute linked combinations including the two.
#
# You must obey the GNU Lesser General Public License in all respects for
# all of the code used other than OpenSSL.
#
# ksycopg2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.

import datetime
import time

ZERO = datetime.timedelta(0)


class FixedOffsetTimezone(datetime.tzinfo):
    """Fixed offset in minutes east from UTC.

    This is exactly the implementation__ found in Python 2.3.x documentation,
    with a small change to the `!__init__()` method to allow for pickling
    and a default name in the form ``sHH:MM`` (``s`` is the sign.).

    The implementation also caches instances. During creation, if a
    FixedOffsetTimezone instance has previously been created with the same
    offset and name that instance will be returned. This saves memory and
    improves comparability.

    .. __: https://docs.python.org/library/datetime.html
    """
    _name = None
    _offset = ZERO

    _cache = {}

    def __init__(self, offset=None, name=None):
        if offset is not None:
            self._offset = datetime.timedelta(minutes=offset)
        if name is not None:
            self._name = name

    def __new__(cls, offset=None, name=None):
        """Return a suitable instance created earlier if it exists
        """
        key = (offset, name)
        try:
            return cls._cache[key]
        except KeyError:
            tz = super(FixedOffsetTimezone, cls).__new__(cls, offset, name)
            cls._cache[key] = tz
            return tz

    def __repr__(self):
        offset_mins = self._offset.seconds // 60 + self._offset.days * 24 * 60
        return "ksycopg2.tz.FixedOffsetTimezone(offset=%r, name=%r)" \
            % (offset_mins, self._name)

    def __getinitargs__(self):
        offset_mins = self._offset.seconds // 60 + self._offset.days * 24 * 60
        return offset_mins, self._name

    def utcoffset(self, dt):
        return self._offset

    def tzname(self, dt):
        if self._name is not None:
            return self._name
        else:
            seconds = self._offset.seconds + self._offset.days * 86400
            hours, seconds = divmod(seconds, 3600)
            minutes = seconds / 60
            if minutes:
                return "%+03d:%d" % (hours, minutes)
            else:
                return "%+03d" % hours

    def dst(self, dt):
        return ZERO


STDOFFSET = datetime.timedelta(seconds=-time.timezone)
if time.daylight:
    DSTOFFSET = datetime.timedelta(seconds=-time.altzone)
else:
    DSTOFFSET = STDOFFSET
DSTDIFF = DSTOFFSET - STDOFFSET


class LocalTimezone(datetime.tzinfo):
    """Platform idea of local timezone.

    This is the exact implementation from the Python 2.3 documentation.
    """
    def utcoffset(self, dt):
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt):
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt):
        return time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        tt = (dt.year, dt.month, dt.day,
              dt.hour, dt.minute, dt.second,
              dt.weekday(), 0, -1)
        stamp = time.mktime(tt)
        tt = time.localtime(stamp)
        return tt.tm_isdst > 0


LOCAL = LocalTimezone()

# TODO: pre-generate some interesting time zones?
