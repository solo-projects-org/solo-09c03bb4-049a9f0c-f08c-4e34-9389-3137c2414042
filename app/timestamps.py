"""Canonical UTC timestamp formatting and tolerant parsing.

The archive stores timestamps as text so that they can be compared
chronologically.  The canonical form is::

    0001-01-01T00:00:00.000000Z

i.e. a four-digit year, microsecond precision, ``Z`` suffix.  Four-digit
padding matters: ``strftime("%Y")`` (used by older releases) prints year 1 as
``1``, which both breaks fixed-format parsing and makes text comparisons
wrong.  This module formats manually and parses both canonical values and
legacy values whose year has no leading zeros.

Legacy stamps are interpreted as the instant they actually denote: a stored
``1-01-01T00:00:00.000000Z`` is year 1, not year 2001 or "now".  Parsing is
deliberately a little more permissive than formatting (a missing fraction is
read as zero microseconds; an explicit offset is converted to UTC) so that an
old volume can be upgraded without rewriting identifiers or relations.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

# Years are 1..9999.  The legacy format emitted by ``strftime("%Y-...")``
# leaves years below 1000 unpadded (``1-``, ``12-``, ``999-``).
_TS_RE = re.compile(
    r"^(?P<year>\d{1,4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,6}))?"
    r"(?P<offset>Z|z|[+-]\d{2}:?\d{2})?$"
)


def to_ts(value: datetime) -> str:
    """Format ``value`` as canonical UTC text (four-digit year, microseconds)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC)
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f".{value.microsecond:06d}Z"
    )


def from_ts(value: str) -> datetime:
    """Parse a stored timestamp into an aware UTC ``datetime``.

    Accepts canonical four-digit years as well as legacy unpadded years and
    explicit offsets.  A missing zone is treated as UTC.  Raises ``ValueError``
    for anything else so a corrupt archive fails loudly instead of being read
    as a different instant.
    """
    match = _TS_RE.match(value)
    if match is None:
        raise ValueError(f"invalid stored timestamp: {value!r}")

    fraction = match.group("fraction") or ""
    microsecond = int(fraction.ljust(6, "0")) if fraction else 0
    offset = match.group("offset") or "Z"
    if offset in ("Z", "z"):
        tz = UTC
    else:
        sign = 1 if offset[0] == "+" else -1
        digits = offset[1:].replace(":", "")
        tz = timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            microsecond,
            tzinfo=tz,
        ).astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid stored timestamp: {value!r}") from exc
    return parsed


def is_canonical_ts(value: str) -> bool:
    """Whether ``value`` is already in canonical form."""
    try:
        return to_ts(from_ts(value)) == value
    except ValueError:
        return False
