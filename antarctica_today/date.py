import datetime as dt

from antarctica_today.constants.dates import (
    MELT_END_MMDD,
    MELT_START_MMDD,
)


def date_within_melt_period(date: dt.datetime) -> bool:
    mmdd = (date.month, date.day)

    # IMPORTANT: Tuple comparison works for this _only_ because we're calculating the
    # negative. The melt season crosses the year boundary, and tuple comparison doesn't
    # know the year rolls over at 12. For example, we can _not_ calculate that
    # 2020-01-01 is within the melt period by seeing if it is _greater_ than the melt
    # start month and day (10, 1). We _can_ determine whether is _not_ within the melt
    # period by checking if (1, 1) is greater than (4, 30) and less than (10, 1). Since
    # this is false, it is within the melt period.
    not_in_melt = mmdd > MELT_END_MMDD and mmdd < MELT_START_MMDD

    return not not_in_melt
