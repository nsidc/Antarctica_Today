import datetime as dt

import pytest

from antarctica_today.date import date_within_melt_period


@pytest.mark.parametrize(
    "date,expected",
    [
        (dt.datetime(1980, 12, 1), True),
        (dt.datetime(2010, 12, 15), True),
        (dt.datetime(2020, 12, 31), True),
        (dt.datetime(1980, 1, 1), True),
        (dt.datetime(2010, 1, 15), True),
        (dt.datetime(2020, 1, 31), True),
        (dt.datetime(1980, 4, 1), True),
        (dt.datetime(2010, 4, 15), True),
        (dt.datetime(2020, 4, 30), True),
        (dt.datetime(1980, 5, 1), False),
        (dt.datetime(2010, 5, 15), False),
        (dt.datetime(2020, 5, 31), False),
        (dt.datetime(1980, 9, 1), False),
        (dt.datetime(2010, 9, 15), False),
        (dt.datetime(2020, 9, 30), False),
        (dt.datetime(1980, 10, 1), True),
        (dt.datetime(2010, 10, 15), True),
        (dt.datetime(2020, 10, 31), True),
    ],
)
def test_date_in_melt_period(date, expected):
    assert date_within_melt_period(date) == expected
