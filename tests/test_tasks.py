from datetime import date

from app.tasks.retrain import _previous_month_range


def test_previous_month_range_simple():
    """Test normal case (e.g. March -> February)."""
    today = date(2023, 3, 15)
    start, end = _previous_month_range(today)

    assert start == date(2023, 2, 1)
    assert end == date(2023, 2, 28)


def test_previous_month_range_january():
    """Test January rollover (Jan 2023 -> Dec 2022)."""
    today = date(2023, 1, 10)
    start, end = _previous_month_range(today)

    assert start == date(2022, 12, 1)
    assert end == date(2022, 12, 31)


def test_previous_month_range_leap_year():
    """Test leap year (March 2024 -> Feb 2024)."""
    today = date(2024, 3, 1)
    start, end = _previous_month_range(today)

    assert start == date(2024, 2, 1)
    assert end == date(2024, 2, 29)
