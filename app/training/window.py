from __future__ import annotations

from datetime import date


def shift_months(d: date, months: int) -> date:
    """Shift date by N months, keeping day in range for target month."""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _days_in_month(year, month))

    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)

    return (nxt - date(year, month, 1)).days


def window_start(end_date: date, window_months: int) -> date:
    """
    For an end_date, compute start_date of trailing window:
    start_date = end_date shifted by -window_months + 1 day? (we keep inclusive range in pipeline)
    """
    if window_months <= 0:
        raise ValueError("window_months must be > 0")

    return shift_months(end_date, -window_months)
