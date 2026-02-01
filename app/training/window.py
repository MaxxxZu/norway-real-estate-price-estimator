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
