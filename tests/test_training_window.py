from datetime import date

from app.training.window import window_start


def test_window_start_12_months_simple():
    assert window_start(date(2026, 1, 31), 12) == date(2025, 1, 31)
    assert window_start(date(2026, 2, 1), 12) == date(2025, 2, 1)


def test_window_start_clamps_day():
    assert window_start(date(2026, 3, 31), 1) == date(2026, 2, 28)
