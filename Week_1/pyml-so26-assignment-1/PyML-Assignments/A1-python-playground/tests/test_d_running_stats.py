from statistics import StatisticsError
import math
import pytest
from d_running_stats import RunningStats


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_len_is_zero():
    pass


def test_mean_raises_before_any_update():
    pass


def test_variance_raises_before_any_update():
    pass


def test_std_raises_before_any_update():
    pass


def test_repr_no_values_shows_na():
    pass


# ---------------------------------------------------------------------------
# Single update
# ---------------------------------------------------------------------------


def test_len_after_one_update():
    pass


def test_mean_after_one_update():
    pass


def test_variance_raises_after_one_update():
    pass


def test_repr_one_value_shows_na_for_std():
    pass


# ---------------------------------------------------------------------------
# Multiple updates — correctness
# ---------------------------------------------------------------------------

STREAM = [2, 4, 4, 4, 5, 5, 7, 9]  # mean=5.0, sample variance=32/7


def _make_rs(values=STREAM):
    rs = RunningStats()
    for x in values:
        rs.update(x)
    return rs


def test_mean_correct_after_multiple_updates():
    pass


def test_variance_correct_after_multiple_updates():
    pass


def test_std_is_sqrt_of_variance():
    pass


def test_len_tracks_number_of_updates():
    pass


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


def test_update_raises_on_string():
    pass


def test_update_raises_on_none():
    pass


def test_update_accepts_int():
    pass


def test_update_accepts_float():
    pass


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_format_after_several_updates():
    pass


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_restores_initial_state():
    pass


def test_reset_allows_fresh_updates():
    pass
