import pytest
from f_sensor_processing import compose, pipeline, apply_if, window_filter


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_compose_applies_right_to_left():
    pass


def test_compose_three_functions():
    pass


def test_compose_single_function_is_identity():
    pass


def test_compose_raises_on_no_arguments():
    pass


def test_compose_raises_on_non_callable():
    pass


def test_compose_returns_callable():
    pass


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_pipeline_applies_left_to_right():
    pass


def test_pipeline_single_stage():
    pass


def test_pipeline_kwargs_forwarded():
    pass


def test_pipeline_raises_on_non_tuple_stage():
    pass


def test_pipeline_raises_on_stage_without_dict():
    pass


def test_pipeline_raises_on_non_callable_fn():
    pass


# ---------------------------------------------------------------------------
# apply_if
# ---------------------------------------------------------------------------


def test_apply_if_transforms_matching_elements():
    pass


def test_apply_if_substitutes_default_for_non_matching():
    pass


def test_apply_if_custom_default_value():
    pass


def test_apply_if_all_match():
    pass


def test_apply_if_none_match():
    pass


def test_apply_if_preserves_order_and_length():
    pass


def test_apply_if_raises_on_non_callable_predicate():
    pass


def test_apply_if_raises_on_non_callable_transform():
    pass


# ---------------------------------------------------------------------------
# window_filter
# ---------------------------------------------------------------------------


def test_window_filter_passes_matching_windows():
    pass


def test_window_filter_replaces_failing_windows_with_default():
    pass


def test_window_filter_custom_default_value():
    pass


def test_window_filter_correct_number_of_windows():
    pass


def test_window_filter_size_equals_len_data():
    pass


def test_window_filter_size_one():
    pass


def test_window_filter_raises_on_size_zero():
    pass


def test_window_filter_raises_on_size_greater_than_data():
    pass


def test_window_filter_raises_on_non_callable_predicate():
    pass
