import pytest
from e_batch_sampler import chunks, take, first_satisfying, EpochSampler


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------


def test_chunks_evenly_divisible():
    pass


def test_chunks_with_remainder():
    pass


def test_chunks_size_equals_len():
    pass


def test_chunks_size_one():
    pass


def test_chunks_is_a_generator():
    pass


def test_chunks_raises_on_size_zero():
    pass


def test_chunks_raises_on_negative_size():
    pass


def test_chunks_works_on_string():
    pass


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------


def test_take_fewer_than_available():
    pass


def test_take_more_than_available_stops_early():
    pass


def test_take_zero_returns_empty():
    pass


def test_take_returns_list():
    pass


def test_take_raises_on_negative_n():
    pass


def test_take_uses_iter_and_next():
    pass


# ---------------------------------------------------------------------------
# first_satisfying
# ---------------------------------------------------------------------------


def test_first_satisfying_returns_element_and_index():
    pass


def test_first_satisfying_first_element():
    pass


def test_first_satisfying_last_element():
    pass


def test_first_satisfying_no_match_returns_none_neg1():
    pass


def test_first_satisfying_raises_on_non_callable():
    pass


# ---------------------------------------------------------------------------
# EpochSampler
# ---------------------------------------------------------------------------


def test_epoch_sampler_yields_full_batches_only():
    pass


def test_epoch_sampler_covers_all_indices():
    pass


def test_epoch_sampler_drops_last_incomplete_batch():
    pass


def test_epoch_sampler_reproducible_with_seed():
    pass


def test_epoch_sampler_seeded_same_each_iteration():
    pass


def test_epoch_sampler_different_shuffle_each_iteration():
    pass


def test_epoch_sampler_is_iterable_not_iterator():
    pass


def test_epoch_sampler_raises_on_invalid_n_samples():
    pass


def test_epoch_sampler_raises_on_invalid_batch_size():
    pass
