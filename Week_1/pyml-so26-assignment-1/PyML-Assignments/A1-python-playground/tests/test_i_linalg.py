import math
import pytest
from i_linalg import gram_schmidt, gaussian_elimination


# ---------------------------------------------------------------------------
# gram_schmidt
# ---------------------------------------------------------------------------


def test_gram_schmidt_empty_input_returns_empty():
    pass


def test_gram_schmidt_output_vectors_are_unit_length():
    pass


def test_gram_schmidt_output_vectors_are_orthogonal():
    pass


def test_gram_schmidt_spans_same_subspace():
    pass


def test_gram_schmidt_single_vector():
    pass


def test_gram_schmidt_standard_basis_unchanged():
    pass


def test_gram_schmidt_raises_on_unequal_lengths():
    pass


def test_gram_schmidt_raises_on_linearly_dependent_vectors():
    pass


# ---------------------------------------------------------------------------
# gaussian_elimination
# ---------------------------------------------------------------------------


def test_gaussian_elimination_2x2_system():
    pass


def test_gaussian_elimination_3x3_system():
    pass


def test_gaussian_elimination_identity_matrix():
    pass


def test_gaussian_elimination_solution_satisfies_ax_equals_b():
    pass


def test_gaussian_elimination_raises_on_non_square_matrix():
    pass


def test_gaussian_elimination_raises_on_dimension_mismatch():
    pass


def test_gaussian_elimination_raises_on_singular_matrix():
    pass
