"""
Finding the Best Split

`candidate_thresholds` is provided. Implement `best_split`, which performs
an exhaustive search over every feature and candidate threshold, returning
the split with the highest information gain.

DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

from collections.abc import Callable

from .impurity import information_gain


def candidate_thresholds(values: list[float]) -> list[float]:
    """
    Returns candidate split thresholds for a single feature column.

    Candidates are the midpoints between every pair of consecutive unique
    sorted values. Any threshold within a gap produces the same partition,
    so the midpoint is a canonical representative.

    Arguments:
        values -- feature values from one column

    Returns:
        list[float] -- sorted candidate thresholds; empty if all values
                       are identical
    """
    unique_sorted = sorted(set(values))
    return [(a + b) / 2 for a, b in zip(unique_sorted, unique_sorted[1:])]


def best_split(
    X: list[list],
    y: list[int],
    criterion: Callable[[list[int]], float],
    feature_types: list[str] | None = None,
) -> tuple[int, float | str, float] | None:
    """
    Finds the (feature, split_value) pair that maximizes information gain.

    For numeric features, searches every candidate threshold from
    `candidate_thresholds`; samples with feature value <= threshold go
    left, the rest go right.

    For categorical features, each unique value is a candidate; samples
    where feature == value go left, the rest go right.

    Arguments:
        X             -- feature vectors (n_samples x n_features)
        y             -- labels (length n_samples)
        criterion     -- impurity callable, e.g. `gini` or `entropy`
        feature_types -- per-feature type strings, either 'numeric' or
                         'categorical'; None treats all as numeric

    Returns:
        tuple(feature_index, split_value, gain) -- the best split found,
            where split_value is a float threshold for numeric features
            and a string category value for categorical features
        None -- if no split produces strictly positive gain
    """

    # ------ WRITE YOUR CODE HERE ------

    n_samples = len(y)
    n_features = len(X[0])

    if feature_types is None:
        feature_types = ["numeric"] * n_features

    best_feature = None
    best_value = None
    best_gain = 0.0

    # Recorremos cada feature
    for j in range(n_features):
        column = [row[j] for row in X]
        ftype = feature_types[j]

        # -------------------------
        # NUMERIC FEATURE
        # -------------------------
        if ftype == "numeric":
            thresholds = candidate_thresholds(column)

            for t in thresholds:
                y_left = [y[i] for i in range(n_samples) if column[i] <= t]
                y_right = [y[i] for i in range(n_samples) if column[i] > t]

                if not y_left or not y_right:
                    continue  # split inválido

                gain = information_gain(y, y_left, y_right, criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_value = t

        # -------------------------
        # CATEGORICAL FEATURE
        # -------------------------
        else:
            categories = set(column)

            for cat in categories:
                y_left = [y[i] for i in range(n_samples) if column[i] == cat]
                y_right = [y[i] for i in range(n_samples) if column[i] != cat]

                if not y_left or not y_right:
                    continue

                gain = information_gain(y, y_left, y_right, criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_value = cat

    # Si no hubo ganancia positiva
    if best_gain <= 0:
        return None

    return (best_feature, best_value, best_gain)
