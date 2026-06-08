"""
Classification Metrics

After training a classifier you need to measure how well it performs.
Below you will implement the standard binary and multiclass evaluation
metrics used throughout ML — the same ones reported by scikit-learn's
classification_report and logged by most experiment trackers.

Resources to understand potentially unfamiliar concepts that are used in this
assignment are found below.

More information about the `@dataclass` decorator can be found here:

    https://docs.python.org/3/library/dataclasses.html

More information about the `@property` decorator can be found here:

    https://www.stratascratch.com/blog/how-to-use-python-property-decorator-with-examples

More information about precision, recall, and f1 metrics can be found here:

    https://en.wikipedia.org/wiki/Precision_and_recall


DO NOT MODIFY THE FUNCTION SIGNATURES OR THE DATACLASS FIELD DEFINITIONS.
"""

import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. ConfusionCounts  (dataclass)
# ---------------------------------------------------------------------------


@dataclass
class ConfusionCounts:
    """
    Stores the four outcomes of binary classification for a single class.

    Fields (set automatically by @dataclass — do not add __init__):
        tp -- true  positives (predicted positive, actually positive)
        fp -- false positives (predicted positive, actually negative)
        fn -- false negatives (predicted negative, actually positive)
        tn -- true  negatives (predicted negative, actually negative)

    Implement the three properties below. Each should return 0.0 whenever
    its denominator would be zero.

    Properties to implement:
        precision -- tp / (tp + fp)
        recall    -- tp / (tp + fn)
        f1        -- harmonic mean of precision and recall:
                     2 * precision * recall / (precision + recall)
    """

    tp: int
    fp: int
    fn: int
    tn: int

    # ------ SOLUTION GOES HERE!  ------
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom != 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom != 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        denom = p + r
        return 2 * p * r / denom if denom != 0 else 0.0

# ---------------------------------------------------------------------------
# 2. binary_confusion
# ---------------------------------------------------------------------------


def binary_confusion(y_true: list, y_pred: list, pos_label) -> ConfusionCounts:
    """
    Computes TP, FP, FN, TN by comparing two parallel label lists, treating
    pos_label as the positive class (all other values are negative).

    Use a match / case statement to classify each (true, pred) pair into
    one of the four outcomes. Match on a tuple of two booleans:
        (true == pos_label, pred == pos_label)

    Arguments:
        y_true    -- list of ground-truth labels
        y_pred    -- list of predicted labels (same length as y_true)
        pos_label -- the value treated as the positive class

    Returns:
        ConfusionCounts

    Raises:
        ValueError -- if y_true and y_pred differ in length
        ValueError -- if either list is empty

    Example:
        >>> binary_confusion([1, 0, 1, 1], [1, 1, 1, 0], pos_label=1)
        ConfusionCounts(tp=2, fp=1, fn=1, tn=0)
    """

    # ------ SOLUTION GOES HERE!  ------
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if len(y_true) == 0:
        raise ValueError("Lists must not be empty")

    tp = fp = fn = tn = 0

    for t, p in zip(y_true, y_pred):
        match (t == pos_label, p == pos_label):
            case (True, True):
                tp += 1
            case (False, True):
                fp += 1
            case (True, False):
                fn += 1
            case (False, False):
                tn += 1

    return ConfusionCounts(tp, fp, fn, tn)

# ---------------------------------------------------------------------------
# 3. multiclass_report
# ---------------------------------------------------------------------------


def multiclass_report(y_true: list, y_pred: list) -> dict[str, ConfusionCounts]:
    """
    Computes per-class ConfusionCounts for every label that appears in
    y_true or y_pred (the one-vs-rest convention: for each class, that
    class is positive and all others are negative).

    Use a set to determine the unique classes, then call binary_confusion()
    for each one. Return results as a dict comprehension whose keys are
    sorted so that output order is deterministic.

    Arguments:
        y_true -- list of ground-truth labels
        y_pred -- list of predicted labels (same length as y_true)

    Returns:
        dict -- maps each class label to its ConfusionCounts,
                with keys in sorted order

    Raises:
        ValueError -- if the lists are empty or of unequal length

    Example:
        >>> r = multiclass_report(["a","b","a"], ["a","a","b"])
        >>> r["a"]
        ConfusionCounts(tp=1, fp=1, fn=1, tn=0)
    """

    # ------ SOLUTION GOES HERE!  ------
def multiclass_report(y_true: list, y_pred: list) -> dict[str, ConfusionCounts]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if len(y_true) == 0:
        raise ValueError("Lists must not be empty")

    classes = sorted(set(y_true) | set(y_pred))

    return {
        cls: binary_confusion(y_true, y_pred, pos_label=cls)
        for cls in classes
    }

# ---------------------------------------------------------------------------
# 4. aggregate_metrics
# ---------------------------------------------------------------------------


def aggregate_metrics(
    report: dict[str, ConfusionCounts],
    y_true: list,
    strategy: str,
) -> dict[str, float]:
    """
    Aggregates a per-class report (from multiclass_report) into a single
    set of summary statistics using one of three strategies.

    Use a match / case statement to select the strategy. Each branch should
    compute summary precision, recall, and f1:

        "macro"    -- unweighted mean of per-class precision, recall, f1

        "weighted" -- mean weighted by each class's support, i.e. the number
                      of times that class appears in y_true

        "micro"    -- pool all TP, FP, FN counts across classes first, then
                      compute precision/recall/f1 from the totals
                      (hint: you can construct a temporary ConfusionCounts)

        any other  -- raise ValueError

    Arguments:
        report   -- dict returned by multiclass_report()
        y_true   -- the original ground-truth labels (needed for support)
        strategy -- one of "macro", "weighted", or "micro"

    Returns:
        dict with keys "precision", "recall", "f1", each rounded to 4 d.p.

    Raises:
        ValueError -- if strategy is not one of the three valid strings
    """

    # ------ SOLUTION GOES HERE!  ------
    # support por clase
    support = {cls: y_true.count(cls) for cls in report}

    match strategy:
        # ---------------- MACRO ----------------
        case "macro":
            precisions = [c.precision for c in report.values()]
            recalls    = [c.recall    for c in report.values()]
            f1s        = [c.f1        for c in report.values()]

            precision = sum(precisions) / len(precisions)
            recall    = sum(recalls)    / len(recalls)
            f1        = sum(f1s)        / len(f1s)

        # ---------------- WEIGHTED ----------------
        case "weighted":
            total = sum(support.values())

            precision = sum(report[cls].precision * support[cls] for cls in report) / total
            recall    = sum(report[cls].recall    * support[cls] for cls in report) / total
            f1        = sum(report[cls].f1        * support[cls] for cls in report) / total

        # ---------------- MICRO ----------------
        case "micro":
            tp = sum(c.tp for c in report.values())
            fp = sum(c.fp for c in report.values())
            fn = sum(c.fn for c in report.values())
            tn = sum(c.tn for c in report.values())

            pooled = ConfusionCounts(tp, fp, fn, tn)

            precision = pooled.precision
            recall    = pooled.recall
            f1        = pooled.f1

        # ---------------- INVALID ----------------
        case _:
            raise ValueError("Invalid strategy: must be 'macro', 'weighted', or 'micro'")

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }

if __name__ == "__main__":
    # Small 3-class dataset: ground-truth labels vs. model predictions
    y_true = ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "dog"]
    y_pred = ["cat", "cat", "cat", "bird", "dog", "dog", "bird", "dog"]

    # Build a per-class confusion matrix using one-vs-rest: each class is
    # treated as "positive" while the other two are "negative"
    report = multiclass_report(y_true, y_pred)

    # Print a table with precision, recall, F1, and raw counts for each class
    print(f"{'Class':<8} {'Prec':>7} {'Rec':>7} {'F1':>7}  Counts")
    print("-" * 50)
    for cls, counts in report.items():
        print(f"{cls:<8} {counts.precision:>7.3f} {counts.recall:>7.3f} {counts.f1:>7.3f}  {counts}")

    # Collapse per-class metrics into a single summary score.
    # The aggregation strategy can be passed as a CLI argument; macro is the
    # default because it treats every class equally regardless of frequency:
    #   python 6_metrics.py macro      (unweighted mean across classes)
    #   python 6_metrics.py weighted   (mean weighted by class frequency)
    #   python 6_metrics.py micro      (pool all TP/FP/FN, then compute)
    strategy = sys.argv[1] if len(sys.argv) > 1 else "macro"
    summary = aggregate_metrics(report, y_true, strategy)
    print(f"\nAggregate ({strategy}): {summary}")
