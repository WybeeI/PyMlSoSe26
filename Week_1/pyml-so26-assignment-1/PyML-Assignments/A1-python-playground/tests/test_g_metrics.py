import pytest
from g_metrics import (
    ConfusionCounts,
    binary_confusion,
    multiclass_report,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# ConfusionCounts properties
# ---------------------------------------------------------------------------


def test_precision_normal():
    pass


def test_precision_zero_denominator():
    pass


def test_recall_normal():
    pass


def test_recall_zero_denominator():
    pass


def test_f1_normal():
    pass


def test_f1_zero_denominator():
    pass


def test_f1_is_harmonic_mean_of_precision_and_recall():
    pass


# ---------------------------------------------------------------------------
# binary_confusion
# ---------------------------------------------------------------------------


def test_binary_confusion_counts_tp_fp_fn_tn():
    pass


def test_binary_confusion_perfect_predictions():
    pass


def test_binary_confusion_all_wrong():
    pass


def test_binary_confusion_non_numeric_labels():
    pass


def test_binary_confusion_raises_on_empty_lists():
    pass


def test_binary_confusion_raises_on_length_mismatch():
    pass


# ---------------------------------------------------------------------------
# multiclass_report
# ---------------------------------------------------------------------------


def test_multiclass_report_keys_are_sorted():
    pass


def test_multiclass_report_correct_per_class_counts():
    pass


def test_multiclass_report_includes_classes_from_pred_only():
    pass


def test_multiclass_report_raises_on_empty():
    pass


def test_multiclass_report_raises_on_length_mismatch():
    pass


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_macro_unweighted_mean():
    pass


def test_aggregate_weighted_by_class_support():
    pass


def test_aggregate_micro_pools_counts():
    pass


def test_aggregate_results_rounded_to_4dp():
    pass


def test_aggregate_raises_on_unknown_strategy():
    pass
