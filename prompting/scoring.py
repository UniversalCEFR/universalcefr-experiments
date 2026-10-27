"""Original CEFR label extraction and metrics, with explicit coverage."""

LABELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def extract_label(response):
    """Return the first A1..C2 substring in label order, as in the original code.

    This retains case sensitivity and label-order precedence. Ambiguous
    responses and substrings such as A10 therefore need caution.
    """
    if not isinstance(response, str):
        raise ValueError("model response must be a string")
    return next((label for label in LABELS if label in response), "UNKNOWN")


def compute_metrics(records):
    """Exclude UNKNOWN predictions and macro-average over all six CEFR labels."""
    valid = [row for row in records if row["prediction"] != "UNKNOWN"]
    result = {"sample_count": len(records), "recognized_count": len(valid),
              "unknown_count": len(records) - len(valid),
              "coverage": len(valid) / len(records) if records else 0.0,
              "accuracy": None, "precision_macro": None,
              "recall_macro": None, "f1_macro": None}
    if not valid:
        return result
    result["accuracy"] = sum(row["prediction"] == row["gold_label"] for row in valid) / len(valid)
    precision, recall, f1 = [], [], []
    for label in LABELS:
        true_positive = sum(row["prediction"] == row["gold_label"] == label for row in valid)
        predicted = sum(row["prediction"] == label for row in valid)
        actual = sum(row["gold_label"] == label for row in valid)
        p = true_positive / predicted if predicted else 0.0
        r = true_positive / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    result.update(precision_macro=sum(precision) / len(LABELS),
                  recall_macro=sum(recall) / len(LABELS),
                  f1_macro=sum(f1) / len(LABELS))
    return result
