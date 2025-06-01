"""This module contains metrics for the LLM."""

from Levenshtein import distance as levenshtein_distance


def calculate_exact_match(ground_truth: str, prediction: str) -> float:
    """Calculate the exact match metric."""
    return float(ground_truth == prediction)


def calculate_levenshtein_distance(ground_truth: str, prediction: str) -> int:
    """Calculate the Levenshtein distance between the ground truth and prediction."""
    return levenshtein_distance(ground_truth, prediction)


def calculate_partial_match(ground_truth: str, prediction: str) -> float:
    """
    Calculates a partial match score based on common items in comma-separated strings.
    It can also handle lists of strings for ground_truth and prediction.

    Args:
        ground_truth: The comma-separated string or list of expected items.
        prediction: The comma-separated string or list of predicted items.

    Returns:
        A float score between 0.0 and 1.0, where 1.0 means all ground truth items
        are present in the prediction, and 0.0 means no ground truth items are present.
        Returns 0.0 if the ground_truth is None, empty, or whitespace-only (or an empty list).
    """
    if isinstance(ground_truth, list):
        ground_truth = ",".join(str(item) for item in ground_truth)
    if isinstance(prediction, list):
        prediction = ",".join(str(item) for item in prediction)

    if not ground_truth or ground_truth.isspace():
        return 0.0  # Or handle as an error/None, depending on desired behavior

    gt_items = {item.strip() for item in ground_truth.split(",") if item.strip()}
    if not gt_items:  # Handles cases where ground_truth might be just commas like ",,"
        return 0.0

    if not prediction or prediction.isspace():
        pred_items = set()
    else:
        pred_items = {item.strip() for item in prediction.split(",") if item.strip()}

    common_items = gt_items.intersection(pred_items)

    return len(common_items) / len(gt_items)
