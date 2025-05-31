"""This module contains metrics for the LLM."""

from Levenshtein import distance as levenshtein_distance


def calculate_exact_match(ground_truth: str, prediction: str) -> float:
    """Calculate the exact match metric."""
    return float(ground_truth == prediction)


def calculate_levenshtein_distance(ground_truth: str, prediction: str) -> int:
    """Calculate the Levenshtein distance between the ground truth and prediction."""
    return levenshtein_distance(ground_truth, prediction)
