import pytest

from src.metrics import calculate_partial_match


def test_partial_match_with_comma_separated_strings():
    gt = "apple, banana, orange"
    pred = "banana, kiwi, orange"
    assert calculate_partial_match(gt, pred) == pytest.approx(2 / 3)


def test_partial_match_with_lists():
    gt = ["alpha", "beta", "gamma"]
    pred = ["gamma", "beta", "delta"]
    assert calculate_partial_match(gt, pred) == pytest.approx(2 / 3)


def test_partial_match_empty_prediction():
    gt = "apple, banana"
    pred = "   "
    assert calculate_partial_match(gt, pred) == 0.0
