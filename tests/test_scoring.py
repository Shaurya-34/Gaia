import pytest

from scoring import is_recovered, jaccard, match_injection


def test_jaccard_values():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(2 / 6)
    assert jaccard([], []) == 0.0
    assert jaccard([1, 1, 2], [2]) == 0.5  # ids are a set


def test_match_picks_highest_jaccard():
    groups = [{"members": [1, 2, 9], "cst": 10.0}, {"members": [1, 2, 3, 4], "cst": 4.0}]
    assert match_injection([1, 2, 3, 4], groups) == (1, 1.0)


def test_match_tie_goes_to_higher_cst():
    groups = [{"members": [1, 2], "cst": 3.5}, {"members": [3, 4], "cst": 8.0}]
    assert match_injection([1, 2, 3, 4], groups) == (1, 0.5)


def test_no_overlap_is_no_match():
    assert match_injection([1, 2], [{"members": [5, 6], "cst": 9.0}]) == (None, 0.0)
    assert match_injection([1, 2], []) == (None, 0.0)


def test_recovered_threshold():
    assert is_recovered(0.5) and not is_recovered(0.49)
