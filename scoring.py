"""Jaccard scorer and injection-to-group matcher (PREREG section 5)."""
import numpy as np

from config import load_config

JACCARD_RECOVERED = load_config()["definitions"]["jaccard_recovered"]


def jaccard(a, b) -> float:
    """|A & B| / |A | B| of two collections of source_ids (0 if both empty)."""
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_injection(member_ids, groups):
    """Match an injection's surviving members to the retrieved group with the
    highest Jaccard; ties go to the higher CST. `groups` is a list of dicts
    with "members" (source_ids) and "cst". Returns (index, jaccard), or
    (None, 0.0) when no group shares a member."""
    best, best_key = None, (0.0, -np.inf)
    for i, g in enumerate(groups):
        key = (jaccard(member_ids, g["members"]), g["cst"])
        if key[0] > 0 and key > best_key:
            best, best_key = i, key
    return best, best_key[0]


def is_recovered(jaccard_value) -> bool:
    return jaccard_value >= JACCARD_RECOVERED
