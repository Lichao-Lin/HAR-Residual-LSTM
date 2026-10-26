"""Chronological holdouts with four excluded forecast origins at each boundary."""

import numpy as np

SPLIT_GAP = 4


def split_indices(n):
    """Keep 70%/85% cutoffs and skip four observations before each holdout.

    The gap counts observed trading dates, not calendar days. Historical input
    windows may overlap; the five-observation future target windows do not.
    """
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    validation_start = train_end + SPLIT_GAP
    test_start = validation_end + SPLIT_GAP
    if train_end < 1 or validation_start >= validation_end or test_start >= n:
        raise ValueError("Not enough observations for three nonempty splits and four-day gaps")
    return (
        np.arange(train_end),
        np.arange(validation_start, validation_end),
        np.arange(test_start, n),
    )
