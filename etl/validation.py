from typing import Tuple


def check_balance(labels) -> Tuple[float, float]:
    """
    Finds the number of labels which are 0 vs. 1 for binary classification.

    :param labels: binary labels
    :return: percentages of 0s and 1s,
    """
    # Assess data (im)balance
    num_data = len(labels)
    num_ones = sum(labels)

    pct_ones = 100.0 * (num_ones / num_data)

    return 100.0 - pct_ones, pct_ones
