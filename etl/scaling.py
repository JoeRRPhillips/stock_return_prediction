import pandas as pd
from typing import List, Tuple


def normalise(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        colnames: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs min-max normalisation over the train and validation datasets.
    The normalisation for the validation set is applied using summary
    statistics from the training set to avoid data snooping.

    Note: the unnormalised columns are not dropped here to allow flexibility for
    indicators to use normalised or unnormalised values. Consideration is given
    to dropping the raw values after feature processing.

    :param train_df:
    :param validation_df:
    :param colnames: which columns to normalise
    :return:
    """

    for col in colnames:
        min_val = min(train_df[col])
        max_val = max(train_df[col])

        train_df[f"{col}_normalised"] = (train_df[col] - min_val) / (max_val - min_val)
        validation_df[f"{col}_normalised"] = (validation_df[col] - min_val) / (max_val - min_val)

    return train_df, validation_df


def standardise(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        colnames: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardises data to (mean, std) = (0, 1).
    The standardisation for the validation set is applied using summary
    statistics from the training set to avoid data snooping.

    :param train_df:
    :param validation_df:
    :param colnames: which columns to standardise
    :return:
    """
    for col in colnames:
        mean_train = train_df[col].mean()
        std_train = train_df[col].std()

        train_df[f"{col}_standardised"] = (train_df[col] - mean_train) / std_train
        validation_df[f"{col}_standardised"] = (validation_df[col] - mean_train) / std_train

    return train_df, validation_df
