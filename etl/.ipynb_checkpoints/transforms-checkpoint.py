import numpy as np
import pandas as pd
from typing import Dict

# Disable unnecessary warning for pandas==1.4.2
pd.options.mode.chained_assignment = None


def rename_columns(df: pd.DataFrame, columns_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    :param df: dataframe
    :param columns_mapping: key = from, value = to.
    :return: dataframe
    """
    df.rename(columns=columns_mapping, inplace=True)
    return df


def convert_date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def compute_returns(df: pd.DataFrame, price_colname: str) -> pd.DataFrame:
    assert price_colname in df.columns
    df["returns_t"] = df[price_colname].pct_change(1)
    return df


def compute_log_returns(df: pd.DataFrame, price_colname: str) -> pd.DataFrame:
    assert price_colname in df.columns
    df["log_returns_t"] = np.log(df[price_colname]/df[price_colname].shift(1))
    return df


def compute_direction(df: pd.DataFrame) -> pd.DataFrame:
    df["direction_t"] = df.apply(lambda row: 0 if row["log_returns_t"] < 0 else 1, axis=1)
    return df


def label_dataset(df: pd.DataFrame, labels_colname: str, column_to_label: str) -> pd.DataFrame:
    # Shifting introduces NaN at end range. Arbitrarily fill with 0.
    df[labels_colname] = df[column_to_label].shift(-1, fill_value=0)
    return df
