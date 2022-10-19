import pandas as pd
from typing import List


def past_returns(df: pd.DataFrame, colname: str, time_periods: List[int]) -> pd.DataFrame:
    """
    :param df: dataframe
    :param colname: names of the returns column to lag by k.
    :param time_periods: list of time steps to lag past returns by.
    :return: dataframe augmented with past return data.
    """
    assert colname in df.columns

    for k in time_periods:
        df[f"{colname}-{k}"] = df[colname].shift(k)

    df.dropna(inplace=True)
    return df
