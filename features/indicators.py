import math
import pandas as pd


def sharpe_ratio(
        df: pd.DataFrame,
        returns_colname: str,
        risk_free_rate_colname: str,
        sharpe_ratio_colname: str = "sharpe_ratio_t",
) -> pd.DataFrame:
    """
    Computes Sharpe Ratio expanding over time dimension.
    E.g. for daily data, the calculation uses summary statistics up to day i, where i \in {0,t}.

    Leverages the Martingale property: E[X_{n+1} | X_1, ... ,X_n] = X_n to use returns at time t
    in place of expected returns from time t+1 onwards.

    :param df: dataframe
    :param returns_colname: name of the column containing the (log) returns of the asset.
    :param risk_free_rate_colname: name of the column containing the (log) returns of a risk free baseline asset.
    :param sharpe_ratio_colname: name of the new column which will contain the computed Sharpe Ratio.
    :return:
    """
    returns = df[returns_colname].expanding().mean()
    risk_free_rate = df[risk_free_rate_colname].expanding().mean()
    std = df[returns_colname].expanding().std()

    df[sharpe_ratio_colname] = (returns - risk_free_rate) / std
    return df


def sortino_ratio(
        df: pd.DataFrame,
        returns_colname: str,
        risk_free_rate_colname: str,
        sortino_ratio_colname: str = "sortino_ratio_t",
) -> pd.DataFrame:
    """
    Computes Sortino Ratio expanding over time dimension.
    E.g. for daily data, the calculation uses summary statistics up to day i, where i \in {0,t}.

    Leverages the Martingale property: E[X_{n+1} | X_1, ... ,X_n] = X_n to use returns at time t
    in place of expected returns from time t+1 onwards.

    :param df: dataframe
    :param returns_colname: name of the column containing the (log) returns of the asset.
    :param risk_free_rate_colname: name of the column containing the (log) returns of a risk free baseline asset.
    :param sortino_ratio_colname: name of the new column which will contain the computed Sortino Ratio.
    :return:
    """
    assert returns_colname in df.columns

    returns = df[returns_colname].expanding().mean()
    risk_free_rate = df[risk_free_rate_colname].expanding().mean()

    downside_dev_df = df[df[returns_colname] < 0]
    downside_dev = downside_dev_df[returns_colname].std()

    # Scaling can result in no negative values.
    assert not math.isnan(downside_dev)

    df[sortino_ratio_colname] = (returns - risk_free_rate) / downside_dev
    return df


def price_spread_open_close(df: pd.DataFrame, open_colname: str, close_colname: str) -> pd.DataFrame:
    assert open_colname in df.columns
    assert close_colname in df.columns

    df["spread_open_close_t"] = df[open_colname] - df[close_colname]
    return df


def price_spread_high_low(df: pd.DataFrame, high_colname: str, low_colname: str) -> pd.DataFrame:
    assert high_colname in df.columns
    assert low_colname in df.columns

    df["spread_high_low_t"] = df[high_colname] - df[low_colname]
    return df


def momentum(df: pd.DataFrame, colname: str, span: int) -> pd.DataFrame:
    """
    Args:
        df: dataframe
        colname: data column to apply EMA over.
        span: duration of time to roll the MA over.
    """
    assert colname in df.columns

    df[f"momentum_{span}_{colname}"] = df[colname] - df[colname].shift(span)
    df.dropna(inplace=True)
    return df
