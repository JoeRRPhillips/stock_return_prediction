import pandas as pd

from datetime import datetime
from yahoofinancials import YahooFinancials

from etl import transforms


def compute_risk_free_rate(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "zeros":
        # Near zero values in recent years
        df["risk_free_rate_t"] = 0.0

    elif method == "us_treasury_note_10yr":
        # Get US Treasury Note data
        start_date = datetime.strftime(df.index[0], format="%Y-%m-%d")
        end_date = datetime.strftime(df.index[-1], format="%Y-%m-%d")
        tnx_df = get_data_us_treasury_note_10yr(start_date, end_date)

        # Left join data on date
        df = pd.merge(df, tnx_df, left_index=True, right_index=True)

    else:
        raise ValueError("Unsupported method for risk free rate calculation")

    return df


def get_data_us_treasury_note_10yr(start_date: str, end_date: str, ticker: str = "TNX") -> pd.DataFrame:
    """
    Use the 10-Year US Treasury Note to indicate risk free return
    """
    yahoo_financials = YahooFinancials(ticker)
    data = yahoo_financials.get_historical_price_data(
        start_date=start_date, end_date=end_date, time_interval="daily"
    )
    df = pd.DataFrame(data[ticker]["prices"])

    # Reset date for convenience in joining with tick data.
    df["Date"] = pd.to_datetime(df["formatted_date"])
    df = df.drop("date", axis=1).set_index("Date")

    df = df \
        .pipe(transforms.compute_returns, price_colname="adjclose") \
        .pipe(transforms.rename_columns, columns_mapping={"returns_t": "risk_free_returns_t"}) \
        .fillna(0.0)

    return df[["risk_free_rate_t"]]
