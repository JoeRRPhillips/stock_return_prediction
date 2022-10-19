import math
import pandas as pd
from typing import Optional, Tuple


def train_validation_split(df: pd.DataFrame, validation_ratio) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    assert validation_ratio >= 0.0
    if validation_ratio == 0.0:
        return df, None

    num_train = math.floor(len(df) * (1.0 - validation_ratio))
    train_df = df.iloc[:num_train, :]
    validation_df = df.iloc[num_train:, :]

    return train_df, validation_df
