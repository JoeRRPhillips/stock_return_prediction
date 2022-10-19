import pandas as pd
import pickle
from pathlib import Path


def save_df_to_csv(df: pd.DataFrame, filepath: str, index: bool) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)


def save_model(model, filepath: str) -> None:
    filepath = Path(f"{filepath}/saved_model.pickle")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
