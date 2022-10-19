import pandas as pd
import numpy as np
import sklearn
from omegaconf import DictConfig
from typing import Dict

from etl import storage


def train_and_save(
        model_cfg: DictConfig,
        max_iter: int,
        train_features_df: pd.DataFrame,
        train_labels_df: pd.DataFrame,
        validation_features_df: pd.DataFrame,
        validation_labels_df: pd.DataFrame,
        save_model_dir: str,
        model_name: str,
) -> Dict[str, float]:
    """
    Creates a logistic regression model and applies the appropriate penalty defined in the model config.
    Trains the model and saves the coefficients in 2 formats:
    1. Pickle: for reuse of the model.
    2. Df to csv: for inspection.

    :param model_cfg:
    :param max_iter:
    :param train_features_df:
    :param train_labels_df:
    :param validation_features_df:
    :param validation_labels_df:
    :param save_model_dir:
    :param model_name:
    :return:
    """
    model = sklearn.linear_model.LogisticRegression(
        penalty=model_cfg.penalty,
        solver=model_cfg.solver,
        C=model_cfg.regularisation,
        l1_ratio=model_cfg.l1_ratio,  # Only used for ElasticNet
        max_iter=max_iter,
    )

    model.fit(train_features_df, train_labels_df)

    # 1. Save models for reuse
    storage.save_model(model=model, filepath=f"{save_model_dir}/models")

    # 2. Save models coefficients for inspection
    coefficients_df = pd.concat([
        pd.DataFrame(train_features_df.columns),
        pd.DataFrame(np.transpose(model.coef_))
    ], axis=1)
    storage.save_df_to_csv(df=coefficients_df, filepath=f"{save_model_dir}/coeffs/{model_name}.csv", index=False)

    # 3. Log training metrics
    train_log_loss = model.score(train_features_df, train_labels_df)
    train_acc = accuracy(model, train_features_df, train_labels_df)

    # 4. Log validation metrics
    val_log_loss = model.score(validation_features_df, validation_labels_df)
    val_acc = accuracy(model, validation_features_df, validation_labels_df)

    metrics = {
        "log_loss": train_log_loss,
        "val_log_loss": val_log_loss,
        "acc": train_acc,
        "val_acc": val_acc,
    }

    return metrics


def accuracy(model, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> float:
    y_true = labels_df.to_numpy()
    y_pred = model.predict(features_df)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    return acc
