 import hydra
import pandas as pd
import yfinance as yf
from omegaconf import DictConfig, OmegaConf

from etl import partitions, scaling, storage, transforms, validation
from features import indicators, risk_free_rate, temporal
from features import moving_averages as ma
from model.train import train_and_save


def compute_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    # Add returns and log returns data.
    df = df \
        .pipe(transforms.compute_returns, price_colname=cfg.returns.price_colname) \
        .pipe(transforms.compute_log_returns, price_colname=cfg.returns.price_colname) \
        .pipe(transforms.compute_direction) \
        .pipe(transforms.label_dataset, labels_colname=cfg.data.labels_colname, column_to_label="direction_t") \
        .dropna()

    # Add technical indicators.
    if cfg.features.sharpe_ratio:
        df = df \
            .pipe(risk_free_rate.compute_risk_free_rate, method="zeros") \
            .pipe(
                indicators.sharpe_ratio,
                returns_colname="log_returns_t",
                risk_free_rate_colname="risk_free_rate_t",
                sharpe_ratio_colname="sharpe_ratio_t"
            )

    if cfg.features.sortino_ratio:
        df = df \
            .pipe(risk_free_rate.compute_risk_free_rate, method="zeros") \
            .pipe(
                indicators.sortino_ratio,
                returns_colname="log_returns_t",
                risk_free_rate_colname="risk_free_rate_t",
                sortino_ratio_colname="sortino_ratio_t"
            )

    if cfg.features.price_spread_open_close:
        df = df.pipe(indicators.price_spread_open_close, open_colname="Open", close_colname="price_t")

    if cfg.features.price_spread_high_low:
        df = df.pipe(indicators.price_spread_high_low, high_colname="High", low_colname="Low")

    if cfg.features.ema:
        for span in cfg.ema.spans:
            df = df.pipe(ma.exponential_moving_average, colname=cfg.ema.colname, span=span)

    if cfg.features.sma:
        for span in cfg.sma.spans:
            df = df.pipe(ma.simple_moving_average, colname=cfg.sma.colname, span=span)

    if cfg.features.momentum:
        for span in cfg.momentum.spans:
            df = df.pipe(indicators.momentum, colname=cfg.momentum.colname, span=span)

    # Drop raw features that are no longer required.
    df.drop(cfg.features.drop_columns, axis=1, inplace=True)

    # If the raw feature was normalised, a normalised column will have been computed. Drop this as well.
    for col in cfg.features.drop_columns:
        if col in cfg.scaling.normalise_min_max:
            df.drop(f"{col}_normalised", axis=1, inplace=True)

    # Columns to drop raw value but keep scaled volume.
    df.drop(["Volume"], axis=1, inplace=True)

    return df


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    df = yf.download(cfg.data.ticker, start=cfg.data.start_date, end=cfg.data.end_date, progress=False)
    storage.save_df_to_csv(df, cfg.storage.raw_data_fp, index=False)

    # Preprocess data: add returns, log_returns, direction columns.
    df = df \
        .pipe(transforms.rename_columns, columns_mapping={"Adj Close": "price_t"}) \
        .dropna()

    # Separate datasets from training and model validation
    train_df, validation_df = partitions.train_validation_split(df, validation_ratio=cfg.train.validation_ratio)

    # Scale data using summary statistics from the training set only. Normalise appropriate raw data.
    train_df, validation_df = scaling.normalise(train_df, validation_df, colnames=cfg.scaling.normalise_min_max)

    train_df = compute_features(train_df, cfg)
    validation_df = compute_features(validation_df, cfg)

    # Scale data using summary statistics from the training set only. Standardise appropriate features.
    train_df, validation_df = scaling.standardise(train_df, validation_df, colnames=cfg.scaling.standardise)

    # Add past returns data after standardisation
    if cfg.features.past_returns:
        train_df = train_df \
            .pipe(
                temporal.past_returns,
                colname=cfg.past_returns.colname,
                time_periods=cfg.past_returns.time_periods
            )
        validation_df = validation_df \
            .pipe(
                temporal.past_returns,
                colname=cfg.past_returns.colname,
                time_periods=cfg.past_returns.time_periods
            )

    # Columns to drop raw value but keep scaled volume.
    train_df.drop(["sharpe_ratio_t", "sortino_ratio_t"], axis=1, inplace=True)
    validation_df.drop(["sharpe_ratio_t", "sortino_ratio_t"], axis=1, inplace=True)

    # Assess data (im)balance
    pct_decreases_train, pct_increases_train = validation.check_balance(train_df["direction_t"])
    pct_decreases_val, pct_increases_val = validation.check_balance(validation_df["direction_t"])
    print(
        f"Percentage Decreases Train = {pct_decreases_train} | Percentage Increases Train = {pct_increases_train}"
        f"\nPercentage Decreases Validation = {pct_decreases_val} | Percentage Increases Validation = {pct_increases_val}"
    )

    # Write intermediate data to disk for inspection.
    storage.save_df_to_csv(df, cfg.storage.processed_data_fp, index=False)
    storage.save_df_to_csv(train_df, cfg.storage.train_data_fp, index=False)
    storage.save_df_to_csv(validation_df, cfg.storage.validation_data_fp, index=False)

    # Split data -> (features, labels).
    train_features_df = train_df.copy()
    train_labels_df = train_features_df.pop(cfg.data.labels_colname)

    validation_features_df = validation_df.copy()
    validation_labels_df = validation_features_df.pop(cfg.data.labels_colname)

    # Fit models and compare results.
    # Baseline Logistic Regression with no regularisation.
    log_reg_metrics = train_and_save(
        model_cfg=cfg.logistic_regression,
        max_iter=cfg.train.max_iter,
        train_features_df=train_features_df,
        train_labels_df=train_labels_df,
        validation_features_df=validation_features_df,
        validation_labels_df=validation_labels_df,
        save_model_dir=cfg.storage.save_model_dir,
        model_name="LogisticRegression",
    )

    # LASSO L1 Regularisation.
    lasso_metrics = train_and_save(
        model_cfg=cfg.lasso,
        max_iter=cfg.train.max_iter,
        train_features_df=train_features_df,
        train_labels_df=train_labels_df,
        validation_features_df=validation_features_df,
        validation_labels_df=validation_labels_df,
        save_model_dir=cfg.storage.save_model_dir,
        model_name="LASSO",
    )

    # Ridge L2 Regularisation.
    ridge_metrics = train_and_save(
        model_cfg=cfg.ridge,
        max_iter=cfg.train.max_iter,
        train_features_df=train_features_df,
        train_labels_df=train_labels_df,
        validation_features_df=validation_features_df,
        validation_labels_df=validation_labels_df,
        save_model_dir=cfg.storage.save_model_dir,
        model_name="Ridge",
    )

    # Elastic Net Model with L1 & L2 Regularisation.
    elasticnet_metrics = train_and_save(
        model_cfg=cfg.elasticnet,
        max_iter=cfg.train.max_iter,
        train_features_df=train_features_df,
        train_labels_df=train_labels_df,
        validation_features_df=validation_features_df,
        validation_labels_df=validation_labels_df,
        save_model_dir=cfg.storage.save_model_dir,
        model_name="ElasticNet",
    )

    print("\n\nMETRICS: ")
    print("\n\tRandom Baseline:")
    print(
        "\t\tlog_loss = 0.693"
        "\n\t\tacc = 0.5"
    )
    print("\n\tLogistic Regression:")
    for k, v in log_reg_metrics.items():
        print(f"\t\t{k} = {v}")

    print("\n\tLASSO:")
    for k, v in lasso_metrics.items():
        print(f"\t\t{k} = {v}")

    print("\n\tRidge:")
    for k, v in ridge_metrics.items():
        print(f"\t\t{k} = {v}")

    print("\n\tElasticNet:")
    for k, v in elasticnet_metrics.items():
        print(f"\t\t{k} = {v}")


if __name__ == "__main__":
    main()
