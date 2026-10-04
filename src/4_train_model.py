"""Train and evaluate injury-risk and performance prediction models."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def chronological_split(df, test_fraction=0.2):
    """Reserve each participant's most recent dates for testing."""
    train_parts = []
    test_parts = []
    for _, participant_df in df.groupby("participant_id", sort=False):
        split_index = max(1, int(len(participant_df) * (1 - test_fraction)))
        train_parts.append(participant_df.iloc[:split_index])
        test_parts.append(participant_df.iloc[split_index:])
    return pd.concat(train_parts), pd.concat(test_parts)


def get_model_features(df):
    excluded = {
        "participant_id",
        "date",
        "injury_today",
        "injury_event",
        "injury_next_7d",
        "performance_index",
        "performance_index_next_day",
    }
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded
    ]


def train_models(input_csv="data/processed/final_features_dataset.csv"):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Fichier '{input_csv}' introuvable. Exécute d'abord 3_features.py."
        )

    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(
        subset=[
            "participant_id",
            "date",
            "injury_next_7d",
            "performance_index_next_day",
        ]
    )
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)

    features = get_model_features(df)
    if not features:
        raise ValueError("Aucune variable numérique disponible pour entraîner les modèles.")

    train_df, test_df = chronological_split(df)
    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)

    injury_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=3,
        random_state=42,
    )
    injury_model.fit(X_train, train_df["injury_next_7d"].astype(int))
    injury_predictions = injury_model.predict(X_test)
    injury_probabilities = injury_model.predict_proba(X_test)[:, 1]

    performance_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
    )
    performance_model.fit(X_train, train_df["performance_index_next_day"])
    performance_predictions = performance_model.predict(X_test)

    y_injury = test_df["injury_next_7d"].astype(int)
    metrics = {
        "injury_precision": precision_score(
            y_injury, injury_predictions, zero_division=0
        ),
        "injury_recall": recall_score(
            y_injury, injury_predictions, zero_division=0
        ),
        "injury_f1": f1_score(y_injury, injury_predictions, zero_division=0),
        "injury_pr_auc": average_precision_score(y_injury, injury_probabilities),
        "performance_mae": mean_absolute_error(
            test_df["performance_index_next_day"], performance_predictions
        ),
        "performance_rmse": np.sqrt(
            mean_squared_error(test_df["performance_index_next_day"], performance_predictions)
        ),
        "performance_r2": r2_score(
            test_df["performance_index_next_day"], performance_predictions
        ),
    }
    metrics["injury_roc_auc"] = (
        roc_auc_score(y_injury, injury_probabilities)
        if y_injury.nunique() == 2
        else np.nan
    )

    predictions = test_df[
        [
            "participant_id",
            "date",
            "injury_next_7d",
            "performance_index_next_day",
        ]
    ].copy()
    predictions["injury_probability"] = injury_probabilities
    predictions["injury_prediction"] = injury_predictions
    predictions["performance_prediction"] = performance_predictions
    predictions = predictions.rename(
        columns={"performance_index_next_day": "performance_target"}
    )
    os.makedirs("data/processed", exist_ok=True)
    predictions.to_csv("data/processed/model_test_predictions.csv", index=False)

    print("=== Modèle de risque de blessure ===")
    for name in [
        "injury_precision",
        "injury_recall",
        "injury_f1",
        "injury_roc_auc",
        "injury_pr_auc",
    ]:
        print(f"{name}: {metrics[name]:.3f}")

    print("\n=== Modèle d'indice de performance ===")
    for name in ["performance_mae", "performance_rmse", "performance_r2"]:
        print(f"{name}: {metrics[name]:.3f}")

    print(f"\nVariables utilisées: {len(features)}")
    print("Prédictions sauvegardées dans data/processed/model_test_predictions.csv")
    return injury_model, performance_model, metrics


if __name__ == "__main__":
    train_models()
