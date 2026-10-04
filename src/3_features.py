"""Clean merged data and calculate features and targets for model training.

-> final_features_dataset.csv
"""

import os

import numpy as np
import pandas as pd


def _scale(series, minimum, maximum):
    return ((series - minimum) / (maximum - minimum) * 100).clip(0, 100)


def _performance_index(df):
    """Build an explainable daily fitness index ranging from 0 to 100."""
    components = {}
    if "readiness" in df:
        components["readiness"] = _scale(df["readiness"], 0, 10)
    if "sleep_quality" in df:
        components["sleep_quality"] = _scale(df["sleep_quality"], 1, 5)
    if "sleep_duration_h" in df:
        components["sleep_duration"] = (100 - (df["sleep_duration_h"] - 8).abs() * 25).clip(0, 100)
    if "mood" in df:
        components["mood"] = _scale(df["mood"], 1, 5)
    if "fatigue" in df:
        components["fatigue"] = 100 - _scale(df["fatigue"], 1, 5)
    if "soreness" in df:
        components["soreness"] = 100 - _scale(df["soreness"], 1, 5)

    if not components:
        return pd.Series(np.nan, index=df.index)

    component_frame = pd.DataFrame(components, index=df.index)
    return component_frame.mean(axis=1, skipna=True).round(2)


def _add_injury_target(df, window_days=7):
    """Create a binary target indicating whether an injury occurs within the next 7 days."""
    df["injury_event"] = pd.to_numeric(df.get("injury_today", 0), errors="coerce").fillna(0).astype(int)
    df["injury_today"] = df["injury_event"]
    df["injury_next_7d"] = 0
    for _, participant_index in df.groupby("participant_id").groups.items():
        participant_dates = df.loc[participant_index, "date"]
        injury_dates = participant_dates[df.loc[participant_index, "injury_event"].eq(1)].to_numpy()
        df.loc[participant_index, "injury_next_7d"] = [
            int(any((injury_date > current_date)
                    and (injury_date <= current_date + pd.Timedelta(days=window_days))
                    for injury_date in injury_dates))
            for current_date in participant_dates
        ]
    return df


def _date_rolling(df, column, window, aggregation="mean"):
    """Calculate a rolling window in calendar days for each participant."""
    result = pd.Series(index=df.index, dtype=float)
    for _, participant_index in df.groupby("participant_id").groups.items():
        participant = df.loc[participant_index].sort_values("date")
        values = (
            participant.set_index("date")[column]
            .rolling(window, min_periods=1)
            .agg(aggregation)
            .to_numpy()
        )
        result.loc[participant.index] = values
    return result


def process_features(
    input_csv="data/processed/global_daily_dataset.csv",
    output_csv="data/processed/final_features_dataset.csv"
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Fichier '{input_csv}' introuvable. Lance 2_merge_dataset.py d'abord.")

    print("1. Chargement du dataset fusionné...")
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["participant_id", "date"]).sort_values(["participant_id", "date"]).reset_index(drop=True)

    print("2. Nettoyage, indicateurs de valeurs manquantes et outliers...")
    for column in ["meal_types", "image_names"]:
        if column in df:
            df[column] = df[column].fillna("")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    protected_columns = {"injury_today", "injury_event", "injury_next_7d"}
    for column in numeric_columns:
        if column in protected_columns:
            continue
        df[f"{column}_missing"] = df[column].isna().astype(int)
        if df[column].notna().sum() == 0:
            df[column] = 0.0
            continue
        participant_median = df.groupby("participant_id")[column].transform("median")
        df[column] = df[column].fillna(participant_median).fillna(df[column].median())
        first_quartile = df[column].quantile(0.25)
        third_quartile = df[column].quantile(0.75)
        iqr = third_quartile - first_quartile
        if iqr > 0:
            df[column] = df[column].clip(first_quartile - 1.5 * iqr, third_quartile + 1.5 * iqr)

    for column in ["num_meals", "total_calories_consumed"]:
        if column in df:
            df[column] = df[column].clip(lower=0)

    print("3. Variables temporelles et tendances glissantes...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    if {"total_calories_consumed", "calories_burned"}.issubset(df.columns):
        df["net_calorie_balance"] = df["total_calories_consumed"] - df["calories_burned"]

    rolling_columns = ["total_calories_consumed", "steps", "sleep_hours", "training_load", "readiness"]
    for column in rolling_columns:
        if column in df:
            df[f"{column}_7d_avg"] = _date_rolling(df, column, "7D")

    if "training_load" in df:
        df["training_load_7d_sum"] = _date_rolling(df, "training_load", "7D", "sum")
        chronic_load = _date_rolling(df, "training_load", "28D")
        df["training_load_28d_avg"] = chronic_load
        df["acwr"] = (df["training_load_7d_sum"] / (chronic_load * 7).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    print("4. Création des cibles de modélisation...")
    df = _add_injury_target(df)
    df["performance_index"] = _performance_index(df)
    df["performance_index_next_day"] = (
        df.groupby("participant_id")["performance_index"].shift(-1)
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset features généré : {output_csv} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


if __name__ == "__main__":
    final_dataset = process_features()
    print(final_dataset[["participant_id", "date", "injury_next_7d", "performance_index"]].head())