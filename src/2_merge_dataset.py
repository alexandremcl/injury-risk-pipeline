"""
Merge nutrition, Fitbit, wellness, training, and self-report data sources.

-> global_daily_dataset.csv
"""

import json
import os
from pathlib import Path

import pandas as pd


def _date_column(values):
    """Convert heterogeneous date values to timezone-naive calendar dates."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _daily_numeric_json(path, value_column, aggregation="sum", date_column="dateTime"):
    if not path.exists():
        return pd.DataFrame(columns=["date", value_column])
    records = _load_json(path)
    df = pd.DataFrame(records)
    if df.empty or date_column not in df or "value" not in df:
        return pd.DataFrame(columns=["date", value_column])
    df["date"] = _date_column(df[date_column])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    grouped = df.dropna(subset=["date"]).groupby("date")["value"]
    result = getattr(grouped, aggregation)().reset_index(name=value_column)
    return result


def load_food_data(food_csv_path="data/processed/food_images_summary.csv"):
    """Aggregate meal photos and nutrition estimates by participant and day."""
    if not os.path.exists(food_csv_path):
        return pd.DataFrame(columns=["participant_id", "date"])

    df_food = pd.read_csv(food_csv_path)
    if df_food.empty:
        return pd.DataFrame(columns=["participant_id", "date"])
    df_food["date"] = _date_column(df_food["date"])
    calories_column = "calories_estimated" if "calories_estimated" in df_food else None
    aggregations = {"num_meals": ("file_name", "count")}
    if calories_column:
        aggregations["total_calories_consumed"] = (calories_column, "sum")
    for column in ["meal_types", "image_description", "meal_description"]:
        if column in df_food:
            aggregations["meal_types"] = (column, lambda values: ", ".join(dict.fromkeys(values.dropna().astype(str))))
            break
    aggregations["image_names"] = ("file_name", lambda values: ", ".join(values.dropna().astype(str)))
    return df_food.dropna(subset=["date"]).groupby(["participant_id", "date"]).agg(**aggregations).reset_index()


def load_fitbit_data(participant_path):
    """Aggregate Fitbit minute-level, sleep, and activity data by day."""
    fitbit_path = participant_path / "fitbit"
    frames = []
    for filename, column, aggregation in [
        ("calories.json", "calories_burned", "sum"),
        ("distance.json", "distance", "sum"),
        ("steps.json", "steps", "sum"),
        ("heart_rate.json", "heart_rate_mean", "mean"),
        ("resting_heart_rate.json", "resting_heart_rate", "mean"),
    ]:
        frames.append(_daily_numeric_json(fitbit_path / filename, column, aggregation))

    sleep_score_path = fitbit_path / "sleep_score.csv"
    if sleep_score_path.exists():
        sleep_score = pd.read_csv(sleep_score_path)
        sleep_score["date"] = _date_column(sleep_score["timestamp"])
        numeric = [column for column in ["overall_score", "deep_sleep_in_minutes"] if column in sleep_score]
        frames.append(sleep_score.groupby("date")[numeric].mean().reset_index().rename(columns={"overall_score": "sleep_score"}))

    sleep_path = fitbit_path / "sleep.json"
    if sleep_path.exists():
        sleep = pd.DataFrame(_load_json(sleep_path))
        if not sleep.empty:
            sleep["date"] = _date_column(sleep["dateOfSleep"])
            sleep["sleep_hours"] = pd.to_numeric(sleep["minutesAsleep"], errors="coerce") / 60
            frames.append(sleep.groupby("date")["sleep_hours"].sum().reset_index())

    exercise_path = fitbit_path / "exercise.json"
    if exercise_path.exists():
        exercise = pd.DataFrame(_load_json(exercise_path))
        if not exercise.empty:
            exercise["date"] = _date_column(exercise["startTime"])
            exercise["exercise_count"] = 1
            exercise["exercise_duration_min"] = pd.to_numeric(exercise["duration"], errors="coerce") / 60000
            frames.append(exercise.groupby("date")[
                ["exercise_count", "calories", "exercise_duration_min", "steps"]
            ].sum(min_count=1).reset_index().rename(columns={"calories": "exercise_calories", "steps": "exercise_steps"}))

    result = None
    for frame in frames:
        if frame.empty:
            continue
        result = frame if result is None else result.merge(frame, on="date", how="outer")
    return result if result is not None else pd.DataFrame(columns=["date"])


def load_pmsys_data(participant_path):
    """Aggregate wellness, sRPE, and injury data by day."""
    pmsys_path = participant_path / "pmsys"
    frames = []
    wellness_path = pmsys_path / "wellness.csv"
    if wellness_path.exists():
        wellness = pd.read_csv(wellness_path)
        wellness["date"] = _date_column(wellness["effective_time_frame"])
        numeric = [column for column in ["fatigue", "mood", "readiness", "sleep_duration_h", "sleep_quality", "soreness", "stress"] if column in wellness]
        frames.append(wellness.groupby("date")[numeric].mean().reset_index())

    srpe_path = pmsys_path / "srpe.csv"
    if srpe_path.exists():
        srpe = pd.read_csv(srpe_path)
        srpe["date"] = _date_column(srpe["end_date_time"])
        srpe["training_load"] = pd.to_numeric(srpe["perceived_exertion"], errors="coerce") * pd.to_numeric(srpe["duration_min"], errors="coerce")
        srpe["training_sessions"] = 1
        frames.append(srpe.groupby("date")[["training_load", "training_sessions"]].sum().reset_index())

    injury_path = pmsys_path / "injury.csv"
    if injury_path.exists():
        injury = pd.read_csv(injury_path)
        injury["date"] = _date_column(injury["effective_time_frame"])
        injury["injury_today"] = injury["injuries"].fillna("{}").ne("{}").astype(int)
        frames.append(injury.groupby("date")["injury_today"].max().reset_index())

    result = None
    for frame in frames:
        result = frame if result is None else result.merge(frame, on="date", how="outer")
    return result if result is not None else pd.DataFrame(columns=["date"])


def load_reporting_data(participant_path):
    reporting_path = participant_path / "googledocs" / "reporting.csv"
    if not reporting_path.exists():
        return pd.DataFrame(columns=["date"])
    reporting = pd.read_csv(reporting_path)
    reporting["date"] = pd.to_datetime(reporting["date"], dayfirst=True, errors="coerce").dt.normalize()
    reporting["reported_meals"] = reporting["meals"].fillna("").astype(str).str.count(",") + reporting["meals"].notna().astype(int)
    reporting["alcohol_consumed"] = reporting["alcohol_consumed"].eq("Yes").astype(int)
    numeric = [column for column in ["weight", "glasses_of_fluid", "reported_meals", "alcohol_consumed"] if column in reporting]
    return reporting.dropna(subset=["date"]).groupby("date")[numeric].mean().reset_index()


def merge_all_datasets(
    food_csv="data/processed/food_images_summary.csv",
    raw_dir="data/raw",
    output_csv="data/processed/global_daily_dataset.csv"
):
    """Build a multi-source daily table for each participant."""
    raw_path = Path(raw_dir)
    participants = sorted(path for path in raw_path.glob("p*") if path.is_dir())
    all_participants = []
    food_daily = load_food_data(food_csv)

    for participant_path in participants:
        participant_id = participant_path.name
        sources = [load_fitbit_data(participant_path), load_pmsys_data(participant_path), load_reporting_data(participant_path)]
        participant_food = food_daily[food_daily["participant_id"] == participant_id].drop(columns="participant_id", errors="ignore")
        frames = [participant_food, *sources]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            continue
        participant_daily = frames[0]
        for frame in frames[1:]:
            participant_daily = participant_daily.merge(frame, on="date", how="outer")
        participant_daily.insert(0, "participant_id", participant_id)
        all_participants.append(participant_daily)

    if not all_participants:
        raise RuntimeError("Aucune donnée exploitable n'a été trouvée dans data/raw.")
    merged = pd.concat(all_participants, ignore_index=True).sort_values(["participant_id", "date"])
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Fusion terminée : {len(merged)} lignes et {len(merged.columns)} colonnes dans {output_csv}")
    return merged


if __name__ == "__main__":
    df = merge_all_datasets()
    print(df.head())