"""Analyze pipeline output and generate descriptive statistics and exploratory plots."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def run_eda(input_csv="data/processed/final_features_dataset.csv"):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Fichier '{input_csv}' introuvable. Exécute d'abord l'étape 3.")

    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'])

    print("=== 📊 STATISTIQUES DESCRIPTIVES ===")
    print(df[['num_meals', 'total_calories_consumed', 'total_calories_consumed_7d_avg']].describe())

    os.makedirs("reports/figures", exist_ok=True)

    # -------------------------------------------------------------
    # Charts 1 and 2: Raw calorie intake and rolling average over time
    # -------------------------------------------------------------
    excluded_dates = (
        df['date'].dt.year.isin([2026, 2027])
        | df['date'].between('2019-11-01', '2020-02-29')
    )
    timeline_df = df[~excluded_dates]

    plt.figure(figsize=(12, 5))
    for participant_id, group in timeline_df.groupby('participant_id'):
        plt.plot(
            group['date'],
            group['total_calories_consumed'],
            alpha=0.6,
            label=participant_id,
        )

    plt.title("Daily Calorie Intake (Raw)")
    plt.xlabel("Date")
    plt.ylabel("Calories (kcal)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/daily_calorie_intake_raw.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for participant_id, group in timeline_df.groupby('participant_id'):
        plt.plot(
            group['date'],
            group['total_calories_consumed_7d_avg'],
            linewidth=2,
            label=participant_id,
        )

    plt.title("Daily Calorie Intake (7-Day Average)")
    plt.xlabel("Date")
    plt.ylabel("Calories (kcal)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/daily_calorie_intake_7_day_average.png")
    plt.close()

    # -------------------------------------------------------------
    # Chart 3: Relationship between meal photos and calorie intake
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(df['num_meals'], df['total_calories_consumed'], alpha=0.7, color='teal')
    plt.title("Relationship Between Daily Meal Photos and Estimated Calorie Intake")
    plt.xlabel("Nombre de photos prises par jour")
    plt.ylabel("Total calories (kcal)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/meal_photos_vs_calorie_intake.png")
    plt.close()

    print("\n✅ Analyse terminée ! Les graphiques ont été enregistrés dans 'reports/figures/'.")


if __name__ == "__main__":
    run_eda()