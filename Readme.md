# Athlete Performance and Injury Risk Analysis

This project combines daily nutrition, activity, training, wellness, and self-report data for each participant. It builds a daily dataset, engineers features, trains exploratory models for injury risk and a performance index, and generates exploratory plots.

## Pipeline

The main pipeline runs the following stages in order:

1. **`1_nutrition.py` — meal-photo processing**
   - Reads photo timestamps from EXIF metadata, falling back to the file modification time.
   - Keeps photos dated from February 1 through March 31, 2020.
   - Assigns a meal period based on the time of day and estimates calories using a heuristic range, scaled to target about 3,500 kcal per participant-day on the current study dataset.
   - It does not recognize food from image content; calorie values are rough estimates, not nutritional measurements or individual recommendations.
2. **`2_merge_dataset.py` — daily data merging**
   - Aggregates nutrition photos, Fitbit data, PMSYS wellness/training/injury data, and Google Docs self-reports by participant and date.
   - Writes `data/processed/global_daily_dataset.csv`.
3. **`3_features.py` — cleaning and feature engineering**
   - Adds missing-value indicators, imputes numeric values, and caps outliers with the IQR rule.
   - Creates calendar variables, rolling summaries, net calorie balance, training-load measures, and the model targets.
   - Writes `data/processed/final_features_dataset.csv`.
4. **`4_train_model.py` — model training and evaluation**
   - Uses a chronological split per participant: the earlier records train the models and the latest records test them.
   - Trains a random-forest classifier for `injury_next_7d` and a random-forest regressor for `performance_index_next_day`.
   - Prints evaluation metrics and writes test predictions to `data/processed/model_test_predictions.csv`.

`run_pipeline.py` runs these four stages and also copies `final_features_dataset.csv` to `data/processed/test_dataset.csv`. Run `5_eda.py` separately to print descriptive statistics and create plots in `reports/figures/`.

## Targets and interpretation

- **`injury_next_7d`** is 1 when an injury is recorded after the current date and within the following seven calendar days; otherwise it is 0. The classifier learns patterns from the available dataset to estimate this outcome. Its output is not a medical assessment.
- **`performance_index`** is a hand-designed 0–100 composite based on available readiness, sleep quality and duration, mood, fatigue, and soreness values. Higher fatigue and soreness reduce the score; a sleep duration near eight hours increases it.
- **`performance_index_next_day`** is the performance index on the participant's next recorded row. If a calendar date is missing, that row may not represent the next calendar day. The regressor learns to predict this target.

These are exploratory targets and should not be interpreted as validated medical or athletic-performance measures. The 3,500 kcal/day calibration is a rough dataset-level adjustment based on the requested general athlete range; it is not inferred from participant physiology, and model metrics depend on the small study dataset and cannot be assumed to generalize.

## Project structure

```text
.
├── data/
│   ├── raw/                         # Participant source data and meal photos
│   └── processed/                   # Generated CSV datasets and predictions
├── notebooks/
│   └── exploratory_analysis.ipynb   # Notebook for analysis and presentation
├── reports/
│   └── figures/                     # EDA plots
├── src/
│   ├── 1_nutrition.py               # Photo timestamps and heuristic meal estimates
│   ├── 2_merge_dataset.py           # Daily aggregation and source merging
│   ├── 3_features.py                # Cleaning, engineered features, and targets
│   ├── 4_train_model.py             # Model training, evaluation, and test predictions
│   ├── 5_eda.py                     # Descriptive statistics and plots
│   ├── meal_clustering.py           # Standalone time- and image-based photo clustering
│   └── run_pipeline.py              # Runs the four main pipeline stages
├── requirements.txt                 # Python dependencies
└── Readme.md
```

`meal_clustering.py` is a separate photo-clustering experiment. It writes `data/processed/clean_daily_features.csv`; the main pipeline does not call it or consume that output.

## Installation and usage

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_pipeline.py
```

To run exploratory analysis after the pipeline:

```bash
python src/5_eda.py
```

The stages can also be run individually, in this order:

```bash
python src/1_nutrition.py
python src/2_merge_dataset.py
python src/3_features.py
python src/4_train_model.py
python src/5_eda.py
```

For a different environment or data folder, adjust the paths passed to each script's processing function or provide the expected data under `data/raw/`.

## Dependencies

`requirements.txt` contains the packages used by the scripts and notebook. No pretrained computer-vision model is used by the current nutrition script, so PyTorch, Transformers, and a model download are not required.
