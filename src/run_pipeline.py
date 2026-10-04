"""Run nutrition processing, data merging, feature engineering, and model training in order."""

from importlib import import_module
import shutil
from pathlib import Path

process_features = import_module("3_features").process_features
merge_all_datasets = import_module("2_merge_dataset").merge_all_datasets
process_food_images = import_module("1_nutrition").process_food_images
train_models = import_module("4_train_model").train_models


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    food_summary = PROCESSED_DIR / "food_images_summary.csv"
    food_data = process_food_images(str(RAW_DIR))
    food_data.to_csv(food_summary, index=False)
    print(f"Nutrition : {len(food_data)} photos traitees")

    merge_all_datasets(
        food_csv=str(food_summary),
        raw_dir=str(RAW_DIR),
        output_csv=str(PROCESSED_DIR / "global_daily_dataset.csv"),
    )
    process_features(
        input_csv=str(PROCESSED_DIR / "global_daily_dataset.csv"),
        output_csv=str(PROCESSED_DIR / "final_features_dataset.csv"),
    )

    final_features = PROCESSED_DIR / "final_features_dataset.csv"
    shutil.copyfile(final_features, PROCESSED_DIR / "test_dataset.csv")
    train_models(input_csv=str(final_features))
    print("Pipeline terminee. Les fichiers sont dans data/processed/")


if __name__ == "__main__":
    main()
