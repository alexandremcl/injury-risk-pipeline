"""Extract meal photo dates and estimate meal categories and calorie counts."""

import os
import re
import random
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

STUDY_START = datetime(2020, 2, 1)
STUDY_END = datetime(2020, 3, 31, 23, 59, 59)
CALORIE_ESTIMATE_SCALE = 2.1156

# ==========================================
# 1. Heuristic estimation (time -> calories)
# ==========================================
def estimate_meal_from_time(dt):
    """
    Estimate a meal category and scaled calorie value from the photo timestamp.
    """
    if dt is None:
        return round(500 * CALORIE_ESTIMATE_SCALE), "Repas"
    
    hour = dt.hour
    
    # Breakfast (5 a.m. - 10 a.m.)
    if 5 <= hour < 10:
        kcal = random.randint(300, 500)
        description = "Petit-déjeuner"
    # Lunch (11 a.m. - 2 p.m.)
    elif 11 <= hour < 14:
        kcal = random.randint(600, 900)
        description = "Déjeuner"
    # Dinner (6 p.m. - 10 p.m.)
    elif 18 <= hour < 22:
        kcal = random.randint(500, 850)
        description = "Dîner"
    # Snack (all other times)
    else:
        kcal = random.randint(150, 350)
        description = "Collation / Encas"
        
    return round(kcal * CALORIE_ESTIMATE_SCALE), description

# ==========================================
# 2. EXIF metadata
# ==========================================
def get_exif_date(image_path):
    try:
        with Image.open(image_path) as image:
            exif_data = image._getexif()
            if exif_data is not None:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    
    timestamp = os.path.getmtime(image_path)
    return datetime.fromtimestamp(timestamp)

def process_food_images(raw_data_dir="data/raw"):
    records = []
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Dossier '{raw_data_dir}' introuvable.")

    for participant_dir in sorted(os.listdir(raw_data_dir)):
        participant_path = os.path.join(raw_data_dir, participant_dir)
        
        if os.path.isdir(participant_path) and re.match(r"^p\d+$", participant_dir):
            food_images_path = os.path.join(participant_path, "food-images")
            
            if not os.path.exists(food_images_path):
                continue
            
            for root, _, files in os.walk(food_images_path):
                for file in sorted(files):
                    if file.startswith('~$'):
                        continue
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
                        file_path = os.path.join(root, file)
                        image_date = get_exif_date(file_path)
                        if image_date is None or not STUDY_START <= image_date <= STUDY_END:
                            continue
                        
                        # Seed the random generator with the filename for reproducible results
                        random.seed(file)
                        kcal, description = estimate_meal_from_time(image_date)
                        
                        records.append({
                            'participant_id': participant_dir,
                            'file_name': file,
                            'date': image_date.strftime('%Y-%m-%d') if image_date else None,
                            'time': image_date.strftime('%H:%M:%S') if image_date else None,
                            'image_description': description,
                            'calories_estimated': kcal,
                            'file_path': file_path
                        })
                
    return pd.DataFrame(records)

# ==========================================
# 3. Execution and export
# ==========================================
if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    OUTPUT_CSV = "data/processed/food_images_summary.csv"
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    print("1️⃣ Traitement des images et estimation heuristique...")
    df_food = process_food_images(RAW_DATA_DIR)
    
    if not df_food.empty:
        df_food.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Terminé ! {len(df_food)} images traitées et sauvegardées sous : '{OUTPUT_CSV}'")
        print("\nAperçu des premières lignes :")
        print(df_food.head(10))
    else:
        print("❌ Aucune image trouvée.")
