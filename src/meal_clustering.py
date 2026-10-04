"""Group nearby, visually similar meal photos and calculate daily totals."""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def _perceptual_hash(image_path):
    """Create a low-frequency image hash tolerant of minor visual changes."""
    size = 32
    with Image.open(image_path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L")
        pixels = np.asarray(
            grayscale.resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )

    positions = np.arange(size)
    frequencies = np.arange(8)[:, None]
    transform = np.cos(
        np.pi * (2 * positions + 1) * frequencies / (2 * size)
    )
    transform[0] *= 1 / np.sqrt(2)
    transform *= np.sqrt(2 / size)
    low_frequencies = (transform @ pixels @ transform.T)[:8, :8].flatten()[1:]
    median = np.median(low_frequencies)
    image_hash = 0
    for coefficient in low_frequencies:
        image_hash = (image_hash << 1) | int(coefficient > median)
    return image_hash


def _hash_distance(first_hash, second_hash):
    """Return the number of differing bits between two perceptual hashes."""
    return bin(first_hash ^ second_hash).count("1")


def cluster_meals_by_time(
    input_csv="data/processed/food_images_summary.csv",
    output_csv="data/processed/clean_daily_features.csv",
    time_threshold_minutes=20,
    max_image_hash_distance=20,
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Fichier '{input_csv}' introuvable.")

    df = pd.read_csv(input_csv)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    aberrant_dates = df['datetime'].dt.year.isin([2026, 2027])
    removed_photos = df.loc[aberrant_dates, ['participant_id', 'file_name', 'datetime']]
    if not removed_photos.empty:
        print("\nPhotos removed because of aberrant dates (2026–2027):")
        for photo in removed_photos.itertuples(index=False):
            print(
                f"- Participant: {photo.participant_id} | "
                f"File: {photo.file_name} | Date: {photo.datetime}"
            )
    df = df[~aberrant_dates].copy()
    df['image_hash'] = df['file_path'].map(_perceptual_hash)
    df = df.sort_values(by=['participant_id', 'datetime']).reset_index(drop=True)

    clustered_records = []

    # Group photos by participant and time window
    for participant_id, group in df.groupby('participant_id'):
        current_cluster = []
        
        for _, row in group.iterrows():
            if not current_cluster:
                current_cluster.append(row)
            else:
                first_photo = current_cluster[0]
                last_time = current_cluster[-1]['datetime']
                time_diff = (row['datetime'] - last_time).total_seconds() / 60.0
                image_distance = _hash_distance(
                    first_photo['image_hash'],
                    row['image_hash'],
                )
                
                # Compare each candidate with the first photo to prevent chaining.
                if (
                    row['date'] == first_photo['date']
                    and time_diff <= time_threshold_minutes
                    and image_distance <= max_image_hash_distance
                ):
                    current_cluster.append(row)
                else:
                    # Save the previous group and start a new cluster
                    clustered_records.append(process_cluster(current_cluster, participant_id))
                    current_cluster = [row]
                    
        if current_cluster:
            clustered_records.append(process_cluster(current_cluster, participant_id))

    df_clustered = pd.DataFrame(clustered_records)

    print("\nPhotos included in each meal cluster:")
    for _, cluster in df_clustered.iterrows():
        if cluster['photo_count'] <= 1:
            continue
        print(
            f"- {cluster['participant_id']} | {cluster['meal_id']} "
            f"({cluster['photo_count']} photo(s))"
        )
        for photo in cluster['photos']:
            print(f"  - {photo}")

    # Aggregate clustered records by day
    daily_clean = df_clustered.groupby(['participant_id', 'date']).agg(
        num_distinct_meals=('meal_id', 'count'),
        total_photos_taken=('photo_count', 'sum'),
        total_calories_consumed=('cluster_calories', 'sum'),
        meal_types=('cluster_type', lambda x: ", ".join(list(dict.fromkeys(x))))
    ).reset_index()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    daily_clean.to_csv(output_csv, index=False)
    print(f"✅ Clustering terminé ! Dataset corrigé sauvegardé sous : '{output_csv}'")
    
    return daily_clean


def process_cluster(cluster_rows, participant_id):
    """Calculate a single calorie value for a cluster of nearby photos."""
    df_cluster = pd.DataFrame(cluster_rows)
    
    # Use the cluster's average calories to avoid double-counting
    cluster_kcal = int(df_cluster['calories_estimated'].mean())
    cluster_type = df_cluster['image_description'].iloc[0]
    
    return {
        'participant_id': participant_id,
        'meal_id': f"{participant_id}_{df_cluster['datetime'].iloc[0].strftime('%Y%m%d_%H%M')}",
        'date': df_cluster['date'].iloc[0],
        'time': df_cluster['time'].iloc[0],
        'photo_count': len(df_cluster),
        'photos': [
            f"{row['file_name']} ({row['datetime']:%Y-%m-%d %H:%M:%S})"
            for _, row in df_cluster.iterrows()
        ],
        'cluster_calories': cluster_kcal,
        'cluster_type': cluster_type
    }


if __name__ == "__main__":
    df_clean = cluster_meals_by_time()
    print("\nAperçu après ajustement des doublons de photos :")
    print(df_clean.head())