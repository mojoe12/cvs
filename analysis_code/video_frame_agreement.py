import os
import pandas as pd
import numpy as np

def process_directory(directory):
    frame_path = os.path.join(directory, 'frame.csv')
    video_path = os.path.join(directory, 'video.csv')

    if not os.path.isfile(frame_path) or not os.path.isfile(video_path):
        return None

    # Read the CSV files
    frame_df = pd.read_csv(frame_path).drop('frame_id', axis=1)
    video_df = pd.read_csv(video_path).drop(['confidence_rater1', 'confidence_rater2', 'confidence_rater3'], axis=1)

    # Perform OR operation across all rows of frame.csv
    combined_frame = frame_df.astype(bool).any(axis=0).astype(int)

    # Check equality
    video_col = video_df.iloc[0, :].astype(int)
    comparison = combined_frame.equals(video_col)

    return {
        'directory': directory,
        'match': comparison,
        'combined_frame': combined_frame.tolist(),
        'video_column': video_col.tolist()
    }

def main():
    results = []
    base_dir = os.getcwd()
    match_count = 0
    total_count = 0

    for root, dirs, files in os.walk(base_dir):
        if 'frame.csv' in files and 'video.csv' in files:
            result = process_directory(root)
            if result:
                total_count += 1
                if result['match']:
                    match_count += 1
                results.append(result)

    for res in results:
        print(f"Directory: {res['directory']}")
        print(f"Match: {res['match']}")
        print(f"Row-wise AND (frame.csv): {res['combined_frame']}")
        print(f"Video column (video.csv): {res['video_column']}")
        print("-" * 40)

    if total_count > 0:
        match_percent = (match_count / total_count) * 100
        print(f"✅ Matched: {match_count}/{total_count} directories ({match_percent:.2f}%)")
    else:
        print("⚠️ No valid directories found.")

if __name__ == "__main__":
    main()

