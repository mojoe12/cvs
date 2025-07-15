import os
import pandas as pd
import shutil

# Paths
csv_path = 'analysis/train_mlc_data.csv'
source_dir = 'sages_cvs_challenge_2025/frames/'
destination_dir = 'seg_yolo_labels/images/train/'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Load the CSV and get the image filenames
df = pd.read_csv(csv_path)
image_filenames = df['image'].unique()  # use .unique() to avoid duplicates

# Copy each image
for filename in image_filenames:
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(destination_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found.")

print("Image copying complete.")

