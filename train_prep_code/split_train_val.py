import os
import shutil
import random
import glob
import json
import pandas as pd

# Parameters
val_ratio = 0.2  # 20%
base_dir = 'sages_cvs_challenge_2025'
label_base_dir = 'sages_cvs_challenge_2025/labels/'

# Get list of directories (excluding train and val if they already exist)
all_dirs = [
    d for d in os.listdir(label_base_dir)
    if os.path.isdir(os.path.join(label_base_dir, d)) and d not in {'train', 'val'}
]

# Shuffle and split
random.shuffle(all_dirs)
val_count = int(len(all_dirs) * val_ratio)
val_dirs = all_dirs[:val_count]
train_dirs = all_dirs[val_count:]

# Paths
coco_json_path = os.path.join(base_dir, 'segmentation_labels', 'sages_cvs_segmentation_2025.json')  # change if needed
train_output = os.path.join('analysis', 'instances_train.json')
val_output = os.path.join('analysis', 'instances_val.json')

# Load JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

train_names = set(train_dirs)
val_names = set(val_dirs)

# Prepare data structures
def filter_coco(coco_data, prefixes):
    # Filter images
    images = [
        img for img in coco_data['images']
        if any(img['file_name'].startswith(prefix) for prefix in prefixes)
    ]
    image_ids = set(img['id'] for img in images)

    # Filter annotations based on image_id
    annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in image_ids
    ]

    return {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': images,
        'annotations': annotations,
    }

# Split and save
train_coco = filter_coco(coco_data, train_names)
val_coco = filter_coco(coco_data, val_names)

with open(train_output, 'w') as f:
    json.dump(train_coco, f)

with open(val_output, 'w') as f:
    json.dump(val_coco, f)

print(f"Saved split COCO annotations: {train_output}, {val_output}")

base_dir = 'analysis/'
csv_path = os.path.join(base_dir, 'combined_frame.csv')
train_csv_path = os.path.join(base_dir, 'train_frames.csv')
val_csv_path = os.path.join(base_dir, 'val_frames.csv')

# Load CSV
df = pd.read_csv(csv_path)

# Split based on 'dir_name' column
train_df = df[df['dir_name'].isin(train_names)]
val_df = df[df['dir_name'].isin(val_names)]

# Save the two CSVs
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

print(f"Split complete. Saved to:\n- {train_csv_path}\n- {val_csv_path}")

