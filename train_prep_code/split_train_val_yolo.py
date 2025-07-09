import os
import shutil
import pandas as pd

# Paths
image_dir = "yolo_dataset/labels/cvs"
csv_path = "analysis/train_mlc_data.csv"
train_dir = "yolo_dataset/labels/cvs_train"
val_dir = "yolo_dataset/labels/cvs_val"

# Create target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load CSV and get the list of training filenames
df = pd.read_csv(csv_path)

# Auto-detect the column containing filenames
if "file_name" in df.columns:
    train_filenames = set(df["file_name"].astype(str))
else:
    # Use the first column if no header
    train_filenames = set(df.iloc[:, 0].astype(str))

# Iterate over all .jpg files in the image directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".txt"):
        src_path = os.path.join(image_dir, filename)
        new_filename = filename.replace(".txt", ".jpg")
        if new_filename in train_filenames:
            dst_path = os.path.join(train_dir, filename)
        else:
            dst_path = os.path.join(val_dir, filename)
        shutil.move(src_path, dst_path)

print(f"Split complete. Images moved to '{train_dir}' and '{val_dir}'.")

