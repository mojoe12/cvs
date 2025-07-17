import os
import pandas as pd
import shutil

# Paths

assignments = []
assignments.append(("sages_cvs_challenge_2025/frames", "analysis/train_mlc_data_interpolated.csv", "seg_yolo_labels/images/train"))
assignments.append(("sages_cvs_challenge_2025/frames", "analysis/val_mlc_data.csv", "seg_yolo_labels/images/val"))
assignments.append(("endoscapes/train", "analysis/endo_train_mlc_data.csv", "seg_yolo_labels/images/endo_train"))
assignments.append(("endoscapes/val", "analysis/endo_val_mlc_data.csv", "seg_yolo_labels/images/endo_val"))
assignments.append(("endoscapes/test", "analysis/endo_test_mlc_data.csv", "seg_yolo_labels/images/endo_test"))

for source_dir, csv_file, output_dir in assignments:
    # Ensure the destination directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV and get the image filenames
    df = pd.read_csv(csv_file)
    image_filenames = df['image'].unique()  # use .unique() to avoid duplicates

    # Copy each image
    for filename in image_filenames:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {src_path} not found.")

print("Image copying complete.")

