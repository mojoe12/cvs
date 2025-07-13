import os
import shutil
import pandas as pd

# Paths
csv_paths = ['analysis/train_mlc_data.csv', 'analysis/val_mlc_data.csv']
frames_dirs = ['sages_cvs_challenge_2025/frames', 'sages_cvs_challenge_2025/frames']
output_base_dirs = ['yolo_cls_dataset/train/', 'yolo_cls_dataset/val/']

for i in range(len(csv_paths)):
    csv_path = csv_paths[i]
    frames_dir = frames_dirs[i]
    output_base_dir = output_base_dirs[i]
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Round c1, c2, c3 to 0 or 1
    for col in ['c1', 'c2', 'c3']:
        df[col] = df[col].round().astype(int)

    # Create a combination string for each row, e.g., "010"
    df['class'] = df[['c1', 'c2', 'c3']].astype(str).agg(''.join, axis=1)

    # Create output directories and move files
    for _, row in df.iterrows():
        img_name = row['image']
        combo_class = row['class']
        src_path = os.path.join(frames_dir, img_name)
        dest_dir = os.path.join(output_base_dir, combo_class)
        dest_path = os.path.join(dest_dir, img_name)

        # Make sure the target directory exists
        os.makedirs(dest_dir, exist_ok=True)

        # Move the image
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: {src_path} does not exist.")

print("Done organizing images into class directories.")

