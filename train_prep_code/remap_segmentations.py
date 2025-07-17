import os
import cv2
from glob import glob
from pathlib import Path
from ultralytics import YOLO  # Make sure you have YOLOv8 installed
import numpy as np
import pandas as pd

# Load the YOLOv8 segmentation model
model = YOLO("train_code/yolo11s_cvs.pt")  # Replace with your trained YOLOv8 segmentation model

# Directory paths
assignments = []
assignments.append(("sages_cvs_challenge_2025/frames", "analysis/train_mlc_data_interpolated.csv", "seg_yolo_labels/labels/train"))
assignments.append(("sages_cvs_challenge_2025/frames", "analysis/val_mlc_data.csv", "seg_yolo_labels/labels/val"))
assignments.append(("endoscapes/train", "analysis/endo_train_mlc_data.csv", "seg_yolo_labels/labels/endo_train"))
assignments.append(("endoscapes/val", "analysis/endo_val_mlc_data.csv", "seg_yolo_labels/labels/endo_val"))
assignments.append(("endoscapes/test", "analysis/endo_test_mlc_data.csv", "seg_yolo_labels/labels/endo_test"))

# Define class name to index mapping
original_classes = {
    0: "cystic_plate",
    1: "calot_triangle",
    2: "cystic_artery",
    3: "cystic_duct",
    4: "gallbladder",
    5: "tool"
}

# New class mapping (11 classes)
new_class_map = {
    ("cystic_plate", 1): 0,
    ("cystic_plate", 0): 1,
    ("calot_triangle", 1): 2,
    ("calot_triangle", 0): 3,
    ("cystic_artery", 1): 4,
    ("cystic_artery", 0): 5,
    ("cystic_duct", 1): 6,
    ("cystic_duct", 0): 7,
    ("gallbladder", 1): 8,
    ("gallbladder", 0): 9,
    ("tool", None): 10
}

# Helper to remap classes
def remap_class(class_name, c1, c2, c3):
    if class_name == "cystic_plate":
        return new_class_map[(class_name, round(c3))]
    elif class_name == "calot_triangle":
        return new_class_map[(class_name, round(c2))]
    elif class_name == "cystic_artery":
        return new_class_map[(class_name, round(c1))]
    elif class_name == "cystic_duct":
        return new_class_map[(class_name, round(c1))]
    elif class_name == "gallbladder":
        return new_class_map[(class_name, round(c3))]
    elif class_name == "tool":
        return new_class_map[(class_name, None)]
    else:
        return None  # Skip anything unexpected

# Process each image
for img_dir, csv_file, output_dir in assignments:
    os.makedirs(output_dir, exist_ok=True)

    # Metadata: maps image name to [c1, c2, c3]
    my_df = pd.read_csv(csv_file)
    image_info = {
        row['image']: [row['c1'], row['c2'], row['c3']]
        for _, row in my_df.iterrows()
    }
    for img_path in glob(os.path.join(img_dir, "*.jpg")):
        img_name = os.path.basename(img_path)
        if img_name not in image_info:
            continue

        c1, c2, c3 = image_info[img_name]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        results = model(img_path, verbose=False)[0]

        txt_path = os.path.join(output_dir, img_name.replace(".jpg", ".txt"))
        with open(txt_path, "w") as f:
            if results.masks is None:
                continue

            for cls_id, box, mask in zip(results.boxes.cls.cpu().numpy().astype(int),
                                         results.boxes.xywhn.cpu().numpy(),
                                         results.masks.xy):
                class_name = original_classes[cls_id]
                new_class_id = remap_class(class_name, c1, c2, c3)
                if new_class_id is None:
                    continue

                x_center, y_center, width, height = box  # normalized

                # Normalize polygon coordinates
                poly_coords = []
                for point in mask:  # each point is [x, y] in absolute coords
                    norm_x = point[0] / w
                    norm_y = point[1] / h
                    poly_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(poly_coords) + "\n"
                f.write(line)

