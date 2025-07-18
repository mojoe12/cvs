import os
from pathlib import Path
from glob import glob
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Paths
val_img_dir = "seg_yolo_labels/images/endo_val"
model_path = "runs/detect/train6/weights/best.pt"  # Replace with your model path
csv_file = "analysis/endo_val_mlc_data.csv" #CHANGE TO TRAIN

# Your metadata dictionary with ground truth
# Metadata: maps image name to [c1, c2, c3]
my_df = pd.read_csv(csv_file)
image_info = {
    row['image']: [round(row['c1']), round(row['c2']), round(row['c3'])]
    for _, row in my_df.iterrows()
}

# Class ID to name mapping
class_ids = {
    "cystic_plate_c3_one": 0,
    "cystic_plate_c3_zero": 1,
    "calot_triangle_c2_one": 2,
    "calot_triangle_c2_zero": 3,
    "cystic_artery_c1_one": 4,
    "cystic_artery_c1_zero": 5,
    "cystic_duct_c1_one": 6,
    "cystic_duct_c1_zero": 7,
    "gallbladder_c3_one": 8,
    "gallbladder_c3_zero": 9,
    "tool": 10,
}
id_to_class = {v: k for k, v in class_ids.items()}

# Load model
model = YOLO(model_path)

# Evaluation storage
y_true = {"c1": [], "c2": [], "c3": []}
y_pred = {"c1": [], "c2": [], "c3": []}

# Inference
for img_path in glob(os.path.join(val_img_dir, "*.jpg")):
    img_name = os.path.basename(img_path)
    if img_name not in image_info:
        continue

    gt_c1, gt_c2, gt_c3 = image_info[img_name]
    result = model(img_path, verbose=False)[0]

    detected_ids = set(result.boxes.cls.cpu().numpy().astype(int)) if result.boxes else set()
    detected_classes = {id_to_class[i] for i in detected_ids}

    pred_c3 = 1 if "cystic_plate_c3_one" in detected_classes or "gallbladder_c3_one" in detected_classes else 0
    pred_c2 = 1 if "calot_triangle_c2_one" in detected_classes else 0
    pred_c1 = 1 if ("cystic_artery_c1_one" in detected_classes or "cystic_duct_c1_one" in detected_classes) else 0
    #print("c1", y_true["c1"], "pred_c1", pred_c1)

    y_true["c1"].append(gt_c1)
    y_true["c2"].append(gt_c2)
    y_true["c3"].append(gt_c3)

    y_pred["c1"].append(pred_c1)
    y_pred["c2"].append(pred_c2)
    y_pred["c3"].append(pred_c3)

# Compute metrics
for key in ["c1", "c2", "c3"]:
    f1 = f1_score(y_true[key], y_pred[key])
    try:
        ap = average_precision_score(y_true[key], y_pred[key])
    except ValueError:
        ap = float("nan")
    print(f"{key} - F1: {f1:.3f}, mAP: {ap:.3f}")

