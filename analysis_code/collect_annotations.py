import json
import csv
from collections import defaultdict

# === Load COCO JSON ===
with open("sages_cvs_challenge_2025/segmentation_labels/sages_cvs_segmentation_2025.json") as f:
    coco = json.load(f)

# === Build category mapping ===
cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
category_names = sorted(set(cat["name"] for cat in coco["categories"]))

# === Create image metadata map ===
image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

# === Initialize counts ===
image_class_counts = defaultdict(lambda: defaultdict(int))

# === Count object annotations per image ===
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    cat_name = cat_id_to_name[cat_id]
    image_class_counts[img_id][cat_name] += 1

# === Write CSV ===
with open("image_class_counts.csv", "w", newline="") as csvfile:
    fieldnames = ["image_id", "file_name"] + category_names
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for img_id in image_id_to_file:
        row = {"image_id": img_id, "file_name": image_id_to_file[img_id]}
        for cat_name in category_names:
            row[cat_name] = image_class_counts[img_id].get(cat_name, 0)
        writer.writerow(row)

print("CSV saved as image_class_counts.csv")

