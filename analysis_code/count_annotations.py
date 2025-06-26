import json
from collections import Counter, defaultdict

# Load the JSON file
with open("sages_cvs_challenge_2025/segmentation_labels/sages_cvs_segmentation_2025.json") as f:
    coco = json.load(f)

# Create a mapping from category id to category name
category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

# Count the number of annotations per category
annotation_counts = Counter()

for ann in coco["annotations"]:
    annotation_counts[ann["category_id"]] += 1

# Sort and print class imbalance information
sorted_counts = sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True)

print("Class Imbalance Summary:\n")
for cat_id, count in sorted_counts:
    class_name = category_id_to_name[cat_id]
    print(f"{class_name}: {count} instances")

import matplotlib.pyplot as plt

labels = [category_id_to_name[cid] for cid, _ in sorted_counts]
values = [count for _, count in sorted_counts]

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=45, ha="right")
plt.title("Class Distribution in COCO Dataset")
plt.ylabel("Number of Instances")
plt.tight_layout()
plt.show()

