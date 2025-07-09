import json

# Paths
input_path = "sages_cvs_challenge_2025/segmentation_labels/sages_cvs_segmentation_2025.json"
output_path = "sages_cvs_challenge_2025/segmentation_labels/fixed_coco_annotations_seg.json"

# Load the original JSON
with open(input_path, "r") as f:
    data = json.load(f)

# Step 1: Build a set of image_ids with at least one valid segmentation annotation
valid_image_ids = set()
for ann in data["annotations"]:
    seg = ann.get("segmentation")
    if seg:  # non-empty segmentation
        valid_image_ids.add(ann["image_id"])

# Step 2: Filter images to only include those with valid segmentations
all_images = data.get("images", [])
filtered_images = [img for img in all_images if img["id"] in valid_image_ids]

# Step 3: (Optional but recommended) Filter annotations to only include those for remaining images
filtered_annotations = [ann for ann in data["annotations"] if ann["image_id"] in valid_image_ids]

# Step 4: Update the dataset and save
data["images"] = filtered_images
data["annotations"] = filtered_annotations

print(f"Kept {len(filtered_images)} images with segmentation annotations.")
print(f"Kept {len(filtered_annotations)} annotations.")

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Filtered dataset saved to {output_path}")

