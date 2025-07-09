import json
import os

# Paths
image_dir = "cvs/"
json_path = "../../sages_cvs_challenge_2025/segmentation_labels/annotation_polygons.json"

# Load COCO JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Get set of allowed file names from the JSON
allowed_filenames = set(img["file_name"] for img in data.get("images", []))

# List all files in the image directory
all_files = os.listdir(image_dir)

# Identify and delete .jpg files not in the allowed list
deleted_files = 0
for filename in all_files:
    if filename.lower().endswith(".jpg") and filename not in allowed_filenames:
        print("wanting to remove", filename)
        file_path = os.path.join(image_dir, filename)
        #os.remove(file_path)
        deleted_files += 1

print(f"Deleted {deleted_files} JPG files not in the COCO annotation JSON.")

