import json
import pycocotools.mask as maskUtils
import numpy as np
from PIL import Image
from skimage import measure

# Load your JSON
with open("sages_cvs_challenge_2025/segmentation_labels/fixed_coco_annotations_seg.json", "r") as f:
    data = json.load(f)

def rle_to_polygon(rle, height, width):
    # Decode the RLE to a binary mask
    binary_mask = maskUtils.decode(rle)

    # Convert binary mask to contours (polygons)

    contours = measure.find_contours(binary_mask, 0.5)
    segmentation = []
    for contour in contours:
        contour = np.flip(contour, axis=1)  # (row, col) to (x, y)
        segmentation.append(contour.ravel().tolist())
    return segmentation

# Convert all RLE segmentations to polygons
for ann in data["annotations"]:
    seg = ann.get("segmentation")
    if isinstance(seg, dict) and "counts" in seg:
        image_id = ann["image_id"]
        image_info = next((img for img in data["images"] if img["id"] == image_id), None)
        if image_info:
            height = image_info["height"]
            width = image_info["width"]
            polygon = rle_to_polygon(seg, height, width)
            if polygon:
                ann["segmentation"] = polygon
            else:
                fail
                ann["segmentation"] = []  # fallback if conversion failed

# Save to new file
with open("sages_cvs_challenge_2025/segmentation_labels/annotation_polygons.json", "w") as f:
    json.dump(data, f, indent=2)

print("Converted RLE segmentations to polygon format and saved.")

