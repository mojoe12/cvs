from pylabel import importer

# Specify the path to your COCO JSON annotation file
path_to_annotations = "sages_cvs_challenge_2025/segmentation_labels/annotation_polygons.json"

# Specify the path to your images (if they are not in the same folder as the annotations)
path_to_images = "yolo_dataset/images/cvs/" # Optional

# Import the dataset
dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images)

print(dataset.df.head(5).ann_segmentation)

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes:{dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")
print(f"Path to annotations:\n{dataset.path_to_annotations}")

dataset.export.ExportToYoloV5(segmentation=True, cat_id_index=0)
