from ultralytics import YOLO

# Load a pre-trained model (choose one: yolov8n.pt, yolov8m.pt, yolov11.pt, etc.)
model = YOLO('yolo11n.pt')

# Evaluate using COCO-format annotations (without training)

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="sages_test.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Show results
print(results)
