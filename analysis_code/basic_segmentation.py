import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pycocotools.coco import COCO
from torchvision.models.densenet import densenet121
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

# ---- Custom COCO Segmentation Dataset ----
class CocoSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, root_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())
        self.resize = transforms.Resize((720, 1280))  # Height x Width

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")
        
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann in anns:
            cat_id = ann["category_id"]
            mask_seg = self.coco.annToMask(ann)
            mask[mask_seg == 1] = cat_id

        if self.transforms:
            image = self.transforms(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

def main():
    # ---- CONFIG ----
    coco_annotation_path = 'segmentation_labels/sages_cvs_segmentation_2025.json'
    image_root_dir = 'segmentation_visualization/'  # Set this to the folder with your images
    num_classes = 6  # Background + foreground (or more if multi-class)
    batch_size = 4
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Transforms ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ---- Load and Split Dataset ----
    full_dataset = CocoSegmentationDataset(coco_annotation_path, image_root_dir, transforms=transform)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---- Use DeepLabV3 + DenseNet Backbone ----
    # Torchvision doesn't provide DeepLab with DenseNet, so we hack in a custom model

    def get_custom_deeplab_densenet(num_classes):
        base_model = densenet121(pretrained=True)
        backbone = base_model.features
        model = deeplabv3_resnet50(pretrained=False)
        model.backbone = backbone
        model.classifier = DeepLabHead(1024, num_classes)  # DenseNet121 last feature map has 1024 channels
        return model

    model = get_custom_deeplab_densenet(num_classes).to(device)

    # ---- Optimizer and Loss ----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional; only needed if using pyinstaller
    main()
