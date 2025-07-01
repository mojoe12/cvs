import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
from pycocotools.coco import COCO
from torchvision.models.densenet import densenet121
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import skimage.transform as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import random

class DenseNetBackbone(torch.nn.Module):
    def __init__(self, densenet_features):
        super().__init__()
        self.features = densenet_features

    def forward(self, x):
        x = self.features(x)
        return {"out": x}   # Wrap output in a dict with key 'out'

class CocoSegmentationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, height, width, augment):
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.image_ids = list(self.coco.imgs.keys())
        self.h = height
        self.w = width
        self.transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
        ])
        self.augment = augment

        # Only photometric (image-only)
        self.image_only_transforms = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Initialize blank mask
        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        for ann in anns:
            cat_id = ann["category_id"]
            mask_seg = self.coco.annToMask(ann)
            mask_resized = st.resize(
                mask_seg,
                (self.h, self.w),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            )
            mask[mask_resized == 1] = cat_id

        mask = torch.from_numpy(mask).long()
        # Apply joint image-mask augmentations
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = mask.unsqueeze(0)          # [H, W] → [1, H, W]
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            mask = mask.squeeze(0)            # [1, H, W] → [H, W]

            # Apply image-only transforms
            image = self.image_only_transforms(image)
        # Convert mask to torch tensor
        return image, mask

def custom_collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

class MultiLabelClassifier(nn.Module):
    def __init__(self, base_model, num_classes): # this will be 3
        super().__init__()
        self.backbone = base_model.backbone  # keep encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),  # adjust input size as needed
            nn.Sigmoid()  # sigmoid for multilabel outputs
        )

    def forward(self, x):
        features = self.backbone(x)['out']  # get feature map
        pooled = self.pool(features)
        out = self.classifier(pooled)
        return out

def get_custom_deeplab_densenet(num_classes, base_model=None, saved_classifier=None):
    if base_model is None:
        base_model = densenet121(pretrained=True)
    backbone = DenseNetBackbone(base_model.features)  # wrap backbone

    model = deeplabv3_resnet50(pretrained=False)
    model.backbone = backbone
    if saved_classifier is None:
        saved_classifier = DeepLabHead(1024, num_classes)  # DenseNet121 last feature channels = 1024
    model.classifier = saved_classifier
    return model

def grad_norm(params):
    total_norm = 0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

def train_segmentation(num_epochs, log_dir, saved_backbone, saved_classifier):
    # ---- CONFIG ----
    coco_annotation_path = 'segmentation_labels/sages_cvs_segmentation_2025.json'
    image_root_dir = 'segmentation_visualization/'  # Set this to the folder with your images
    num_classes = 7 # Background + foreground (or more if multi-class)
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = 224, 224

    # Full dataset object (no split yet)
    full_dataset = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=False)

    # Create a consistent train/val split
    total_indices = list(range(len(full_dataset)))
    random.shuffle(total_indices)
    val_size = int(0.1 * len(full_dataset))
    val_indices = total_indices[:val_size]
    train_indices = total_indices[val_size:]

    # Create two dataset instances with different augment flags
    train_dataset_full = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=False)
    val_dataset_full = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=False)

    # Subset them using the precomputed indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Set to 0 while debugging
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    # ---- Use DeepLabV3 + DenseNet Backbone ----
    # Torchvision doesn't provide DeepLab with DenseNet, so we hack in a custom model

    model = get_custom_deeplab_densenet(num_classes, saved_backbone, saved_classifier).to(device)

    # ---- Optimizer and Loss ----
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
    ])
    criterion = torch.nn.CrossEntropyLoss()

    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())

    freeze_backbone(model)

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        backbone_grad_norm = 0.0
        classifier_grad_norm = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            backbone_grad_norm += grad_norm(backbone_params)
            classifier_grad_norm += grad_norm(classifier_params)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}")
        backbone_grad_norm = backbone_grad_norm / len(train_loader)
        classifier_grad_norm = classifier_grad_norm / len(train_loader)
        print(f"Backbone grad norm: {backbone_grad_norm:.4f}, Classifier grad norm: {classifier_grad_norm:.4f}")

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

        if epoch == 4: #the fifth time
            unfreeze_backbone(model)
            print("Unfroze the backbone")

    os.makedirs(log_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            # Save visualizations of incorrect predictions
            for i in range(images.size(0)):
                pred_mask = preds[i].cpu().numpy()
                true_mask = masks[i].cpu().numpy()
                incorrect = (pred_mask != true_mask)

                if np.any(incorrect):
                    image_np = images[i].cpu().permute(1, 2, 0).numpy()
                    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
                    axs[0].imshow(image_np)
                    axs[0].set_title("Image")
                    axs[1].imshow(true_mask, cmap='tab20')
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(pred_mask, cmap='tab20')
                    axs[2].set_title("Prediction")
                    axs[3].imshow(incorrect, cmap='gray')
                    axs[3].set_title("Incorrect")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"{log_dir}/val_{batch_idx}_{i}.png")
                    plt.close()

            # Collect labels for confusion matrix
            all_preds.append(preds.flatten().cpu().numpy())
            all_labels.append(masks.flatten().cpu().numpy())

    # Compute confusion matrix
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp_labels = ["Background", "Cystic plate", "Calot triangle", "Cystic artery", "Cystic duct", "Gallbladder", "Tool"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/confusion_matrix.png")
    plt.close()
    return model.backbone, model.classifier

def main():
    saved_backbone, saved_classifier = train_segmentation(50, "segmentation_results", None, None)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional; only needed if using pyinstaller
    main()
