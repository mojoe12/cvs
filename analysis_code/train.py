import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, average_precision_score
from pycocotools.coco import COCO
import skimage.transform as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import random

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

def getSegmentationLoaders(batch_size, h, w):
    # ---- CONFIG ----
    coco_annotation_path = 'sages_cvs_challenge_2025/segmentation_labels/sages_cvs_segmentation_2025.json'
    image_root_dir = 'sages_cvs_challenge_2025/segmentation_visualization/'  # Set this to the folder with your images

    # Full dataset object (no split yet)
    full_dataset = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=False)

    # Create a consistent train/val split
    total_indices = list(range(len(full_dataset)))
    random.shuffle(total_indices)
    val_size = int(0.2 * len(full_dataset))
    val_indices = total_indices[:val_size]
    train_indices = total_indices[val_size:]

    # Create two dataset instances with different augment flags
    train_dataset_full = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=True)
    val_dataset_full = CocoSegmentationDataset(coco_annotation_path, image_root_dir, h, w, augment=False)

    # Subset them using the precomputed indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Set to 0 while debugging
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, labels_and_confidences, height, width, augment):
        # labels_and_confidences = {filename: ([0, 1, 1], [1.0, 0.6, 0.9])}
        self.image_dir = image_dir
        self.data = labels_and_confidences
        self.image_filenames = list(labels_and_confidences.keys())
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
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            # Apply image-only transforms
            image = self.image_only_transforms(image)
        label, confidence = self.data[filename]
        label = torch.tensor(label, dtype=torch.float32)
        confidence = torch.tensor(confidence, dtype=torch.float32)
        return image, label, confidence

def getMLCLoaders(batch_size, h, w):
    # Transforms
    lc_df = pd.read_csv('analysis/mlc_data.csv')

    labels_confidences_dict = {
        row['image']: (
            [row['c1'], row['c2'], row['c3']],
            [row['weight_c1'], row['weight_c2'], row['weight_c3']]
        )
        for _, row in lc_df.iterrows()
    }

    # Datasets
    image_path = 'sages_cvs_challenge_2025/frames'
    full_dataset = MultiLabelImageDataset(image_path, labels_confidences_dict, h, w, augment=False)
    # Create a consistent train/val split
    total_indices = list(range(len(full_dataset)))
    random.shuffle(total_indices)
    val_size = int(0.2 * len(full_dataset))
    val_indices = total_indices[:val_size]
    train_indices = total_indices[val_size:]

    # Create two dataset instances with different augment flags
    train_dataset_full = MultiLabelImageDataset(image_path, labels_confidences_dict, h, w, augment=True)
    val_dataset_full = MultiLabelImageDataset(image_path, labels_confidences_dict, h, w, augment=False)

    # Subset them using the precomputed indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Set to 0 while debugging
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader

class EfficientNetMultiHead(nn.Module):
    def __init__(self, num_labels, num_classes):
        super().__init__()
        base = models.efficientnet_b3(pretrained=True)

        # Remove the last 3 MBConv blocks
        cutoff_layer, num_backbone_features = 3, 136
        self.features = nn.Sequential(*list(base.features.children())[:-cutoff_layer])

        # Use adaptive pooling to collapse spatial dims
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)  # Multi-label logits
        )

        # --- Segmentation head ---
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(num_backbone_features, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=0),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x, task='classification'):
        features = self.features(x)

        if task == 'classification':
            pooled = self.pool(features)
            return self.classifier(pooled)

        elif task == 'segmentation':
            return self.segmentation_head(features)

        elif task == 'both':
            pooled = self.pool(features)
            cls_out = self.classifier(pooled)
            seg_out = self.segmentation_head(features)
            return cls_out, seg_out

        else:
            raise ValueError(f"Unknown task: {task}")

def trainSegmentationStep(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images, 'segmentation')
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validateSegmentationStep(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, 'segmentation')
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# ==== Training ====
def trainMLCStep(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels, confidences in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        confidences = confidences.to(device)
        optimizer.zero_grad()
        outputs = model(images, 'classification')
        loss_raw = criterion(outputs, labels)
        loss = (loss_raw * confidences).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validateMLCStep(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, confidences in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            confidences = confidences.to(device)

            outputs = model(images, 'classification')
            loss_raw = criterion(outputs, labels)
            loss = (loss_raw * confidences).mean()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_preds.append((probs > 0.5).cpu())
            all_labels.append((labels > 0.5).cpu())

    avg_loss = total_loss / len(dataloader)

    # Stack all batches
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute macro F1
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Compute AP per class
    per_class_ap = average_precision_score(all_labels, all_probs, average=None)

    return avg_loss, f1, per_class_ap

def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = True

def logSegmentationResults(log_dir, seg_val_loader, model, device):
    os.makedirs(log_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(seg_val_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, 'segmentation')
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

def train_loop(num_epochs, model, optimizer, num_classes, unfreeze_epoch, device, seg_train_loader, seg_val_loader, mlc_train_loader, mlc_val_loader):
    # ---- Optimizer and Loss ----
    mlc_criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss
    seg_criterion = torch.nn.CrossEntropyLoss()

    if unfreeze_epoch > 0:
        freeze_backbone(model)

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            unfreeze_backbone(model)
            print("Unfroze the backbone")

        seg_train_loss = trainSegmentationStep(model, seg_train_loader, seg_criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Segmentation Train Loss: {seg_train_loss:.4f}")
        seg_val_loss = validateSegmentationStep(model, seg_val_loader, seg_criterion, device)
        print(f"Segmentation Validation Loss: {seg_val_loss:.4f}")

        mlc_train_loss = trainMLCStep(model, mlc_train_loader, mlc_criterion, optimizer, device)
        print(f"CVS Classification Train Loss: {mlc_train_loss:.4f}")
        mlc_val_loss, val_f1, val_ap = validateMLCStep(model, mlc_val_loader, mlc_criterion, device)
        print(f"CVS Classification Validation Loss: {mlc_val_loss:.4f}, F1: {val_f1:.4f}")
        for i, ap in enumerate(val_ap):
            print(f"Average Precision for c{i+1}: {ap:.4f}")

def main():
    seg_batch_size, mlc_batch_size = 20, 40
    h, w = 300, 300
    train_seg_loader, val_seg_loader = getSegmentationLoaders(seg_batch_size, h, w)
    train_mlc_loader, val_mlc_loader = getMLCLoaders(mlc_batch_size, h, w)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetMultiHead(num_labels=num_labels, num_classes=num_classes).to(device)
    num_epochs = 50
    unfreeze_epoch = 5
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ])
    train_loop(num_epochs, model, optimizer, num_classes, unfreeze_epoch, device, train_seg_loader, val_seg_loader, train_mlc_loader, val_mlc_loader)
    logSegmentationResults('segmentation_results', seg_val_loader, model, device)

if __name__ == "__main__":
    main()
