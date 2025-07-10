import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.transform as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import random

class PadToSquareHeight:
    def __call__(self, img):
        w, h = img.size
        if h >= w:
            return img  # already tall or square
        pad = (w - h) // 2
        padding = (0, pad, 0, w - h - pad)  # left, top, right, bottom
        return TF.pad(img, padding, fill=0, padding_mode='constant')

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, labels_and_confidences, height, width, augment):
        # labels_and_confidences = {filename: ([0, 1, 1], [1.0, 0.6, 0.9])}
        self.image_dir = image_dir
        self.data = labels_and_confidences
        self.image_filenames = list(labels_and_confidences.keys())
        self.h = height
        self.w = width
        self.transform = transforms.Compose([
            PadToSquareHeight(),
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
        ])
        self.augment = augment

        # Only photometric (image-only)
        self.image_only_transforms = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

        yolo_model_path='train_code/yolo11s_cvs.pt'
        self.yolo = YOLO(yolo_model_path)  # YOLOv8 instance segmentation model

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

        seg = self.yolo.predict(torch.unsqueeze(image, 0), verbose=False)[0]
        masked_image = image # fallback
        if seg.masks is not None:
            masks = seg.masks.data.to(image.device)  # (N, H, W)
            # Combine all masks (e.g., union over all instance masks)
            combined_mask = masks.sum(dim=0).clamp(max=1.0)  # (H, W)

            # Create soft mask using sigmoid to emphasize foreground
            sharpness = 2 # these could be learned, if I find more free time
            resistance = 0.5 # 0 to 1, 1 is sharpest quality
            soft_mask = (resistance + 1) * torch.sigmoid(combined_mask * sharpness) - resistance  # (H, W)
            masked_image = image * soft_mask.unsqueeze(0)  # Apply mask to all 3 channels

            print_masked_image = True
            if print_masked_image:
                # Create subplots
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

                # Convert (C, H, W) to (H, W, C)
                axes[0].imshow(image.permute(1, 2, 0).numpy())
                axes[0].axis('off')  # Hide axis
                axes[0].set_title(f"Original")

                im = axes[1].imshow(soft_mask.unsqueeze(0).permute(1, 2, 0).numpy())
                axes[1].axis('off')  # Hide axis
                axes[1].set_title(f"Mask")
                fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                # Convert (C, H, W) to (H, W, C)
                axes[2].imshow(masked_image.permute(1, 2, 0).numpy())
                axes[2].axis('off')  # Hide axis
                axes[2].set_title(f"Masked image")

                plt.tight_layout()
                plt.show()

        return masked_image, label, confidence

def getMLCLoader(csv_file, image_path, augment, h, w, batch_size):
    # Transforms
    my_df = pd.read_csv(csv_file)

    labels_confidences_dict = {
        row['image']: (
            [row['c1'], row['c2'], row['c3']],
            [row['weight_c1'], row['weight_c2'], row['weight_c3']]
        )
        for _, row in my_df.iterrows()
    }

    # Create dataset instance
    dataset = MultiLabelImageDataset(image_path, labels_confidences_dict, h, w, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return loader

class MLCModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_labels)
        )

    def forward(self, x):
        return self.base(x)

# ==== Training ====
def trainMLCStep(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels, confidences in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        confidences = confidences.to(device)
        optimizer.zero_grad()
        outputs = model(images)
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

            outputs = model(images)
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
    for param in model.base.features.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.base.features.parameters():
        param.requires_grad = True

def train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders):
    # ---- Optimizer and Loss ----
    mlc_criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss

    if unfreeze_epoch > 0:
        freeze_backbone(model)

    optimizer = optimizer_adamw

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            unfreeze_backbone(model)
            print("Unfroze the backbone")

        if epoch == sgd_epoch:
            optimizer = optimizer_sgd
            print("Switched to SGD optimizer")

        print(f"Epoch {epoch+1}/{num_epochs}")
        for d, train_mlc_loader in enumerate(train_mlc_loaders):
            train_mlc_loss = trainMLCStep(model, train_mlc_loader, mlc_criterion, optimizer, device)
            print(f"Dataset {d} CVS Classification Train Loss: {train_mlc_loss:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            val_mlc_loss, val_f1, val_ap = validateMLCStep(model, val_mlc_loader, mlc_criterion, device)
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, F1: {val_f1:.4f}, Average Precisions:", end='')
            for ap in val_ap:
                print(f" {ap:.4f}", end='')
            print("")

def main():
    mlc_batch_size = 32 if torch.cuda.is_available() else 1
    h, w = 640, 640
    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, h, w, mlc_batch_size)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, h, w, mlc_batch_size)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, h, w, mlc_batch_size)
    cvs_train_mlc_loader = getMLCLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', True, h, w, mlc_batch_size)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, h, w, mlc_batch_size)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLCModel(num_labels=num_labels).to(device)
    num_epochs = 50
    unfreeze_epoch = 10
    sgd_epoch = 20

    # Define AdamW optimizer (for first 20 epochs)
    optimizer_adamw = torch.optim.AdamW([
        {'params': model.base.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': model.base.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ])

    # Define SGD optimizer (to be used after 20 epochs)
    optimizer_sgd = torch.optim.SGD([
        {'params': model.base.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': model.base.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ], momentum=0.9, nesterov=True)

    train_mlc_loaders = [cvs_train_mlc_loader]
    val_mlc_loaders = [cvs_val_mlc_loader]
    train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders)

if __name__ == "__main__":
    main()
