import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image
import os
import pandas as pd
import random

# ==== Dummy Dataset ====
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

class EfficientNetMultiLabel(nn.Module):
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
def train(model, dataloader, criterion, optimizer, device):
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

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels, confidences in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            confidences = confidences.to(device)
            outputs = model(images)
            loss_raw = criterion(outputs, labels)
            loss = (loss_raw * confidences).mean()
            total_loss += loss.item()
    return total_loss / len(dataloader)

def freeze_backbone(model):
    for param in model.base.features.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.base.features.parameters():
        param.requires_grad = True

# ==== Main ====
def train_mlc(num_epochs, log_dir, saved_backbone, saved_classifier):
    # Configuration
    num_labels = 3
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    full_dataset = MultiLabelImageDataset(image_path, labels_confidences_dict, 224, 224, augment=False)
    # Create a consistent train/val split
    total_indices = list(range(len(full_dataset)))
    random.shuffle(total_indices)
    val_size = int(0.0 * len(full_dataset))
    val_indices = total_indices[:val_size]
    train_indices = total_indices[val_size:]

    # Create two dataset instances with different augment flags
    train_dataset_full = MultiLabelImageDataset(image_path, labels_confidences_dict, 224, 224, augment=False)
    val_dataset_full = MultiLabelImageDataset(image_path, labels_confidences_dict, 224, 224, augment=True)

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

    # Model
    model = EfficientNetMultiLabel(num_labels=num_labels).to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss
    optimizer = torch.optim.Adam([
        {'params': model.base.features.parameters(), 'lr': 1e-5},
        {'params': model.base.classifier.parameters(), 'lr': 1e-4},
    ])

    backbone_params = list(model.base.features.parameters())
    classifier_params = list(model.base.classifier.parameters())

    freeze_backbone(model)

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch == 4: #the fifth time
            unfreeze_backbone(model)
            print("Unfroze the backbone")

    return model.self.features, model.self.classifier

def main():
    saved_backbone, saved_classifier = train_mlc(50, "mlc_results", None, None)

if __name__ == "__main__":
    main()
