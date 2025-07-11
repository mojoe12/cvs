import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO
#from ultralytics.nn.modules.block import Concat, Detect, C2f, C3, Conv  # etc.
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
            transforms.ToTensor()
        ])
        self.augment = augment
        self.rand_augment = transforms.RandAugment(num_ops=16, magnitude=4)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            image = self.rand_augment(image)
        image = self.transform(image)
        label, confidence = self.data[filename]
        label = torch.tensor(label, dtype=torch.float32)
        confidence = torch.tensor(confidence, dtype=torch.float32)

        return image, label, confidence

class CustomCollateFn:
    def __init__(self, model_path, print_masked_image, sharpness, resistance, h, w):
        self.yolo = YOLO(model_path)
        self.print_masked_image = print_masked_image
        self.sharpness = sharpness
        self.resistance = resistance
        self.h = h
        self.w = w

    def __call__(self, batch):

        """
        Custom collate function for a batch of images and variable-sized targets (e.g., bounding boxes).

        Args:
            batch: A list of tuples (image, target), where:
                - image is a Tensor of shape (C, H, W)
                - target is a dict (e.g., with boxes, labels, masks, etc.)

        Returns:
            images: A Tensor of stacked images (if sizes match) or list if variable sizes
            targets: A list of target dictionaries
        """
        images = [TF.to_pil_image(item[0]) for item in batch]
        labels = [item[1] for item in batch]
        confidences = [item[2] for item in batch]

        resize_t = transforms.Resize((self.h, self.w))
        masked_images = [0.5 * (1 - self.resistance) * resize_t(item[0]) for item in batch] # fallback

        segs = self.yolo.predict(images, verbose=False)

        for index, seg in enumerate(segs):
            if seg.masks is not None:
                image = images[index]
                masks = seg.masks.data  # (N, H, W)
                classes = seg.boxes.cls.to(torch.int)    # (N,) - class IDs for each mask
                # Filter out masks with class id == 5
                keep_indices = (classes != 5).nonzero(as_tuple=True)[0]
                if keep_indices.numel() > 0:
                    filtered_masks = masks[keep_indices]  # (M, H, W) where M <= N

                    combined_mask = filtered_masks.sum(dim=0).clamp(max=1.0)  # (H, W)

                    # Create soft mask using sigmoid to emphasize foreground
                    soft_mask = (self.resistance + 1) * torch.sigmoid(combined_mask * self.sharpness) - resistance
                    masked_images[index] = masked_images[index] * resize_t(soft_mask.unsqueeze(0)).to('cpu')

                    if self.print_masked_image:
                        # Create subplots
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

                        # Convert (C, H, W) to (H, W, C)
                        axes[0].imshow(image)
                        axes[0].axis('off')  # Hide axis
                        axes[0].set_title(f"Original")

                        im = axes[1].imshow(soft_mask.unsqueeze(0).permute(1, 2, 0).numpy())
                        axes[1].axis('off')  # Hide axis
                        axes[1].set_title(f"Mask")
                        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                        # Convert (C, H, W) to (H, W, C)
                        axes[2].imshow(masked_images[index].permute(1, 2, 0).numpy())
                        axes[2].axis('off')  # Hide axis
                        axes[2].set_title(f"Masked image")

                        plt.tight_layout()
                        plt.show()

        return torch.stack(masked_images, dim=0), torch.stack(labels, dim=0), torch.stack(confidences, dim=0)

def getMLCLoader(csv_file, image_path, augment, h, w, batch_size, collate_fn):
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
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    return loader

class YOLOBackbone(nn.Module):
    def __init__(self, model_spec, cutoff=10):
        super().__init__()
        yolo = YOLO(model_spec)
        # Extract raw backbone layers (excluding heads)
        self.layers = nn.ModuleList(list(yolo.model.model)[:cutoff])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class MLCModel(nn.Module):
    def __init__(self, num_labels, model_type, model_spec):
        super().__init__()
        self.model_type = model_type
        if self.model_type == "efficientnet":
            if model_spec == "b3":
                self.base = models.efficientnet_b3(pretrained=True)
            else:
                fail #invalid model_spec
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_labels)
            )
        elif self.model_type == "resnet":
            if model_spec == "50":
                self.base = models.resnet50(pretrained=True)
            elif model_spec == "18":
                self.base = models.resnet18(pretrained=True)
            else:
                fail #invalid model_spec
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_labels)
        elif self.model_type == "yolo":
            self.base = YOLOBackbone(model_spec)
            num_out_features = 512
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(num_out_features, num_labels)
            )
        else:
            fail # invalid model type

    def forward(self, x):
        if self.model_type == "yolo":
            x = self.base(x)
            return self.classifier(x)
        else:
            return self.base(x)

    def set_backbone(self, requires_grad):
        if self.model_type == "efficientnet":
            for param in self.base.features.parameters():
                param.requires_grad = requires_grad
        elif self.model_type == "resnet":
            for param in self.base.parameters():
                param.requires_grad = requires_grad
            for param in self.base.fc.parameters():
                param.requires_grad = True
        elif self.model_type == "yolo":
            for param in self.base.parameters():
                param.requires_grad = True
        else:
            fail # invalid self.model_type


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


def train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders):
    # ---- Optimizer and Loss ----
    mlc_criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss

    if unfreeze_epoch > 0:
        model.set_backbone(requires_grad=False)

    optimizer = optimizer_adamw

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch and unfreeze_epoch > 0:
            model.set_backbone(requires_grad=True)
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
    yolo_intermediate = False
    yolo_model_path = 'train_code/yolo11s_cvs.pt'
    model_type = "yolo"
    model_spec = yolo_model_path
    mlc_batch_size = 32 if torch.cuda.is_available() else 1
    if model_type == "yolo":
        h, w = 640, 640
    elif model_type == "efficientnet":
        if model_spec == "b3":
            h, w = 300, 300
        else:
            fail #invalid model_spec
    elif model_type == "resnet":
        h, w = 224, 224
    else:
        fail #invalid model_type

    collate_fn = None
    if yolo_intermediate:
        sharpness = 2
        resistance = 0.5
        collate_fn = CustomCollateFn(yolo_model_path, print_masked_image, sharpness, resistance, h, w)
        h, w = 640, 640
    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, h, w, mlc_batch_size, collate_fn)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, h, w, mlc_batch_size, collate_fn)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, h, w, mlc_batch_size, collate_fn)
    cvs_train_mlc_loader = getMLCLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', True, h, w, mlc_batch_size, collate_fn)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, h, w, mlc_batch_size, collate_fn)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLCModel(num_labels=num_labels, model_type=model_type, model_spec=model_spec).to(device)
    num_epochs = 20
    unfreeze_epoch = 10
    sgd_epoch = 20

    # Define AdamW optimizer (for first 20 epochs)
    optimizer_adamw = torch.optim.AdamW([
        #{'params': model.base.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-9},
        {'params': model.base.parameters(), 'lr': 1e-5, 'weight_decay': 1e-9},
    ])

    # Define SGD optimizer (to be used after 20 epochs)
    optimizer_sgd = torch.optim.SGD([
        {'params': model.base.parameters(), 'lr': 1e-5, 'weight_decay': 1e-9},
        #{'params': model.base.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
        #{'params': model.base.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    ], momentum=0.9, nesterov=True)

    train_mlc_loaders = [cvs_train_mlc_loader]
    val_mlc_loaders = [cvs_val_mlc_loader]
    train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
