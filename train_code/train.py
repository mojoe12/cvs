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
        shuffle=True,
        num_workers=4
    )
    return loader

class YOLOBackbone(nn.Module):
    def __init__(self, model_spec, cutoff):
        super().__init__()
        yolo = YOLO(model_spec)
        # Extract raw backbone layers (excluding heads)
        self.layers = nn.ModuleList(list(yolo.model.model)[:cutoff])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print(i, type(layer), getattr(layer, 'name', None))
            x = layer(x)
        return x

class MLCModel(nn.Module):
    def __init__(self, num_labels, model_type, model_spec):
        super().__init__()
        self.model_type = model_type
        self.base = YOLOBackbone(model_spec, 11)
        num_out_features = 512 # this might be different based on model_type. not sure yet!
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_out_features, num_labels)
        )

    def forward(self, x):
        feats = self.base(x)
        return self.classifier(feats)

    def set_backbone(self, requires_grad):
        for param in self.base.parameters():
            param.requires_grad = True


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
    model_type = "11s"
    model_spec = "train_code/yolo11s_cvs.pt"
    #model_spec = "train_code/yolo11m_cvs_endo.pt"
    num_epochs = 20
    unfreeze_epoch = 10
    sgd_epoch = 20
    mlc_batch_size = 32 if torch.cuda.is_available() else 1
    h, w = 640, 640
    base_adam_lr = 1e-5
    classifier_adam_lr = 1e-5
    base_sgd_lr = 1e-5
    classifier_sgd_lr = 1e-5
    base_weight_decay = 0
    classifier_weight_decay = 0

    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, h, w, mlc_batch_size)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, h, w, mlc_batch_size)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, h, w, mlc_batch_size)
    cvs_train_mlc_loader = getMLCLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', True, h, w, mlc_batch_size)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, h, w, mlc_batch_size)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLCModel(num_labels=num_labels, model_type=model_type, model_spec=model_spec).to(device)

    # Define AdamW optimizer (for first 20 epochs)
    optimizer_adamw = torch.optim.AdamW([
        {'params': model.base.parameters(), 'lr': base_adam_lr, 'weight_decay': base_weight_decay},
        {'params': model.classifier.parameters(), 'lr': classifier_adam_lr, 'weight_decay': classifier_weight_decay},
    ])

    # Define SGD optimizer (to be used after 20 epochs)
    optimizer_sgd = torch.optim.SGD([
        {'params': model.base.parameters(), 'lr': base_sgd_lr, 'weight_decay': base_weight_decay},
        {'params': model.classifier.parameters(), 'lr': classifier_sgd_lr, 'weight_decay': classifier_weight_decay},
    ], momentum=0.9, nesterov=True)

    train_mlc_loaders = [cvs_train_mlc_loader]
    val_mlc_loaders = [cvs_val_mlc_loader]
    train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
