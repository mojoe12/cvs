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
import argparse
import copy

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
        self.transform = transforms.Compose([
            PadToSquareHeight(),
            transforms.Resize((height, width)),
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
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(list(model.model.model)[:])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print(i, type(layer), getattr(layer, 'name', None))
            x = layer(x)
        return x

def _ensure_tensor(out):
    return out[1] if isinstance(out, tuple) else out

class MLCModel(nn.Module):
    def __init__(self, num_labels, model_spec, shared_backbone):
        super().__init__()
        yolo = YOLO("yolo11s-cls").load(model_spec)
        yolo.reshape_outputs(yolo.model, nc=1)

        # Deep copy 3 separate branches
        c1 = copy.deepcopy(yolo.model)
        c2 = copy.deepcopy(yolo.model)
        c3 = copy.deepcopy(yolo.model)

        # Optional: add Sigmoid for multilabel probability output
        self.c1 = nn.Sequential(c1.model[:])
        self.c2 = nn.Sequential(c2.model[:])
        self.c3 = nn.Sequential(c3.model[:])

    def forward(self, x):
        out1 = _ensure_tensor(self.c1(x))
        out2 = _ensure_tensor(self.c2(x))
        out3 = _ensure_tensor(self.c3(x))

        return torch.cat([out1, out2, out3], dim=1)

    def set_backbone(self, requires_grad):
        return None

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


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--model_spec', type=str, default="train_code/yolo11s_cvs.pt", help='Path to model specification')
    parser.add_argument('--num_epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--unfreeze_epoch', type=int, default=10, help='Epoch at hwich to unfreeze layers')
    parser.add_argument('--sgd_epoch', type=int, default=20, help='Epoch at which to switch to SGD')

    parser.add_argument('--mlc_batch_size', type=int, default=32 if torch.cuda.is_available() else 1,
                        help='Batch size for multi-label classification')

    parser.add_argument('--base_adam_lr', type=float, default=1e-4, help='Base Adam learning rate')
    parser.add_argument('--classifier_adam_lr', type=float, default=1e-3, help='Classifier Adam learning rate')
    parser.add_argument('--base_sgd_lr', type=float, default=1e-4, help='Base SGD learning rate')
    parser.add_argument('--classifier_sgd_lr', type=float, default=1e-3, help='Classifier SGD learning rate')

    parser.add_argument('--base_weight_decay', type=float, default=5e-4, help='Base weight decay')
    parser.add_argument('--classifier_weight_decay', type=float, default=5e-4, help='Classifier weight decay')

    parser.add_argument('--shared_backbone', action='store_true', help='Share same backbone weights for c1, c2, c3')

    parser.add_argument('--use_endoscapes', action='store_true', help='Add in samples from endoscapes cvs 201')

    return parser.parse_args()

def main():

    args = parse_args()
    height, width = 640, 640 # yolo specific
    print(f"Model spec: {args.model_spec} with {"shared" if args.shared_backbone else "different"} backbone")
    print(f"Epochs: {args.num_epochs}, Unfreeze at: {args.unfreeze_epoch}, SGD at: {args.sgd_epoch}")
    print(f"Batch size: {args.mlc_batch_size}, Image size: {height}x{width}")
    print(f"Learning rates: Adam (base={args.base_adam_lr}, clf={args.classifier_adam_lr}), "
          f"SGD (base={args.base_sgd_lr}, clf={args.classifier_sgd_lr})")
    print(f"Weight decay: base={args.base_weight_decay}, clf={args.classifier_weight_decay}")
    if args.use_endoscapes:
        print("Using endoscapes cvs 201 for additional training data")

    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, height, width, args.mlc_batch_size)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, height, width, args.mlc_batch_size)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, height, width, args.mlc_batch_size)
    cvs_train_mlc_loader = getMLCLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', True, height, width, args.mlc_batch_size)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, height, width, args.mlc_batch_size)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLCModel(num_labels=num_labels, model_spec=args.model_spec, shared_backbone=args.shared_backbone).to(device)

    # Define AdamW optimizer (for first 20 epochs)
    optimizer_adamw = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.base_adam_lr, 'weight_decay': args.base_weight_decay},
    ])

    # Define SGD optimizer (to be used after 20 epochs)
    optimizer_sgd = torch.optim.SGD([
        {'params': model.parameters(), 'lr': args.base_sgd_lr, 'weight_decay': args.base_weight_decay},
    ], momentum=0.9, nesterov=True)

    if args.use_endoscapes:
        train_mlc_loaders = [cvs_train_mlc_loader, endo_train_mlc_loader, endo_val_mlc_loader, endo_test_mlc_loader]
    else:
        train_mlc_loaders = [cvs_train_mlc_loader]
    val_mlc_loaders = [cvs_val_mlc_loader]
    train_loop(args.num_epochs, model, optimizer_adamw, optimizer_sgd, args.sgd_epoch, num_classes, args.unfreeze_epoch, device, train_mlc_loaders, val_mlc_loaders)

if __name__ == "__main__":
    main()
