import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from ultralytics import YOLO
import timm
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from pycocotools.coco import COCO
from torchmetrics import AveragePrecision
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

class CropToSquareHeight:
    def __call__(self, img):
        w, h = img.size
        if h >= w:
            return img  # already tall or square
        return transforms.CenterCrop(h)(img)

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, labels_and_confidences, height, width, augment, pad_and_not_crop):
        # labels_and_confidences = {filename: ([0, 1, 1], [1.0, 0.6, 0.9])}
        self.image_dir = image_dir
        self.data = labels_and_confidences
        self.image_filenames = list(labels_and_confidences.keys())
        square_transform = PadToSquareHeight() if pad_and_not_crop else CropToSquareHeight()
        self.transform = transforms.Compose([
            square_transform,
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        self.augment = augment
        self.image_only_transforms = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.rand_augment = transforms.RandAugment(num_ops=16, magnitude=4)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            # Apply image-only transforms
            image = self.image_only_transforms(image)
            #image = self.rand_augment(image)
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
    dataset = MultiLabelImageDataset(image_path, labels_confidences_dict, h, w, augment, pad_and_not_crop=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return loader

class MultiScaleHead(nn.Module):
    """
    A head layer that takes three feature maps of different spatial resolutions
    (HxW = 40x40, 80x80, 160x160), performs adaptive pooling, concatenates them,
    and outputs multilabel classification logits for `num_labels` labels.
    """
    def __init__(self, detect, num_labels=3):
        super(MultiScaleHead, self).__init__()

        self.cv3 = detect.cv3
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 80x80 -> 40x40
            nn.Conv2d(11, 128, kernel_size=3, padding=1),  # 80x80 -> 40x40
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 40x40 -> 20x20
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 40x40 -> 20x20
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 80x80 -> 40x40
            nn.Conv2d(11, 128, kernel_size=3, padding=1),  # 40x40 -> 20x20
            nn.ReLU()
        )
        self.conv_final = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 20x20 -> 10x10
            nn.Conv2d(256 + 128 + 11, 512, kernel_size=3, stride=1, padding=1), # 20x20 -> 10x10
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 10x10 -> 1x1
            nn.Flatten(),
            nn.Linear(512, num_labels)
        )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple of torch.Tensor): (x1, x2, x3) feature maps from
                three scales with shapes:
                x1: [N, C1, 40, 40]
                x2: [N, C2, 80, 80]
                x3: [N, C3, 160, 160]

        Returns:
            logits: [N, num_labels]
        """
        x1, x2, x3 = inputs
        conv1 = self.conv1(self.cv3[0](x1))
        conv2 = self.conv2(self.cv3[1](x2))
        conv3 = self.cv3[2](x3)
        conv_all = torch.cat((conv1, conv2, conv3), dim=1)
        return self.conv_final(conv_all)

class MultiLabelCNN(nn.Module):
    def __init__(self, num_inputs, num_labels):
        super(MultiLabelCNN, self).__init__()

        ni1, ni2, ni3 = num_inputs
        in_channels = ni1 + ni2 + ni3
        # Conv Layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 80x80 → 40x40

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 40x40 → 20x20
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 256 x 20 x 20 = 102400
            nn.Linear(256 * 20 * 20, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def forward(self, inputs):
        x1, x2, x3 = inputs
        x2_upscale = F.interpolate(x2, scale_factor=2, mode='nearest')
        x3_upscale = F.interpolate(x3, scale_factor=4, mode='nearest')
        x = torch.cat((x1, x2_upscale, x3_upscale), dim=1)

        x = self.features(x)
        return self.classifier(x)

# Example usage:
# model = MultiLabelCNN(num_classes=10)
# output = model(torch.randn(8, 33, 80, 80))  # batch of 8
# print(output.shape)  # -> torch.Size([8, 10])

class YoloMLCModel(nn.Module):
    def __init__(self, num_labels, model_name, use_y12):
        super().__init__()
        self.model_name = model_name
        yolo = YOLO(self.model_name)
        self.output_indices = set([4, 6, 10])
        self.head_layer = MultiLabelCNN((256, 256, 512), num_labels)
        max_output_index = max(self.output_indices)
        self.backbone_layers = nn.ModuleList([
            yolo.model.model[index] for index in range(max_output_index + 1)
        ])

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.backbone_layers):
            x = layer(x)
            if idx in self.output_indices:
                outputs.append(x)
        return self.head_layer(outputs)

    def backbone_parameters(self):
        return [p for layer in self.backbone_layers for p in layer.parameters()]

    def classifier_parameters(self):
        return self.head_layer.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

    def export(self, output_file):
        # Load base YOLO model
        yolo = YOLO(self.model_name)

        # Replace backbone and head
        for i, layer in enumerate(self.layers):
            yolo.model.model[i] = self.layers[i]

        # Save the model
        yolo.save(output_file)
        print(f"Model exported to: {output_file}")

class TransformerModel(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        # Load pretrained model
        #self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
        #tested models: 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
        #               'vit_base_patch16_224.mae'
        #               'mambaout_femto.in1k'
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feature_channels = 1152 #self.backbone.feature_info[-1]['num_chs']
        self.head = nn.Linear(feature_channels, num_labels)
        # Replace classifier with multilabel output (3 labels)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.head.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

    def export(self, output_file):
        TODO

# ==== Training ====
def trainStep(model, dataloader, criterion, optimizer, device):
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

def validateMLCStep(model, dataloader, criterion, device, precision):
    model.eval()
    total_loss = 0

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

            precision.update(torch.sigmoid(outputs), (labels > 0.5))

    avg_loss = total_loss / len(dataloader)

    # Compute statistics

    return avg_loss

def train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, scheduler_sgd, sgd_epoch, unfreeze_epoch, num_classes, num_labels, device, train_mlc_loaders, val_mlc_loaders):
    # ---- Optimizer and Loss ----
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss
    precision = AveragePrecision(task='multilabel', num_labels=num_labels, average='none')

    optimizer = optimizer_adamw

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        if epoch == sgd_epoch:
            optimizer = optimizer_sgd
            print("Switched to SGD optimizer")

        if epoch == unfreeze_epoch:
            model.set_backbone(True)
            print("Unfroze backbone")


        print(f"Epoch {epoch+1}/{num_epochs}")
        for d, train_mlc_loader in enumerate(train_mlc_loaders):
            train_loss = trainStep(model, train_mlc_loader, criterion, optimizer, device)
            print(f"Dataset {d} CVS Classification Train Loss: {train_loss:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            precision.reset()
            val_mlc_loss = validateMLCStep(model, val_mlc_loader, criterion, device, precision)
            map_score = precision.compute()  # Tensor of size [num_classes] or scalar if average="macro"
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, Average Precisions: {map_score}")

        if len(val_mlc_loaders) > 0 and epoch >= sgd_epoch and sgd_epoch > -1:
            scheduler_sgd.step(val_mlc_loss)

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--model_name', type=str, required=True, help='Path to MLC model specification')
    parser.add_argument('--image_size', type=int, required=True, help='Image Size. YOLO=640, transformers vary')
    parser.add_argument('--num_epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--sgd_epoch', type=int, default=-1, help='Epoch at which to switch to SGD')
    parser.add_argument('--unfreeze_epoch', type=int, default=-1, help='Epoch at which to unfreeze the backbone')

    parser.add_argument('--mlc_batch_size', type=int, default=32 if torch.cuda.is_available() else 1,
                        help='Batch size for multi-label classification')

    parser.add_argument('--backbone_adam_lr', type=float, default=1e-5, help='Base Adam learning rate')
    parser.add_argument('--classifier_adam_lr', type=float, default=1e-3, help='Classifier Adam learning rate')
    parser.add_argument('--backbone_sgd_lr', type=float, default=1e-3, help='Base SGD learning rate')
    parser.add_argument('--classifier_sgd_lr', type=float, default=1e-2, help='Classifier SGD learning rate')

    parser.add_argument('--backbone_weight_decay', type=float, default=5e-4, help='Base weight decay')
    parser.add_argument('--classifier_weight_decay', type=float, default=5e-4, help='Classifier weight decay')

    parser.add_argument('--use_endoscapes', action='store_true', help='Add in samples from endoscapes cvs 201')
    parser.add_argument('--use_interpolated_cvs', action='store_true', help='Add in interpolated samples from CVS')

    parser.add_argument('--use_yolo12', action='store_true', help='Use YOLO 12 structure')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer')

    parser.add_argument('--output_file', type=str, default="", help='Path to model specification')

    return parser.parse_args()

def main():

    args = parse_args()
    height, width = args.image_size, args.image_size
    print(f"MLC Model name: {args.model_name}")
    print(f"Num epochs: {args.num_epochs}")
    if args.sgd_epoch >= 0:
        print(f"Switch to SGD at: {args.sgd_epoch}")
    if args.unfreeze_epoch >= 0:
        print(f"Unfreeze backbone at: {args.unfreeze_epoch}")
    print(f"Batch size: {args.mlc_batch_size}, Image size: {height}x{width}")
    print(f"Learning rates: Adam (backbone={args.backbone_adam_lr}, clf={args.classifier_adam_lr})")
    if args.sgd_epoch >= 0:
        print(f"SGD (backbone={args.backbone_sgd_lr}, clf={args.classifier_sgd_lr})")
    print(f"Weight decay: backbone={args.backbone_weight_decay}, clf={args.classifier_weight_decay}")
    if args.use_endoscapes:
        print("Using endoscapes cvs 201 for additional training data")
    if args.use_interpolated_cvs:
        print("Using interpolated data from CVS")
    if args.use_yolo12:
        print("Using YOLO 12 structure")
    if len(args.output_file) > 0:
        print(f"Logging to {args.output_file}")

    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, height, width, args.mlc_batch_size)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, height, width, args.mlc_batch_size)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, height, width, args.mlc_batch_size)
    train_mlc_data_csv = 'analysis/train_mlc_data_interpolated.csv' if args.use_interpolated_cvs else 'analysis/train_mlc_data.csv'
    cvs_train_mlc_loader = getMLCLoader(train_mlc_data_csv, 'sages_cvs_challenge_2025/frames', True, height, width, args.mlc_batch_size)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, height, width, args.mlc_batch_size)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_transformer:
        model = TransformerModel(num_labels, args.model_name).to(device)
    else:
        model = YoloMLCModel(num_labels, args.model_name, args.use_yolo12).to(device)
    model.set_backbone(False)

    optimizer_adamw = torch.optim.AdamW([
        {'params': model.backbone_parameters(), 'lr': args.backbone_adam_lr, 'weight_decay': args.backbone_weight_decay},
        {'params': model.classifier_parameters(), 'lr': args.classifier_adam_lr, 'weight_decay': args.classifier_weight_decay},
    ])

    optimizer_sgd = torch.optim.SGD([
        {'params': model.backbone_parameters(), 'lr': args.backbone_sgd_lr, 'weight_decay': args.backbone_weight_decay},
        {'params': model.classifier_parameters(), 'lr': args.classifier_sgd_lr, 'weight_decay': args.classifier_weight_decay},
    ], momentum=0.9, nesterov=True)

    scheduler_sgd = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sgd,
        mode='min',          # minimize val_loss
        factor=0.5,          # reduce LR by this factor
        patience=2,          # wait N epochs before reducing
        min_lr=1e-6          # don’t reduce below this
    )

    if args.use_endoscapes:
        train_mlc_loaders = [cvs_train_mlc_loader, endo_train_mlc_loader, endo_val_mlc_loader, endo_test_mlc_loader]
    else:
        train_mlc_loaders = [endo_train_mlc_loader]
    val_mlc_loaders = [endo_val_mlc_loader, endo_test_mlc_loader]

    train_loop(args.num_epochs, model, optimizer_adamw, optimizer_sgd, scheduler_sgd, args.sgd_epoch, args.unfreeze_epoch, num_classes, num_labels, device, train_mlc_loaders, val_mlc_loaders)
    if len(args.output_file) > 0:
        model.export(args.output_file)

if __name__ == "__main__":
    main()
