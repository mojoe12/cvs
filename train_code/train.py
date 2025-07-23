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
        self.image_only_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandAugment(num_ops=16, magnitude=4),
        ])
        self.random_erase = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        basename = os.path.splitext(os.path.basename(filename))[0]
        video, frame_str = basename.rsplit('_', 1)
        frame = int(frame_str)
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
        image = self.transform(image)
        if self.augment:
            image = self.random_erase(image)
        label, confidence = self.data[filename]
        label = torch.tensor(label, dtype=torch.float32)
        confidence = torch.tensor(confidence, dtype=torch.float32)
        #confidence = confidence * torch.tensor([3.19852941, 4.46153846, 2.79518072], dtype=torch.float32)

        return image, label, confidence, video, frame

def getMLCImageLoader(csv_file, image_path, augment, h, w, batch_size):
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
    return loader, dataset

class MultiLabelVideoDataset(Dataset):
    def __init__(self, features_labels_and_confidences):
        # features_labels_and_confidences = [(features, labels, confidences)]
        self.data = features_labels_and_confidences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels, confidences = self.data[idx]
        return features, labels, confidences

def getMLCVideoLoader(image_dataset, model, batch_size, device):
    features_labels_and_confidences = []

    curr_video = None
    prev_frame = -1
    video_index = 0
    video_images, video_labels, video_confidences = [], [], []

    def compute_video_features():
        assert video_index == 18, f"Expected 18 frames, got {video_index} for video {curr_video}"
        with torch.no_grad():
            video_features = model(torch.stack(video_images).to(device), return_hidden=True).cpu()
        new_video = (video_features, torch.stack(video_labels), torch.stack(video_confidences))
        features_labels_and_confidences.append(new_video)

    for image, label, confidence, video, frame in image_dataset:
        if curr_video is not None and video != curr_video:
            # Finalize previous video
            compute_video_features()
            # Reset for new video
            video_images, video_labels, video_confidences = [], [], []
            video_index = 0
            prev_frame = -1

        assert prev_frame < frame, f"Frame order issue in video {video}: {frame} after {prev_frame}"
        prev_frame = frame
        curr_video = video
        video_index += 1
        assert video_index <= 18, f"Too many frames in video {video}: {video_index}"

        video_images.append(image)
        video_labels.append(label)
        video_confidences.append(confidence)

    # Handle last video
    compute_video_features()

    # Create dataset instance
    dataset = MultiLabelVideoDataset(features_labels_and_confidences)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return loader

class YoloSimpleMLCModel(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        self.model_name = model_name
        yolo = YOLO(self.model_name)
        self.num_features = 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_features, num_labels)
        )
        max_head_index = 11
        self.backbone_layers = nn.ModuleList([
            yolo.model.model[index] for index in range(min(max_head_index, len(yolo.model.model)-1))
        ])

    def forward(self, x, return_hidden=False):
        for idx, layer in enumerate(self.backbone_layers):
            x = layer(x)
        return x if return_hidden else self.head(x)

    def backbone_parameters(self):
        return [p for layer in self.backbone_layers for p in layer.parameters()]

    def classifier_parameters(self):
        return self.head.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

class YoloTransformerMLCModel(nn.Module):
    def __init__(self, num_labels, yolo_model_name, transformer_model_name):
        super().__init__()
        self.yolo_model_name = yolo_model_name
        yolo = YOLO(self.yolo_model_name)
        num_out_features = 512
        self.flatten_yolo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        max_head_index = 11
        self.backbone_layers = nn.ModuleList([
            yolo.model.model[index] for index in range(min(max_head_index, len(yolo.model.model)-1))
        ])
        self.backbone = timm.create_model(transformer_model_name, pretrained=True)
        feature_channels = self.backbone.feature_info[-1]['num_chs']
        self.num_features = feature_channels + num_out_features
        self.backbone.reset_classifier(0)
        self.head = nn.Linear(self.num_features, num_labels, bias=True)

    def forward(self, x, return_hidden=False):
        yolo_x = x
        for idx, layer in enumerate(self.backbone_layers):
            yolo_x = layer(yolo_x)
        yolo_x = self.flatten_yolo(yolo_x)
        trans_x = self.backbone(x)
        head_x = torch.cat([yolo_x, trans_x], dim=1)
        return head_x if return_hidden else self.head(head_x)

    def backbone_parameters(self):
        return [p for layer in self.backbone_layers for p in layer.parameters()] + list(self.backbone.parameters())

    def classifier_parameters(self):
        return self.head.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

class TransformerMLCModel(nn.Module):
    def __init__(self, num_labels, model_name):
        super().__init__()
        # Load pretrained model
        #self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
        #tested models: 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
        #               'vit_base_patch16_224.mae'
        #               'mambaout_femto.in1k'
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.num_features = self.backbone.feature_info[-1]['num_chs']
        self.backbone.reset_classifier(0)
        self.head = nn.Linear(self.num_features, num_labels, bias=True)
        # Replace classifier with multilabel output (3 labels)

    def forward(self, x, return_hidden=False):
        feats = self.backbone(x)
        return feats if return_hidden else self.head(feats)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def classifier_parameters(self):
        return self.head.parameters()

    def set_backbone(self, requires_grad):
        for param in self.backbone_parameters():
            param.requires_grad = requires_grad

class TemporalMLCPredictor(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_labels, num_layers=2, num_heads=4):
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):  # x: (B, 18, hidden_dim)
        x = self.projection(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, hidden_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (B, 18, hidden_dim)
        out = self.classifier(x)  # (B, 18, num_labels)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.downsample = nn.Identity()  # Keep for structure
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):  # x: [B, C, T]
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual  # Residual connection
        return x

class TemporalMLCTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_blocks):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dilation=2**i) for i in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, num_labels, kernel_size=1)

    def forward(self, x):  # x: [B, 18, 1536]
        x = x.transpose(1, 2)         # [B, 1536, 18] → [B, C, T]
        x = self.input_proj(x)        # [B, hidden_dim, 18]
        x = self.blocks(x)            # temporal modeling
        x = self.output_proj(x)      # [B, 3, 18]
        x = x.transpose(1, 2)         # [B, 18, 3]
        return x

# ==== Training ====
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    '''CutMix for 3D (B, L, F) or 4D (B, C, H, W) inputs.
    Returns mixed inputs, pairs of targets, and lambda.'''

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    if x.dim() == 4:  # Image: (B, C, H, W)
        _, _, H, W = x.size()
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    elif x.dim() == 3:  # Sequence: (B, L, F)
        _, L, _ = x.size()
        cut_len = int(L * np.sqrt(1 - lam))
        cx = np.random.randint(L)
        start = np.clip(cx - cut_len // 2, 0, L)
        end = np.clip(cx + cut_len // 2, 0, L)
        x[:, start:end, :] = x[index, start:end, :]
        lam = 1 - ((end - start) / L)
    else:
        raise ValueError(f'Unsupported input dimension {x.dim()}, expected 3D or 4D tensor.')
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def trainStep(model, dataloader, criterion, optimizer, device, max_norm):
    model.train()
    total_loss = 0
    for images, labels, confidences, *rest in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        confidences = confidences.to(device)
        # Apply MixUp or CutMix
        if random.random() < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
        else:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=0.4)
        optimizer.zero_grad()
        outputs = model(images)
        loss_raw = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss = (loss_raw * confidences).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validateMLCStep(model, dataloader, criterion, device, precision):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, confidences, *rest in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            confidences = confidences.to(device)

            outputs = model(images)
            loss_raw = criterion(outputs, labels)
            loss = (loss_raw * confidences).mean()
            total_loss += loss.item()

            if outputs.dim() == 3:
                for index in range(outputs.shape[1]):
                    precision.update(torch.sigmoid(outputs[:, index]), (labels[:, index] > 0.5))
            elif outputs.dim() == 2:
                precision.update(torch.sigmoid(outputs), (labels > 0.5))
            else:
                assert False # outputs dim should be 2 or 3
    return total_loss / len(dataloader)

def train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, unfreeze_epoch, num_labels, device, train_mlc_loaders, val_mlc_loaders, output_file, output_file_ext):
    # ---- Optimizer and Loss ----
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss
    precision = AveragePrecision(task='multilabel', num_labels=num_labels, average='none')
    max_norm = 5
    scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=num_epochs)
    scheduler_sgd = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sgd,
        mode='min',          # minimize val_loss
        factor=0.5,          # reduce LR by this factor
        patience=2,          # wait N epochs before reducing
        min_lr=1e-6          # don’t reduce below this
    )
    optimizer = optimizer_adamw
    best_val_loss = float('inf')

    # ---- Training Loop ----
    for epoch in range(num_epochs):
        if epoch == sgd_epoch:
            optimizer = optimizer_sgd
            print("Switched to SGD optimizer")

        if epoch == unfreeze_epoch and unfreeze_epoch != 0:
            model.set_backbone(True)
            print("Unfroze backbone")

        print(f"Epoch {epoch+1}/{num_epochs}")
        for d, train_mlc_loader in enumerate(train_mlc_loaders):
            train_loss = trainStep(model, train_mlc_loader, criterion, optimizer, device, max_norm)
            print(f"Dataset {d} CVS Classification Train Loss: {train_loss:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            precision.reset()
            val_mlc_loss = validateMLCStep(model, val_mlc_loader, criterion, device, precision)
            map_score = precision.compute()  # Tensor of size [num_labels] or scalar if average="macro"
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, Average Precisions: {map_score}")

            if d == 0 and val_mlc_loss < best_val_loss and len(output_file) > 0:
                best_val_loss = val_mlc_loss
                torch.save(model.state_dict(), output_file + output_file_ext)
                print(f"Model improved. Weights saved to: {output_file + output_file_ext}")

        if len(val_mlc_loaders) > 0:
            if epoch >= sgd_epoch and sgd_epoch > -1:
                scheduler_sgd.step(val_mlc_loss)
            else:
                scheduler_adamw.step()

    if len(val_mlc_loaders) == 0 and len(output_file) > 0:
        torch.save(model.state_dict(), output_file + output_file_ext)
        print(f"Model weights exported to: {output_file + output_file_ext}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--transformer_model', type=str, default="", help='Path to Transformer model specification')
    parser.add_argument('--yolo_model', type=str, default="", help='Path to YOLO model specification')
    parser.add_argument('--saved_weights', type=str, default="", help='Path to file representing model weights')
    parser.add_argument('--image_size', type=int, required=True, help='Image Size. Depends on model')
    parser.add_argument('--num_epochs', type=int, default=20, help='Total number of static training epochs')
    parser.add_argument('--temporal_epochs', type=int, default=0, help='Total number of temporal training epochs')
    parser.add_argument('--sgd_epoch', type=int, default=-1, help='Epoch at which to switch to SGD')
    parser.add_argument('--unfreeze_epoch', type=int, default=0, help='Epoch at which to unfreeze the backbone')

    parser.add_argument('--mlc_batch_size', type=int, default=32 if torch.cuda.is_available() else 1,
                        help='Batch size for multi-label classification')

    parser.add_argument('--backbone_adam_lr', type=float, default=1e-5, help='Base Adam learning rate')
    parser.add_argument('--classifier_adam_lr', type=float, default=1e-3, help='Classifier Adam learning rate')
    parser.add_argument('--backbone_sgd_lr', type=float, default=1e-3, help='Base SGD learning rate')
    parser.add_argument('--classifier_sgd_lr', type=float, default=1e-2, help='Classifier SGD learning rate')

    parser.add_argument('--backbone_weight_decay', type=float, default=5e-4, help='Base weight decay')
    parser.add_argument('--classifier_weight_decay', type=float, default=5e-4, help='Classifier weight decay')

    parser.add_argument('--use_endoscapes', action='store_true', help='Add in samples from endoscapes cvs 201')
    parser.add_argument('--only_endoscapes', action='store_true', help='Train and validate on endoscapes only')
    parser.add_argument('--use_interpolated', action='store_true', help='Add in interpolated training samples')

    parser.add_argument('--output_file', type=str, default="", help='Path to model specification')

    return parser.parse_args()

def main():

    args = parse_args()
    height, width = args.image_size, args.image_size
    print(f"Num epochs for per-frame training: {args.num_epochs}")
    print(f"Num epochs for per-video training: {args.temporal_epochs}")
    if args.sgd_epoch >= 0:
        print(f"Switch to SGD at: {args.sgd_epoch}")
    if args.unfreeze_epoch > 0:
        print(f"Unfreeze backbone at: {args.unfreeze_epoch}")
    print(f"Batch size: {args.mlc_batch_size}, Image size: {height}x{width}")
    print(f"Learning rates: Adam (backbone={args.backbone_adam_lr}, clf={args.classifier_adam_lr})")
    if args.sgd_epoch >= 0:
        print(f"SGD (backbone={args.backbone_sgd_lr}, clf={args.classifier_sgd_lr})")
    print(f"Weight decay: backbone={args.backbone_weight_decay}, clf={args.classifier_weight_decay}")
    if args.use_endoscapes:
        print("Using endoscapes cvs 201 for additional training data")
    if args.only_endoscapes:
        print("Training and validating only on endoscapes")
    assert not args.use_endoscapes or not args.only_endoscapes

    if args.use_interpolated:
        print("Using interpolated training data")
    if len(args.output_file) > 0:
        assert ".pth" not in args.output_file # library will add that
        print(f"Logging to {args.output_file}")

    endo_train_mlc_data_csv = 'analysis/endo_train_mlc_data_interpolated.csv' if args.use_interpolated else 'analysis/endo_train_mlc_data.csv'
    endo_train_mlc_loader, _ = getMLCImageLoader(endo_train_mlc_data_csv, 'endoscapes/train', True, height, width, args.mlc_batch_size)
    endo_val_mlc_loader, _ = getMLCImageLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', args.use_endoscapes, height, width, args.mlc_batch_size)
    endo_test_mlc_loader, _ = getMLCImageLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', args.use_endoscapes, height, width, args.mlc_batch_size)
    train_mlc_data_csv = 'analysis/train_mlc_data_interpolated.csv' if args.use_interpolated else 'analysis/train_mlc_data.csv'
    cvs_train_mlc_loader, _ = getMLCImageLoader(train_mlc_data_csv, 'sages_cvs_challenge_2025/frames', True, height, width, args.mlc_batch_size)
    cvs_val_mlc_loader, cvs_val_mlc_dataset = getMLCImageLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, height, width, args.mlc_batch_size)

    num_labels = 3
    #num_classes = 7 # irrelevant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(args.transformer_model) > 0:
        print(f"Using transformer model {args.transformer_model} as backbone")
        if len(args.yolo_model) > 0:
            print(f"Using yolo model {args.yolo_model} as backbone")
            model = YoloTransformerMLCModel(num_labels, args.yolo_model, args.transformer_model).to(device)
        else:
            model = TransformerMLCModel(num_labels, args.transformer_model).to(device)
    else:
        if len(args.yolo_model) > 0:
            print(f"Using yolo model {args.yolo_model} as backbone")
            model = YoloSimpleMLCModel(num_labels, args.yolo_model).to(device)
        else:
            assert len(args.transformer_model) > 0 or len(args.yolo_model) > 0
    model.set_backbone(args.unfreeze_epoch == 0)
    if len(args.saved_weights) > 0:
        model.load_state_dict(torch.load(args.saved_weights))
    assert len(args.saved_weights) > 0 or args.num_epochs > 0

    assert args.num_epochs > 0 or args.temporal_epochs > 0
    if args.num_epochs > 0:
        optimizer_adamw = torch.optim.AdamW([
            {'params': model.backbone_parameters(), 'lr': args.backbone_adam_lr, 'weight_decay': args.backbone_weight_decay},
            {'params': model.classifier_parameters(), 'lr': args.classifier_adam_lr, 'weight_decay': args.classifier_weight_decay},
        ])
        optimizer_sgd = torch.optim.SGD([
            {'params': model.backbone_parameters(), 'lr': args.backbone_sgd_lr, 'weight_decay': args.backbone_weight_decay},
            {'params': model.classifier_parameters(), 'lr': args.classifier_sgd_lr, 'weight_decay': args.classifier_weight_decay},
        ], momentum=0.9, nesterov=True)

        if args.use_endoscapes:
            train_mlc_loaders = [cvs_train_mlc_loader, endo_train_mlc_loader, endo_val_mlc_loader, endo_test_mlc_loader]
        elif args.only_endoscapes:
            train_mlc_loaders = [endo_train_mlc_loader]
        else:
            train_mlc_loaders = [cvs_train_mlc_loader]
        if args.only_endoscapes:
            val_mlc_loaders = [endo_val_mlc_loader, endo_test_mlc_loader]
        else:
            val_mlc_loaders = [cvs_val_mlc_loader]

        train_loop(args.num_epochs, model, optimizer_adamw, optimizer_sgd, args.sgd_epoch, args.unfreeze_epoch, num_labels, device, train_mlc_loaders, val_mlc_loaders, args.output_file, ".static_method.pth")
        if len(args.output_file) > 0:
            model.load_state_dict(torch.load(args.output_file + ".static_method.pth"))

    if args.temporal_epochs > 0:
        # eval the hidden weights first
        _, cvs_train_mlc_dataset = getMLCImageLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, height, width, args.mlc_batch_size)
        train_mlc_video = getMLCVideoLoader(cvs_train_mlc_dataset, model, 16, device)
        val_mlc_video = getMLCVideoLoader(cvs_val_mlc_dataset, model, 16, device)
        print("Computed features of video frames")
        #temporal_model = TemporalMLCPredictor(model.num_features, 192, num_labels).to(device)
        temporal_model = TemporalMLCTCN(model.num_features, 128, num_labels, 3).to(device)

        optimizer_adamw = torch.optim.AdamW([
            {'params': temporal_model.parameters(), 'lr': args.classifier_adam_lr, 'weight_decay': args.classifier_weight_decay},
        ])
        optimizer_sgd = torch.optim.SGD([
            {'params': temporal_model.parameters(), 'lr': args.classifier_sgd_lr, 'weight_decay': args.classifier_weight_decay},
        ], momentum=0.9, nesterov=True)

        train_loop(args.temporal_epochs, temporal_model, optimizer_adamw, optimizer_sgd, -1, -1, num_labels, device, [train_mlc_video], [val_mlc_video], args.output_file, ".temporal_model.pth")

if __name__ == "__main__":
    main()
