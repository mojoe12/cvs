import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, average_precision_score
from torchvision.ops import FeaturePyramidNetwork
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.transform as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import random

def pad_to_square_height(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    if h >= w:
        return mask  # already square or portrait
    pad_total = w - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return np.pad(mask, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

class PadToSquareHeight:
    def __call__(self, img):
        w, h = img.size
        if h >= w:
            return img  # already tall or square
        pad = (w - h) // 2
        padding = (0, pad, 0, w - h - pad)  # left, top, right, bottom
        return TF.pad(img, padding, fill=0, padding_mode='constant')

class CocoSegmentationDataset(Dataset):
    def __init__(self, annotation_file, root_dir, height, width, augment):
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.image_ids = list(self.coco.imgs.keys())
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
            mask_seg = pad_to_square_height(mask_seg)
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

def getSegmentationLoader(json_file, image_dir, augment, h, w, batch_size):
    # Create dataset instance
    dataset = CocoSegmentationDataset(json_file, image_dir, h, w, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return loader

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

# ----------------------------
# Attention Gate Module
# ----------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ----------------------------
# Modified Multi-Task Model with Attention Gating
# ----------------------------
class EfficientNetMultiHead(nn.Module):
    def __init__(self, num_labels, num_classes):
        super(EfficientNetMultiHead, self).__init__()
        base = models.efficientnet_b0(pretrained=True)
        self.backbone = nn.Sequential(*list(base.features.children()))

        self.attn_classification = AttentionGate(F_g=1280, F_l=1280, F_int=256)
        self.attn_segmentation = AttentionGate(F_g=1280, F_l=1280, F_int=256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, num_classes, kernel_size=1)
         )

    def forward(self, x, task):
        features = self.backbone(x)

        if task == 'classification':
            gated = self.attn_classification(features, features)
            pooled = self.pool(gated)
            return self.classifier(pooled)

        elif task == 'segmentation':
            gated = self.attn_segmentation(features, features)
            return self.segmentation_head(gated)

        else:
            raise ValueError("Task must be either 'classification' or 'segmentation'")

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
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

def logSegmentationResults(log_dir, val_seg_loader, model, device):
    os.makedirs(log_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_seg_loader):
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

def train_loop(num_epochs, model, optimizer, num_classes, unfreeze_epoch, device, train_seg_loaders, val_seg_loaders, train_mlc_loaders, val_mlc_loaders):
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

        print(f"Epoch {epoch+1}/{num_epochs}")
        for d, train_seg_loader in enumerate(train_seg_loaders):
            train_seg_loss = trainSegmentationStep(model, train_seg_loader, seg_criterion, optimizer, device)
            print(f"Dataset {d} Segmentation Train Loss: {train_seg_loss:.4f}")

        for d, train_mlc_loader in enumerate(train_mlc_loaders):
            train_mlc_loss = trainMLCStep(model, train_mlc_loader, mlc_criterion, optimizer, device)
            print(f"Dataset {d} CVS Classification Train Loss: {train_mlc_loss:.4f}")

        for d, val_seg_loader in enumerate(val_seg_loaders):
            val_seg_loss = validateSegmentationStep(model, val_seg_loader, seg_criterion, device)
            print(f"Dataset {d} Segmentation Validation Loss: {val_seg_loss:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            val_mlc_loss, val_f1, val_ap = validateMLCStep(model, val_mlc_loader, mlc_criterion, device)
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, F1: {val_f1:.4f}")
            for i, ap in enumerate(val_ap):
                print(f"Average Precision for c{i+1}: {ap:.4f}")

def main():
    seg_batch_size, mlc_batch_size = 20, 20
    h, w = 240, 240
    endo_train_seg_loader = getSegmentationLoader('endoscapes/train_seg/annotation_coco.json', 'endoscapes/train_seg', True, h, w, seg_batch_size)
    endo_val_seg_loader = getSegmentationLoader('endoscapes/val_seg/annotation_coco.json', 'endoscapes/val_seg', False, h, w, seg_batch_size)
    endo_test_seg_loader = getSegmentationLoader('endoscapes/test_seg/annotation_coco.json', 'endoscapes/test_seg', False, h, w, seg_batch_size)
    cvs_train_seg_loader = getSegmentationLoader('analysis/instances_train.json', 'sages_cvs_challenge_2025/segmentation_visualization/', True, h, w, seg_batch_size)
    cvs_val_seg_loader = getSegmentationLoader('analysis/instances_val.json', 'sages_cvs_challenge_2025/segmentation_visualization/', False, h, w, seg_batch_size)
    endo_train_mlc_loader = getMLCLoader('analysis/endo_train_mlc_data.csv', 'endoscapes/train', True, h, w, mlc_batch_size)
    endo_val_mlc_loader = getMLCLoader('analysis/endo_val_mlc_data.csv', 'endoscapes/val', False, h, w, mlc_batch_size)
    endo_test_mlc_loader = getMLCLoader('analysis/endo_test_mlc_data.csv', 'endoscapes/test', False, h, w, mlc_batch_size)
    cvs_train_mlc_loader = getMLCLoader('analysis/train_mlc_data.csv', 'sages_cvs_challenge_2025/frames', True, h, w, mlc_batch_size)
    cvs_val_mlc_loader = getMLCLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, h, w, mlc_batch_size)

    num_labels = 3
    num_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetMultiHead(num_labels=num_labels, num_classes=num_classes).to(device)
    num_epochs = 50
    unfreeze_epoch = 5
    optimizer = torch.optim.SGD([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ], momentum=0.9, nesterov=True)
    train_seg_loaders = [cvs_train_seg_loader, endo_train_seg_loader, endo_val_seg_loader, endo_test_seg_loader]
    val_seg_loaders = [cvs_val_seg_loader]
    train_mlc_loaders = []
    val_mlc_loaders = []
    train_loop(num_epochs, model, optimizer, num_classes, unfreeze_epoch, device, train_seg_loaders, val_seg_loaders, train_mlc_loaders, val_mlc_loaders)
    logSegmentationResults('segmentation_results', endo_val_seg_loader, model, device)

if __name__ == "__main__":
    main()
