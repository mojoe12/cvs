import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
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

def compute_iou(pred_mask, true_mask, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        intersection = (pred_cls & true_cls).sum().item()
        union = (pred_cls | true_cls).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Will be ignored in mAP
        else:
            ious.append(intersection / union)
    return ious

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

class EfficientNetMultiHead(nn.Module):
    def __init__(self, num_labels, num_classes):
        super().__init__()

        self.features = deeplabv3_mobilenet_v3_large(pretrained=True)
        #self.features.backbone = base.features #OPTIONAL
        seg_num_backbone_features = 960
        self.features.classifier = DeepLabHead(seg_num_backbone_features, num_classes)
        base = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Sequential(
            *list(base.features.children()),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_labels),
        )

    def forward(self, x, task):
        seg = self.features(x)["out"]
        if task == 'classification':
            softmax = torch.softmax(seg, dim=1)
            sharpness = 50
            mask = torch.sigmoid((1 - softmax[:, 0, :, :]) * sharpness)
            return self.classifier(x * torch.unsqueeze(mask, 1))
        elif task == 'segmentation':
            return seg
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

def validateSegmentationStep(model, dataloader, criterion, device, num_classes, iou_threshold=0.5):
    model.eval()
    val_loss = 0.0
    all_ap = [[] for _ in range(num_classes)]  # AP for each class
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, 'segmentation')  # (B, C, H, W)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            for pred, target in zip(preds, masks):
                pred = pred.cpu()
                target = target.cpu()

                ious = compute_iou(pred, target, num_classes)
                for cls in range(num_classes):
                    iou = ious[cls]
                    if not np.isnan(iou):
                        all_ap[cls].append(iou >= iou_threshold)
    # Compute AP per class
    ap_per_class = []
    for cls_iou_matches in all_ap:
        if len(cls_iou_matches) == 0:
            ap = float('nan')
        else:
            precision = np.mean(cls_iou_matches)
            ap = precision  # With binary TP/FP and thresholded IoU
        ap_per_class.append(ap)

    # Filter NaNs and compute mAP
    valid_aps = [ap for ap in ap_per_class if not np.isnan(ap)]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    return val_loss / len(dataloader), mAP

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
    for param in model.features.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.features.backbone.parameters():
        param.requires_grad = True

def logSegmentationResults(log_dir, val_seg_loader, model, device, num_classes):
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

def train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_seg_loaders, val_seg_loaders, train_mlc_loaders, val_mlc_loaders):
    # ---- Optimizer and Loss ----
    mlc_criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep per-label loss
    seg_criterion = torch.nn.CrossEntropyLoss()

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
        for d, train_seg_loader in enumerate(train_seg_loaders):
            train_seg_loss = trainSegmentationStep(model, train_seg_loader, seg_criterion, optimizer, device)
            print(f"Dataset {d} Segmentation Train Loss: {train_seg_loss:.4f}")

        for d, train_mlc_loader in enumerate(train_mlc_loaders):
            train_mlc_loss = trainMLCStep(model, train_mlc_loader, mlc_criterion, optimizer, device)
            print(f"Dataset {d} CVS Classification Train Loss: {train_mlc_loss:.4f}")

        for d, val_seg_loader in enumerate(val_seg_loaders):
            val_seg_loss, val_mAP = validateSegmentationStep(model, val_seg_loader, seg_criterion, device, num_classes)
            print(f"Dataset {d} Segmentation Validation Loss: {val_seg_loss:.4f}, mAP: {val_mAP:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            val_mlc_loss, val_f1, val_ap = validateMLCStep(model, val_mlc_loader, mlc_criterion, device)
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, F1: {val_f1:.4f}, Average Precisions:", end='')
            for ap in val_ap:
                print(f" {ap:.4f}", end='')
            print("")

def main():
    seg_batch_size, mlc_batch_size = 32, 32
    h, w = 224, 224
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
    unfreeze_epoch = 10
    sgd_epoch = 20

    # Define AdamW optimizer (for first 20 epochs)
    optimizer_adamw = torch.optim.AdamW([
        {'params': model.features.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
        #{'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.features.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    ])

    # Define SGD optimizer (to be used after 20 epochs)
    optimizer_sgd = torch.optim.SGD([
        {'params': model.features.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-9},
        #{'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': model.features.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-9},
    ], momentum=0.9, nesterov=True)

    train_seg_loaders = [cvs_train_seg_loader, endo_train_seg_loader, endo_val_seg_loader, endo_test_seg_loader]
    val_seg_loaders = [cvs_val_seg_loader]
    train_mlc_loaders = []
    val_mlc_loaders = []
    train_loop(num_epochs, model, optimizer_adamw, optimizer_sgd, sgd_epoch, num_classes, unfreeze_epoch, device, train_seg_loaders, val_seg_loaders, train_mlc_loaders, val_mlc_loaders)
    logSegmentationResults('cvs_segmentation_results', cvs_val_seg_loader, model, device, num_classes)

if __name__ == "__main__":
    main()
