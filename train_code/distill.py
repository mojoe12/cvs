import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from torchmetrics import AveragePrecision
import numpy as np
import random
import argparse
from losses import AsymmetricLoss, FocalLoss
from loaders import PadToSquareHeight, CropToSquareHeight, MultiLabelImageDataset, getMLCImageLoader, MultiLabelVideoDataset, getMLCVideoLoader
from models import TimmMLCModel, TemporalMLCPredictor, TemporalMLCTCN, TemporalMLCLSTM

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
    elif x.dim() == 5:  # Video: (B, T, C, H, W)
        _, T, C, H, W = x.size()
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x[:, :, :, y1:y2, x1:x2] = x[index, :, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        raise ValueError(f'Unsupported input dimension {x.dim()}, expected 3D or 4D or 5D tensor.')
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def trainStep(model, teacher_model, dataloader, criterion, optimizer, device, max_norm):
    model.train()
    total_loss = 0
    for images, timages, labels, confidences in dataloader:
        images = images.to(device)
        timages = timages.to(device)
        labels = labels.to(device)
        confidences = confidences.to(device)
        # Apply MixUp or CutMix
        if random.random() < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            timages, _, _, _ = mixup_data(timages, labels, alpha=0.4)  # same lam
        else:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=0.4)
            timages, _, _, _ = cutmix_data(timages, labels, alpha=0.4)  # same lam

        with torch.no_grad():
            teacher_outputs = teacher_model(timages)

        optimizer.zero_grad()
        outputs = model(images)

        # Hard loss
        loss_raw = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        hard_loss = (loss_raw * confidences).mean()

        # Soft loss
        temperature = 4.0
        soft_loss = F.kl_div(
            F.log_softmax(outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1),
            reduction="batchmean"
        ) * (temperature ** 2)

        loss = 5 * soft_loss + 0.5 * hard_loss

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
        for images, timages, labels, confidences in dataloader:
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

def train_loop(num_epochs, model, teacher_model, optimizer_adamw, optimizer_sgd, sgd_epoch, unfreeze_epoch, num_labels, device, train_mlc_loaders, val_mlc_loaders, output_file, output_file_ext):
    # ---- Optimizer and Loss ----
    criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
    precision = AveragePrecision(task='multilabel', num_labels=num_labels, average='none')
    max_norm = 5
    num_warmup = 5
    scheduler_adamw = optim.lr_scheduler.SequentialLR(
        optimizer_adamw,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer_adamw, start_factor=1e-2, total_iters=num_warmup),
            optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=num_epochs-num_warmup)
        ],
        milestones=[num_warmup]
    )
    scheduler_sgd = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sgd,
        mode='min',          # minimize val_loss
        factor=0.5,          # reduce LR by this factor
        patience=2,          # wait N epochs before reducing
        min_lr=1e-6          # don’t reduce below this
    )
    optimizer = optimizer_adamw
    best_avg_precision = 0.

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
            train_loss = trainStep(model, teacher_model, train_mlc_loader, criterion, optimizer, device, max_norm)
            print(f"Dataset {d} CVS Classification Train Loss: {train_loss:.4f}")

        for d, val_mlc_loader in enumerate(val_mlc_loaders):
            precision.reset()
            val_mlc_loss = validateMLCStep(model, val_mlc_loader, criterion, device, precision)
            map_score = precision.compute().cpu() # Tensor of size [num_labels] or scalar if average="macro"
            print(f"Dataset {d} CVS Classification Validation Loss: {val_mlc_loss:.4f}, Average Precision is {map_score.mean():.4f}: {map_score}")

            if d == 0 and map_score.mean() > best_avg_precision and len(output_file) > 0:
                best_avg_precision = map_score.mean()
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

    parser.add_argument('--teacher_model', type=str, required=True, help='Path to Timm model specification')
    parser.add_argument('--student_model', type=str, required=True, help='Path to Timm model specification')
    parser.add_argument('--saved_weights', type=str, required=True, help='Path to file representing model weights')
    parser.add_argument('--teacher_image_size', type=int, required=True, help='Image Size. Depends on model')
    parser.add_argument('--student_image_size', type=int, required=True, help='Image Size. Depends on model')
    parser.add_argument('--num_epochs', type=int, required=True, help='Total number of static training epochs')
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
    height, width = args.student_image_size, args.teacher_image_size
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
    cvs_train_mlc_loader, cvs_train_mlc_dataset = getMLCImageLoader(train_mlc_data_csv, 'sages_cvs_challenge_2025/frames', True, height, width, args.mlc_batch_size)
    cvs_val_mlc_loader, cvs_val_mlc_dataset = getMLCImageLoader('analysis/val_mlc_data.csv', 'sages_cvs_challenge_2025/frames', False, height, width, args.mlc_batch_size)

    num_labels = 3
    #num_classes = 7 # irrelevant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = TimmMLCModel(num_labels, args.teacher_model).to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False
    assert len(args.saved_weights) > 0
    print(f"Loading model weights from {args.saved_weights}")
    teacher_model.load_state_dict(torch.load(args.saved_weights, map_location=device))
    assert args.num_epochs > 0
    model = TimmMLCModel(num_labels, args.student_model).to(device)
    model.set_backbone(args.unfreeze_epoch == 0)

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

        train_loop(args.num_epochs, model, teacher_model, optimizer_adamw, optimizer_sgd, args.sgd_epoch, args.unfreeze_epoch, num_labels, device, train_mlc_loaders, val_mlc_loaders, args.output_file, ".static_model.pth")
        if len(args.output_file) > 0:
            print(f"Loading model weights from {args.output_file}.static_model.pth")
            model.load_state_dict(torch.load(args.output_file + ".static_model.pth", map_location=device))

    if args.temporal_epochs > 0:
        model.set_backbone(False)
        # eval the hidden weights first
        batch_size = max(2, args.mlc_batch_size // 9) # need at least 2 for cutmix. 9 is 18 // 2 for there are 18 frames
        print(f"Using batch size {batch_size} for temporal model training")
        train_mlc_video = getMLCVideoLoader(cvs_train_mlc_dataset, batch_size, device)
        val_mlc_video = getMLCVideoLoader(cvs_val_mlc_dataset, batch_size, device)
        #temporal_model = TemporalMLCPredictor(model.num_features, 192, num_labels).to(device)
        temporal_model = TemporalMLCLSTM(model, 128, num_labels, 3).to(device)

        optimizer_adamw = torch.optim.AdamW([
            {'params': temporal_model.parameters(), 'lr': args.classifier_adam_lr, 'weight_decay': args.classifier_weight_decay},
        ])
        optimizer_sgd = torch.optim.SGD([
            {'params': temporal_model.parameters(), 'lr': args.classifier_sgd_lr, 'weight_decay': args.classifier_weight_decay},
        ], momentum=0.9, nesterov=True)

        train_loop(args.temporal_epochs, temporal_model, optimizer_adamw, optimizer_sgd, -1, -1, num_labels, device, [train_mlc_video], [val_mlc_video], args.output_file, ".temporal_model.pth")

if __name__ == "__main__":
    main()
