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
import json
import time

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
    def __init__(self, image_dir, image_json, height, width, pad_and_not_crop):
        self.image_dir = image_dir

        with open(image_json, 'r') as f:
            data = json.load(f)

        self.image_filenames = [img["file_name"] for img in data["images"]]
        square_transform = PadToSquareHeight() if pad_and_not_crop else CropToSquareHeight()
        self.transform = transforms.Compose([
            square_transform,
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

    def getFilenames(self):
        return self.image_filenames

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, filename

def getMLCImageLoader(image_path, image_json, h, w, batch_size):
    # Create dataset instance
    dataset = MultiLabelImageDataset(image_path, image_json, h, w, pad_and_not_crop=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return loader, dataset

class MultiLabelVideoDataset(Dataset):
    def __init__(self, image_dataset, video_indices):
        self.image_dataset = image_dataset
        self.video_indices = video_indices

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        video_images, video_filenames = [], []
        for index in self.video_indices[idx]:
            image, filename = self.image_dataset[index]
            video_images.append(image)
            video_filenames.append(filename)
        return torch.stack(video_images), video_filenames

def getMLCVideoLoader(image_dataset, batch_size, device):
    image_filenames = image_dataset.getFilenames()
    video_frames = {}
    for index, filename in enumerate(image_filenames):
        basename = os.path.splitext(os.path.basename(filename))[0]
        video, frame_str = basename.rsplit('_', 1)
        frame = int(frame_str)
        if video not in video_frames:
            video_frames[video] = {}
        video_frames[video][frame] = index
    video_indices = []
    for video_name, frames in video_frames.items():
        video_indices.append(list(frames.values()))

    # Create dataset instance
    dataset = MultiLabelVideoDataset(image_dataset, video_indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
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
    def __init__(self, model, hidden_dim, num_labels, num_layers=2, num_heads=4):
        super().__init__()
        self.static_model = model
        self.projection = nn.Linear(model.num_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):  # x: [B, 18, 3, 384, 384]
        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.static_model(x_reshaped, return_hidden=True).view(x.size(0), x.size(1), -1) # [B, 18, hidden_dim]
        x = self.projection(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch, hidden_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to [B, 18, hidden_dim]
        out = self.classifier(x)  # [B, 18, num_labels]
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
    def __init__(self, model, hidden_dim, num_labels, num_blocks):
        super().__init__()
        self.static_model = model
        self.input_proj = nn.Conv1d(model.num_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dilation=2**i) for i in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, num_labels, kernel_size=1)

    def forward(self, x):  # x: [B, 18, 3, 384, 384]
        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.static_model(x_reshaped, return_hidden=True).view(x.size(0), x.size(1), -1)
        x = x.transpose(1, 2)         # [B, 1536, 18] → [B, C, T]
        x = self.input_proj(x)        # [B, hidden_dim, 18]
        x = self.blocks(x)            # temporal modeling
        x = self.output_proj(x)      # [B, 3, 18]
        x = x.transpose(1, 2)         # [B, 18, 3]
        return x

class TemporalMLCLSTM(nn.Module):
    def __init__(self, model, hidden_dim=256, num_labels=3, num_layers=1, bidirectional=False):
        super().__init__()
        self.static_model = model
        self.lstm = nn.LSTM(input_size=model.num_features,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)

    def forward(self, x):  # x: [B, 18, 3, 384, 384]
        # 1. Select fewer frames for the backbone
        useful_indices = torch.tensor([1, 3, 5, 7, 15], device=x.device)
        x_subset = x[:, useful_indices]  # [B, 5, 3, H, W]

        x_reshaped = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        feats = self.static_model(x_reshaped, return_hidden=True)

        feats = feats.view(x.size(0), x.size(1), -1)
        
        # 3. LSTM over reduced time steps
        lstm_out, _ = self.lstm(feats)  # [B, 5, hidden]
        
        # 4. Interpolate back to 18 steps
        lstm_out_interp = F.interpolate(
            lstm_out.transpose(1, 2),  # [B, hidden, 5]
            size=18,                    # 18
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # [B, 18, hidden]
        
        # 5. Classify
        out = self.classifier(lstm_out_interp)  # [B, 18, num_labels]
        return out

def evalModel(model, dataloader, device, output_file):
    model.eval()
    results = []

    start_time = time.time()  # Start timing
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)  # [batch_size, 3]
            probs = torch.sigmoid(outputs).cpu().numpy()

            for filename_index in range(len(filenames)):
                filename_list = filenames[filename_index]
                for filename, prob in zip(filename_list, probs[:, filename_index]):
                    pred = [int(p >= 0.5) for p in prob]
                    results.append({
                        "file_name": filename,
                        "pred_ds_prob": prob.tolist(),
                        "pred_ds": pred
                    })

    end_time = time.time()  # End timing
    elapsed = (end_time - start_time) * 1.0 / len(results)
    print(f"Inference took {elapsed:.2f} seconds per operation")

    final_json = {"images": results}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_json, f, indent=4)

    print(f"Saved predictions to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--transformer_model', type=str, default="", help='Path to Transformer model specification')
    parser.add_argument('--yolo_model', type=str, default="", help='Path to YOLO model specification')
    parser.add_argument('--saved_weights', type=str, required=True, help='Path to file representing model weights')
    parser.add_argument('--image_size', type=int, required=True, help='Image Size. Depends on model')
    parser.add_argument('--mlc_batch_size', type=int, default=32 if torch.cuda.is_available() else 1,
                        help='Batch size for multi-label classification')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input frames')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input json listing useful frames')
    parser.add_argument('--output_json', type=str, required=True, help='Path to output file for writing outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    height, width = args.image_size, args.image_size
    print(f"Batch size: {args.mlc_batch_size}, Image size: {height}x{width}")
    cvs_val_mlc_loader, cvs_val_mlc_dataset = getMLCImageLoader(args.input_dir, args.input_json, height, width, args.mlc_batch_size)
    num_labels = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
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
    model.set_backbone(False)
    temporal_model = TemporalMLCLSTM(model, 128, num_labels, 3).to(device)
    assert len(args.saved_weights) > 0
    print(f"Loading model weights from {args.saved_weights}")
    temporal_model.load_state_dict(torch.load(args.saved_weights, map_location=device))

    val_mlc_video = getMLCVideoLoader(cvs_val_mlc_dataset, args.mlc_batch_size, device)
    evalModel(temporal_model, val_mlc_video, device, args.output_json)

if __name__ == "__main__":
    print(os.cpu_count())
    torch.set_num_threads(os.cpu_count())  # Or use os.cpu_count()
    torch.set_num_interop_threads(os.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    main()
