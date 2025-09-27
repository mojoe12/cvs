import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
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

    def getFilenames(self):
        return self.image_filenames

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
        image = self.transform(image)
        if self.augment:
            image = self.random_erase(image)
        label, confidence = self.data[filename]
        label = torch.tensor(label, dtype=torch.float32)
        confidence = torch.tensor(confidence, dtype=torch.float32)

        return image, label, confidence

def getMLCImageLoader(num_labels, csv_file, image_path, augment, h, w, batch_size):
    # Transforms
    my_df = pd.read_csv(csv_file)
    labels_confidences_dict = {}

    for _, row in my_df.iterrows():
        # Dynamically create lists of labels and weights based on num_labels
        labels = [row[f'c{i+1}'] for i in range(num_labels)]

        # Check if weight columns exist, default to 1 if not
        weights = []
        for i in range(num_labels):
            col_name = f'weight_c{i+1}'
            weight = row[col_name] if col_name in row and pd.notna(row[col_name]) else 1.0
            weights.append(weight)

        labels_confidences_dict[row['image']] = (labels, weights)

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
    def __init__(self, image_dataset, video_indices):
        self.image_dataset = image_dataset
        self.video_indices = video_indices

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        video_images, video_labels, video_confidences = [], [], []
        for index in self.video_indices[idx]:
            image, label, confidence = self.image_dataset[index]
            video_images.append(image)
            video_labels.append(label)
            video_confidences.append(confidence)
        return torch.stack(video_images), torch.stack(video_labels), torch.stack(video_confidences)

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
        shuffle=True,
        num_workers=4
    )
    return loader

