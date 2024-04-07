import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, is_training=True):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training

        if self.is_training:
            self.data = self.data.sample(frac=1, random_state=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.data.iloc[idx, 1]}.jpg")
        image = Image.open(img_name)
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scores = torch.tensor(self.data.iloc[idx, 2:12].values.astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, scores

# Train & Validation, Test Transformation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize
    transforms.RandomHorizontalFlip(),  # Random Horizontal Hlip
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize
    transforms.ToTensor(),
])

# Define Dataset and Dataloader
def get_data_loader(csv_path, image_dir, is_training, batch_size):
    if is_training == True:
        dataset = CustomDataset(csv_path, image_dir, transform=train_transform, is_training=is_training)
    
    else:
        dataset = CustomDataset(csv_path, image_dir, transform=test_transform, is_training=is_training)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    
    return data_loader