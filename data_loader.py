import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
        scores = torch.tensor(self.data.iloc[idx, 2:12].values.astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, scores

# Transformation for image preprocessing
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Define dataset and dataloader
def get_data_loader(csv_path, image_dir, is_training, batch_size):
    dataset = CustomDataset(csv_path, image_dir, transform=data_transform, is_training=is_training)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    return data_loader