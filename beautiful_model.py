import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

from functools import partial
from time import time
from tqdm import tqdm
import argparse
import numpy as np
import os

import trainer
import data_loader
import make_graph

import timm
from timm.models import create_model

device = torch.device("cuda")

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def main():
    epoch = 3
    batch_size = 1
    lr = 1e-3
    input_image_size = 224
    use_amp = True
    image_dir = './images'
    train_csv = './AVA_train.csv'
    validation_csv = './AVA_validation.csv'

    model = create_model("efficientnet_b0", pretrained=True, num_classes=10) 
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), 
        nn.Linear(1280, 10),
        )
    model.to(device)
    softmax = nn.Softmax().to(device)

    # print(model)
    # print("model parameters: ", num_parameters(model))

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    
    train_loader = data_loader.get_data_loader(train_csv, image_dir, True, batch_size)
    test_loader = data_loader.get_data_loader(validation_csv, image_dir, False, batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epoch):
        train_loss = trainer.train(train_loader, model, softmax, ce, mse, optimizer, scaler, use_amp, epoch)
        test_loss = trainer.test(test_loader, model, softmax, ce, mse)

        train_loss = (train_loss/len(train_loader))
        test_loss = (test_loss/len(test_loader))

        print(f"epoch: {epoch+1},\
                train loss: {train_loss},\
                test loss: {test_loss}")

if __name__=='__main__':

    main()
