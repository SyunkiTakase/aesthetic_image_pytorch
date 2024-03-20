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
train_losses = []
validation_losses = []
save_path = './model/'

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def main(args):
    alpha = args.alpha
    beta = args.beta
    use_amp = args.amp

    # Setting Dataset
    train_loader = data_loader.get_data_loader(args.train_csv, args.img_dir, True, args.batch_size)
    validation_loader = data_loader.get_data_loader(args.validation_csv, args.img_dir, False, args.batch_size)

    # Setting Model
    model = create_model("efficientnet_b0", pretrained=True, num_classes=10) 
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), 
        nn.Linear(1280, 10),
        )
    model.to(device)
    softmax = nn.Softmax().to(device)

    print(model)
    print("model parameters: ", num_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Setting Loss
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Training & Validation Loop
    for epoch in range(args.epoch):
        train_loss = trainer.train(train_loader, model, softmax, ce, mse, optimizer, scaler, use_amp, alpha, beta)
        validation_loss = trainer.test(validation_loader, model, softmax, ce, mse, alpha, beta)

        train_loss = (train_loss/len(train_loader))
        validation_loss = (validation_loss/len(validation_loader))

        print(f"epoch: {epoch+1},\
                train loss: {train_loss},\
                validation loss: {validation_loss}")

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        save_model_path = os.path.join(save_path + 'weights/',"{}.tar".format(epoch + 1))
        torch.save({
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch
            },save_model_path)
        
        make_graph.draw_loss_graph(train_losses, validation_losses)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--img_dir", type=str, default='./images')
    parser.add_argument("--train_csv", type=str, default='./AVA_train.csv')
    parser.add_argument("--validation_csv", type=str, default='./AVA_validation.csv')
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--beta", type=int, default=1e-5)
    parser.add_argument('--amp', action='store_true')
    args=parser.parse_args()
    main(args)
