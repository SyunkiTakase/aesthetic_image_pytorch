import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from functools import partial
from time import time
from tqdm import tqdm
import numpy as np
from PIL import Image

device = torch.device("cuda")

def train(train_loader, model, softmax, ce, mse, optimizer, scaler, use_amp, epoch):
    model.train()
    sum_loss = 0.0
    count = 0
    alpha = 1
    beta=1e-5
    
    for img,score in tqdm(train_loader):
        img = img.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        labels = torch.arange(1, 11, dtype=torch.float32).to(device)
        
        # print('img')
        # print(img.shape)
        # print('score')
        # print(score.shape)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(img)
            loss_ce = ce(logit, score)

            pred = softmax(logit)
            pred = torch.sum(pred * labels, axis=1)
            score_mean = torch.sum(score * labels, axis=1)
            loss_mse = mse(score_mean, pred)
            
            l2 = torch.tensor(0., requires_grad=True)
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2
            
            loss = loss_ce + loss_mse + (beta * l2)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()

    return sum_loss

def test(test_loader, model, softmax, ce, mse):
    model.eval()
    sum_loss = 0.0
    count = 0
    alpha = 1
    beta=1e-5
    
    with torch.no_grad():
        for img, score in tqdm(test_loader):
            img = img.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            labels = torch.arange(1, 11, dtype=torch.float32).to(device)
            
            logit = model(img)
            loss_ce = ce(logit, score)

            pred = softmax(logit)
            pred = torch.sum(pred * labels, axis=1)
            score_mean = torch.sum(score * labels, axis=1)
            loss_mse = mse(score_mean, pred)
        
            l2 = torch.tensor(0., requires_grad=True)
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2
            
            loss = loss_ce + loss_mse + (beta * l2)
            
            sum_loss += loss.item()

    return sum_loss