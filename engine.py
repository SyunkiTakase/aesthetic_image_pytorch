import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

from PIL import Image
from time import time
from tqdm import tqdm
from functools import partial

device = torch.device("cuda")

def train(train_loader, model, softmax, ce, mse, optimizer, scaler, use_amp, alpha, beta):
    model.train()
    sum_loss = 0.0
    
    for img, score in tqdm(train_loader):
        img = img.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        labels = torch.arange(1, 11, dtype=torch.float32).to(device)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(img)
            pred = softmax(logit)
            
            score_mean = torch.sum(score * labels, dim=1)
            score_mean_bucketized = torch.round(score_mean) - 1
            pred_mean = torch.sum(pred * labels, axis=1)

            loss_ce = ce(pred, score_mean_bucketized.long())
            loss_mse = mse(score_mean, pred_mean)
            
            l2 = torch.tensor(0., requires_grad=True) # L2 Regularization
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2
            
            loss = loss_ce + (alpha * loss_mse) + (beta * l2)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_loss += loss.item()

    return sum_loss

def validation(validation_loader, model, softmax, ce, mse, alpha, beta):
    model.eval()
    sum_loss = 0.0
    
    with torch.no_grad():
        for img, score in tqdm(validation_loader):
            img = img.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            labels = torch.arange(1, 11, dtype=torch.float32).to(device)
            
            logit = model(img)
            pred = softmax(logit)
            
            score_mean = torch.sum(score * labels, dim=1)
            score_mean_bucketized = torch.round(score_mean) - 1
            pred_mean = torch.sum(pred * labels, axis=1)

            loss_ce = ce(pred, score_mean_bucketized.long())
            loss_mse = mse(score_mean, pred_mean)
            
            l2 = torch.tensor(0., requires_grad=True) # L2 Regularization
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2
            
            loss = loss_ce + (alpha * loss_mse) + (beta * l2)
            
            sum_loss += loss.item()

        for i in range(img.size(0)): 
            score_gt = score_mean[i].item()
            score_pred = pred_mean[i].item()
            print(f"Image {i+1}: Ground Truth Score = {score_gt}, Predicted Score = {score_pred}")

    return sum_loss

def test(test_loader, model, softmax):
    model.eval()
    
    with torch.no_grad():
        for img, score in tqdm(test_loader):
            img = img.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            labels = torch.arange(1, 11, dtype=torch.float32).to(device)
            
            logit = model(img)
            pred = softmax(logit)
            
            score_mean = torch.sum(score * labels, dim=1)
            score_mean_bucketized = torch.round(score_mean) - 1
            pred_mean = torch.sum(pred * labels, axis=1)

        for i in range(img.size(0)): 
            score_gt = score_mean[i].item()
            score_pred = pred_mean[i].item()
            
            print(f"Image {i+1}: Ground Truth Score = {score_gt}, Predicted Score = {score_pred}")