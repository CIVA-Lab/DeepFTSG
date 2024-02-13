import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as O


# tversky loss
def tverskyLoss(pred, target, weight_map=None, alpha = 0.5, beta = 0.5, smooth = 1.):
    if weight_map is None:
        weight_map = torch.ones_like(pred)
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    TP = (pred * target * weight_map).sum(dim=2).sum(dim=2)    
    FP = ((1-target) * pred * weight_map).sum(dim=2).sum(dim=2)
    FN = (weight_map * target * (1-pred)).sum(dim=2).sum(dim=2)
    
    loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - loss
    
    return loss.mean() 


# calculate overall loss
def calcLoss(pred, target, stats, weight_map=None, bceWeight=0.5):
    if weight_map is None:
        weight_map = torch.ones_like(pred)
        
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    bce =  torch.sum(bce * weight_map) / torch.sum(weight_map)

    pred = torch.sigmoid(pred)
    tversky = tverskyLoss(pred, target, weight_map=weight_map)

    loss = bceWeight * bce  + (1 - bceWeight) * tversky 

    stats['bce'] += bce.data.cpu().numpy() * target.size(0)
    stats['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    stats['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
    

def tverskyLossVISIR(pred, target, alpha = 0.5, beta = 0.5, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    
    TP = (pred * target).sum(dim=2).sum(dim=2)    
    FP = ((1-target) * pred).sum(dim=2).sum(dim=2)
    FN = (target * (1-pred)).sum(dim=2).sum(dim=2)
    
    loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - loss
    
    return loss.mean() 


# calculate overall loss
def calcLossVISIR(pred, target, stats, bceWeight=0.5):

    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    tversky = tverskyLossVISIR(pred, target)

    loss = bce * bceWeight + tversky * (1 - bceWeight)

    stats['bce'] += bce.data.cpu().numpy() * target.size(0)
    stats['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    stats['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
     

def calcLossWithFocal(pred, target, stats, weight_map=None, bceWeight=0.5):
    if weight_map is None:
        weight_map = torch.ones_like(pred)
    
    focalLoss = O.sigmoid_focal_loss(pred, target, alpha = 0.25, gamma = 2, reduction='none')
    focalLoss =  torch.sum(focalLoss * weight_map) / torch.sum(weight_map)
    
    pred = torch.sigmoid(pred)
    tversky = tverskyLoss(pred, target, weight_map=weight_map)

    loss = bceWeight * focalLoss  + (1 - bceWeight) * tversky 

    stats['focal'] += focalLoss.data.cpu().numpy() * target.size(0)
    stats['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    stats['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss
    
# print loss values
def printStats(stats, epochSamples, phase):
   
    outStats = []

    for i in stats.keys():
        outStats.append("{}: {:3f}".format(i, stats[i] / epochSamples))

    print("{}: {}".format(phase, ", ".join(outStats)))
