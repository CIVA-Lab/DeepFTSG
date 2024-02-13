import glob
import pandas as pd
import torch
import torch.hub
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary

from data.dataLoader_ss import (DeepFTSG_1_PathLoader, DeepFTSG_1_TupleLoader)
from nets.DeepFTSG_1 import DeepFTSG
from trainers.trainer import trainSSModel

# inputs and labels path
# put your input and label inside data folder
# change extensions accordingly

modelName = 'DeepFTSG_1'
print('**************************')
print(modelName)
print('**************************')

# network input size
imgWidth = 480
imgHeight = 320

# get video path from Flist.txt
fileName = './files/CD2014.txt'
df = pd.read_csv(fileName, names=['filename'])
nameListTrain = df['filename']

# train 90% and validation 10%
lengths = [int(200*0.9), int(200*0.1)]

trainDataset = []
valDataset =[]

for videoPath in nameListTrain:

    print(videoPath)

    folderData = sorted(glob.glob("./datasets/train/CD2014/INPUTS/" + videoPath + "/*.jpg"))
    folderBgSub = sorted(glob.glob("./datasets/train/CD2014/BGS/" + videoPath + "/*.png"))
    folderFlux = sorted(glob.glob("./datasets/train/CD2014/FLUX/" + videoPath + "/*.png"))
    folderMask = sorted(glob.glob("./datasets/train/CD2014/GT/" + videoPath + "/*.png")) 
    
    print(folderData[0])
    print(folderBgSub[0])
    print(folderFlux[0])
    print(folderMask[0])

    print(folderData[-1])
    print(folderBgSub[-1])
    print(folderFlux[-1])
    print(folderMask[-1])

    dataset = DeepFTSG_1_PathLoader(folderData, folderBgSub, folderFlux, folderMask)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

    trainDataset.extend(train_dataset)
    valDataset.extend(val_dataset)

    del dataset
    del train_dataset
    del val_dataset

print(len(trainDataset))
print(len(valDataset))

train_data = DeepFTSG_1_TupleLoader(trainDataset, img_size=(imgHeight, imgWidth))
val_data = DeepFTSG_1_TupleLoader(valDataset, img_size=(imgHeight, imgWidth))

print(len(train_data))
print(len(val_data))

# set batch size
batchSize = 16

trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_data, batch_size=batchSize, shuffle=True)

dataLoaders = {
    'train': trainLoader,
    'val': valLoader
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pretrained se-resnet50 on ImageNet
se_resnet_hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)

se_resnet_base_layers = [] 

# traverse each element of hub model and add it to a list
for name, m in se_resnet_hub_model.named_children():
    se_resnet_base_layers.append(m)  

numClass = 1
model = DeepFTSG(numClass, se_resnet_base_layers).to(device)

# summarize the network
summary(model, [(3, imgHeight, imgWidth)])

# using Adam optimizer with learning rate 1e-4
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# decrease learning rate by 0.5 after each 40th epoch
lrScheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
# train model with 120 epoch
trainSSModel(dataLoaders, model, optimizer, lrScheduler, earlyStopNumber=20, numEpochs=120, modelN = modelName, lossFunction = 'bce')
