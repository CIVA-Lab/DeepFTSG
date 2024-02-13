import torch
import numpy as np
import glob
import os
import pandas as pd
import imageio
import torch.nn.functional as F
import time
import torch.hub

from datetime import timedelta

from data.dataLoader_test import DeepFTSG_1_TestDataLoader
from nets.DeepFTSG_1 import DeepFTSG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

modelName = 'DeepFTSG_1'

# network input size
imgWidth = 480
imgHeight = 320

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

# load trained model
model.load_state_dict(torch.load('./models/' + modelName + '.pt'))
model.eval() 

# get video path from Flist.txt
fileName = './files/SBI.txt'
df = pd.read_csv(fileName, names=['filename'])
nameList = df['filename']

sbiInputExt = "/*.png"

for videoPath in nameList:

    # start timer 
    startTime = time.time()

    print(videoPath)

    if "CAVIAR2" in videoPath:
        sbiInputExt = "/*.jpg"
        print("***** Found CAVIAR2 *****")
    else:
        sbiInputExt = "/*.png"

    # path to the test image
    # change folder name and extension according to your test images
    folderData = sorted(glob.glob("./datasets/test/SBI2015/DATASET/" + videoPath + "input" + sbiInputExt))
    folderBgSub = sorted(glob.glob("./datasets/test/SBI2015/BGS/" + videoPath + "/*.png"))
    folderFlux = sorted(glob.glob("./datasets/test/SBI2015/FLUX/" + videoPath + "/*.png"))

    print(len(folderData))

    startIndex = len(folderData) - len(folderBgSub)
    print(startIndex)

    folderData = folderData[startIndex:]
    print(len(folderData))

    print(folderData[0])
    print(folderBgSub[0])
    print(folderFlux[0])

    print(folderData[-1])
    print(folderBgSub[-1])
    print(folderFlux[-1])

    testDataset = DeepFTSG_1_TestDataLoader(folderData, folderBgSub, folderFlux, img_size=(imgHeight, imgWidth))
    print(len(testDataset))

    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)

    # set mask path
    maskDir = os.path.join('./output/SBI2015/' + modelName + '/', videoPath)
    # create path if not exist
    if not os.path.exists(maskDir):
            os.makedirs(maskDir)

    for i, (inputs) in enumerate(testLoader):
        
        inputs = inputs.to(device)
        inputs = inputs.float()
       
        # Predict
        pred = model(inputs)
        
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        
        outPred = pred[0].squeeze()
        outPredNorm = 255 * outPred
        outPredUint8 = outPredNorm.astype(np.uint8)

        # get frame name from original frame and replace in with bin and extension of jpg to png
        # change accordingly replace functions for your test inputs
        fname = os.path.basename(folderData[i]).replace('in','bin').replace('jpg','png')

        print(maskDir + fname)
        
        imageio.imwrite(maskDir + fname, outPredUint8)

    finalTime = time.time() - startTime
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(finalTime))
    print(msg)




