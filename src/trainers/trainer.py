import copy
import time
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np

from utils import calcLoss, calcLossWithFocal, printStats


class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def trainSSModel(dataloaders, model, optimizer, scheduler, earlyStopNumber=10, numEpochs=40, modelN='DeepFTSG_1', lossFunction = 'bce'):

    # set initial best model weights and loss
    bestModelWeights = copy.deepcopy(model.state_dict())
    bestValLoss = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    early_stopper = EarlyStopper(earlyStopNumber)
    stop_early = False
    
    # iterate thorugh epoches
    for epoch in range(numEpochs):
        print('*' * 20)
        print('Epoch {}/{}'.format(epoch + 1, numEpochs))
        print('*' * 20)

        startTime = time.time()

        # look for training and validation phase in each epoch
        for phase in ['train', 'val']:
            # if phase is training
            if phase == 'train':
                scheduler.step()
                for groupParam in optimizer.param_groups:
                    print("Learning rate:", groupParam['lr'])
                # train model
                model.train()
            else:
                # clean model for validation phase
                model.eval()

            stats = defaultdict(float)
            epochSamples = 0

            for inputs, labels, weight in dataloaders[phase]: #tqdm(dataloaders[phase], desc=phase): # 
                inputs = inputs.to(device)
                labels = labels.to(device)
                weight = weight.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    inputs = inputs.float()
                    outputs = model(inputs)

                    labels = labels.unsqueeze(1)
                    labels = labels.type_as(outputs)
                    
                    # calculate loss
                    if lossFunction == 'bce':
                        loss = calcLoss(outputs, labels, stats, weight_map=weight)
                    else:
                        loss = calcLossWithFocal(outputs, labels, stats, weight_map=weight)

                    # backward and optimize in traning phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epochSamples += inputs.size(0)

                del inputs, labels, weight


            # print statistics
            printStats(stats, epochSamples, phase)

            # calculate epoch loss
            epochLoss = stats['loss'] / epochSamples

            # save the best model
            if phase == 'val' and epochLoss < bestValLoss:
                print("Saving best model")
                bestValLoss = epochLoss
                bestModelWeights = copy.deepcopy(model.state_dict())
                torch.save(bestModelWeights, './models/' + modelN + '.pt')
                
            if phase == 'val' and early_stopper.early_stop(epochLoss):  
                print("We are at epoch:", epoch) 
                stop_early = True      

        finalTime = time.time() - startTime
        print('{:.0f}m {:.0f}s'.format(finalTime // 60, finalTime % 60))
        
        if stop_early:
          print("Early Stopping Called")   
          break

    print('Best validation loss: {:3f}'.format(bestValLoss))
    

def trainMSModel(dataloaders, model, optimizer, scheduler, earlyStopNumber=10, numEpochs=40, modelN='DeepFTSG_2', lossFunction = 'bce'):
    
    # set initial best model weights and loss
    bestModelWeights = copy.deepcopy(model.state_dict())
    bestValLoss = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    early_stopper = EarlyStopper(earlyStopNumber)
    stop_early = False

    # iterate thorugh epoches
    for epoch in range(numEpochs):
        print('*' * 20)
        print('Epoch {}/{}'.format(epoch + 1, numEpochs))
        print('*' * 20)

        startTime = time.time()

        # look for training and validation phase in each epoch
        for phase in ['train', 'val']:
            # if phase is training
            if phase == 'train':
                scheduler.step()
                for groupParam in optimizer.param_groups:
                    print("Learning rate:", groupParam['lr'])
                # train model
                model.train()
            else:
                # clean model for validation phase
                model.eval()

            stats = defaultdict(float)
            epochSamples = 0

            for inputs, inputs2, labels, weight in dataloaders[phase]: # tqdm(dataloaders[phase], desc=phase): # 
                inputs = inputs.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
                weight = weight.to(device)

                # set parameter gradients to zero
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    
                    inputs = inputs.float()
                    inputs2 = inputs2.float()
                    outputs = model(inputs, inputs2)
                    
                    labels = labels.unsqueeze(1)
                    labels = labels.type_as(outputs)
                    
                    # calculate loss
                    if lossFunction == 'bce':
                        loss = calcLoss(outputs, labels, stats, weight_map=weight)
                    else:
                        loss = calcLossWithFocal(outputs, labels, stats, weight_map=weight)

                    # backward and optimize in traning phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epochSamples += inputs.size(0)
                
                del inputs, inputs2, labels, weight
            

            # print statistics
            printStats(stats, epochSamples, phase)

            # calculate epoch loss
            epochLoss = stats['loss'] / epochSamples

            # save the best model
            if phase == 'val' and epochLoss < bestValLoss:
                print("Saving best model")
                bestValLoss = epochLoss
                bestModelWeights = copy.deepcopy(model.state_dict())
                torch.save(bestModelWeights, './models/' + modelN + '.pt')
                
            if phase == 'val' and early_stopper.early_stop(epochLoss):  
                print("Stop at epoch:", epoch) 
                stop_early = True      

        finalTime = time.time() - startTime
        print('{:.0f}m {:.0f}s'.format(finalTime // 60, finalTime % 60))
        
        if stop_early:
          print("Early Stopping Called")   
          break

        finalTime = time.time() - startTime
        print('{:.0f}m {:.0f}s'.format(finalTime // 60, finalTime % 60))

    print('Best validation loss: {:3f}'.format(bestValLoss))
