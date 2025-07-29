import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from config import *
from utils import * 
from torch import optim 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from UNETR import *

# loading data for test
def DataLoad():
    
    print ("Data Loading ... ")
    
    X_train = torch.Tensor(np.load("data/Input_train.npy"))
    X_test = torch.Tensor(np.load("data/Input_test.npy"))
    Y_train = torch.Tensor(np.load("data/output_Temp_train.npy"))
    Y_test = torch.Tensor(np.load("data/output_Temp_test.npy"))


    print ("Preparing Dataset ... ")
    
    # split train & foreseen test data and unforeseen test data
    X_test_foreseen, _, Y_test_foreseen, _ = train_test_split(X_train, Y_train, train_size=1000, shuffle=True)
    
    X_test_foreseen = torch.Tensor(X_test_foreseen)
    X_test_unforeseen = torch.Tensor(X_test)
    
    Y_test_foreseen = torch.Tensor(Y_test_foreseen)
    Y_test_unforeseen = torch.Tensor(Y_test)
    
    dataset = [X_test_foreseen, X_test_unforeseen, Y_test_foreseen, Y_test_unforeseen]
    
    return dataset

# test trained model
def TestModel(model, datasetm):

    with torch.no_grad():
        
        model.eval()
        
        mse = nn.MSELoss()
        
        total_loss_foreseen = []
        total_loss_unforeseen = []
        image_foreseen = []
        image_unforeseen = []
        
        X_test_foreseen = dataset[0]
        X_test_unforeseen = dataset[1]
        Y_test_foreseen = dataset[2]
        Y_test_unforeseen = dataset[3]
        
        for i in tqdm(range(len(X_test_foreseen))):
            
            # test foreseen data
            data = X_test_foreseen[i:i+1].to(device)
            targets = Y_test_foreseen[i:i+1].to(device)
            
            outputs = model(data)
        
            total_loss_foreseen.append(mse(outputs, targets).cpu())
            
            image_foreseen.append(np.array(outputs.detach().cpu()))
            
            # test unforeseen data
            data = X_test_unforeseen[i:i+1].to(device)
            targets = Y_test_unforeseen[i:i+1].to(device)
            
            outputs = model(data)
        

            total_loss_unforeseen.append(mse(outputs, targets).cpu())
            
            image_unforeseen.append(np.array(outputs.detach().cpu()))
        
        
        mse_foreseen = np.mean(total_loss_foreseen)
        mse_unforeseen = np.mean(total_loss_unforeseen)
        print("MSE Loss (foreseen data) : {} % \n".format(mse_foreseen))
        print("MSE Loss (unforeseen data) : {} % \n".format(mse_unforeseen))
    
    return image_foreseen, image_unforeseen

    
 
if __name__ == "__main__":
    
    
    # model
    model = UNETR().to(device)
    model.load_state_dict(torch.load(f'model/UNETR_Temp.pth'), map_location=device)
    
    # test
    dataset = DataLoad()
    image_foreseen, image_unforeseen = TestModel(model, dataset)
    ImagePlot(25, image_unforeseen, dataset[3], 0, 18)
    
    # calculate inference time
    InfernceTime(dataset[0][0:1], model)