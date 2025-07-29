import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

from config import *
from utils import *
from tqdm import tqdm 
from torch import optim 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from UNETR import *


# training data load
def DataLoad(batch):
    
    print ("Data Loading ... ")
    
    X_train = torch.Tensor(np.load("data/Input_train.npy"))
    X_valid = torch.Tensor(np.load("data/Input_valid.npy"))
    Y_train = torch.Tensor(np.load("data/output_Temp_train.npy"))
    Y_valid = torch.Tensor(np.load("data/output_Temp_valid.npy"))
    
    print ("Preparing Dataset ... ")

    
    ds_train = TensorDataset(X_train, Y_train)
    ds_valid = TensorDataset(X_valid, Y_valid)
    
    loader_train = DataLoader(ds_train, batch_size = batch, shuffle = True)
    loader_valid = DataLoader(ds_valid, batch_size = batch, shuffle = False)
    
    return loader_train, loader_valid


def TrainModel(model, learning_rate, epoch_num, device, loader_train, loader_valid, patience=20):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    temploss = nn.MSELoss()
    
    train_loss_list = []
    valid_loss_list = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epoch_num):
        
        train_loss = 0
        valid_loss = 0
        
        model.train()
        with tqdm(loader_train, desc=f"Epoch {epoch+1}/{epoch_num}") as tepoch :
            start_time = time.time()
            
            for data, target in tepoch:
                data = data.to(device) 
                target = target.to(device) 
                optimizer.zero_grad()
                outputs = model(data)
                loss = temploss(target, outputs) 
                loss.backward() 
                optimizer.step()
                train_loss += loss.item()
        
            train_loss_mean = train_loss / len(loader_train)
            print(f"{epoch+1} epoch Train Loss : {train_loss_mean:.6f}")
            train_loss_list.append(train_loss_mean)
        
        model.eval()
        with torch.no_grad():
            for data, target in loader_valid:
                data = data.to(device) 
                target = target.to(device) 
                outputs = model(data)
                loss = temploss(target, outputs)
                valid_loss += loss.item()
            valid_loss_mean = valid_loss / len(loader_valid)
            print(f"{epoch+1} Validation epoch Loss : {valid_loss_mean:.6f}\n")
            valid_loss_list.append(valid_loss_mean)
        
        if valid_loss_mean < best_val_loss:
            best_val_loss = valid_loss_mean
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_loss_list, valid_loss_list


if __name__ == "__main__":
    
    # model
    model = UNETR().to(device)
    
    # training
    loader_train, loader_valid = DataLoad(batch_size)
    train_loss_list, valid_loss_list = TrainModel(model, learning_rate, epoch_num, device, loader_train, loader_valid)
    model_name = f'{train_date}_Temp_UNETR_{epoch_num}epoch'
    torch.save(model.state_dict(), f'model/{model_name}.pth')
    
    # trainig graph
    TrainGraph(train_loss_list, 'Temp_train loss')
    TrainGraph(valid_loss_list, 'Temp_valid_loss')