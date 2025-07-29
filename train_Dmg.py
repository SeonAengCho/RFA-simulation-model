import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

from config import *
from utils import *
from tqdm import tqdm
from loss import diceloss 
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
    Y_train = torch.Tensor(np.load("data/output_Dmg_train.npy"))
    Y_valid = torch.Tensor(np.load("data/output_Dmg_valid.npy"))
    
    print ("Preparing Dataset ... ")
    
    ds_train = TensorDataset(X_train, Y_train)
    ds_valid = TensorDataset(X_valid, Y_valid)
    
    loader_train = DataLoader(ds_train, batch_size = batch, shuffle = True)
    loader_valid = DataLoader(ds_valid, batch_size = batch, shuffle = False)
    
    return loader_train, loader_valid


def TrainModel(model, learning_rate, epoch_num, device, diceloss, loader_train, loader_valid, patience=20):
    import copy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    valid_loss_list = []

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0  # early stopping counter

    for epoch in range(epoch_num):
        train_loss = 0
        valid_loss = 0

        model.train()
        with tqdm(loader_train, desc=f"Epoch {epoch+1}/{epoch_num}") as tepoch:
            start_time = time.time()

            for data, target in tepoch:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = diceloss(target, outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            end_time = time.time()
            execution_time = end_time - start_time
            train_loss_mean = train_loss / len(loader_train)
            print(f"{epoch+1} epoch Train Loss : {train_loss_mean:.4f}")
            print("Execution time:", execution_time, "seconds")
            train_loss_list.append(train_loss_mean)

        model.eval()
        with torch.no_grad():
            for data, target in loader_valid:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = diceloss(target, outputs)
                valid_loss += loss.item()
            valid_loss_mean = valid_loss / len(loader_valid)
            print(f"{epoch+1} Validation Loss : {valid_loss_mean:.4f}")
            valid_loss_list.append(valid_loss_mean)

        # Early Stopping üũ
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            print("Validation loss improved. Model weights updated.")
        else:
            counter += 1
            print(f"No improvement in validation loss. Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return train_loss_list, valid_loss_list


if __name__ == "__main__":
    
    # model
    model = UNETR().to(device)
    
    # training
    loader_train, loader_valid = DataLoad(batch_size)
    train_loss_list, valid_loss_list = TrainModel(model, learning_rate, epoch_num, device, diceloss, loader_train, loader_valid)
    model_name = f'{train_date}_Dmg_UNETR_{epoch_num}epoch'
    torch.save(model.state_dict(), f'model/{model_name}.pth')
    
    # trainig graph
    TrainGraph(train_loss_list, 'Dmg_train_loss')
    TrainGraph(valid_loss_list, 'Dmg_valid_loss')