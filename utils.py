import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience 
        self.verbose = verbose 
        self.delta = delta 
        self.counter = 0
        self.best_loss = None 
        self.early_stop = False
        self.best_model = None 

    def __call__(self, loss, model):
        
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = model.state_dict()
            
        elif loss > self.best_loss - self.delta: 
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else: 
            self.best_loss = loss
            self.best_model = model.state_dict()
            self.counter = 0
        
        

def TrainGraph(loss, loss_name):
    
    plt.figure(figsize=(10,5))
    plt.title(f"{loss_name} Loss")
    plt.plot(loss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'train_graph/{loss_name}.png', dpi=600, pad_inches=0)
    plt.close()


def InfernceTime(random_data, model):
    
    device = next(model.parameters()).device
    
    start_time = time.time()
    output = model(random_data.to(device))
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    
    
def ImagePlot(image_num, pred, label, version, timestep):
    
    pred = np.array(pred).reshape(1000, timestep, 41, 41, 41)
    label = np.array(label)
    
    rows, cols = 3, 6  

    if version == 0:  
        max_pred = np.max(pred[image_num])
        max_label = np.max(label[image_num])
        max_temp = max(max_pred, max_label)
        min_temp = 37
        
        # pred subplot
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(timestep):
            axes[i].imshow(pred[image_num][i][20], vmin=min_temp, vmax=max_temp, cmap='viridis')
            axes[i].set_title(f"Pred t={i}")
            axes[i].axis('off')
        fig.suptitle("Prediction", fontsize=16)
        plt.tight_layout()
        plt.show()

        # label subplot
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(timestep):
            axes[i].imshow(label[image_num][i][20], vmin=min_temp, vmax=max_temp, cmap='viridis')
            axes[i].set_title(f"Label t={i}")
            axes[i].axis('off')
        fig.suptitle("Label", fontsize=16)
        plt.tight_layout()
        plt.show()

    else:  # Damage
        # pred subplot
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(timestep):
            axes[i].imshow(pred[image_num][i][20])
            axes[i].set_title(f"Pred t={i}")
            axes[i].axis('off')
        fig.suptitle("Prediction (Damage)", fontsize=16)
        plt.tight_layout()
        plt.show()

        # label subplot
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(timestep):
            axes[i].imshow(label[image_num][i][20])
            axes[i].set_title(f"Label t={i}")
            axes[i].axis('off')
        fig.suptitle("Label (Damage)", fontsize=16)
        plt.tight_layout()
        plt.show()