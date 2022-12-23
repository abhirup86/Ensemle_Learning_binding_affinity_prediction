from __future__ import division
from __future__ import unicode_literals

from torch.utils import data
from torch_geometric.data import Data
from torch.nn import BatchNorm1d
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch

from torch.nn import Sequential as Seq, Linear, ReLU, CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, NNConv, GATConv, TopKPooling,  global_add_pool
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader, Dataset

import os
import torch
import pickle
import collections
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from pysmiles import read_smiles
import pandas as pd
import logging
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from itertools import repeat, product, chain
from tqdm.notebook import trange, tqdm
from ogb.utils import smiles2graph
import multiprocessing
from model import *
from train_data import *



# turn this off
def base_model_train(data):
    
    for batch_data in train_data_loader:
        
        batch_data=batch_data.to(device)
        optimizer.zero_grad()
        x = batch_data.x.float()
        edge_index = batch_data.edge_index
        batch_index  = batch_data.batch
        #prediction, embedding = model(x, edge_index,batch_index )
        prediction = model(x, edge_index,batch_index )
        loss_val = loss_function(prediction, batch_data.y)
        loss_val.backward()
        optimizer.step()
    return loss_val
#     return loss_val, embedding

epochs = 200
hist_loss = []

for epoch in range(epochs):
    #loss, h = model_train(data)
    loss = base_model_train(data)
    hist_loss.append(loss)
    if epoch%10 == 0:
        print("Completed Epochs: {} -> Loss: {}".format(epoch, loss))
        
# Save the Trained Model
torch.save(baseModel, "models/kinase_baseModel_ensemble_Model3.pkl")

# Baseline Model was trained over 2500 epochs
hist_losses = [float(loss.cpu().detach().numpy()) for loss in hist_loss[:]]
loss_index = [i for i in range(len(hist_losses))]

# Plotting 
fig, ax = plt.subplots(figsize=(6, 4),dpi=100)
p2 = sns.lineplot(loss_index, hist_losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
fig.savefig('images/training_loss_ensemble_Model3.png',dpi=400)


## ensemble learning ###

# turn this off
def model_train(data):
    
    for batch_data in train_data_loader:
        
        batch_data=batch_data.to(device)
        optimizer.zero_grad()
        x = batch_data.x.float()
        edge_index = batch_data.edge_index
        batch_index  = batch_data.batch
        #prediction, embedding = model(x, edge_index,batch_index )
        prediction = model(x, edge_index,batch_index )
        loss_val = loss_function(prediction, batch_data.y)
        loss_val.backward()
        optimizer.step()
    return loss_val
#     return loss_val, embedding

epochs = 200
hist_loss = []

for epoch in range(epochs):
    #loss, h = model_train(data)
    loss = model_train(data)
    hist_loss.append(loss)
    if epoch%10 == 0:
        print("Completed Epochs: {} -> Loss: {}".format(epoch, loss))
        
# Save the Trained Model
torch.save(baseModel, "models/kinase_baseModel_ensemble_Model3.pkl")

# Baseline Model was trained over 2500 epochs
hist_losses = [float(loss.cpu().detach().numpy()) for loss in hist_loss[:]]
loss_index = [i for i in range(len(hist_losses))]

# Plotting 
fig, ax = plt.subplots(figsize=(6, 4),dpi=100)
p2 = sns.lineplot(loss_index, hist_losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
fig.savefig('images/training_loss_ensemble_Model3.png',dpi=400)



# ensemble trainging
def ensemble_model_train(data):
    
    for batch_data in train_data_loader:
        
        batch_data=batch_data.to(device)
        optimizer.zero_grad()
        x = batch_data.x.float()
        edge_index = batch_data.edge_index
        batch_index  = batch_data.batch
        #prediction, embedding = model(x, edge_index,batch_index )
        prediction = model(x, edge_index,batch_index )
        loss_val = loss_function(prediction, batch_data.y)
        loss_val.backward()
        optimizer.step()
    return loss_val
#     return loss_val, embedding

epochs = 200
hist_loss = []

for epoch in range(epochs):
    #loss, h = model_train(data)
    loss = ensemble_model_train(data)
    hist_loss.append(loss)
    if epoch%10 == 0:
        print("Completed Epochs: {} -> Loss: {}".format(epoch, loss))
        
# Save the Trained Model
torch.save(baseModel, "models/kinase_baseModel_ensemble_Model3.pkl")

# Baseline Model was trained over 2500 epochs
hist_losses = [float(loss.cpu().detach().numpy()) for loss in hist_loss[:]]
loss_index = [i for i in range(len(hist_losses))]

# Plotting 
fig, ax = plt.subplots(figsize=(6, 4),dpi=100)
p2 = sns.lineplot(loss_index, hist_losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
fig.savefig('images/training_loss_ensemble_Model3.png',dpi=400)