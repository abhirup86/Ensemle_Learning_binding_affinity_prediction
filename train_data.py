from __future__ import division
from __future__ import unicode_literals
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

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn import metrics

from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from rdkit.Chem import PandasTools, AllChem as Chem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator



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
from model import *

df =  pd.read_csv('data/kinase_JAK.csv') #read dataset
### 2.2.2 Creating torch geometric data objects
####  Splititng the dataset for each kinase
# dataset for JAK1
mask_JAK1 = df['Kinase_name'] =='JAK1'
df_JAK1 = df[mask_JAK1]
df_JAK1_Y = list(df_JAK1['measurement_value'])

# dataset for JAK2
mask_JAK2 = df['Kinase_name'] =='JAK2'
df_JAK2 = df[mask_JAK2]
df_JAK2_Y = list(df_JAK2['measurement_value'])

# dataset for JAK3
mask_JAK3 = df['Kinase_name'] =='JAK3'
df_JAK3 = df[mask_JAK3]
df_JAK3_Y = list(df_JAK3['measurement_value'])

# dataset for TYK1
mask_TYK2 = df['Kinase_name'] =='TYK2'
df_TYK2 = df[mask_TYK2]
df_TYK2_Y = list(df_TYK2['measurement_value'])


# SMILES and Ys for the graph creation

## All Data
X_smiles = list(df['SMILES']) #get smiles strings from file
Y = np.asarray(df['measurement_value']) #get solubility values from file
df_Y = list(df['measurement_value'])
X_smiles = df['SMILES']
data_mols = [Chem.MolFromSmiles(s) for s in X_smiles]
data_all = [mol2vec(m,s) for m,s in tqdm(zip(data_mols,X_smiles))]

for i in tqdm(range(0,len(df))):
        data_all[i].y = torch.tensor([[list(df.measurement_value)[i]]], dtype=torch.float)
# # JAK1
X_smiles = df_JAK1['SMILES']
data_mols = [Chem.MolFromSmiles(s) for s in X_smiles]
data_JAK1 = [mol2vec(m,s) for m,s in tqdm(zip(data_mols,X_smiles))]

for i in tqdm(range(0,len(df_JAK1))):
        data_JAK1[i].y = torch.tensor([[list(df_JAK1.measurement_value)[i]]], dtype=torch.float)

# JAK2
X_smiles = df_JAK2['SMILES']
data_mols = [Chem.MolFromSmiles(s) for s in X_smiles]
data_JAK2 = [mol2vec(m,s) for m,s in tqdm(zip(data_mols,X_smiles))]

for i in tqdm(range(0,len(df_JAK2))):
        data_JAK2[i].y = torch.tensor([[list(df_JAK2.measurement_value)[i]]], dtype=torch.float)

# JAK3
X_smiles = df_JAK3['SMILES']
data_mols = [Chem.MolFromSmiles(s) for s in X_smiles]
data_JAK3 = [mol2vec(m,s) for m,s in tqdm(zip(data_mols,X_smiles))]

for i in tqdm(range(0,len(df_JAK3))):
        data_JAK3[i].y = torch.tensor([[list(df_JAK3.measurement_value)[i]]], dtype=torch.float)

# TYK2
X_smiles = df_TYK2['SMILES']
data_mols = [Chem.MolFromSmiles(s) for s in X_smiles]
data_TYK2 = [mol2vec(m,s) for m,s in tqdm(zip(data_mols,X_smiles))]

for i in tqdm(range(0,len(df_TYK2))):
        data_TYK2[i].y = torch.tensor([[list(df_TYK2.measurement_value)[i]]], dtype=torch.float)



#Get the data
data = data_all  

# Print how the data looks like
print("Inspecting a Molecule Graph data sample..\nFirst Molecule Graph Data in dataset: {}\nNumber of Nodes in Molecule Graph: {} | Number of Edges in Molecule Graph: {}".format(data[0], data[0].num_nodes, data[0].num_edges))
print("\nInspect the features in the data sample:")
print("\nThe target variable for the Graph Data sample:")
# print(data[0].edge_index.t()) # Get the Edge Insformation i.e. tuple of nodes connected by an edge.
print("The graph data has 32 nodes & each node is comprised of Node-features which consists of a tensor of 9 features for each node.")

# Use the GPU for Model training if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
baseModel = baseModel.to(device)

# Define the Objective Function for Base Model compilation
# MSE -> Objective / Loss Funciton
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(baseModel.parameters(), lr = 0.0005)

# Use Data Loader to collate Data into batches
data_size = len(data)
batch_size = 64

# Pass 75% of the data as Train to the Data Loader
train_data_loader = DataLoader(data[:int(data_size * 0.75)], batch_size=batch_size, shuffle=True)

# Pass 25% of the data as Test to the Data Loader
test_data_loader = DataLoader(data[int(data_size * 0.75):], batch_size=batch_size, shuffle=True)