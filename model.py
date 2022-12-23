import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch_geometric.data import Data
from torch.nn import BatchNorm1d

from torch.nn import Sequential as Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp



class baseGCN(torch.nn.Module):
    def __init__(self, n_features, embedding_size):
        super(baseGCN,self).__init__()
        torch.manual_seed(95)

        # Initialise the Base Model Architecture
        self.in_conv = GCNConv(n_features, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)
        # self.fc1 = Linear(embedding_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.bn2 = BatchNorm1d(embedding_size)
        # self.fc2 = Linear(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.bn3 = BatchNorm1d(embedding_size)
        # self.fc3 = Linear(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.bn4 = BatchNorm1d(embedding_size)
        
        # output layer of the Graph Neural Net
        self.out = Linear(embedding_size*2, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        hidden = self.in_conv(x, edge_index)
        hidden = self.bn1(hidden)
        # hidden = self.fc1(hidden)
        hidden = F.tanh(hidden)
        
        # Graph Message Passing
        hidden = self.conv1(hidden, edge_index)
        hidden = self.bn2(hidden)
        # hidden = self.fc2(hidden)
        hidden = F.tanh(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = self.bn3(hidden)
        # hidden = self.fc3(hidden)
        hidden = F.tanh(hidden)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = self.bn4(hidden)
        hidden = F.tanh(hidden)
        
        # Apply Pooling at Graph-level
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        
        # Apply a final (linear) regressor
        out = self.out(hidden)
        
        return out, hidden
    
emb_size = 256
num_features = 75
baseModel = baseGCN(num_features, emb_size)

# display(baseModel)
class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, modelD, inputs):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD

        self.fc1 = Linear(inputs, 128)
        self.fc2 = Linear(128, 1)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, 1)

    def forward(self, x, edge_index,batch_index ):
        out1,_ = self.modelA(x, edge_index,batch_index ) #since model returns prediction and embedding but we only want prediction 
        out2,_ = self.modelB(x, edge_index,batch_index )
        out3,_ = self.modelC(x, edge_index,batch_index )
        out4,_ = self.modelD(x, edge_index,batch_index )

        out = out1 + out2 + out3 + out4

        # print(f'size {out.size()} size1 {out1.size()}')
        x_out = self.fc1(out)
        x_out = self.fc2(x_out)
        # print(f'out1: {out1} out2: {out2} out3: {out3} out4: {out4} finalout: {out} x_out: {x_out}')
#         x_out = self.fc3(x_out) #F.relu(self.fc3(x_out))
#         x_out = self.fc4(x_out)
        return x_out
        #return torch.softmax(x_out, dim=1)
    
    
embedding_size = 256
num_features = 75

## models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

JAK1_model = torch.load("models/kinase_baseModel_JAK1_Model3.pkl")
JAK2_model = torch.load("models/kinase_baseModel_JAK2_Model3.pkl")
JAK3_model = torch.load("models/kinase_baseModel_JAK3_Model3.pkl")
TYK2_model = torch.load("models/kinase_baseModel_TYK2_Model3.pkl")
inputdata=None
model = MyEnsemble(JAK1_model, JAK2_model,JAK3_model,TYK2_model,1)
# display(model)