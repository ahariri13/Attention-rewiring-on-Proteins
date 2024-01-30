# -*- coding: utf-8 -*-
"""Att_rewire.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18NVaEIRde7UdfEZSZRKmx-w56hgQuTml
"""

# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
# !pip install einops
# !pip install wandb

### Attention-based Rewiring
## In this notebook we plan to rewire the input graph based on the attention weights from latent to output.

# Install required packages.
import os
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)


import torch
from torch_geometric.datasets import TUDataset, LRGBDataset
import os.path as osp
import torch_geometric.transforms as T
import wandb
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import math
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool, ChebConv,global_add_pool
#from torch_sparse import SparseTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import argparse
import random
import torch_geometric
from torch_geometric.utils import dropout_adj,dense_to_sparse
from torch_geometric.utils import to_dense_adj,dense_to_sparse #dropout_adj,

parser = argparse.ArgumentParser()
parser.add_argument('--comp', type=int, default=6, help='Latent Bottleneck')
parser.add_argument('--hidden', type=int, default=290, help='Latent Dimension')
parser.add_argument('--seed', type=int, default=12345, help='Latent Dimension')
parser.add_argument('--batch_size', type=int, default=32, help='Latent Dimension')
parser.add_argument('--k', type=int, default=16, help='number of vectors')
parser.add_argument('--laplace', type=bool, default=False, help='Use laplacian PE')
parser.add_argument('--FA', type=bool, default=False, help='Use FA Layer')
parser.add_argument('--learnable', type=bool, default=False, help='Use FA Layer')

parser.add_argument('--use_graph', type=bool, default=True, help='Use graph infos')
parser.add_argument('--patches', type=bool, default=True, help='Use graph infos')
parser.add_argument('--pretrain', type=bool, default=False, help='Use graph infos')

parser.add_argument('--use_weights', type=bool, default=False, help='Use graph infos')
parser.add_argument('--gConv', type=str, default='GAT', help='graph conv between latents')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--lamb', type=float, default=0.7, help='Regularizer')
parser.add_argument('--webdata', type=str, default='sdeuycw1fw', help='which data')
args = parser.parse_args([])

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=args.k)
torch.manual_seed(args.seed)


def random_walk(A, n_iter):
    # Geometric diffusion features with Random Walk
    Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
    RW = A * Dinv
    M = RW
    M_power = M
    # Iterate
    PE = [torch.diagonal(M)]
    for _ in range(n_iter-1):
        M_power = torch.matmul(M_power, M)
        PE.append(torch.diagonal(M_power))
    PE = torch.stack(PE, dim=-1)
    return PE


def RWSE(edge_index, pos_enc_dim, num_nodes):
    """
        Initializing positional encoding with RWSE
    """
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        PE = random_walk(A, pos_enc_dim)
    return PE


def pe(datasetz):
  outs=[]
  ins=[]
  dataset2=[]
  for p in range(len(datasetz)):
    try:
      tempo=datasetz[p]
      tempo['laplace']= RWSE(datasetz[p].edge_index, args.k, tempo.num_nodes)#check(datasetz[p]).laplacian_eigenvector_pe#torch.cat((datasetz[p].x,check(datasetz[p]).laplacian_eigenvector_pe),dim=1)
      dataset2.append(tempo)
      ins.append(p)
    except:
      print(p)
      outs.append(p)
  return dataset2


def falayer(datasetz):
  outs=[]
  ins=[]
  dataset2=[]
  for p in range(len(datasetz)):
    try:
      tempo=datasetz[p]
      tempo['edge_index']= dense_to_sparse(torch.ones(tempo.x.shape[0],tempo.x.shape[0]))[0]
      dataset2.append(tempo)
      ins.append(p)
    except:
      print(p)
      outs.append(p)
  return dataset2


def patches(N, M,device):

    # Calculate the number of connections per group
    connections_per_group = N // M

    # Generate the indices of connected nodes
    row_indices = []
    col_indices = []

    for i in range(M):
        start_index = i * connections_per_group
        end_index = start_index + connections_per_group

        row_indices.extend(range(start_index, end_index))
        col_indices.extend([N + i] * connections_per_group)

    if (N+M)%2 !=0:
            row_indices.append(N-1)
            perm = torch.randperm(len(col_indices))
            idx = perm[0]
            indi = col_indices[idx]

            col_indices.append(indi)

    # Create the sparse tensor
    random.shuffle(col_indices)
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)

    return indices



from torch_geometric.nn import global_add_pool

BN = True


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


from torch_scatter import scatter
class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x


my_dataset='Peptides-func'
dataset1 = LRGBDataset(root='./', name=my_dataset, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset, split="test")#.shuffle()


num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes


def adjacency_matrix(matrix):
    # Get the shape of the input matrix
    n= matrix.shape[0]

    # Generate the fully connected adjacency matrix
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    edge_index = torch.stack([row, col], dim=0)


    return edge_index

def patchData(dat,k, weights, device):
    updated_data=[]
    for g in range(len(dat)):
        timpo=dat[g]
        top,bot=patches(timpo.x.shape[0],k,device)
        adj=torch.stack((top,bot)).to(device)
        timpo['patch_adj']=adj.t() ### input to latent edges
        #timpo['full_graph']=torch.cat((timpo.x.to(device),weights)).to(device) ### Full graph]
        timpo['latent_adj']= adjacency_matrix(weights).to(device) ### edges between latent nodes
        updated_data.append(timpo)
    return updated_data

#weights=torch.rand(args.comp,args.hidden,requires_grad=True).to(device)
# dataset1 = patchData(dataset1,args.comp,weights,device)
# validation_set1 = patchData(validation_set1,args.comp,weights,device)
# test_set1 = patchData(test_set1,args.comp,weights,device)

if args.laplace==True:
  dataset1 = pe(dataset1)
  validation_set1 = pe(validation_set1)
  test_set1 = pe(test_set1)

if args.FA==True:
  dataset1 = falayer(dataset1)
  validation_set1 = falayer(validation_set1)
  test_set1 = falayer(test_set1)


from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=args.batch_size, shuffle=False)


from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context , mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
class HNO(torch.nn.Module):

    def __init__(self,):
        super(HNO, self).__init__()

        if args.laplace==True:
          self.conv1 = GCNConv(num_feats+args.k, int(args.hidden))#,K=8)
          self.Inlin = Linear(num_feats+args.k, int(args.hidden))
          self.Inlin2 = Linear(int(args.hidden), int(args.hidden))
          # self.gin1 = GINConv(nn.Sequential(nn.Linear(num_feats+args.k, int(args.hidden)),
          #                                     #nn.ReLU(),
          #                                     #nn.Linear(args.hidden, args.hidden),
          #                                     nn.ReLU(),
          #                                     nn.Linear(int(args.hidden), int(2*args.hidden))))
        else:
          self.conv1 = GCNConv(num_feats, int(args.hidden))#,K=8)
          self.Inlin = Linear(num_feats, int(args.hidden))
          self.Inlin2 = Linear(int(args.hidden), int(args.hidden))
          # self.gin1 = GINConv(nn.Sequential(nn.Linear(num_feats, int(args.hidden)),
          #                                     #nn.ReLU(),
          #                                     #nn.Linear(args.hidden, args.hidden),
          #                                     nn.ReLU(),
          #                                     nn.Linear(int(args.hidden), int(args.hidden))))

        #self.conv1 = GCNConv(num_feats, int(args.hidden))
        self.conv2 = GCNConv(int(args.hidden), int(args.hidden))#,K=8)
        self.conv3 = GCNConv(int(args.hidden), args.hidden)#,K=8)
        self.conv4 = GCNConv(int(args.hidden), args.hidden)#,K=8)
        # self.conv5 = ChebConv(int(args.hidden), args.hidden,K=8)
        #self.conv4 = SAGEConv(args.hidden, args.hidden)
        self.convGraph =  GCNConv(args.hidden,args.hidden)
        self.convGraph2 =  GCNConv(args.hidden,args.comp)

        self.Fconv = GCNConv(args.hidden*args.comp, num_classes)

        self.conv_sTc= GATConv(args.hidden, args.hidden)#GATConv(args.hidden, args.hidden)
        #self.Inlin = Linear(dataset.num_features, int(args.hidden))

        self.lin = Linear(args.hidden, int(args.hidden))
        self.lin2 = Linear(int(args.hidden), args.hidden)

        self.mlp= Linear(args.hidden*args.comp, num_classes)
        self.mlp2= Linear(args.hidden, num_classes)

        self.bano1 = torch.nn.BatchNorm1d(num_features= int(args.hidden))
        self.bano2 = torch.nn.BatchNorm1d(num_features= int(args.hidden))
        self.bano3 = torch.nn.BatchNorm1d(num_features= int(args.hidden))

        self.embLin=Linear(args.hidden, int(args.hidden))

        self.LaToGr= GATConv(int(args.hidden), int(args.hidden))

        #self.cross=CrossAttention(int(args.hidden),context_dim=int(args.hidden)) ## query dim, key_dim

        # if args.gConv=='GIN':
        #   self.conv_cTc = GINConv(nn.Sequential(nn.Linear(int(args.hidden), int(args.hidden)),
        #                                       #nn.ReLU(),
        #                                       #nn.Linear(args.hidden, args.hidden),
        #                                       nn.ReLU(),
        #                                       nn.Linear(int(args.hidden), int(args.hidden))))
        # elif args.gConv=='SAGE':
        #   self.conv_cTc= SAGEConv(args.hidden, args.hidden)

        if args.gConv=='GAT':
          self.conv_cTc= GATConv(args.hidden, args.hidden)

        # self.gin2 = GINConv(nn.Sequential(nn.Linear(args.hidden, args.hidden),
        #                                     nn.ReLU(),
        #                                     nn.Linear(args.hidden, args.hidden),
        #                                     nn.ReLU(),
        #                                     nn.Linear(args.hidden, args.hidden)))
        self.mlpRep = MLP(
            int(args.hidden), num_classes, nlayer=2, with_final_activation=False)


    def connect(self,adTop):
      temp=[]
      for k in range(args.comp):
        T = torch.cat([adTop[:k], adTop[k+1:]])
        temp+=T
      return torch.stack(temp)

    def extend(self,Gnodes,comp,device):
      adTop=torch.arange(0, Gnodes.shape[0]).to(device)
      adBot=torch.arange(Gnodes.shape[0], Gnodes.shape[0]+comp).to(device)

      top=adTop.repeat(comp)
      bot=adBot.repeat(Gnodes.shape[0],1).t().reshape(-1)
      indices = torch.stack((top, bot))
      return indices#top,bot

    def adjacency_matrix(self,matrix):
        # Get the shape of the input matrix
        n= matrix.shape[0]

        # Generate the fully connected adjacency matrix
        row = torch.arange(n).repeat_interleave(n)
        col = torch.arange(n).repeat(n)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def my_patches(self,N,M,device):
      rows=torch.arange(0,N)
      div=N%M ## comp 
      ent=N-div
      g2=torch.arange(N,+N+M)
      g2=g2.repeat_interleave(int(ent/M), dim=0)
      if div !=0:
        g2=torch.cat((g2,g2[-1].repeat(div)))
      indices = torch.tensor([rows.tolist(), g2.tolist()], dtype=torch.long)
      return indices


    def patches(self,N, M,device):

        # Calculate the number of connections per group
        connections_per_group = N // M

        # Generate the indices of connected nodes
        row_indices = []
        col_indices = []

        for i in range(M):
            start_index = i * connections_per_group
            end_index = start_index + connections_per_group

            row_indices.extend(range(start_index, end_index))
            col_indices.extend([N + i] * connections_per_group)

        if (N+M)%2 !=0:
                row_indices.append(N-1)
                perm = torch.randperm(len(col_indices))
                idx = perm[0]
                indi = col_indices[idx]

                col_indices.append(indi)


        # Create the sparse tensor
        #random.shuffle(col_indices)            ### When commented, removes random shuffling of columns in the end
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)

        return indices

    ###  insert the weights at the end of each graph in the batch
    def stack_rows(self,matrix, row_indices, weight):
        n, d = matrix.shape
        p = len(row_indices)
        new_n = n #+ k * p

        # Generate the stacked matrix
        stacked_matrix = torch.zeros(new_n, d, dtype=matrix.dtype, device=matrix.device)
        stacked_matrix[:n] = matrix

        # Stack the new matrix at the specified row indices
        k=weight.shape[0]
        for i, idx in enumerate(row_indices):
            stacked_matrix = torch.cat((stacked_matrix[:idx + i * k], weight, stacked_matrix[idx + i * k -k+ k:]))

        return stacked_matrix


    def transform_indices(self,A, B):

        n = len(A)
        m = max(B)

        shift = m - n+1

        # Step 2: Create mapping dictionary
        mapping = {}
        new_index = 0
        for value in set(B):
            mapping[value] = new_index
            new_index += 1

        # Step 3: Replace elements in B with new indices
        B_transformed = [mapping[value] for value in B]

        # Step 4: Create new tensor A
        A_transformed = list(range(shift, m + 1))

        return A_transformed, B_transformed

    def forward(self, x, edge_index, batch, params,batch_size,device,data,pretrain):

        if args.use_graph==True:
          x = self.conv1(x, edge_index)
          x = F.relu(x)
          x=self.bano1(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv2(x, edge_index)
          x = F.relu(x)
          x=self.bano2(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv3(x, edge_index)
          x = F.relu(x)
          x = self.bano3(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv4(x, edge_index)
          # x = F.leaky_relu(x)
          # x=self.bano3(x)
          # x = F.dropout(x, training=self.training,p=0.2)

          # x = self.conv5(x, edge_index)

          # x_graph=self.convGraph(x,edge_index)
          # x_graph=self.convGraph2(x_graph,edge_index)

          # x_graph=torch.sigmoid(x_graph)

          # x_graph=(x_graph > 0.5)

        else:
          x=self.Inlin(x)
          x = x.relu()
          x = F.dropout(x, training=self.training,p=0.3)
          x=self.Inlin2(x)
          x = x.relu()
          x = F.dropout(x, training=self.training,p=0.3)

        params=self.embLin(params)
        params = params.relu()

        fuser=[]
        all_atts=[]
        lens=[] ## lengths of each unique graph
        latLens=[] ## lengths of each unique graph
        patch=[]



        c1=0
        c2=0
        co=0
        merged_samples=[]
        for i in range(batch_size):
          samples=x[batch==i]
          lens.append(samples.shape[0])
          latLens.append(torch.arange(c1+samples.shape[0],c1+samples.shape[0]+args.comp))


          if args.patches==True:
             patch.append(self.my_patches(samples.shape[0],args.comp,device) +c1)
          else:
              patch.append(self.extend(samples,args.comp,device) +c1)

          c1+=samples.shape[0]+args.comp
          co+=data[i].x.shape[0]
          merged_samples.append(samples)


        merged_samples=torch.cat(merged_samples)
        cumLens=torch.cumsum(torch.tensor(lens), dim=0).tolist()

        ## We stacked latent arrays after each graph G_i
        x2 = self.stack_rows(x.clone(), cumLens, params)

        patch=torch.cat(patch,dim=1).to(device)

        """
        Input to latent
        """
        newG=self.conv_sTc(x2,patch)    #,return_attention_weights=True)
        latLens=torch.stack(latLens).reshape(-1)

        """
        Latent-to-Latent
        """
        latentNodes=newG[latLens].clone().to(device)
        temp=torch_geometric.utils.dense_to_sparse(torch.ones(len(lens),args.comp,args.comp))[0]
        CompNodes=F.relu(self.conv_cTc(latentNodes,temp.cuda()))
        CompNodes=CompNodes.reshape(len(lens),args.comp,args.hidden)
        CompNodes = torch.sum(CompNodes, dim=1)

        # """
        # Latent-to-Input
        # """
        # topLI,botLI=patch

        # # top2,bot2=self.transform_indices(topLI.tolist(), botLI.tolist())
        # # adj_latent_input=torch.stack((torch.tensor(bot2),torch.tensor(top2))).to(device)
        # x3 = self.stack_rows(x.clone(), cumLens, CompNodes)
        # #patch2 = to_undirected(patch)

        # # newG_out= torch.cat((CompNodes,merged_samples)).to(device)

        # #adj_latent_input=torch.stack((torch.tensor(botLI),torch.tensor(topLI))).to(device)
        # patch[[0,1]] = patch[[1,0]]

        # final,atts=self.LaToGr(x3,patch.long(),return_attention_weights=True)

        # mem=[]
        # c0=0
        # for el in lens:
        #   mem.append(torch.arange(c0,el+c0))
        #   c0+=el+args.comp
        # mem=torch.cat(mem)

        ##checker=torch.arange(0,len(lens)).repeat(final,1).t().reshape(-1).to(device)
        #final1 = global_add_pool(final[mem], batch)
        final2 = global_add_pool(x, batch)
        #print(final2.shape)
        #print(CompNodes.shape)
        final3=final2+ args.lamb*CompNodes #+ final1*0.8 #torch.cat((final2,CompNodes),dim=1)

        classifier=self.mlpRep(final3)
        #classifier=self.mlp(CompNodes)
        # classifier=F.leaky_relu(classifier)
        # classifier= F.dropout(classifier, training=self.training,p=0.3)
        # classifier=self.mlp2(classifier)
        cdd=0

        return classifier,lens, lens


def stack_rows(matrix, row_indices, k):
    n, d = matrix.shape
    p = len(row_indices)
    new_n = n #+ k * p

    # Generate the stacked matrix
    stacked_matrix = torch.zeros(new_n, d, dtype=matrix.dtype, device=matrix.device)
    stacked_matrix[:n] = matrix

    # Stack the new matrix at the specified row indices
    for i, idx in enumerate(row_indices):
        stacked_matrix = torch.cat((stacked_matrix[:idx + i * k], torch.ones(k, d), stacked_matrix[idx + i * k -k+ k:]))

    return stacked_matrix


if args.laplace==True:
  eigs=args.k
else:
  eigs=0


from sklearn.metrics import average_precision_score
import numpy as np
def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''
    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)



#import matplotlib.pyplot as plt
lplot=[]
vplot=[]

model = HNO().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

criterion = torch.nn.CrossEntropyLoss()

config = dict (
  Changes="weights2",
  Latent_nodes=args.comp,
  hidden_dim=args.hidden,
  Laplacian=args.laplace,
  k_eigs=eigs,
  batch_size=args.batch_size,

  patches=args.patches,
  use_GNN=args.use_graph,
  learning_rate = args.lr,
  gConv = args.gConv,
  pretrain = args.pretrain,
  seed = args.seed
)


wandb.init(
project="ICML_Table1_Boost",
name="4GATConv_lamb0.8",
config=config,
)


temp=0
for epoch in range(350):
  model.train()
  correct = 0
  precision=0
  real=[]
  pred=[]
  for i, data in enumerate(trainloader):

    data=data.to(device)

    optimizer.zero_grad()

    weights=torch.rand(args.comp,args.hidden,requires_grad=False).to(device)
    #weights=torch.zeros(args.comp,args.hidden,requires_grad=False).to(device)

    if args.laplace==True:
      batch_pos_enc = data.laplace.cuda()
      sign_flip = torch.rand(batch_pos_enc.size(1)).cuda()
      sign_flip[sign_flip >= 0.5] = 1.0
      sign_flip[sign_flip < 0.5] = -1.0
      data.laplace = batch_pos_enc * sign_flip.unsqueeze(0)
      feats=data.x.cuda()
      lap=data.laplace.cuda()
      feats=torch.cat((feats,lap),dim=1)
      del lap
    else:
      feats=data.x.float()

    # if args.laplace==True:
    #   feats=data.laplace
    # else:
    #   feats=data.x.float()

    #patches=data.patch_adj

    if args.pretrain==True:
      classify=model(feats,data.edge_index,data.batch,weights,data.batch.unique().shape[0],device,data,pretrain=args.pretrain)
    else:
      classify,atts, fuser=model(feats,data.edge_index,data.batch,weights,data.batch.unique().shape[0],device,data,pretrain=args.pretrain)

    loss = criterion(classify, data.y)  # Compute the loss

    loss.backward()

    optimizer.step()

    real.append(data.y)
    pred.append(classify)

  y_true = torch.cat(real, dim=0)
  y_pred = torch.cat(pred, dim=0)
  train_perf = eval_ap(y_true=y_true, y_pred=y_pred)
  del real
  del pred


  train_acc=precision / (i+1)
  if epoch>=40:
    if epoch %10==0:
      optimizer.param_groups[0]["lr"]=optimizer.param_groups[0]["lr"]*0.95

  val_correct=0
  val_precision=0
  valreal=[]
  valpred=[]
  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    if args.laplace==True:
      batch_pos_enc = valdata.laplace.cuda()
      sign_flip = torch.rand(batch_pos_enc.size(1)).cuda()
      sign_flip[sign_flip >= 0.5] = 1.0
      sign_flip[sign_flip < 0.5] = -1.0
      valdata.laplace = batch_pos_enc * sign_flip.unsqueeze(0)
      valfeats=valdata.x.cuda()
      lap=valdata.laplace.cuda()
      valfeats=torch.cat((valfeats,lap),dim=1)
      del lap
    else:
      valfeats=valdata.x.float()

    # if args.laplace==True:
    #   valfeats=valdata.laplace
    # else:
    #   valfeats=valdata.x.float()

    #Valweights=torch.rand(args.comp,args.hidden).to(device)
    #Valweights=torch.zeros(args.comp,args.hidden,requires_grad=False).to(device)

    if args.pretrain==True:
      val_classify =model(valfeats,valdata.edge_index,valdata.batch,weights,valdata.batch.unique().shape[0],device,valdata,pretrain=args.pretrain)
    else:
      val_classify,val_atts,val_fuser=model(valfeats,valdata.edge_index,valdata.batch,weights,valdata.batch.unique().shape[0],device,valdata,pretrain=args.pretrain)

    val_loss = criterion(val_classify, valdata.y)
    val_pred = val_classify.argmax(dim=1)
    val_precision+=eval_ap(valdata.y,val_classify)
    valreal.append(valdata.y)
    valpred.append(val_classify)

  val_y_true = torch.cat(valreal, dim=0)
  val_y_pred = torch.cat(valpred, dim=0)
  val_perf = eval_ap(y_true=val_y_true, y_pred=val_y_pred)

  if val_perf>=temp:
    temp=val_perf
    when=epoch
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_weights':weights,
          }, './temp_weights/'+args.webdata+'_'+str(epoch)+'.pth')

  #lplot.append(loss)
  #vplot.append(val_loss)

  print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f},Train Acc: {train_perf:.4f}, Val_Loss: {val_loss.item():.4f},Val Acc: {val_perf:.4f}')
  wandb.log({"Train Acc": train_perf})
  wandb.log({"Val Acc": val_perf})
  wandb.log({"Train Loss": loss})
  wandb.log({"Val Loss": val_loss})
  wandb.log({"Epoch": epoch})

device="cuda"
checkpoint = torch.load('./temp_weights/'+args.webdata+'_'+str(when)+'.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

weights2=checkpoint['best_weights']
weights2=weights2.to(device)

test_precision=0
tr=[]
tp=[]
with torch.no_grad():
  for k, testdata in enumerate(testloader):
    model.eval()
    model=model.to(device)
    testdata=testdata.to(device)
    testedgeAtt=testdata.edge_attr.float()

    if args.laplace==True:
      batch_pos_enc = testdata.laplace.cuda()
      sign_flip = torch.rand(batch_pos_enc.size(1)).cuda()
      sign_flip[sign_flip >= 0.5] = 1.0
      sign_flip[sign_flip < 0.5] = -1.0
      testdata.laplace = batch_pos_enc * sign_flip.unsqueeze(0)
      testfeats=testdata.x.cuda()
      lap=testdata.laplace.cuda()
      testfeats=torch.cat((testfeats,lap),dim=1)
      del lap
    else:
      testfeats=testdata.x.float()

    # if args.laplace==True:
    #   testfeats=testdata.laplace
    # else:
    #   testfeats=testdata.x.float()

    if args.pretrain==True:
      test_classify=model(testfeats,testdata.edge_index,testdata.batch,weights2,testdata.batch.unique().shape[0],device,pretrain=args.pretrain)
    else:
      #testfeats=add_lap(testdata)
       test_classify,val_atts,val_fuser=model(testfeats,testdata.edge_index,testdata.batch,weights2,testdata.batch.unique().shape[0],device,testdata,pretrain=args.pretrain)

    tr.append(testdata.y)
    tp.append(test_classify)

  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)
  #test_precision+=eval_ap(testdata.y,test_classify)
  #test_acc=test_precision / (k+1)
  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)
  test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)

wandb.log({"Test Acc": test_perf})
wandb.log({"conv between latents": str(model.get_submodule('conv_cTc'))[:8]})
wandb.finish()