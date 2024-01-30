
import torch
from torch_geometric.datasets import TUDataset, LRGBDataset
import os.path as osp
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import math
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool,GatedGraphConv,ChebConv
#from torch_sparse import SparseTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import argparse
import random
import torch_geometric
import scipy as sp
import wandb 
parser = argparse.ArgumentParser()
parser.add_argument('--comp', type=int, default=8, help='Latent Bottleneck')
parser.add_argument('--hidden', type=int, default=120, help='Latent Dimension')
parser.add_argument('--seed', type=int, default=0, help='Latent Dimension')
parser.add_argument('--heads', type=int, default=5, help='Latent Dimension')
parser.add_argument('--batch_size', type=int, default=16, help='Latent Dimension')
parser.add_argument('--k', type=int, default=16, help='number of vectors')
parser.add_argument('--laplace', type=bool, default=False, help='Use laplacian PE')

parser.add_argument('--use_graph', type=bool, default=True, help='Use graph infos')
parser.add_argument('--patches', type=bool, default=True, help='Use graph infos')
parser.add_argument('--pretrain', type=bool, default=False, help='Use graph infos')

parser.add_argument('--use_weights', type=bool, default=False, help='Use graph infos')
parser.add_argument('--gConv', type=str, default='SAGE', help='graph conv between latents')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--webdata', type=str, default='oieurouw', help='which data')
args = parser.parse_args([])
torch.manual_seed(args.seed)

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=args.k)

def LapPE(edge_index, pos_enc_dim, num_nodes):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros
    return PE


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

best_last=182


def pe(datasetz):
  outs=[]
  ins=[]
  dataset2=[]
  for p in range(len(datasetz)):
    try:
      tempo=datasetz[p]
      posi=RWSE(datasetz[p].edge_index, args.k, datasetz[p]['x'].shape[0])
      tempo['laplace']=torch.cat((datasetz[p].x,posi),dim=1)
      dataset2.append(tempo)
      ins.append(p)
    except:
      print(p)
      outs.append(p)
  return dataset2



from torch_geometric.nn import global_add_pool

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

my_dataset='PascalVOC-SP'
dataset1 = LRGBDataset(root='./', name=my_dataset, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes

if args.laplace==True:
  dataset1 = pe(dataset1)
  validation_set1 = pe(validation_set1)
  test_set1 = pe(test_set1)

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

class HNO(torch.nn.Module):

    def __init__(self,):
        super(HNO, self).__init__()

        if args.laplace==True:
          self.conv1 = GCNConv(num_feats+args.k, int(args.hidden))
          self.Inlin = Linear(num_feats+args.k, int(2*args.hidden))
          self.Inlin2 = Linear(int(args.hidden), int(args.hidden))
          # self.gin1 = GINConv(nn.Sequential(nn.Linear(num_feats+args.k, int(args.hidden)),
          #                                     #nn.ReLU(),
          #                                     #nn.Linear(args.hidden, args.hidden),
          #                                     nn.ReLU(),
          #                                     nn.Linear(int(args.hidden), int(2*args.hidden))))
        else:
          self.conv1 = ChebConv(num_feats, int(args.hidden),K=4)
          self.Inlin = Linear(num_feats, int(args.hidden))
          self.Inlin2 = Linear(int(args.hidden), int(args.hidden))
          # self.gin1 = GINConv(nn.Sequential(nn.Linear(num_feats, int(args.hidden)),
          #                                     #nn.ReLU(),
          #                                     #nn.Linear(args.hidden, args.hidden),
          #                                     nn.ReLU(),
          #                                     nn.Linear(int(args.hidden), int(args.hidden))))

        #self.conv1 = GCNConv(num_feats, int(2*args.hidden))
        self.conv2 = ChebConv(int(args.hidden), int(args.hidden),K=4)
        self.conv3 = ChebConv(int(args.hidden), args.hidden,K=4)
        self.conv4 = ChebConv(args.hidden, args.hidden,K=4)
        #self.conv5 = GCNConv(args.hidden, args.hidden)

        #self.trans=GraphTransfer(int(args.hidden), args.hidden)
        self.conv_sTc= GATConv(args.hidden, args.hidden)#GATConv(args.hidden, args.hidden)
        #self.conv_cTc= SAGEConv(args.hidden, args.hidden)



        self.LaToGr= GATConv(int(args.hidden), int(args.hidden))
        #self.Inlin = Linear(dataset.num_features, int(args.hidden))

        # self.lin = Linear(args.hidden, int(2*args.hidden))
        # self.lin2 = Linear(int(2*args.hidden), args.hidden)

        self.mlp= Linear(args.hidden*args.comp, num_classes)

        self.mlpRep = MLP(
            args.hidden, num_classes, nlayer=2, with_final_activation=False)

        #self.bano1 = torch.nn.BatchNorm1d(num_features= int(args.hidden))
        self.bano1 = torch.nn.BatchNorm1d(num_features= int(args.hidden))
        self.bano2 = torch.nn.BatchNorm1d(num_features= int(args.hidden))
        self.bano3 = torch.nn.BatchNorm1d(num_features= int(args.hidden))

        #self.cross=CrossAttention(int(args.hid den),context_dim=int(args.hidden)) ## query dim, key_dim


        #self.outLin1 = Linear(int(2*2*args.hidden), num_classes)

        self.outLin1a = Linear(int(args.hidden), int(args.hidden))
        self.outLin2 = Linear(int(args.hidden), num_classes)

        self.outGr = GCNConv(int(args.hidden), num_classes)
        #self.outLin2 = Linear(int(args.hidden), num_classes)


        if args.gConv=='GIN':
          self.conv_cTc = GINConv(nn.Sequential(nn.Linear(int(args.hidden), int(args.hidden)),
                                              #nn.ReLU(),
                                              #nn.Linear(args.hidden, args.hidden),
                                              nn.ReLU(),
                                              nn.Linear(int(args.hidden), int(args.hidden))))

        elif args.gConv=='SAGE':
          self.conv_cTc= SAGEConv(args.hidden, args.hidden)

        elif args.gConv=='GAT':
          self.conv_cTc= GATConv(args.hidden, args.hidden)


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
      return top,bot

    def adjacency_matrix(self,matrix):
        # Get the shape of the input matrix
        n= matrix.shape[0]

        # Generate the fully connected adjacency matrix
        row = torch.arange(n).repeat_interleave(n)
        col = torch.arange(n).repeat(n)
        edge_index = torch.stack([row, col], dim=0)


        return edge_index

    def reconstruct(self,graph,comp,device):
        adTop=torch.arange(0, comp).to(device)
        adBot=torch.arange(comp, comp+graph.shape[0]).to(device)

        top=adTop.repeat(graph.shape[0])
        bot=adBot.repeat(comp,1).t().reshape(-1)
        return top,bot

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
        random.shuffle(col_indices)
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)

        return indices


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
          x = F.dropout(x, training=self.training,p=0.4)

          x = self.conv2(x, edge_index)
          x = F.relu(x)
          x=self.bano2(x)
          x = F.dropout(x, training=self.training,p=0.3)

          x = self.conv3(x, edge_index)
          x = F.relu(x)
          x=self.bano3(x)
          x = F.dropout(x, training=self.training,p=0.4)

          x = self.conv4(x, edge_index)
          # x = F.relu(x)
          # x=self.bano3(x)
          # x = F.dropout(x, training=self.training,p=0.2)

          #x = self.conv5(x, edge_index)
        else:
          x=self.Inlin(x)
          x = x.relu()
          x = F.dropout(x, training=self.training,p=0.3)
          x=self.Inlin2(x)
          x = x.relu()
          x = F.dropout(x, training=self.training,p=0.3)

        fuser=[]
        all_atts=[]
        lens=[] ## lengths of each unique graph
        latLens=[] ## lengths of each unique graph
        patch=[]


        c1=0
        c2=0
        for i in range(batch_size):
          samples=x[batch==i]

          lens.append(samples.shape[0])

          latLens.append(torch.arange(c1+samples.shape[0],c1+samples.shape[0]+args.comp))

          patch.append(self.my_patches(samples.shape[0],args.comp,device) +c1)

          c1+=samples.shape[0]+args.comp

        cumLens=torch.cumsum(torch.tensor(lens), dim=0).tolist()

        x2 = self.stack_rows(x, cumLens, params) ## We stacked latent arrays after each graph G_i

        patch=torch.cat(patch,dim=1).to(device)

        newG,atts=self.conv_sTc(x2,patch.long(),return_attention_weights=True)

        latLens=torch.stack(latLens).reshape(-1)

        #print(latLens)

        #latentNodes=newG[-params.shape[0]*len(lens):].clone().to(device)
        latentNodes=newG[latLens].clone().to(device)

        temp=torch_geometric.utils.dense_to_sparse(torch.ones(len(lens),args.comp,args.comp))[0]#data.latent_adj#self.adjacency_matrix(latentNodes).long()
        CompNodes=self.conv_cTc(latentNodes,temp.cuda())
        #CompNodes=self.conv_cTc(CompNodes,temp.cuda())

        CompNodes=CompNodes.reshape(batch_size,args.comp,-1)


        #CompNodes=self.bano3(CompNodes)
        #CompNodes=F.relu(self.conv_cTc(latentNodes,temp.cuda()))
        #CompNodes=self.conv_cTc(CompNodes,temp.cuda())

        recs=[]
        for i in range(batch_size):
          samples=x[batch==i]
          #print(samples.shape)

          top,bot=self.patches(samples.shape[0],CompNodes[i].shape[0],device)
          #print(top)
          #print(bot)

          top2,bot2=self.transform_indices(top.tolist(), bot.tolist())

          new_adj_out=torch.stack((torch.tensor(bot2),torch.tensor(top2))).to(device)

          newG_out= torch.cat((CompNodes[i],samples)).to(device)
          #newG,atts=self.conv_sTc(newG,new_adj.long())#,return_attention_weights=True)
          final=self.LaToGr(newG_out,new_adj_out.long())#,return_attention_weights=True)

          recs.append(final[CompNodes[i].shape[0]:]+samples)

        recs=torch.cat(recs,dim=0)


        # classifier=self.outLin1a(recs) #,edge_index)
        classifier=self.mlpRep(recs)
        # #classifier=self.outGr(recs,edge_index)
        # classifier=F.relu(classifier)
        # classifier=self.outLin2(classifier) #,edge_index)
        # classifier=F.relu(classifier)
        return classifier,classifier,classifier  #classifier.log_softmax(dim=-1)


          # query=self.cross(samples.clone().unsqueeze_(0),CompNodes.clone().unsqueeze_(0))
          # query=query[0]
          # query=self.cross(query.clone().unsqueeze_(0),CompNodes.clone().unsqueeze_(0))
          # query=query[0]
          # recs.append(query)

          # recs=torch.cat(recs)


          # top2,bot2=self.transform_indices(top.tolist(), bot.tolist())

          # new_adj_out=torch.stack((torch.tensor(bot2),torch.tensor(top2))).to(device)

          # newG_out= torch.cat((latentNodes,x2)).to(device)



          #final=self.LaToGr(newG_out,new_adj_out.long())#,return_attention_weights=True)

          #newG,atts=self.conv_sTc(newG,new_adj.long())#,return_attention_weights=True)
          #final=self.LaToGr(newG_out,new_adj_out.long())#,return_attention_weights=True)

          # topFin,botFin=self.reconstruct(samples,multiCompNodes.shape[0],device)
          # adjFin=torch.stack((topFin,botFin))

          # fingraph=torch.cat((samples,multiCompNodes))
          # final=self.LaToGr(fingraph,adjFin)
          #final=F.dropout(final,p=0.2)""

          #output=torch.cat((x,final[:x.shape[0]]),dim=1)#final[:x.shape[0]]

          #output=torch.mean(torch.stack((final[:samples.shape[0]],samples)),0)#final[:x.shape[0]]

        #classifier=self.outLin1a(recs) #,edge_index)
        #classifier=F.leaky_relu(classifier)
        #classifier=F.dropout(classifier,p=0.3)

import numpy as np
if args.laplace==True:
  eigs=args.k
else:
  eigs=0


config = dict (
  Changes="Operator",
  Latent_nodes=args.comp,
  hidden_dim=args.hidden,
  Laplacian=args.laplace,
  k_eigs=args.k,
  use_GNN=args.use_graph,
  learning_rate = args.lr,
  conv_latents= args.gConv,
  pretrain=args.pretrain,
  seed=args.seed
)

wandb.init(
  project="ICML_PascaVOC",
  config=config,
  name='4 GATconv'
)

#import matplotlib.pyplot as plt
lplot=[]
vplot=[]


model = HNO().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

criterion = torch.nn.CrossEntropyLoss()

from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=True)
valoader = DataLoader(validation_set1, batch_size=8, shuffle=False,drop_last=True)
testloader = DataLoader(test_set1, batch_size=8, shuffle=False,drop_last=True)

from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional.classification import multiclass_f1_score
f1 = MulticlassF1Score(num_classes=21).cuda()

#multiclass_f1_score(preds, target, num_classes=3, multidim_average='samplewise')

def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    bce_loss = nn.BCEWithLogitsLoss()
    is_labeled = true == true  # Filter our nans.
    return bce_loss(pred[is_labeled], true[is_labeled].float()), pred

import torch
import torch.nn.functional as F

def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                  weight=weight[true])
        return loss, pred #torch.sigmoid(pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

def getF1(pred,true,batch,bs):
  avg=0
  for b in range(bs):
    truez=true[batch==b]
    preds=pred[batch==b]
    f1=f1_score(truez,preds,average='macro')
    avg+=f1
  return avg/(b+1)

import torch
import torch.nn.functional as F

def calculate_f1_score(predictions, labels):
    tp = torch.sum(predictions * labels)
    fp = torch.sum(predictions) - tp
    fn = torch.sum(labels) - tp

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return f1_score

def calculate_macro_f1_score(graph_predictions, graph_labels):
    num_graphs = len(graph_predictions)
    macro_f1 = 0.0

    for i in range(num_graphs):
        predictions = graph_predictions[i]
        labels = graph_labels[i]

        f1_score = calculate_f1_score(predictions, labels)
        macro_f1 += f1_score

    macro_f1 /= num_graphs

    return macro_f1

def eval_F1(seq_ref, seq_pred):
    # '''
    #     compute F1 score averaged over samples
    # '''

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {'precision': np.average(precision_list),
            'recall': np.average(recall_list),
            'F1': np.average(f1_list)}

from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
temp=0
for epoch in range(450):
  model.train()
  correct = 0
  precision=0
  totalLoss=0
  allF1s=0

  allpred=[]
  allref=[]
  for i, data in enumerate(trainloader):

    data=data.to(device)

    weights=torch.rand(args.comp,args.hidden,requires_grad=True).to(device)

    if args.laplace==True:

      feats=data.laplace
    else:
      feats=data.x.float()

    classify,atts, fuser=model(feats,data.edge_index,data.batch,weights,data.batch.unique().shape[0],device,data,pretrain=args.pretrain)

    # allPred=[]
    # totalLoss=0
    # for h in range(args.batch_size):
    loss,pred2 = weighted_cross_entropy(classify, data.y)  # Compute the loss.
    totalLoss+=loss
      # allPred.append(pred)

    #allPred=torch.cat(allPred,dim=0)

    loss.backward()

    if i%6==0:
      torch.nn.utils.clip_grad_norm_(model.parameters(),1)

      optimizer.step()

      optimizer.zero_grad()


    pred = pred2.argmax(dim=1)

    # f1s=[]
    # for j in range(args.batch_size):
    #   p=pred[data.batch==j]
    #   t=data.y[data.batch==j]
    #   score=f1_score(t.cpu().numpy(),p.cpu().numpy(),average='macro')
    #   f1s.append(score)




    allpred.append(pred)
    allref.append(data.y)



    # precision+=np.average(f1_list)#getF1(pred.cpu().numpy(), data.y.cpu().numpy(),data.batch.cpu().numpy(),args.batch_size) #multiclass_f1_score(pred, data.y, num_classes=21)

    #precision+=f1_score(data.y.cpu().numpy(),pred.cpu().numpy(),average='macro')
    #train_acc=precision / (i+1)

    totalLoss+=loss

  allpred=torch.cat(allpred,dim=0)
  allref=torch.cat(allref,dim=0)
  totalLoss=totalLoss / (i+1)
  train_acc= f1_score(allref.cpu().numpy(),allpred.cpu().numpy(),average='macro') #np.array(f1s)/(i+1)*args.batch_size



  # if epoch %10==0:
  #     optimizer.param_groups[0]["lr"]=optimizer.param_groups[0]["lr"]*0.95
  val_correct=0
  val_precision=0
  totalVaLoss=0
  allValref=[]
  allValpred=[]

  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    if args.laplace==True:
      valfeats=valdata.laplace
    else:
      valfeats=valdata.x.float()

    val_classify,val_atts,val_fuser=model(valfeats,valdata.edge_index,valdata.batch,weights,valdata.batch.unique().shape[0],device,valdata,pretrain=args.pretrain)

    val_loss,val_pred2 = weighted_cross_entropy(val_classify,valdata.y)

    totalVaLoss+=val_loss

    val_pred = val_pred2.argmax(dim=1)

    allValpred.append(val_pred)
    allValref.append(valdata.y)

    # val_precision+=getF1(val_pred.cpu().numpy(), valdata.y.cpu().numpy(),valdata.batch.cpu().numpy(),valdata.batch.unique().shape[0]) #multiclass_f1_score(pred, data.y, num_classes=21)

  allValpred=torch.cat(allValpred,dim=0)
  allValref=torch.cat(allValref,dim=0)
  val_acc= f1_score(allValref.cpu().numpy(),allValpred.cpu().numpy(),average='macro') #np.array(f1s)/(i+1)*args.batch_size

  totalVaLoss=totalVaLoss / (j+1)

  if val_acc>=temp:
    temp=val_acc
    when=epoch
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_weights':weights,
          }, './pascal/'+args.webdata+'_'+str(epoch)+'.pth')

  #lplot.append(totalLoss)
  #vplot.append(totalVaLoss)

  print(f'Epoch: {epoch:03d}, Loss: {totalLoss.item():.4f},Train F1: {train_acc:.4f}, Val_Loss: {totalVaLoss.item():.4f},Val F1: {val_acc:.4f}')
  wandb.log({"Train F1": train_acc})
  wandb.log({"Val F1": val_acc})
  wandb.log({"Train Loss": totalLoss})
  wandb.log({"Val Loss": totalVaLoss})

# plt.plot(torch.stack(lplot).detach().cpu().numpy(),label="training")
# plt.plot(torch.stack(vplot).detach().cpu().numpy(),label="validation")
# plt.legend(loc="upper left")

device="cuda"
checkpoint = torch.load('./pascal/'+args.webdata+'_'+str(when)+'.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

weights2=checkpoint['best_weights']
weights2=weights2.to(device)


allTestref=[]
allTestpred=[]

for j, testdata in enumerate(testloader):
  model.eval()
  testdata=testdata.to(device)

  if args.laplace==True:
    testfeats=testdata.laplace
  else:
    testfeats=testdata.x.float()

  test_classify,_,_=model(testfeats,testdata.edge_index,testdata.batch,weights2,testdata.batch.unique().shape[0],device,testdata,pretrain=args.pretrain)

  test_loss,test_pred2 = weighted_cross_entropy(test_classify,testdata.y)

  test_pred = test_pred2.argmax(dim=1)

  allTestpred.append(test_pred)
  allTestref.append(testdata.y)

  # val_precision+=getF1(val_pred.cpu().numpy(), valdata.y.cpu().numpy(),valdata.batch.cpu().numpy(),valdata.batch.unique().shape[0]) #multiclass_f1_score(pred, data.y, num_classes=21)
allTestpred=torch.cat(allTestpred,dim=0)
allTestref=torch.cat(allTestref,dim=0)
test_acc= f1_score(allTestref.cpu().numpy(),allTestpred.cpu().numpy(),average='macro') #np.array(f1s)/(i+1)*args.batch_size
wandb.log({"Test F1": test_acc})
wandb.log({"conv between latents": str(model.get_submodule('conv_cTc'))[:8]})
if args.pretrain==True:
    wandb.log({"best epoch": str(when)})
wandb.finish()