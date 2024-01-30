

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
parser.add_argument('--hidden', type=int, default=160, help='Latent Dimension')
parser.add_argument('--maxAtt', type=int, default=4, help='Max attention')
parser.add_argument('--seed', type=int, default=103, help='Latent Dimension')
parser.add_argument('--batch_size', type=int, default=60, help='Latent Dimension')
parser.add_argument('--k', type=int, default=16, help='number of vectors')
parser.add_argument('--K', type=int, default=8, help='number of vectors')
parser.add_argument('--laplace_RW', type=bool, default=False, help='Use laplacian PE')
parser.add_argument('--laplace', type=bool, default=False, help='Use laplacian PE')
parser.add_argument('--FA', type=bool, default=False, help='Use FA Layer')
parser.add_argument('--learnable', type=bool, default=False, help='Use FA Layer')

parser.add_argument('--use_graph', type=bool, default=True, help='Use graph infos')
parser.add_argument('--patches', type=bool, default=True, help='Use graph infos')
parser.add_argument('--pretrain', type=bool, default=False, help='Use graph infos')
parser.add_argument('--lamb', type=float, default=0.5, help='Regularizer')
parser.add_argument('--use_weights', type=bool, default=False, help='Use graph infos')
parser.add_argument('--gConv', type=str, default='GAT', help='graph conv between latents')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--webdata', type=str, default='jkjnknooi', help='which data')
args = parser.parse_args([])

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=args.k)
torch.manual_seed(args.seed)

from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
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



if args.laplace_RW==True:
  from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
  tf=AddRandomWalkPE(walk_length=15, attr_name='rwe') 
else:
   tf=None

my_dataset='PascalVOC-SP'
dataset1 = LRGBDataset(root='./', name=my_dataset, transform=tf, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset, transform=tf,split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes



#weights=torch.rand(args.comp,args.hidden,requires_grad=True).to(device)
# dataset1 = patchData(dataset1,args.comp,weights,device)
# validation_set1 = patchData(validation_set1,args.comp,weights,device)
# test_set1 = patchData(test_set1,args.comp,weights,device)


def pe(datasetz):
  outs=[]
  ins=[]
  dataset2=[]
  for p in range(len(datasetz)):
    try:
      tempo=datasetz[p]
      tempo['laplace']= torch.cat((datasetz[p].x,datasetz[p].rwe),dim=1)
      dataset2.append(tempo)
      ins.append(p)
    except:
      print(p)
      outs.append(p)
  return dataset2


if args.laplace_RW==True:
  dataset1 = pe(dataset1)
  validation_set1 = pe(validation_set1)
  test_set1 = pe(test_set1)


from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
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

        if args.laplace_RW==True:
          total_feats=num_feats+15
        else:
           total_feats=num_feats

        self.conv1 = SAGEConv(total_feats, int(args.hidden))#,K=args.K)
        self.Inlin = Linear(total_feats, int(args.hidden))
        self.Inlin2 = Linear(int(args.hidden), int(args.hidden))


        self.conv2 = SAGEConv(int(args.hidden), int(args.hidden))#,K=args.K)
        self.conv3 = SAGEConv(int(args.hidden), args.hidden)#,K=args.K)
        self.conv4 = SAGEConv(int(args.hidden), args.hidden)#,K=args.K)
        self.conv5 = SAGEConv(int(args.hidden), args.hidden)#,K=args.K)

        self.convRev = GATConv( args.hidden, args.hidden)#,K=8)
        self.convRev2 = GCNConv(int(args.hidden), int(args.hidden))#,K=8)
        
        
        # self.conv5 = ChebConv(int(args.hidden), args.hidden,K=8)
        #self.conv4 = SAGEConv(args.hidden, args.hidden)
        self.convGraph =  GCNConv(args.hidden,args.hidden)
        self.convGraph2 =  GCNConv(args.hidden,args.comp)

        self.Fconv = GCNConv(args.hidden*args.comp, num_classes)

        self.conv_sTc= GATConv(args.hidden, args.hidden)#GATConv(args.hidden, args.hidden)
        #self.Inlin = Linear(dataset.num_features, int(args.hidden))
        self.mlpComp=Linear(args.hidden, int(args.hidden))

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
            args.hidden, num_classes, nlayer=3, with_final_activation=False)


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

    def select_top_k_values(self, A, B, k):
        selected_values = []
        current_index = 0
        counter=0
        inds=[]
        for sample_size in B:
            # Select top-k values from the current sample
            sample_values = A[current_index : current_index + sample_size]
            selected_sample_values, indices = torch.topk(sample_values, k)

            # Append the selected values to the result
            selected_values.append(selected_sample_values)
            inds.append(indices+current_index-args.comp*counter)

            # Move to the next sample
            current_index += sample_size
            counter+=1

        # Concatenate the selected values from all samples
        result = torch.cat(selected_values)

        flatInd=torch.stack(inds).view(-1)
        # cater=torch.stack((ccc.repeat_interleave(3,dim=0),torch.stack(inds).repeat(1,3).view(-1)))
        return result, flatInd #orch.stack(inds).view(-1)

    def fcadj(self,bs,comp):
      nodes=torch.arange(0,comp*bs)
      top=nodes.repeat_interleave(comp, dim=0)
      bot=nodes.reshape(bs,comp).repeat_interleave(comp, dim=0).reshape(-1)
      return torch.stack((top,bot))

    def forward(self, x1, edge_index, batch, params,batch_size,device,data,pretrain):

        if args.use_graph==True:
          x = self.conv1(x1, edge_index)
          x = F.leaky_relu(x)
          x=self.bano1(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv2(x, edge_index)
          x = F.leaky_relu(x)
          x=self.bano2(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv3(x, edge_index)
          x = F.relu(x)
          x = self.bano3(x)
          x = F.dropout(x, training=self.training,p=0.2)

          x = self.conv4(x, edge_index)
          # x = F.relu(x)
          # x = self.bano3(x)
          # x = F.dropout(x, training=self.training,p=0.2)

          # x = self.conv5(x, edge_index)
          # x = F.relu(x)
          # x = self.bano3(x)
          # #x = F.dropout(x, training=self.training,p=0.2)

        # params=self.embLin(params)
        # params = params.relu()


        classifier=self.mlpRep(x) ### was +x with Cheb

        #classifier=self.mlp(CompNodes)
        # classifier=F.leaky_relu(classifier)
        # classifier= F.dropout(classifier, training=self.training,p=0.3)
        # classifier=self.mlp2(classifier)
        cdd=0

        return classifier,cdd, cdd


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


if args.laplace_RW==True:
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
  Changes="Undirected",
  Latent_nodes=args.comp,
  hidden_dim=args.hidden,
  Laplacian=args.laplace,
  k_eigs=eigs,
  batch_size=args.batch_size,
  hops=args.K,

  patches=args.patches,
  use_GNN=args.use_graph,
  learning_rate = args.lr,
  gConv = args.gConv,
  pretrain = args.pretrain,
  seed = args.seed
)


wandb.init(
project="ICML_PascaVOC",
name="SAGE_only",
config=config,
)

def remove_self(att):
  a,b=att[0],att[1]

  # Find indices where a and b are not equal
  indices = torch.where(a != b)

  # Use the indices to get the desired elements
  result_a = a[indices]
  result_b = b[indices]
  return torch.stack((result_a,result_b))
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

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

temp=0
for epoch in range(400):
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

    if args.laplace_RW==True:

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

    # if i%6==0:
    #   torch.nn.utils.clip_grad_norm_(model.parameters(),1)

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



  # if epoch >=40==0:
  #     if epoch%15==0:
  #       optimizer.param_groups[0]["lr"]=optimizer.param_groups[0]["lr"]*0.95
  val_correct=0
  val_precision=0
  totalVaLoss=0
  allValref=[]
  allValpred=[]

  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    if args.laplace_RW==True:
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
  wandb.log({"Epoch": epoch})

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

  if args.laplace_RW==True:
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