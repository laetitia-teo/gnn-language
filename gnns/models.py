import numpy as np
import torch
import torch.nn.functional as F

import babyai
import gym

from torch.nn import Linear, Sequential, ReLU

env = gym.make('BabyAI-GoToRedBall-v0')

#### sparse reduction ops

def scatter_sum(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    i = torch.LongTensor(batch, torch.arange(nelems))
    
    st = torch.sparse.FloatTensor(
        i,
        x, 
        torch.Size([nbatches, nelems]),
    )
    return torch.sparse.sum(st, dim=1).values()

def scatter_mean(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    i = torch.LongTensor(batch, torch.arange(nelems))
    
    st = torch.sparse.FloatTensor(
        i,
        x, 
        torch.Size([nbatches, nelems]),
    )
    ost = torch.sparse.FloatTensor(
        i,
        torch.ones(nelems), 
        torch.Size([nbatches, nelems]),
    )
    xsum = torch.sparse.sum(st, dim=1).values()
    nx = torch.sparse.sum(ost, dim=1).values().view([-1, 1, 1])
    return xsum / nx

def scatter_softmax(x, batch):
    """
    Computes the softmax-reduction of elements of x as given by the batch index
    tensor.
    """
    nbatches = batch[-1] + 1
    nelems = len(batch)
    i = torch.LongTensor(batch, torch.arange(nelems))
    
    # TODO: patch for numerical stability
    exp = x.exp()
    st = torch.sparse.FloatTensor(
        i,
        exp,
        torch.Size([nbatches, nelems]),
    )
    expsum = torch.sparse.sum(st, dim=1).values()[batch]
    return exp / expsum

def scatter_softmax_nums(x, batch):
    """
    Computes the softmax-reduction of elements of x as given by the batch index
    tensor.
    """
    nbatches = batch[-1] + 1
    nelems = len(batch)
    i = torch.LongTensor(batch, torch.arange(nelems))
    
    # TODO: patch for numerical stability
    exp = x.exp()
    # sustract largest element for numerical stability of softmax
    xm = x.max(0)
    exp -= mx
    st = torch.sparse.FloatTensor(
        i,
        exp,
        torch.Size([nbatches, nelems]),
    )
    expsum = torch.sparse.sum(st, dim=1).values()[batch]
    return exp / expsum

#### ReLU MLP

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(ReLU())
            ls = lin

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#### MHSA and Transformer

# TODO: adapt for variable input size
# idea: use hybrid sparse-dense tensors provided by torch.sparse

class SelfAttentionLayer(torch.nn.Module):
    """
    Multi-head Self-Attention layer.

    TODO: add support for diffrent numbers of objects between queries and
        keys/values.

    Inputs:
        - x: the input data for a minibatch, concatenated in the 0-th dim.
        - batch: an index tensor giving the indices of the elements of x in
            minibatch.
    """
    def __init__(self, Fin, Fqk, Fv, nheads):
        super().__init__()
        self.Fin = Fin # in features
        self.Fqk = Fqk # features for dot product
        self.Fv = Fv # features for values
        self.nheads = nheads

        assert Fqk // nheads == Fqk / nheads, "self-attention features must "\
            " be divisible by number of heads"
        assert Fv // nheads == Fv / nheads, "value features must "\
            " be divisible by number of heads"

        # for now values have the same dim as keys and queries
        self.Ftot = (2*Fqk + Fv)

        self.proj = Linear(Fin, self.Ftot, bias=False)

    def forward(self, x):
        # alternative formulation of forward pass

        B, N, _ = x.shape
        H = self.nheads
        Fh = self.Fqk // H
        Fhv = self.Fv // H

        scaling = float(Fh) ** -0.5
        q, k, v = self.proj(x).split([self.Fqk, self.Fqk, self.Fv], dim=-1)

        q = q * scaling
        q = q.reshape(B, N, H, Fh).transpose(1, 2)
        k = k.reshape(B, N, H, Fh).transpose(1, 2)
        v = v.reshape(B, N, H, Fhv).transpose(1, 2)

        aw = q @ (k.transpose(2, 3))
        aw = torch.softmax(aw, dim=-1)

        print(f"shape: {aw.shape}")

        out = (aw @ v)
        print(f"shape: {out.shape}")
        out = out.transpose(1, 2).reshape(B, N, self.Fv)

        return out

class SelfAttentionLayerSparse(torch.nn.Module):
    """
    Sparse version of the above, for accepting batches with different
    numbers of objects.
    """
    def __init__(self, Fin, Fqk, Fv, nheads):
        super().__init__()
        self.Fin = Fin # in features
        self.Fqk = Fqk # features for dot product
        self.Fv = Fv # features for values
        self.nheads = nheads

        assert Fqk // nheads == Fqk / nheads, "self-attention features must "\
            " be divisible by number of heads"
        assert Fv // nheads == Fv / nheads, "value features must "\
            " be divisible by number of heads"

        # for now values have the same dim as keys and queries
        self.Ftot = (2*Fqk + Fv)

        self.proj = Linear(Fin, self.Ftot, bias=False)

    def forward(self, x, batch, ei):
        # batch is the batch index tensor
        # TODO: implement for source and target tensors ?
        # that way we get rid of excess computation

        src, dest = ei

        B = batch[-1] + 1
        maxbsz = batch.max()
        H = self.nheads
        Fh = self.Fqk // H
        Fhv = self.Fv // H

        scaling = float(Fh) ** -0.5
        q, k, v = self.proj(x).split([self.Fqk, self.Fqk, self.Fv], dim=-1)

        q = q * scaling
        q = q.reshape(-1, H, Fh)
        k = k.reshape(-1, H, Fh)
        v = v.reshape(-1, H, Fhv)

        qs, ks, vs = q[src], k[dest], v[dest]
        # dot product
        aw = qs.view(-1, H, 1, Fh) @ ks.view(-1, H, Fh, 1)
        aw = aw.squeeze(-1)
        # softmax reduction
        eb = batch[src]


class TransformerBlock(torch.nn.Module):
    """
    Implements a full Transformer block, with skip connexions, layernorm
    and an mlp.

    Arguments:
        - d: dimension of a head
        - h: number of heads
    """
    def __init__(self, d, h):
        super().__init__()

        self.d = d
        self.h = h

        # TODO: is this correct ? Check the paper
        self.norm1 = torch.nn.LayerNorm([d * h])
        self.norm2 = torch.nn.LayerNorm([d * h])

        self.mhsa = SelfAttentionLayer(d*h, d, d, h)
        # TODO: check papers for hparams
        self.mlp = MLP([h*d, h*d, h*d])

    def forward(self, x):

        y = self.mhsa(x)
        y = self.norm1(x + y)

        z = self.mlp(y)
        z = self.norm2(y + z)

        return z

#### Full Relational Memory Core

class RMC(torch.nn.Module):
    """
    Relational Memory Core.
    """
    modes = ['RNN', 'LSTM']

    def __init__(self, N, d, h, b, mode='RNN'):
        super().__init__()
        
        # TODO: do we need the batch size in advance ?
        self.N = N # number of slots
        self.d = d # dimension of a head
        self.h = h # number of heads
        self.b = b # batch size

        # initialize memory M
        M = torch.zeros([self.b, self.N, self.d * self.h])
        self.register_buffer('M', M)

        # modules
        self.self_attention = TransformerBlock(d, h)

    def forward(self, x):
        # vanilla recurrent pass
        # x :: [b, N, f]

        M_cat = torch.cat([self.M, x], 1)
        M = self.self_attention(M_cat)[:, :self.N]

        self.register_buffer('M', M)

        # TODO: output

#### Basic tests

# sal = SelfAttentionLayer(10, 10, 10, 2)
# x = torch.rand(12, 7, 10)
# res = sal(x)
# print(res.shape)

# rmc = RMC(5, 13, 3, 7)
# x = torch.rand(7, 5, 13*3)


#### Test that our implem for multi-head self-attention gives the same results
#    as the pytorch one

seed = 0
Fin = 512
nheads = 8
Fv = 512
Fqk = 512

torch.manual_seed(seed)

sal = SelfAttentionLayer(
    Fin=Fin,
    Fqk=Fqk,
    Fv=Fv,
    nheads=nheads,
)
tsal = torch.nn.MultiheadAttention(
    embed_dim=Fin,
    num_heads=nheads,
    bias=False,
)

tsal.in_proj_weight = sal.proj.weight
# set out map to identity
tsal.out_proj.weight = torch.nn.Parameter(torch.eye(Fv))

X = torch.rand(128, 5, 512)
Xt = X.transpose(0, 1)

# res = F.multi_head_attention_forward()
res1 = tsal(Xt, Xt, Xt)[0].transpose(0, 1)
res2 = sal(X)

resf = F.multi_head_attention_forward(
    Xt,
    Xt,
    Xt,
    Fin,
    nheads,
    sal.proj.weight,
    None,
    None,
    None,
    False,
    0.,
    torch.eye(512),
    None)[0].transpose(0, 1)