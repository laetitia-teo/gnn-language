import numpy as np
import torch
import torch.nn.functional as F

import babyai
import gym

from torch.nn import Linear, Sequential, ReLU

env = gym.make('BabyAI-GoToRedBall-v0')

#### sparse summing op

def scatter_sum(x, batch):
    # sums in the 0th dim according to the indices provided in batch.
    # x has to be 2-dimensional.
    
    # TODO: generalize this function to arbitrary number of dims with
    # sparse-dense hybrid tensors and summing over a particular dim.
    bsize = batch[-1]
    nelems = len(batch)

    i = torch.LongTensor(torch.arange(bsize), batch)
    sparse = torch.sparse.FloatTensor(
        i,
        torch.ones(nelems),
        torch.Size([bsize, nelems]),
    )
    return sparse.mm(x)

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

        out = (aw @ v)
        out = out.transpose(1, 2).reshape(B, N, self.Fv)

        return out

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