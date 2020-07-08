import numpy as np
import torch
import torch.nn.functional as F

import babyai
import gym

import utils

from torch.nn import Linear, Sequential, ReLU

env = gym.make('BabyAI-GoToRedBall-v0')

#### sparse reduction ops

def scatter_sum(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    st = torch.sparse.FloatTensor(
        i,
        x, 
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    return torch.sparse.sum(st, dim=1).values()

def scatter_mean(x, batch):
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    st = torch.sparse.FloatTensor(
        i,
        x, 
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    ost = torch.sparse.FloatTensor(
        i,
        torch.ones(nelems), 
        torch.Size([nbatches, nelems]),
    )
    xsum = torch.sparse.sum(st, dim=1).values()
    print(xsum.shape)
    nx = torch.sparse.sum(ost, dim=1).values().view([-1, 1])
    print(nx.shape)
    return xsum / nx

def scatter_softmax(x, batch):
    """
    Computes the softmax-reduction of elements of x as given by the batch index
    tensor.
    """
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    # TODO: patch for numerical stability
    exp = x.exp()
    st = torch.sparse.FloatTensor(
        i,
        exp,
        torch.Size([nbatches, nelems] + list(x.shape[1:])),
    )
    expsum = torch.sparse.sum(st, dim=1).values()[batch]
    return exp / expsum

def scatter_softmax_nums(x, batch):
    """
    Computes the softmax-reduction of elements of x as given by the batch index
    tensor.

    TODO: fix this, values on last dim are not independent
    """
    nbatches = batch[-1] + 1
    nelems = len(batch)
    fx = x.shape[-1]
    i = torch.stack([batch, torch.arange(nelems)])
    
    exp = x.exp()
    # sustract largest element for numerical stability of softmax
    xm = x.max(0).values
    exp = exp - xm
    st = torch.sparse.FloatTensor(
        i,
        exp,
        torch.Size([nbatches, nelems] + list(exp.shape[1:])),
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
        aw = torch.softmax(aw, dim=-2)

        # print(f"shape: {aw.shape}")

        out = (aw @ v)
        # print(f"shape: {out.shape}")
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
        print(f"qs shape: {qs.shape}")
        # dot product
        aw = qs.view(-1, H, 1, Fh) @ ks.view(-1, H, Fh, 1)
        print(f"aw shape: {aw.shape}")
        aw = aw.squeeze()
        print(f"aw shape: {aw.shape}")
        print(f"src: {src}")
        # softmax reduction
        aw = scatter_softmax(aw, src)
        print(f"aw shape, after softmax: {aw.shape}")

        out = aw.view([-1, H, 1]) * vs
        out = scatter_sum(out, src)
        out = out.reshape([-1, H * Fhv])

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

        self.norm1 = torch.nn.LayerNorm([d])
        self.norm2 = torch.nn.LayerNorm([d])

        self.mhsa = SelfAttentionLayer(d, d, d, h)
        # TODO: check papers for hparams
        self.mlp = MLP([d, d, d])

    def forward(self, x):

        y = self.mhsa(x)
        y = self.norm1(x + y)

        z = self.mlp(y)
        z = self.norm2(y + z)

        return z

class TransformerBlockSparse(torch.nn.Module):
    """
    Sparse version of the above, different semantics for the input format.
    """
    def __init__(self, d, h):
        super().__init__()

        self.d = d
        self.h = h

        # TODO: Do those layers work correctly in the sparse case ?
        self.norm1 = torch.nn.LayerNorm([d])
        self.norm2 = torch.nn.LayerNorm([d])

        self.mhsa = SelfAttentionLayerSparse(d, d, d, h)
        # TODO: check papers for hparams
        self.mlp = MLP([d, d, d])

    def forward(self, x, batch, ei):

        y = self.mhsa(x, batch, ei)
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

    def __init__(self, N, d, h, b, mode='RNN', device=torch.device('cpu')):
        super().__init__()
        
        # TODO: do we need the batch size in advance ?
        self.N = N # number of slots
        self.d = d # dimension of a head
        self.h = h # number of heads
        self.b = b # batch size

        self.device = device

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

class RMCSparse(torch.nn.Module):
    """
    Relational Memory Core, sparse version.
    """
    def __init__(self, N, d, h, b, mode='RNN', device=torch.device('cpu')):
        super().__init__()
        
        # TODO: do we need the batch size in advance ?
        self.N = N # number of slots
        self.d = d # dimension of a head
        self.h = h # number of heads
        self.b = b # batch size

        self.device = device

        # initialize memory M (+ batch and edge index)
        M = torch.zeros([self.b * self.N, self.d * self.h])
        Mbatch = torch.ones(b, N) * torch.arange(b).unsqueeze(-1).flatten()

        self.register_buffer('M', M)
        self.register_buffer('Mbatch', Mbatch)
        # TODO: ei init
        # modules
        self.self_attention = TransformerBlockSparse(d, h)

    def forward(self, x, xbatch):
        # vanilla recurrent pass
        # x :: [b, N, f]

        # TODO: add concatenation of batch and ei
        #       concat memory in the correct dims

        M_cat = torch.cat([self.M, x], 0)
        batch_cat = torch.cat([self.Mbatch, xbatch])

        # TODO: only one-way attn between M and X, check it works
        ei_cat = utils.get_all_ei(self.Mbatch, xbatch)
        # TODO: check the following:
        #   - that it runs;
        #   - that the indices selected are the good ones
        M = self.self_attention(M_cat, batch_cat, ei_cat)[:(self.b * self.N)]

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

# sparse reduction ops testing 

x = torch.rand(4, 10)
xd = x.view(2, 2, 10)
batch = torch.LongTensor([0, 0, 1, 1])

# sparse self-attention layer testing

x = torch.rand(4, 100)
xb = x.reshape(2, 2, 100)
batch = torch.LongTensor([0, 0, 1, 1])
ei = torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3],
                       [0, 1, 0, 1, 2, 3, 2, 3]])
ssal = SelfAttentionLayerSparse(Fin=100, Fqk=100, Fv=100, nheads=2)
sal = SelfAttentionLayer(Fin=100, Fqk=100, Fv=100, nheads=2)
sal.proj.weight = ssal.proj.weight

res = ssal(x, batch, ei)
res2 = sal(xb).reshape(4, 100)