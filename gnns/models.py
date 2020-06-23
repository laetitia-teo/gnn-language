import numpy as np
import torch
import babyai
import gym

from torch.nn import Linear, Sequential, ReLU

env = gym.make('BabyAI-GoToRedBall-v0')

### sparse summing op

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

### ReLU MLP

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
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

### MHSA and Transformer

# TODO: adapt for variable input size
# idea: use hybrid sparse-dense tensors provided by torch.sparse

class SelfAttentionLayer(torch.nn.Module):
    """
    Multi-head Self-Attention layer.

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

        # for now values have the same dim as keys and queries
        self.Ftot = (2*Fqk + Fv) * nheads

        self.proj = Linear(Fin, self.Ftot)

    def forward(self, x):
        # x :: [N, Fin]
        N = x.shape[1]
        qkv = self.proj(x) # [N, F]

        # [N, H, F/H]
        qkv = qkv.reshape((-1, N, self.nheads, self.Ftot // self.nheads))
        qkv = qkv.permute(0, 2, 1, 3) # [H, N, F/H]

        q, k, v = qkv.split([self.Fqk, self.Fqk, self.Fv], -1)

        # [H, N, N]
        qk = q @ (k.permute(0, 1, 3, 2))
        qk = qk / np.sqrt(self.Fqk)
        # [H, N, N], what axis ?
        aw = torch.softmax(qk, -1)

        # [H, N, N] x [H, N, F/H] -> [H, N, F/H]
        out = (aw @ v)

        # [N, H, F/H]
        out = out.permute(0, 2, 1, 3)

        # [N, F]
        out = out.reshape((-1, N, self.nheads * self.Fv))

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

        self.mhsa = SelfAttentionLayer(d, d, d*h, h)
        # TODO: check papers for hparams
        self.mlp = MLP([h*d, h*d, h*d])

    def forward(self, x):
        
        y = self.mhsa(x)
        y = self.norm1(x + y)

        z = self.mlp(y)
        z = self.norm2(y + z)

        return z

### Full Relational Memory Core

class RMC(torch.nn.Module):
    """
    Relational Memory Core.
    """
    def __init__(self, N, d, h):
        super().__init__()
        
        # size of the number of slots
        self.N = N
        self.d = d
        self.h = h

        # initialize memory M
        self.M = torch.zeros([self.N, self.d * self.h])

    def forward(self, x):
        # TODO: implement forward pass

### Basic tests

sal = SelfAttentionLayer(5, 5, 2, 2)
x = torch.rand(12, 7, 5)
res = sal(x)
print(res.shape)