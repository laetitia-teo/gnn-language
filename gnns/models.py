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

        assert Fqk // nheads == Fqk / nheads, "self-attention features must "\
            " be divisible by number of heads"
        assert Fv // nheads == Fv / nheads, "value features must "\
            " be divisible by number of heads"

        # for now values have the same dim as keys and queries
        self.Ftot = (2*Fqk + Fv)

        self.proj = Linear(Fin, self.Ftot, bias=False)

    def forward(self, x):

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
        # dot product
        aw = qs.view(-1, H, 1, Fh) @ ks.view(-1, H, Fh, 1)
        aw = aw.squeeze()
        # softmax reduction
        aw = scatter_softmax(aw, src)

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
    Sparse version of the above, different input format.
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

### Base class for relational memory

class RelationalMemory(torch.nn.Module):
    """
    TODO: change this into sparse version.

    Defines the base class for all relational memory modules.
    K is the number of slots in the memory, Fmem is the feature dimension for
    the slots. (B is the batch size when used in batch computation.)

    A subclass has to define:
        - a _one_step method that outputs a tuple of (next_memory, output);
        - optionally, a _mem_init method that initializes the memory.
    """
    def __init__(self, B, K, Fmem):

        super().__init__()
        self.B = B
        self.K = K
        self.Fmem = Fmem

        M = self._mem_init()
        self.register_buffer('M', M)

    def _mem_init(self):
        """
        Default initialization method for the memory: the identity matrix of
        size K is completed with zeros (to ensure different behaviors for each
        of the slots).
        """
        M = torch.cat([
            torch.eye(self.N).expand([self.B, self.N, self.N]),
            torch.zeros([self.B, self.N, self.d * self.h - self.N]),
            -1
        ])

    def _one_step(self, x, M):
        """
        Dummy memory update.
        """
        return M, M[:, 0]

    def forward(self, x):
        M, out = self._one_step(x, self.M)
        self.register_buffer('M', M)

        return out

#### Full Relational Memory Core

class RMC(torch.nn.Module):
    """
    Relational Memory Core.

    Please note that using this model with multiple input vectors, in LSTM
    mode leads to concatenating the inputs in the first dimension when
    computing the gates for the LSTM update. This seems suboptimal from a
    relational architecture point of view, and would orient us towards using
    a per-slot LSTM (where the gates would be computed per-slot) instead.

    -> It breaks the permutation invariance of inputs
    """
    modes = ['RNN', 'LSTM']

    def __init__(self,
                 N,
                 d,
                 h,
                 b,
                 Nx=None,
                 mode='RNN',
                 device=torch.device('cpu')):
        super().__init__()
        
        # TODO: do we need the batch size in advance ?
        self.N = N # number of slots
        if Nx is None: 
            self.Nx = N # number of slots for the input
        else:
            self.Nx = Nx
        self.d = d # dimension of a single head
        self.h = h # number of heads
        self.b = b # batch size

        self.device = device
        self.mode = mode

        # initialize memory M
        M = torch.cat([
            torch.eye(self.N).expand([self.b, self.N, self.N]),
            torch.zeros([self.b, self.N, self.d * self.h - self.N])
        ])
        # M = torch.zeros([self.b, self.N, self.d * self.h])
        self.register_buffer('M', M)

        # modules
        self.self_attention = TransformerBlock(h * d, h)

        if mode in ['LSTM', 'LSTM_noout']:
            # hidden state
            hid = torch.zeros([self.b, self.N, self.d * self.h])
            self.register_buffer('hid', hid)

            # scalar LSTM gates
            self.Wf = Linear(d * h * self.Nx, 1)
            self.Uf = Linear(d * h, 1)

            self.Wi = Linear(d * h * self.Nx, 1)
            self.Ui = Linear(d * h, 1)

            if mode == 'LSTM':
                self.Wo = Linear(d * h * self.Nx, 1)
                self.Uo = Linear(d * h, 1)

    def _forwardRNN(self, x):
        # vanilla recurrent pass
        # x :: [b, N, f]

        M_cat = torch.cat([self.M, x], 1)
        M = self.self_attention(M_cat)[:, :self.N]

        self.register_buffer('M', M)

        # output is flattened memory
        out = M.view(self.b, -1)
        return out

    def _forwardLSTM(self, x):
        # LSTM recurrent pass
        M_cat = torch.cat([self.M, x], 1)
        Mtilde = self.self_attention(M_cat)[:, :self.N]

        x_cat = x.flatten(1).unsqueeze(1)

        f = self.Wf(x_cat) + self.Uf(self.M)
        i = self.Wi(x_cat) + self.Uf(self.M)
        o = self.Wo(x_cat) + self.Uo(self.M)

        M = torch.sigmoid(f) * self.M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = torch.sigmoid(o) * torch.tanh(M)

        self.register_buffer('M', M)

        return hid.view(self.b, -1)

    def _forwardLSTM_noout(self, x):
        # LSTM recurrent pass, no output gate
        M_cat = torch.cat([self.M, x], 1)
        Mtilde = self.self_attention(M_cat)[:, :self.N]
        
        x_cat = x.flatten(1).unsqueeze(1)

        f = self.Wf(x_cat) + self.Uf(self.M)
        i = self.Wi(x_cat) + self.Uf(self.M)

        M = torch.sigmoid(f) * M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = M

        self.register_buffer('M', M)

        return hid.view(self.b, -1)

    def forward(self, x):
        if self.mode == 'RNN':
            return self._forwardRNN(x)
        elif self.mode == 'LSTM':
            return self._forwardLSTM(x)
        elif self.mode == 'LSTM_noout':
            return self._forwardLSTM_noout(x)

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

        # initialize memory M, batch and edge index
        eye = torch.eye(N).expand([b, N, N])
        eyeflatten = eye.reshape(b * N, N)
        M = torch.cat([
            eyeflatten,
            torch.zeros(b * N, d * h - N)])

        Mbatch = torch.arange(b).expand(N, b).transpose(0, 1).flatten()

        # we dont need edge indices, we compute them at each time step (?)
        self.register_buffer('M', M)
        self.register_buffer('Mbatch', Mbatch)

        # modules
        self.self_attention = TransformerBlockSparse(d, h)

        if mode in ['LSTM', 'LSTM_noout']:
            # hidden state
            hid = torch.zeros([self.b, self.N, self.d * self.h])
            self.register_buffer('hid', hid)

            # scalar LSTM gates
            self.Wf = Linear(d * h, 1)
            self.Uf = Linear(d * h, 1)

            self.Wi = Linear(d * h, 1)
            self.Ui = Linear(d * h, 1)

            if mode == 'LSTM':
                self.Wo = Linear(d * h, 1)
                self.Uo = Linear(d * h, 1)

    def _forwardRNN(self, x, xbatch):
        """
        Vanilla recurrent pass.

        Output is concatenation of the memory in the -1 dim.
        """
        M_cat = torch.cat([self.M, x], 0)
        batch_cat = torch.cat([self.Mbatch, xbatch], 0)
        # TODO: only one-way attn between M and X, check it works
        ei_cat = utils.get_all_ei(self.Mbatch, xbatch)

        # TODO: check the following:
        #   - that it runs;
        #   - that the indices selected are the good ones
        M = self.self_attention(M_cat, batch_cat, ei_cat)[:(self.b * self.N)]
        self.register_buffer('M', M)
        
        out = M.reshape(self.b, self.N, self.d * self.h).view(self.b, -1)
        return out

    def _forwardLSTM(self, x, xbatch):
        """
        LSTM recurrent pass.
        """
        M_cat = torch.cat([self.M, x], 0)
        batch_cat = torch.cat([self.Mbatch, xbatch], 0)
        ei_cat = utils.get_all_ei(self.Mbatch, xbatch)

        Mtilde = self.self_attention(M_cat, batch_cat, ei_cat)
        Mtilde = Mtilde[:(self.b * self.N)]

        # for now we use sum
        # TODO: check validity of broadcasting according to M
        x_sum = scatter_sum(x, xbatch)[self.Mbatch]

        f = self.Wf(x_sum) + self.Uf(self.M)
        i = self.Wi(x_sum) + self.Uf(self.M)
        o = self.Wo(x_sum) + self.Uo(self.M)

        M = torch.sigmoid(f) * self.M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = torch.sigmoid(o) * torch.tanh(M)

        self.register_buffer('M', M)

        hid = hid.reshape(self.b, self.N, self.d * self.h).view(self.b, -1)
        return hid

    def _forwardLSTM_noout(self, x, xbatch):
        """
        LSTM recurrent pass, no output gate.
        """
        M_cat = torch.cat([self.M, x], 0)
        batch_cat = torch.cat([self.Mbatch, xbatch], 0)
        ei_cat = utils.get_all_ei(self.Mbatch, xbatch)

        Mtilde = self.self_attention(M_cat, batch_cat, ei_cat)
        Mtilde = Mtilde[:(self.b * self.N)]

        # for now we use sum
        # TODO: check validity of broadcasting according to M
        x_sum = scatter_sum(x, xbatch)[self.Mbatch]

        f = self.Wf(x_sum) + self.Uf(self.M)
        i = self.Wi(x_sum) + self.Uf(self.M)

        M = torch.sigmoid(f) * self.M + torch.sigmoid(i) * torch.tanh(Mtilde)
        hid = M

        self.register_buffer('M', M)

        hid = hid.reshape(self.b, self.N, self.d * self.h).view(self.b, -1)
        return hid

    def forward(self, x, batch):
        if self.mode == 'RNN':
            return self._forwardRNN(x, batch)
        elif self.mode == 'LSTM':
            return self._forwardLSTM(x, batch)
        elif self.mode == 'LSTM_noout':
            return self._forwardLSTM_noout(x, batch)

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

# test dense RMC implem

# rmc = RMC(4, 10, 2, 2)
# rmc2 = RMC(4, 10, 2, 2, mode="LSTM", Nx=7)
# x = torch.rand(2, 7, 20)