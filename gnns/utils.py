import torch
import numpy as np

import babyai
import gym

from scipy.sparse import coo_matrix

# for testing
env = gym.make('BabyAI-GoToRedBall-v0')
obs, _, _, _ = env.step(env.action_space.sample())
x = obs['image']
obs, _, _, _ = env.step(env.action_space.sample())
y = obs['image']
obs, _, _, _ = env.step(env.action_space.sample())
z = obs['image']
X = np.stack([x, y, z], 0)


def get_entities(x, device=torch.device('cpu')):
    """
    Transforms the input array x into a collection of objects.
    Expects a batch of observation arrays (4d) as a numpy array.
    """
    # x shape is Nx7x7x3
    N = x.shape[0]
    x = torch.tensor(x)
    # convert to one hot/embedding
    x = x.float()

    # compute indices for empty space
    testlist = [[0., 0., 0.], [1., 0., 0.]]
    testlist = [torch.tensor(t).expand_as(x) for t in testlist]
    zi = sum([(x == t).prod(-1) for t in testlist])
    nzi = (1 - zi[:, :, :, None]).bool()

    # 2d pos
    a = torch.linspace(-1, 1, 7)
    xy = torch.stack(torch.meshgrid(a, a), -1).expand(N, -1, -1, -1)
    # batch index
    bi = torch.arange(N).float()
    bi = bi[:, None, None] * torch.ones(x.shape[:-1])
    bi = bi[..., None]
    # concatenate all features
    x = torch.cat([x, xy, bi], -1)
    lastdim = x.shape[-1]

    x = torch.masked_select(x, nzi)
    x = x.reshape(-1, lastdim)

    x, batch = x.split([lastdim-1, 1], -1)
    x = x.to(device)
    batch = batch.int().to(device)[:, 0]
    return x, batch

def complete_graph(n, self_edges=False):
    """
    Returns the complete edge index array given the number of nodes.
    """
    I = torch.arange(n)
    ei = torch.stack(torch.meshgrid(I, I), 0).reshape((2, -1))
    # remove self-edges
    ei = ei[:, ei[0] != ei[1]]
    return ei

def get_ei(batch, device=torch.device('cpu')):
    """
    Given a batch index, returns the associated edge index tensor, for a
    fully-connected bi-directional graph with self-edges removed.
    """
    N = len(batch)

    # get numbers of objects from conversion between sparse matrix formats
    coo = coo_matrix((np.empty(N), (batch.numpy(), np.arange(N))))
    cum = coo.tocsr().indptr
    ni = cum[1:] - cum[:-1]
    print(ni)

    # get edge index tensor
    ei = torch.cat([complete_graph(n) + cn for n, cn in zip(ni, cum)], 1)
    ei = ei.to(device)

    return ei

def get_graph(x, device=torch.device('cpu')):
    """
    Takes in a batch of babyai observations as input and outputs the nodes,
    batch, edges, and edge indices corresponding to the underlying complete
    graph without self-edges.
    """
    x, batch = get_entities(x, device)
    ei = get_ei(batch, device)
    
    # edges features are concatenatenations of node features for source
    # and destination nodes
    src, dest = ei
    e = torch.cat([x[src], x[dest]], 1)

    return x, batch, ei, e