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
    if not self_edges:
        ei = ei[:, ei[0] != ei[1]]
    return ei

def complete_crossgraph(n, m, N, bi_directed=True):
    """
    This function computes the edge indices for the following situation:
    there are two graphs with respective number of nodes n and m, the indices
    of nodes in the first graph go from 0 to n-1 and the indices of nodes in
    the second graph go from N to m + N - 1.

    Used when concatenating graphs that belong to the same scenes, e.g. when
    assembling objects and memory in RMC.

    If bi_directed is False, we only return the edges from graph 2 to graph 1.
    """
    In = torch.arange(n)
    Im = torch.arange(m)
    eix, eiy = torch.meshgrid(In, Im)
    eix = eix + N
    ei = torch.stack([eix, eiy], 0).reshape(2, -1)

    if bi_directed:
        # connect back nodes to make bi-directional edges
        ei = torch.cat([ei, ei.flip(0)], 1)

    return ei

b1 = [0, 0, 0, 1, 1]
b2 = [0, 0, 1, 1]

def get_ei(batch, self_edges=True, device=torch.device('cpu')):
    """
    Given a batch index, returns the associated edge index tensor, for a
    fully-connected bi-directional graph with self-edges removed.
    """
    if not isinstance(batch, torch.Tensor):
        batch = torch.tensor(batch)

    N = len(batch)

    # get numbers of objects from conversion between sparse matrix formats
    coo = coo_matrix((np.empty(N), (batch.numpy(), np.arange(N))))
    cum = coo.tocsr().indptr
    ni = cum[1:] - cum[:-1]

    # get edge index tensor
    ei = torch.cat([complete_graph(n, self_edges) + cn \
                    for n, cn in zip(ni, cum)], 1)
    ei = ei.to(device)

    return ei

def get_crossgraph_ei(batch1,
                      batch2,
                      bi_directed=True,
                      device=torch.device('cpu')):
    """
    Get cross graph ei cross_graph edge index for two provided batchtensors.
    """
    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2)

    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    coo1 = coo_matrix((np.empty(N), (batch1.numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    coo2 = coo_matrix((np.empty(M), (batch2.numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]

    # get edge index tensor
    ei = torch.cat(
        [complete_crossgraph(m, n, N, bi_directed) + cn\
            for n, m, cn in zip(ni1, ni2, cum2)],
        1,
    )
    ei = ei.to(device)

    return ei

def get_all_ei(batch1,
               batch2,
               self_edges=True,
               bi_directed=True,
               device=torch.device('cpu')):
    """
    Gets all eis (inter-graph, cross-graph) by combining the two previous
    functions.

    TODO: maybe there's a smarter way to compute this.
    """
    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2)

    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    coo1 = coo_matrix((np.empty(N), (batch1.numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    ei1 = torch.cat([complete_graph(n, self_edges) + cn \
                     for n, cn in zip(ni1, cum1)], 1)

    coo2 = coo_matrix((np.empty(M), (batch2.numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]

    ei2 = torch.cat([complete_graph(n, self_edges) + cn \
                     for n, cn in zip(ni2, cum2)], 1)

    # get edge index tensor
    ei12 = torch.cat(
        [complete_crossgraph(m, n, N, False) \
            + torch.tensor([cm, cn]).view(2, 1) \
            for n, m, cn, cm in zip(ni1, ni2, cum1, cum2)],
        1,
    )
    if bi_directed:
        ei12 = torch.cat([ei12, ei12.flip(0)], 1)

    ei = torch.cat([ei1, ei2 + N, ei12], 1)
    ei = ei.to(device)

    return ei

def get_ei_from(batch1,
                batch2,
                self_edges=True,
                bi_directed=True,
                device=torch.device('cpu')):
    """
    Used in memory module, only keep queries from first input.
    """
    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2)

    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    coo1 = coo_matrix((np.empty(N), (batch1.numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    ei1 = torch.cat([complete_graph(n, self_edges) + cn \
                     for n, cn in zip(ni1, cum1)], 1)

    coo2 = coo_matrix((np.empty(M), (batch2.numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]

    # get cross-graph edge index tensor
    ei12 = torch.cat(
        [complete_crossgraph(m, n, N, False) \
            + torch.tensor([cm, cn]).view(2, 1) \
            for n, m, cn, cm in zip(ni1, ni2, cum1, cum2)],
        1,
    ).flip(0)

    ei = torch.cat([ei1, ei12], 1)
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