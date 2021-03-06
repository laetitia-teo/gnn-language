import time

import torch
import numpy as np

import babyai
import gym

from scipy.sparse import coo_matrix, block_diag
import time

# for testing
env = gym.make('BabyAI-GoToRedBall-v0')
obs, _, _, _ = env.step(env.action_space.sample())
x = obs['image']
obs, _, _, _ = env.step(env.action_space.sample())
y = obs['image']
obs, _, _, _ = env.step(env.action_space.sample())
z = obs['image']
X = np.stack([x, y, z], 0)


### babyai utils

def to_one_hot(x, max_label=[11, 6, 3], device=None):
    """
    Provided with a 2-dimensional integer tensor encoding the classes of a set
    of objects and a max_label tuple of the same length as the last dim of the
    tensor encoding the maximum number of classes, returns the one hot vector
    associated.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    ndims = len(x.shape)

    if ndims == 1:
        x = x.unsqueeze(-1)
    elif ndims != 2:
        raise ValueError("The input tensor must have 1 or 2 dimensions, "
                         f"but {ndims} were given")

    l = len(x)

    # add offset to indices further away
    max_label = torch.tensor([0] + list(max_label), device=device)
    max_label = max_label.cumsum(0)
    offsets = max_label[:-1].expand(l, x.shape[1])

    oh_tensor = torch.zeros(l, max_label[-1], device=device)
    oh_tensor.scatter_(1, x.long() + offsets, 1.)  # TODO check for +1

    return oh_tensor


def get_entities(x, device=None, one_hot=False):
    """
    Transforms the input array x into a collection of objects.
    Expects a batch of observation arrays (4d) as a numpy array.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")
    # x shape is Nx7x7x3
    N = x.shape[0]
    x = torch.tensor(x, device=device)
    # convert to one hot/embedding
    x = x.float()

    # compute indices for empty space
    testlist = [[0., 0., 0.], [1., 0., 0.]]
    testlist = [torch.tensor(t, device=device).expand_as(x) for t in testlist]
    zi = sum([(x == t).prod(-1) for t in testlist])
    nzi = (1 - zi[:, :, :, None]).bool()

    # 2d pos
    a = torch.linspace(-1, 1, 7, device=device)
    xy = torch.stack(torch.meshgrid(a, a), -1).expand(N, -1, -1, -1)
    # batch index
    bi = torch.arange(N, device=device).float()
    bi = bi[:, None, None] * torch.ones(x.shape[:-1], device=device)
    bi = bi[..., None]
    # concatenate all features
    x = torch.cat([x, xy, bi], -1)
    lastdim = x.shape[-1]

    x = torch.masked_select(x, nzi)
    x = x.reshape(-1, lastdim)

    x, batch = x.split([lastdim - 1, 1], -1)
    # x = x.to(device)
    batch = batch.int()[:, 0]

    if one_hot:
        x_oh = to_one_hot(x[:, :-2], device=device).float()
        x = torch.cat([x_oh, x[:, -2:]], -1)
    return x, batch


def get_entities_dense(x, device=None, one_hot=False):
    """
    Preprocess the image as a batch of objects, by flattening the symbolic info
    and concatenating it with the xy position of each object.
    Empty space is also represented.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    N = x.shape[0]
    x = torch.tensor(x, device=device)
    x = x.float()

    # 2d pos
    a = torch.linspace(-1, 1, 7, device=device)
    xy = torch.stack(torch.meshgrid(a, a), -1).expand(N, -1, -1, -1)

    x = torch.cat([x, xy], -1)
    x = x.reshape(-1, 49, x.shape[-1])

    if one_hot:
        x = x.reshape(-1, x.shape[-1])
        x_oh = to_one_hot(x[:, :-2], device=device).float()
        x = torch.cat([x_oh, x[:, -2:]], -1)
        x = x.reshape(-1, 49, x.shape[-1])
    return x


### batch index creation utils

def create_batch_tensor(B, n):
    # B is number of batches
    # n is number of slots per batch
    return torch.arange(B).unsqueeze(1).expand(B, n).flatten()


### edge index creation utils

def complete_graph(n, self_edges=False, device=None):
    """
    Returns the complete edge index array given the number of nodes.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    I = torch.arange(n, device=device)
    ei = torch.stack(torch.meshgrid(I, I), 0).reshape((2, -1))
    # remove self-edges
    if not self_edges:
        ei = ei[:, ei[0] != ei[1]]
    return ei


def complete_crossgraph(n, m, N, bi_directed=True, device=None):
    """
    This function computes the edge indices for the following situation:
    there are two graphs with respective number of nodes n and m, the indices
    of nodes in the first graph go from 0 to n-1 and the indices of nodes in
    the second graph go from N to m + N - 1.

    Used when concatenating graphs that belong to the same scenes, e.g. when
    assembling objects and memory in RMC.

    If bi_directed is False, we only return the edges from graph 2 to graph 1.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    In = torch.arange(n, device=device)
    Im = torch.arange(m, device=device)
    eix, eiy = torch.meshgrid(In, Im)
    eix = eix + N
    ei = torch.stack([eix, eiy], 0).reshape(2, -1)

    if bi_directed:
        # connect back nodes to make bi-directional edges
        ei = torch.cat([ei, ei.flip(0)], 1)

    return ei


b1 = [0, 0, 0, 1, 1]
b2 = [0, 0, 1, 1]


def get_ei(batch, self_edges=True, device=None):
    """
    Given a batch index, returns the associated edge index tensor, for a
    fully-connected bi-directional graph with self-edges removed.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    if not isinstance(batch, torch.Tensor):
        batch = torch.tensor(batch)

    N = len(batch)

    # get numbers of objects from conversion between sparse matrix formats
    coo = coo_matrix((np.empty(N), (batch.cpu().numpy(), np.arange(N))))
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
                      device=None):
    """
    Get cross graph ei cross_graph edge index for two provided batchtensors.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2)

    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    coo1 = coo_matrix((np.empty(N), (batch1.cpu().numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    coo2 = coo_matrix((np.empty(M), (batch2.cpu().numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]

    # get edge index tensor
    ei = torch.cat(
        [complete_crossgraph(m, n, N, bi_directed) + cn \
         for n, m, cn in zip(ni1, ni2, cum2)],
        1,
    )
    ei = ei.to(device)

    return ei


def get_all_ei(batch1,
               batch2,
               self_edges=True,
               bi_directed=True,
               device=None):
    """
    Gets all eis (inter-graph, cross-graph) by combining the two previous
    functions.

    TODO: maybe there's a smarter way to compute this.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2)

    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    coo1 = coo_matrix((np.empty(N), (batch1.cpu().numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    ei1 = torch.cat([complete_graph(n, self_edges) + cn \
                     for n, cn in zip(ni1, cum1)], 1)

    coo2 = coo_matrix((np.empty(M), (batch2.cpu().numpy(), np.arange(M))))
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
                device=None):
    """
    Computes the edge indices:
        - from tokens in batch1 to themselves;
        - from tokens in batch1 to tokens in batch2.

    Note that we do not compute edge indices from tokens in batch2 to 
    themselves or to tokens in batch1.

    Used with a sparse self-attention layer, this means the objects in batch1
    attend to themselves and to tokens in batch2, bit not the other way around.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1, device=device)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2, device=device)
    # batch1 = torch.LongTensor([20, 20, 20, 20, 0, 0, 0, 0])
    # batch2 = torch.LongTensor([20, 20, 0, 0, 0])
    N = len(batch1)
    M = len(batch2)

    # get numbers of objects for both batch tensors
    t0 = time.time()
    coo1 = coo_matrix((np.empty(N), (batch1.cpu().numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]
    t_ni1 = time.time() - t0

    t0 = time.time()
    ei1 = torch.cat([complete_graph(n, self_edges) + cn \
                     for n, cn in zip(ni1, cum1)], 1)
    t_ei1 = time.time() - t0

    t0 = time.time()
    coo2 = coo_matrix((np.empty(M), (batch2.cpu().numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]
    t_ni2 = time.time() - t0

    # get cross-graph edge index tensor
    t0 = time.time()
    ei12 = torch.cat(
        [complete_crossgraph(m, n, N, False) \
         + torch.tensor([cm, cn], device=device).view(2, 1) \
         for n, m, cn, cm in zip(ni1, ni2, cum1, cum2)],
        1,
    ).flip(0)
    t_ei12 = time.time() - t0
    ei = torch.cat([ei1, ei12], 1).long()

    log_time = {'t_ni1': t_ni1, 't_ei1': t_ei1, 't_ni2': t_ni2, 't_ei12': t_ei12}
    return ei, log_time


def get_ei_from_blockdiag(batch1, batch2, device=None):
    """
    Reimplem of get_ei_from using scipy block diagonal matrices.
    """
    device = torch.device("cpu")

    if not isinstance(batch1, torch.Tensor):
        batch1 = torch.tensor(batch1, device=device)
    if not isinstance(batch2, torch.Tensor):
        batch2 = torch.tensor(batch2, device=device)

    N = len(batch1)
    M = len(batch2)

    coo1 = coo_matrix((np.empty(N), (batch1.cpu().numpy(), np.arange(N))))
    cum1 = coo1.tocsr().indptr
    ni1 = cum1[1:] - cum1[:-1]

    coo2 = coo_matrix((np.empty(M), (batch2.cpu().numpy(), np.arange(M))))
    cum2 = coo2.tocsr().indptr
    ni2 = cum2[1:] - cum2[:-1]    

    # create internal edges
    mats = [coo_matrix(np.array(i, i)) for i in ni1]
    S = block_diag(mats)
    row = torch.tensor(S.row, device=device)
    col = torch.tensor(S.col, device=device)
    ei1 = torch.stack([row, col], 0)

    mats = [coo_matrix(np.array(i, j)) for i, j in zip(ni1, ni2)]
    S = block_diag(mats)
    row = torch.tensor(S.row, device=device)
    col = torch.tensor(S.col, device=device)
    ei12 = torch.stack([row, col], 0)

    ei = torch.cat([ei1, ei12], 1).long()

    return ei
    

def get_graph(x, device=None):
    """
    Takes in a batch of babyai observations as input and outputs the nodes,
    batch, edges, and edge indices corresponding to the underlying complete
    graph without self-edges.
    """
    # if device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    x, batch = get_entities(x, device)
    ei = get_ei(batch, device)

    # edges features are concatenatenations of node features for source
    # and destination nodes
    src, dest = ei
    e = torch.cat([x[src.type(torch.IntTensor)], x[dest.type(torch.IntTensor)]], 1)

    return x, batch, ei, e

# def time_blocksparse(n):
#     print(n)
#     a = np.ones((1000, 1000))
#     A = coo_matrix(a)

#     t0 = time.time()
#     C = block_diag([A] * n)
#     t = time.time() - t0

#     return t