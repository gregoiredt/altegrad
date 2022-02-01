import networkx as nx
import numpy as np
import torch
from random import randint
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

def normalize_adjacency(A):

    n = A.shape[0]
    A_tilde = A + sp.identity(n)
    degrees = A_tilde @ np.ones(n)
    inv_degrees = np.power(degrees, -1)
    D_inv = sp.diags(inv_degrees)
    A_normalized = D_inv @ A_tilde
    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    sigmoid = nn.Sigmoid()
    y = list()
    y_pred = list()
    non_zero_ind = adj._indices()
    n = non_zero_ind.size(1)
    
    y.append(torch.ones(n).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[non_zero_ind[0], :], z[non_zero_ind[1], :]), dim=1)))
    random_indices = torch.randint(0, z.size(0), non_zero_ind.size())

    y.append(torch.zeros(n).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[random_indices[0], :], z[random_indices[1], :]), dim=1)))
    y = torch.cat(y, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    loss = mse_loss(y_pred, y)
    return loss

def random_edge(n):
    a=1
    b=1
    while a==b:
        a = np.random.randint(n)
        b = np.random.randint(n)
    return a,b
