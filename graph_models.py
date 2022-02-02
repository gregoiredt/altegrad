import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
import dgl.function as fn
import numpy as np 

import sklearn.metrics

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        x = self.relu(torch.mm(adj, self.fc1(x_in)))
        x = torch.mm(adj, self.fc2(x))
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(self.device)
        out = out.scatter_add_(0, idx, x) 

        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)

class GAE(nn.Module):
    """GAE model"""
    def __init__(self, in_feats, h_feats):
        super(GAE, self).__init__()

        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        self.h_feats = h_feats
        
    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()] # First flow
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()] # Second flow
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

class SageModel(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(SageModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

    def get_hidden(self, graph, x):
        h = F.relu(self.conv1(graph, x))
        h = self.conv2(graph, h)
        return h

class DotPredictor(nn.Module):
    '''
    Reconstructs the adjacency matirx value
     thanks to the embedding h of the given graph
    '''
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


################# UTILS FUNCTIONS #######################

def inference(model, train_dataloader):
    with torch.no_grad():

        result = []
        for _, _, _, mfgs in train_dataloader:
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            result.append(model(mfgs, inputs))

        return torch.cat(result)


def evaluate(emb, label, train_nids, valid_nids, test_nids, num_classes, device):
    classifier = nn.Linear(emb.shape[1], num_classes).to(device)
    opt = torch.optim.LBFGS(classifier.parameters())

    def compute_loss():
        pred = classifier(emb[train_nids].to(device))
        loss = F.cross_entropy(pred, label[train_nids].to(device))
        return loss

    def closure():
        loss = compute_loss()
        opt.zero_grad()
        loss.backward()
        return loss

    prev_loss = float('inf')
    for i in range(1000):
        opt.step(closure)
        with torch.no_grad():
            loss = compute_loss().item()
            if np.abs(loss - prev_loss) < 1e-4:
                print('Converges at iteration', i)
                break
            else:
                prev_loss = loss

    with torch.no_grad():
        pred = classifier(emb.to(device)).cpu()
        label = label
        valid_acc = sklearn.metrics.accuracy_score(label[valid_nids].numpy(), pred[valid_nids].numpy().argmax(1))
        test_acc = sklearn.metrics.accuracy_score(label[test_nids].numpy(), pred[test_nids].numpy().argmax(1))
    return valid_acc, test_acc