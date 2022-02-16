import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, GATConv
import dgl.function as fn
import numpy as np 

import sklearn.metrics


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


class GATModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_heads, nonlinearity):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_feats, h_feats, num_heads)
        self.gat2 = GATConv(h_feats * num_heads, h_feats, num_heads)
        #self.gat3 = GATConv(h_feats * num_heads, h_feats)
        self.h_feats = h_feats
        self.nonlinearity = nonlinearity
        self.num_heads = num_heads

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.gat1(mfgs[0], (x, h_dst))
        h = h.view(-1, h.size(1) * h.size(2))
        h = self.nonlinearity(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.gat2(mfgs[1], (h, h_dst))
        h = torch.mean(h, dim=1)
        return h

    def get_hidden(self, graph, x):
        with torch.no_grad():
            h = F.relu(self.gat1(graph, x))
            h = self.gat2(graph, h)
        return h

class GraphAttentionModel(nn.Module):
    def __init__(self, g, n_layers, num_heads, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()
        self.num_heads = num_heads
        self.g = g
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        self.gat1 = GATConv(input_size, hidden_size, num_heads)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size, num_heads)
        self.gat3 = GATConv(hidden_size * num_heads, output_size, num_heads=num_heads+2)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.gat1(self.g, outputs)
        outputs = outputs.view(-1, outputs.size(1) * outputs.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        outputs = self.nonlinearity(outputs)

        outputs = self.gat2(self.g, outputs)
        outputs = outputs.view(-1, outputs.size(1) * outputs.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        outputs = self.nonlinearity(outputs)

        outputs = self.gat3(self.g, outputs)
        outputs = torch.mean(outputs, dim=1)
        return outputs


class DotPredictor(nn.Module):
    '''
    Reconstructs the adjacency matrix value
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

class MLP(nn.Module):
    def __init__(self, n_hidden, n_input) -> None:
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.f1 = nn.Linear(n_input, n_hidden)
        self.f2 = nn.Linear(n_hidden, n_hidden)
        self.f3 = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.f1(x))
        x = self.dropout(self.relu(self.f2(x)))
        output = self.f3(x)
        return output

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