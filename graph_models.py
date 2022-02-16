import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, GATConv
import dgl.function as fn
import numpy as np 
from tqdm import tqdm
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
            h = self.gat1(graph, x)
            h = h.view(-1, h.size(1) * h.size(2))
            h = self.nonlinearity(h)
            
            h = self.gat2(graph, h)
            h = torch.mean(h, dim=1)
        return h

class DeepGAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_heads, nonlinearity):
        super(DeepGAT, self).__init__()
        self.gat1 = GATConv(in_feats, h_feats, num_heads)
        self.gat2 = GATConv(h_feats * num_heads, h_feats, num_heads)
        self.gat3 = GATConv(h_feats * num_heads, h_feats, num_heads+2)
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
        h = h.view(-1, h.size(1) * h.size(2))
        h = self.nonlinearity(h)

        h_dst = h[:mfgs[2].num_dst_nodes()]
        h = self.gat3(mfgs[2], (h, h_dst))
        h = torch.mean(h, dim=1)
        return h

    def get_hidden(self, graph, x):
        with torch.no_grad():
            h = self.gat1(graph, x)
            h = h.view(-1, h.size(1) * h.size(2))
            h = self.nonlinearity(h)

            h = self.gat2(graph, x)
            h = h.view(-1, h.size(1) * h.size(2))
            h = self.nonlinearity(h)

            h = self.gat3(graph, h)
            h = torch.mean(h, dim=1)
        return h


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


def train_classif(
    model, 
    embeddings, 
    train_dataloader, 
    val_dataloader, 
    criterion, 
    device,
    optimizer, 
    epochs=10, 
    name_model='c1.pt'
    ):

    best_model_path = name_model
    min_loss=np.inf
    for epoch in range(epochs):
        train_losses = []
        val_losses = []
        with tqdm(train_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, _) in enumerate(tq):
                with torch.no_grad():
                    src, dst = pos_graph.edges()
                    src_emb = embeddings[pos_graph.nodes[src].data['_ID']]
                    dst_emb = embeddings[pos_graph.nodes[dst].data['_ID']]
                    x = torch.cat([src_emb, dst_emb], dim=1)
                    n_pos = x.shape[0]

                    src_neg, dst_neg = neg_graph.edges()
                    src_emb_neg = embeddings[neg_graph.nodes[src_neg].data['_ID']]
                    dst_emb_neg = embeddings[neg_graph.nodes[dst_neg].data['_ID']]
                    x_neg = torch.cat([src_emb_neg, dst_emb_neg], dim=1)
                    n_neg = x_neg.shape[0]
                    
                x_tot = torch.cat([x, x_neg], dim=0).to(device)
                y = model(x_tot)
                
                pos_label = torch.ones(n_pos)
                target = torch.cat([pos_label, torch.zeros(n_neg)]).to(device)

                loss = criterion(y.squeeze(), target)
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)
        

        for step, (input_nodes, pos_graph, neg_graph, _) in enumerate(val_dataloader):
            with torch.no_grad():
                src, dst = pos_graph.edges()
                src_emb = embeddings[pos_graph.nodes[src].data['_ID']]
                dst_emb = embeddings[pos_graph.nodes[dst].data['_ID']]
                x = torch.cat([src_emb, dst_emb], dim=1)
                n_pos = x.shape[0]

                src_neg, dst_neg = neg_graph.edges()
                src_emb_neg = embeddings[neg_graph.nodes[src_neg].data['_ID']]
                dst_emb_neg = embeddings[neg_graph.nodes[dst_neg].data['_ID']]
                x_neg = torch.cat([src_emb_neg, dst_emb_neg], dim=1)
                n_neg = x_neg.shape[0]
                
                x_tot = torch.cat([x, x_neg], dim=0).to(device)
                y = model(x_tot)
            
                pos_label = torch.ones(n_pos)
                target = torch.cat([pos_label, torch.zeros(n_neg)]).to(device)

                loss = criterion(y.squeeze(), target)
                val_losses.append(loss.item())
        
        if np.mean(val_losses) < min_loss:
            min_loss = np.mean(val_losses)
            torch.save(model.state_dict(), best_model_path)
        print(f'Epoch {epoch} : Train mean loss {np.mean(train_losses)} : Val mean loss {np.mean(val_losses)}')