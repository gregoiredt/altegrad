import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import Node2Vec

device = 'cpu'


def create_author_graph(text_per_author, author_per_text):
  G_authors = nx.Graph()
  for author in text_per_author.keys(): 
    G_authors.add_node(author)

  for text in author_per_text.keys(): 
    for aut1 in author_per_text[text]: 
      for aut2 in author_per_text[text]: 
        if aut1 != aut2 : 
          #if G_authors.has_edge(aut1, aut2):
          if G_authors.has_edge(aut1, aut2) == False : 
            G_authors.add_edge(aut1, aut2, weight = 1/2)
          else : 
            G_authors[aut1][aut2]['weight']+=1/2
  return G_authors


def create_commun_authors_papers_graph(text_per_author, author_per_text):

  G_papers_ca = nx.Graph()
  for node in author_per_text.keys(): 
    G_papers_ca.add_node(node)

  for author in text_per_author.keys(): 
    for text1 in text_per_author[author]: 
        for text2 in text_per_author[author]: 
          if text1 != text2 : 
            if G_papers_ca.has_edge(text1, text2) == False : 
              G_papers_ca.add_edge(text1, text2, weight = 1/2)
            else : 
              G_papers_ca[text1][text2]['weight']+= 1/2
  return G_papers_ca


def load_author_embeddings_avg(G, text_per_author, author_per_text):
    G_authors = create_author_graph(text_per_author, author_per_text)

    for i, node in enumerate(G_authors.nodes()):
        G_authors.nodes[str(node)]['id'] = i
    G_authors_train = from_networkx(G_authors) # already undirected

    model = Node2Vec(G_authors_train.edge_index, embedding_dim= 64, walk_length=30,
                        context_size=10, walks_per_node=20,
                        num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    model.load_state_dict(torch.load('node2vec_authors_64dim.zip'))
    model.eval()

    authors_embedding = model(torch.arange(G_authors_train.num_nodes, device=device))

    for text in author_per_text.keys(): 
        avg_authors_feature = torch.zeros([64]).to(device)
        for author in author_per_text[text]: 
            avg_authors_feature += authors_embedding[int(G_authors.nodes[str(author)]['id'])].to(device)
        avg_authors_feature /= len(author_per_text[text])
        G.nodes[text]['avg_authors_feature'] = avg_authors_feature.detach().numpy()
    
    return G

def load_common_author_embeddings(G, text_per_author, author_per_text):
    G_papers_ca = create_commun_authors_papers_graph(text_per_author, author_per_text)

    G_papers_ca = from_networkx(G_papers_ca)
    model = Node2Vec(G_papers_ca.edge_index, embedding_dim= 64, walk_length=30,
                    context_size=10, walks_per_node=20,
                    num_negative_samples=1, p=1, q=1, sparse=True)
    model.load_state_dict(torch.load('node2vec_papers_ca_64dim.zip'))
    model.eval()

    papers_ca_embedding = model(torch.arange(G_papers_ca.num_nodes))
    for node in author_per_text : 
        G.nodes[node]['feat_com_authors'] = papers_ca_embedding[node].detach().numpy()
    return G
 






