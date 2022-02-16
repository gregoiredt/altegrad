import networkx as nx
import numpy as np
from random import randint
import dgl
import torch
import csv

def read_graph():
    G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print('Number of nodes:', n)
    print('Number of edges:', m)

    # Read the abstract of each paper
    abstracts = dict()
    with open('abstracts.txt', 'r', encoding="utf8") as f:
        for line in f:
            node, abstract = line.split('|--|')
            abstracts[int(node)] = abstract.replace('\n', '')

    # Map text to set of terms
    #for node in abstracts:
    #    abstracts[node] = set(abstracts[node].split())

    # Authors 
    text_per_author = {}
    author_per_text = {}
    with open('authors.txt', 'r', encoding="utf8") as f:
        for line in f:
            node, authors = line.split('|--|')
            author_per_text[int(node)] = []
            for author in authors.split(','):
                author_per_text[int(node)].append(author)
                if not(author in text_per_author.keys()):
                    text_per_author[author] = []
                text_per_author[author].append(int(node))
    print("Number of authors :", len(text_per_author))

    return G, abstracts, text_per_author, author_per_text

def retrieve_subgraph(G, min_nb_nodes=4):
    degree_sequence = np.array([G.degree(node) for node in G.nodes()])
    print('The minimum degree of the nodes in the graph is :', min(degree_sequence))
    print('The maximum degree of the nodes in the graph is :', max(degree_sequence))
    print('The mean degree of the nodes in the graph is :', np.mean(degree_sequence))
    print('The median degree of the nodes in the graph is :', np.median(degree_sequence))
    sub_nodes = [i for i,v in enumerate(degree_sequence) if v >= min_nb_nodes]
    sub_G = nx.subgraph(G, sub_nodes)
    n = sub_G.number_of_nodes()
    m = sub_G.number_of_edges()
    print('Number of nodes in subgraph:', n)
    print('Number of edges in subgraph:', m)
    return sub_G

def get_dgl_graph(G, features):
    G_dgl = dgl.from_networkx(G)
    # Undirected graph:
    graph = dgl.add_reverse_edges(G_dgl)
    #graph.ndata['feat'] = features
    
    return graph


def retrieve_embeddings(G, embeddings):
    node_pairs = list()
    with open('test.txt', 'r') as f:
        for line in f:
            t = line.split(',')
            node_pairs.append((int(t[0]), int(t[1])))
    x = torch.zeros((len(node_pairs), 2*embeddings.shape[1]))
    for i, (src, dst) in enumerate(node_pairs):
        src_emb = embeddings[G.nodes[src].data['_ID']]
        dst_emb = embeddings[G.nodes[dst].data['_ID']]
        line = torch.cat([src_emb, dst_emb], dim=1)
        x[i,:] = line
    return x




