import networkx as nx
import numpy as np
import pickle
import pandas as pd

def dijkstra_paths(n, m):

    G = nx.grid_2d_graph(n, m)
    nx.set_edge_attributes(G, values=1, name='weight')
    paths = dict(nx.all_pairs_dijkstra_path(G))
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    return paths, lengths