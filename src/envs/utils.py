import networkx as nx
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

path = 'envs/rebalance/'
# xmin, xmax = -74.03, -73.89
# ymin, ymax = 40.69, 40.89

xmin, xmax = 100, 5
ymin, ymax = 100, 5
def dijkstra_paths(n, m):

    G = nx.grid_2d_graph(n, m)
    nx.set_edge_attributes(G, values=1, name='weight')
    paths = dict(nx.all_pairs_dijkstra_path(G))
    lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    return paths, lengths

def load_graph(file_name):
    with open(path+file_name, 'rb') as pickle_file:
        G = pickle.load(pickle_file)
        nodes = list(G.nodes())
        edges = list(G.edges())
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
    return G, A


def draw_map(file_name):
    with open(file_name, "rb") as f:
        load = pickle.load(f)
    fig, ax = plt.subplots(figsize=(50, 50))
    load[1].plot(ax=ax,alpha=0.2)
    fig.show()
    fig.canvas.draw()
    return fig, ax


def draw_image(file_name):
    im = Image.open(path+file_name)
    width, height = im.size
    scale_x = width / (xmax - xmin)
    scale_y = height / (ymax - ymin)
    scale = [scale_x, scale_y]
    return im, scale


def draw_vehicle(image, pos, scale, fill = '#ff6361', r=6):
    scale_x = scale[0]
    scale_y = scale[1]
    y = image.size[1] - (pos[0]-ymin)*scale_y
    x = (pos[1]-xmin)*scale_x
    x_dash = x + r
    y_dash = y + r
    ImageDraw.Draw(image).ellipse([(x-r, y-r), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_order(image, pos, scale, fill='#003f5c', r = 4):
    scale_x = scale[0]
    scale_y = scale[1]
    y = image.size[1] - (pos[0]-ymin)*scale_y
    x = (pos[1]-xmin)*scale_x
    x_dash = x + r
    y_dash = y + r
    ImageDraw.Draw(image).rectangle([(x-r, y-r), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_map_h3(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    df[['h3_res9', 'geometry']].plot(alpha=0.3, ax=ax, facecolor="none",edgecolor='#444e86')

    # fig.show()
    # fig.canvas.draw()
    return fig, ax


def load_file(file_name):
    with open(path+file_name, 'rb') as pickle_file:
        h3 = pickle.load(pickle_file)
    return h3

def fill_cell_im(image, icon, pos, cell_size=None, fill='black', margin=0):
    # assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    # margin *= cell_size
    # x, y, x_dash, y_dash = row + margin, col + margin, row + cell_size - margin, col + cell_size - margin
    # ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)
    image.paste(icon, (int(row)+1, int(col)+1), icon)