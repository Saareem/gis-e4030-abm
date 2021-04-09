import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from covid19_supermarket_abm.utils.shelf_class import Shelf
from covid19_supermarket_abm.utils.create_store_network import dist

data_dir = Path(__file__).parent.parent / 'kmarket_data'


def load_kmarket_store_graph(directed: bool = False) -> nx.Graph:
    """We load our example store graph.
    Note that this uses a different way for important a graph"""
    # Load parameters
    if directed:
        create_using = nx.DiGraph()
        graph_suffix = ''
    else:
        create_using = nx.Graph()
        graph_suffix = ''

    # Load zone
    df_zone = load_kmarket_zones()

    # Load graph
    edge_list_path = os.path.join(data_dir, f'oneway_edges{graph_suffix}.tsv')
    G = nx.read_edgelist(edge_list_path, nodetype=int, create_using=create_using)
    pos = {node: (x, -y) for node, x, y in df_zone.loc[:, ['id', 'x', 'y']].values}

    # Assign positions
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

        # Add edge weight
        weighted_edges = [(u, v, dist(u, v, pos)) for (u, v) in G.edges()]
        G.add_weighted_edges_from(weighted_edges)
    G = G.copy()  # ensures that the nodes are from 0 to n-1
    return G


def load_kmarket_zones():
    df_zone = pd.read_csv(os.path.join(data_dir, f'nodes_abloc.csv'))
    return df_zone


def load_example_paths():
    with open(os.path.join(data_dir, f'zone_paths.json'), 'r') as f:
        zone_paths = json.load(f)
    return zone_paths


def load_shelves(store_id: int, units='cm', suffix='', data_dir='.') -> List[Shelf]:
    """Import shelves from json file"""
    shelves_file_path = os.path.join(data_dir, f'shelves.json')
    with open(shelves_file_path) as f:
        shelves_dict = json.load(f)

    # Convert shelf objects
    all_shelves = []
    for shelf in shelves_dict:
        x = shelf['x']
        y = shelf['y']
        width = shelf['fixtureWidth']
        depth = shelf['fixtureDepth']
        name = shelf['shelfCode']
        angle = shelf['fixtureAngle']
        equipment = shelf['fixtureType']
        if name is None:
            continue
        shelf_obj = Shelf(x, y, width, depth, name, angle, equipment)
        if units == 'cm':
            shelf_obj.convert_to_m()
        all_shelves.append(shelf_obj)
    return all_shelves


def plot_shelves(shelves, ax: Optional[plt.axes] = None, color: str = '#C0C0C0',
                 xdelta: float = 0, ydelta: float = 0, edgecolor="none",
                 with_label=False, plot_special_shelves=True, **kwargs) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for shelf in shelves:
        corners = shelf.corners
        # midpoint_front = (corners[0] + corners[3])/2
        # Shift the coordinates
        corners[:, 0] += xdelta
        corners[:, 1] += ydelta
        if xmin > min(corners[:, 0]):
            xmin = min(corners[:, 0])
        if xmax < max(corners[:, 0]):
            xmax = max(corners[:, 0])
        if ymin > min(corners[:, 1]):
            ymin = min(corners[:, 1])
        if ymax < max(corners[:, 1]):
            ymax = max(corners[:, 1])
        if shelf.equipment == 'Entrance':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='y', hatch='', zorder=0, **kwargs))
        elif shelf.equipment == 'Exit':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='r', hatch='', zorder=0, **kwargs))
        elif shelf.equipment == 'Checkout':
            if plot_special_shelves:
                ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                     fill=True, facecolor='orange', hatch='', zorder=0, **kwargs))
        else:
            ax.add_patch(Polygon(corners, edgecolor=edgecolor, closed=True,
                                 fill=True, facecolor=color, hatch='', zorder=0, **kwargs))
        if with_label:
            ax.annotate(shelf.name, shelf.center, ha='center')
        # ax.plot(*midpoint_front, 'x', color='C1')
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    plt.axis('equal')
    return ax


def load_popular_hours():
    # Get popular hours of the market from google data // Alpo
    kmarket_hours = pd.read_csv(os.path.join(data_dir, f'kmarket_popular_hours.txt'), sep='\t', header=None)
    return kmarket_hours / 100