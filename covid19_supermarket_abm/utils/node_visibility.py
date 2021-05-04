import networkx as nx
import pandas as pd
from pathlib import Path
import os


def node_visibility(G: nx.Graph, data_dir = None, nodes = None):
    """
    Creates a dictionary where keys are the nodes of the graph G and values are all nodes visible from the key node
    """

    visibility = {}

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'kmarket_data'
    if nodes is None:
        nodes = f"nodes_abloc.csv"
    nodes = pd.read_csv(os.path.join(data_dir, nodes))

    # Loops through all nodes to create elements of the dictionary
    for idx1, row1 in nodes.iterrows():
        # Nodes that are visible from node idx1
        for idx2, row2 in nodes.iterrows():
            if abs(row1["x"] - row2["x"]) < 400 and abs(row1["y"] - row2["y"]) < 400:
                visibility[(idx1, idx2)] = 1
            else:
                visibility[(idx1, idx2)] = 0
    return visibility
