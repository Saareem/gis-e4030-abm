import os
from pathlib import Path
from typing import Optional

import pandas as pd
import networkx as nx
import random



def create_weights(G: nx.graph, data_dir: Optional[str] = None, weight_range: Optional[int] = 1, seed: Optional[int] = 0):
    """
    Creates a weight dictionary for nodes based on files in directory specified by data_dir
    Directory needs to contain shelves.json with product field, products.csv listing different products and their
    weights, and shelves_to_nodes.csv for linking shelves to nodes
    If no data_dir is given, random weights are used
    :param data_dir: directory of the files
    :param weight_range: range of integer values used for random weights, value of 1 means uniform weights
    :param seed: if given a non-zero value, will be used as seed for random weights generation
    """

    def _random_weights():
        '''
        Creates random weights
        '''
        randomised_weights = {}
        if seed != 0:
            random.seed(seed)
        for node in G.nodes:
            randomised_weights[node] = random.randint(1, weight_range)
        return randomised_weights

    weights = {}

    # If data_dir is not given, create weights randomly
    random_weights = False
    if data_dir is None:
        weights = _random_weights()
    # If data_dir is given, create weights from the files
    else:
        shelves = pd.read_json(os.path.join(data_dir, f'shelves.json'), orient='records')

        # Create shelves_to_nodes -dictionary
        shelves_to_nodes = {}
        for i, row in pd.read_csv(os.path.join(data_dir, f'shelves_to_nodes.csv')).iterrows():
            shelves_to_nodes[row['shelf']] = row['node']

        # Create product weights dictionary from file
        product_weights = {}
        for i, row in pd.read_csv(os.path.join(data_dir, f'products.csv')).iterrows():
                product_weights[row['product']] = row['weight']

        # Loop through every shelf and add their weight to corresponding node
        for i, row in shelves.iterrows():
            if row['fixtureType'] not in ['Checkout', 'Entrance', 'Exit']:
                weight = product_weights[row['product']]
                node = shelves_to_nodes[row['shelfCode']]
                if node in weights:
                    weights[node] = weights[node] + weight
                else:
                    weights[node] = weight

    # Normalise weights
    weights_sum = sum(weights.values())
    for k, v in weights.items():
        weights[k] = v/weights_sum
    return weights


