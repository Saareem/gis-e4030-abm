import os
from pathlib import Path

import pandas as pd
import random



def create_weights(shelves_fp: str = None, shelves_to_nodes_fp: str = None, products_fp: str = None, random_weights: bool = False):
    """Creates a weight dictionary for nodes
    If no file paths are specified in shelves_fp, shelves_to_nodes_fp, or products_fp, defaults to kmarket data
    If random_weights is set to True, random product weights are used instead of weights from products_fp"""

    data_dir = Path(__file__).parent.parent / 'kmarket_data'

    # If no file path given load kmarket data
    if shelves_fp is None:
        shelves_fp = os.path.join(data_dir, f"shelves.json")
    if shelves_to_nodes_fp is None:
        shelves_to_nodes_fp = os.path.join(data_dir, f"shelves_to_nodes.csv")
    if products_fp is None:
        products_fp = os.path.join(data_dir, f"products")

    weights = {}
    shelves = pd.read_json(shelves_fp, orient="records")

    # Create shelves_to_nodes -dictionary
    shelves_to_nodes = {}
    for i, row in pd.read_csv(shelves_to_nodes_fp).iterrows():
        shelves_to_nodes[row["shelf"]] = row["node"]

    # Create product weights dictionary from file or randomly
    product_weights = {}
    if not random_weights:
        for i, row in pd.read_csv(products_fp).iterrows():
            product_weights[row["product"]] = row["weight"]
    else:
        product_weights = random_product_weights(shelves["product"].unique())

    # Loop through every shelf and add their weight to corresponding node
    for i, row in shelves.iterrows():
        if row["fixtureType"] not in ["Checkout", "Entrance", "Exit"]:
            weight = product_weights[row["product"]]
            node = shelves_to_nodes[row["shelfCode"]]
            if node in weights:
                weights[node] = weights[node] + weight
            else:
                weights[node] = weight

    # Normalise weights
    weights_sum = sum(weights.values())
    for k, v in weights.items():
        weights[k] = v/weights_sum
    return weights


def random_product_weights(products):
    """Creates random product weights"""
    product_weights = {}
    for product in products:
        product_weights[product] = 0.1 + 1.8*random.random()
    return product_weights