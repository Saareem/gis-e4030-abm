from typing import Optional

import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from pathlib import Path
import os


def node_visibility(G: nx.Graph, data_dir: Optional[str] = None):
    """
    Creates a dictionary of node pairs
    The values are 1 if the nodes are visible from each other and 0 otherwise
    Visibility is calculated using rectangular shelves if provided, otherwise all values are 1
    Currently uses shelf fields 'x', 'y', 'fixtureWidth', and 'fixtureDepth' to create the rectangular shelves
    # TODO: Use also other fields, like 'fixtureAngle' for more general shelf geometries

    :param G: The store graph
    :param data_dir: Directory of the shelves.json file
    """

    visibility = {}
    edges = []

    # If shelves is provided, add edges of all shelves to list
    if data_dir is not None:
        for idx, shelf in pd.read_json(os.path.join(data_dir, f'shelves.json'), orient='records').iterrows():
            x1 = shelf['x'] - 0.5*shelf['fixtureWidth']
            y1 = shelf['y'] - 0.5*shelf['fixtureDepth']
            x2 = x1 + shelf['fixtureWidth']
            y2 = y1 + shelf['fixtureDepth']
            edges.append(LineString([(x1, y1), (x2, y1)]))
            edges.append(LineString([(x1, y1), (x1, y2)]))
            edges.append(LineString([(x2, y2), (x2, y1)]))
            edges.append(LineString([(x2, y2), (x1, y2)]))

    # Loops through all nodes and calculates visibility to every node
    nodes = G.nodes(data=True)
    for node1, attributes1 in nodes:
        for node2, attributes2 in nodes:
            # If visibility is already calculated in the opposite direction
            if (node2, node1) in visibility:
                visibility[(node1, node2)] = visibility[(node2, node1)]
            else:
                # Check if any shelf edges intersect line of sight between nodes
                visibility[(node1, node2)] = 1
                x1, y1 = attributes1['pos']
                x2, y2 = attributes2['pos']
                line_of_sight = LineString([(x1, y1), (x2, y2)])
                for edge in edges:
                    if line_of_sight.intersects(edge):
                        visibility[(node1, node2)] = 0
                        break

    return visibility
