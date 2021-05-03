import os

import matplotlib.pyplot as plt
import networkx as nx

def visualize_single_day(G, dictionary, variable):
    """
    Function for visualizing a single-day simulation
    :param G: a networkx graph of the store
    :param dictionary: the results_dict of the simulation
    :param variable: dictionary: the results dictionary of the simulation
    :param variable: the variable being visualized: 1 for 'num_encounters_per_node' and 2 for 'exposure_time_per_node'.
    Default 2.
    :return: 0
    """
    # define the positions of axes
    pos = {node: (y, x) for (node, (x, y)) in nx.get_node_attributes(G, 'pos').items()}

    if variable == 1:
        df = 'df_num_encounters_per_node'
        plt.title('Exposure time per node, 1 iteration')
    if variable == 2:
        df = 'df_exposure_time_per_node'
        plt.title('Number of encounters per node, 1 iteration')

    draw_edges = nx.draw_networkx_edges(G, pos)
    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=dictionary[df], cmap=plt.get_cmap('viridis'))
    draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    plt.colorbar(draw_nodes)

    plt.show()

    return 0




def visualize_multiple_days(G, dictionary, variable=2, categorize=False, ):
    """
    This is a function for visualizing a multiple-day simulation.
    :param G: a networkx graph of the store
    :param dictionary: the results dictionary of the simulation
    :param variable: the variable being visualized: 1 for 'num_encounters_per_node' and 2 for 'exposure_time_per_node'.
    Default: 2.
    :param categorize: if True, the results will be categorize in a predefined manner. Default: False. Should only be
    used with 'exposure_time_per_node' for sane results
    :return: none
    """
    # calculate the mean of the results from the results dict
    mean_results = dictionary[variable].mean(axis=0)

    # define the positions of axes
    pos = {node:(y,x) for (node, (x,y)) in nx.get_node_attributes(G, 'pos').items()}

    if categorize == True:
        mean_results['category'] = mean_results.apply(categorize_variable)
        draw_edges = nx.draw_networkx_edges(G, pos)
        draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=mean_results['category'], cmap=plt.get_cmap('viridis'))
        draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    else:
        draw_edges = nx.draw_networkx_edges(G, pos)
        draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=mean_results, cmap=plt.get_cmap('viridis'))
        draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
        plt.colorbar(draw_nodes)

    plt.show()

    return 0

def categorize_variable(var):
    """
    Categorizes the chosen variable in a predefined, heuristic manner for comparisons between different runs.
    :param var: the value of variable being categorized
    :return: a category color
    TODO: Very high values are sometimes categorized as white. Find out why.
    """
    if var == 0:
        return 'c'
    if 0 < var < 0.5:
        return 'b'
    if 0.5 >= var < 1:
        return 'g'
    if 1 >= var < 1.5:
        return 'y'
    if 1.5 >= var < 2:
        return 'r'
    if var >= 2:
        return 'k'
    else:
        return 'w'
