import base64
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl


def visualize_single_day(G, dictionary, ):
    """
    Function for visualizing a single-day simulation
    :param G: a networkx graph of the store
    :param dictionary: the results_dict of the simulation
    :return: image_list: a list of two base64 encoded images
    """

    # define the positions of axes
    pos = {node: (y, x) for (node, (x, y)) in nx.get_node_attributes(G, 'pos').items()}

    # define images 1 and 2
    fig1 = plt.figure(figsize=[12, 8])
    df1 = 'df_num_encounters_per_node'
    plt.title('Number of encounters per node, 1 iteration', figure = fig1)


    # draw figure 1
    draw_edges = nx.draw_networkx_edges(G, pos)
    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=dictionary[df1], cmap=plt.get_cmap('viridis'))
    draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    cbar = plt.colorbar(draw_nodes)
    cbar.set_label('N:o of encounters')


    # transform fig1 to png and then to bytearray
    img = fig2img(fig1)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='png')
    img = byte_arr.getvalue()

    # fig1 to base64 encoding
    image_data = base64.b64encode(img)
    if not isinstance(image_data, str):
        image_data = image_data.decode()

    # save to list
    result_images = []
    result_images.append('data:image/png;base64,' + image_data)

    fig2 = plt.figure(figsize=[12, 8])
    df2 = 'df_exposure_time_per_node'
    plt.title('Exposure time per node, 1 iteration', figure=fig2)

    # draw figure 2
    draw_edges = nx.draw_networkx_edges(G, pos)
    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=dictionary[df2], cmap=plt.get_cmap('magma'))
    draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    cbar = plt.colorbar(draw_nodes)
    cbar.set_label('minutes')

    # fig2 to bytearray
    img = fig2img(fig2)

    byte_arr = io.BytesIO()
    img.save(byte_arr, format='png')
    img = byte_arr.getvalue()

    # fig2 to base64
    image_data = base64.b64encode(img)
    if not isinstance(image_data, str):
        image_data = image_data.decode()

    # save to list
    result_images.append('data:image/png;base64,' + image_data)

    return result_images

def visualize_multiple_days(G, encounters, exposure_time, days):
    """
    This is a function for visualizing a multiple-day simulation.
    :param G: a networkx graph of the store
    :param encounters: the results dataframe containing the encounter stats
    :param exposure_time: the results dataframe containing the exposure time stats
    :param days: number of simulation days
    :return: list of two base64 images
    """
    # calculate the mean of the results from the results dict
    mean_encounters = encounters.mean(axis=0)
    mean_exposure_time = exposure_time.mean(axis=0)

    # define the positions of axes
    pos = {node:(y,x) for (node, (x,y)) in nx.get_node_attributes(G, 'pos').items()}

    fig1 = plt.figure(figsize=[12, 8])
    plt.title('Mean number of encounters per node, '+str(days)+' iterations', figure=fig1)

    draw_edges = nx.draw_networkx_edges(G, pos)
    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=mean_encounters, cmap=plt.get_cmap('viridis'))
    draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    cbar = plt.colorbar(draw_nodes)
    #draw_nodes.set_clim(0, 14)
    cbar.set_label('N:o of encounters')


    # transform fig1 to png and then to bytearray
    img = fig2img(fig1)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='png')
    img = byte_arr.getvalue()

    # fig1 to base64 encoding
    image_data = base64.b64encode(img)
    if not isinstance(image_data, str):
        image_data = image_data.decode()

    # save to list
    result_images = []
    result_images.append('data:image/png;base64,' + image_data)
    print(len(result_images))

    fig2 = plt.figure(figsize=[12, 8])
    plt.title('Mean number of exposure time node, '+str(days)+' iterations', figure=fig2)

    draw_edges = nx.draw_networkx_edges(G, pos)
    draw_nodes = nx.draw_networkx_nodes(G, pos, node_color=mean_exposure_time, cmap=plt.get_cmap('inferno'))
    draw_labels = nx.draw_networkx_labels(G, pos, font_color='w')
    cbar = plt.colorbar(draw_nodes)
    #draw_nodes.set_clim(0, 3)
    cbar.set_label('minutes')

    # transform fig1 to png and then to bytearray
    img = fig2img(fig2)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='png')
    img = byte_arr.getvalue()

    # fig1 to base64 encoding
    image_data = base64.b64encode(img)
    if not isinstance(image_data, str):
        image_data = image_data.decode()

    # add dataurl and save to list
    result_images.append('data:image/png;base64,' + image_data)

    return result_images

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def chart(series, days):
    """
    This is function for plotting the results of multiple-day simulations.
    :param series:the pandas.series being plot
    :param days: number of days simulated
    :return: img: a base64 encoded string
    """
    # create a figure, visualize and add ticks and label
    fig = plt.figure(figsize=[5, 4])
    plot = series.plot()
    ticks = list(range(1, days+1))
    plt.xticks(series.index, ticks)
    plt.xlabel('Days')

    # convert to bytearray
    img = fig2img(fig)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='png')
    img = byte_arr.getvalue()

    # fig1 to base64 encoding
    image_data = base64.b64encode(img)
    if not isinstance(image_data, str):
        image_data = image_data.decode()

    # return the base64 encoded string with the data url for html
    image_data = 'data:image/png;base64,' + image_data
    return image_data
