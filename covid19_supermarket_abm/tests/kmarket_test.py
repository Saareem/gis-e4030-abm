from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day

# Set parameters
config = {'arrival_rate': 1,  # Poisson rate at which customers arrival
           'traversal_time': 0.05,  # mean wait time per node
           'num_hours_open': 14,  # store opening hours
           'infection_proportion': 0.0011, # proportion of customers that are infectious
          "logging enabled" : True
         }

# load synthetic data

G = load_kmarket_store_graph()
shortest_path_dict = get_all_shortest_path_dicts(G)

# Create a path generator which feeds our model with customer paths
item_nodes = [i for i in range(1, 106) if not i in [1, 23, 52, 55]]
path_generator_function, path_generator_args = get_path_generator(path_generation = "synthetic",
                                                                  G=G,
                                                                  synthetic_path_generator_args = [
                                                                      5,
                                                                      2,
                                                                      [1],
                                                                      [23, 52],
                                                                      [55],
                                                                      item_nodes,
                                                                      shortest_path_dict
                                                                  ])

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)