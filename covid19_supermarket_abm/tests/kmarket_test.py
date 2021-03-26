from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day

# Set parameters
config = {'arrival_rate': 2.55,
          'traversal_time': 0.2,
          'num_hours_open': 14,
          'infection_proportion': 0.0011,
          "logging_enabled": True}

synthetic = True
type = "empirical"
if synthetic is True:
    type = "synthetic"

# load synthetic data

zone_paths = load_example_paths()
G = load_kmarket_store_graph()
shortest_path_dict = get_all_shortest_path_dicts(G)

# Create a path generator which feeds our model with customer paths
item_nodes = [i for i in range(1, 106) if not i in [1, 23, 52, 55]]
path_generator_function, path_generator_args = get_path_generator(path_generation = type,
                                                                  zone_paths = zone_paths,
                                                                  G=G,
                                                                  synthetic_path_generator_args = [
                                                                      1,
                                                                      1,
                                                                      [1],
                                                                      [23, 52],
                                                                      [55],
                                                                      item_nodes,
                                                                      shortest_path_dict
                                                                  ])

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
print(results_dict["logs"])