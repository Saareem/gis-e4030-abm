import os

from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.create_weights import create_weights
from covid19_supermarket_abm.utils.node_visibility import node_visibility
from pathlib import Path


# load synthetic data

data_dir = Path(__file__).parent.parent / 'kmarket_data'
zone_paths = load_example_paths()
G = load_kmarket_store_graph()
shortest_path_dict = get_all_shortest_path_dicts(G)
node_visibility = node_visibility(G, data_dir)

# Set parameters
config = {'arrival_rate': 2.55,
          'traversal_time': 0.2,
          'infection_proportion': 0.0011,
          "logging_enabled": True,
          'day': 6, # 0 = Monday, ..., 6 = Sunday
          'runtime': True,
          'customers_together': 0.2,  # Proportion between [0,1]
          'path_update_interval': 5,
          'shortest_path_dict': shortest_path_dict,
          'avoidance_factor': 2,
          'avoidance_k': 1.5,
          'node_visibility': node_visibility,
          'staff_start_nodes': (27, 27)}  # Start nodes for the staff

# Create a path generator which feeds our model with customer paths
weights = create_weights(G=G, data_dir=None, weight_range=10, seed=10)
item_nodes = [i for i in range(1, 106) if not i in [1, 2, 3, 23, 51, 52, 53, 54, 55]]
synthetic_path_generator_args = [1,
                                 1,
                                 [1],
                                 [51, 52],
                                 [55],
                                 item_nodes,
                                 shortest_path_dict,
                                 weights]
runtime_path_generator_args = [1,
                                 1,
                                 [1],
                                 [51, 52],
                                 [55],
                                 item_nodes,
                                 weights]
path_gen_type = "runtime"
path_generator_function1, path_generator_args1 = get_path_generator(path_generation = 'synthetic',
                                                                  synthetic_path_generator_args=synthetic_path_generator_args,
                                                                  runtime_path_generator_args=runtime_path_generator_args,
                                                                  zone_paths = zone_paths,
                                                                  G=G,
                                                                  )

path_generator_function2, path_generator_args2 = get_path_generator(path_generation = 'runtime',
                                                                  synthetic_path_generator_args=synthetic_path_generator_args,
                                                                  runtime_path_generator_args=runtime_path_generator_args,
                                                                  zone_paths = zone_paths,
                                                                  G=G,
                                                                  )

# Simulate a day and store results in results
#results_dict = simulate_one_day(config, G, path_generator_function1, path_generator_args1)
#print(results_dict["mean_shopping_time"])
a, b, exposure_times1 = simulate_several_days(config, G, path_generator_function2, path_generator_args2, num_iterations=2, use_parallel=False)
#print(results_dict1["df_exposure_time_per_node"].mean(axis=1))
print(exposure_times1.mean().mean())

config['avoidance_factor'] = 0
a, b, exposure_times2 = simulate_several_days(config, G, path_generator_function2, path_generator_args2, num_iterations=2, use_parallel=False)

print(exposure_times2.mean().mean())