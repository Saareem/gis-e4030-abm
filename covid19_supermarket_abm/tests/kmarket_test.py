from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.create_weights import create_weights
from covid19_supermarket_abm.utils.node_visibility import node_visibility

# Set parameters
config = {'arrival_rate': 2.55,
          'traversal_time': 0.2,
          'infection_proportion': 0.0011,
          "logging_enabled": True,
          'day': 6, # 0 = Monday, ..., 6 = Sunday
          'customers_together': 0}  # Proportion between [0,1]



# load synthetic data

zone_paths = load_example_paths()
G = load_kmarket_store_graph()
shortest_path_dict = get_all_shortest_path_dicts(G)
node_visibility = node_visibility(G)

# Create a path generator which feeds our model with customer paths
weights = create_weights(random_weights = True)
item_nodes = [i for i in range(1, 106) if not i in [1, 2, 3, 23, 51, 52, 53, 54, 55]]
path_gen_type = "realtime"
path_generator_function, path_generator_args = get_path_generator(path_generation = path_gen_type,
                                                                  zone_paths = zone_paths,
                                                                  G=G,
                                                                  realtime_path_generator_args = [
                                                                      1,
                                                                      1,
                                                                      [1],
                                                                      [51, 52],
                                                                      [55],
                                                                      item_nodes,
                                                                      weights
                                                                  ])

# Simulate a day and store results in results
results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
#results_dict = simulate_several_days(config, G, path_generator_function, path_generator_args, num_iterations=10, use_parallel=False)
print(results_dict["mean_shopping_time"])