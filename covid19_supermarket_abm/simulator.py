import logging
import multiprocessing
from itertools import repeat
import timeit
import networkx as nx
import numpy as np
import pandas as pd
import simpy
from covid19_supermarket_abm.core import Store, _agent_arrivals, _stats_recorder, \
    _sanity_checks
from tqdm import tqdm
from covid19_supermarket_abm.utils.load_kmarket_data import load_popular_hours


def simulate_one_day(config: dict, G: nx.Graph, path_generator_function, path_generator_args: list):
    # Get parameters
    start = timeit.default_timer()
    if 'day' not in config:
        config['day'] = 0
    popular_hours = load_popular_hours(config.get('num_hours_open', 14)).iloc[:, config['day']]
    num_hours_open = popular_hours.count()
    logging_enabled = config.get('logging_enabled', False)
    raise_test_error = config.get('raise_test_error', False)  # for debugging purposes
    with_node_capacity = config.get('with_node_capacity', False)
    max_customers_in_store_per_sqm = config.get('max_customers_in_store_per_sqm', None)
    floorarea = config.get('floorarea', None)
    if max_customers_in_store_per_sqm is None:
        max_customers_in_store = config.get('max_customers_in_store', None)
    else:
        if floorarea is not None:
            max_customers_in_store = int(max_customers_in_store_per_sqm * floorarea)
        else:
            raise ValueError('If you set the parameter "max_customers_in_store_per_sqm", '
                             'you need to specify the floor area via the "floorarea" parameter in the config.')

    config['staff_start_nodes'] = config.get('staff_start_nodes', ())
    runtime_parameters = {}
    runtime = config.get('runtime', False)
    if runtime:
        runtime_parameters['path_update_interval'] = config.get('path_update_interval', 5)
        runtime_parameters['avoidance_factor'] = config.get('avoidance_factor', 1)
        runtime_parameters['avoidance_k'] = config.get('avoidance_k', 1.5)
        runtime_parameters['node_visibility'] = config.get('node_visibility', None)
        runtime_parameters['shortest_path_dict'] = config.get('shortest_path_dict', None)
        if runtime_parameters['node_visibility'] is None or runtime_parameters['shortest_path_dict'] is None:
            raise Exception('node_visibility and shortest_path_dict need to be specified for runtime path generation')

    # Set up environment and run
    env = simpy.Environment()
    store = Store(env, G, max_customers_in_store=max_customers_in_store, logging_enabled=logging_enabled,
                  staff_start_nodes=config["staff_start_nodes"], runtime=runtime,
                  runtime_parameters=runtime_parameters)
    if with_node_capacity:
        node_capacity = config.get('node_capacity', 2)
        store.enable_node_capacity(node_capacity)
    path_generator = path_generator_function(*path_generator_args)
    env.process(_agent_arrivals(env, store, path_generator, config, popular_hours, num_hours_open))
    env.process(_stats_recorder(store))
    env.run(until=num_hours_open * 60 * 10)

    # Record stats
    _sanity_checks(store, raise_test_error=raise_test_error)
    num_agent = len(store.agents)
    num_S = len(store.number_encounters_with_infected)
    shopping_times = list(store.shopping_times.values())
    waiting_times = np.array(list(store.waiting_times.values()))
    b = np.array(waiting_times) > 0
    num_waiting_people = sum(b)
    if num_waiting_people > 0:
        mean_waiting_time = np.mean(waiting_times[b])
    else:
        mean_waiting_time = 0

    num_contacts_per_agent = [contacts for contacts in store.number_encounters_with_infected.values() if contacts != 0]
    df_num_encounters_per_node = pd.DataFrame(store.number_encounters_per_node, index=[0])
    df_num_encounters_per_node = df_num_encounters_per_node[range(len(G))]
    df_exposure_time_per_node = pd.DataFrame(store.time_with_infected_per_node, index=[0])
    df_exposure_time_per_node = df_exposure_time_per_node[range(len(G))]
    exposure_times = [val for val in list(store.time_with_infected_per_agent.values()) if val > 0]
    results = {'num_agents': num_agent,
               'num_S': num_S,
               'num_I': num_agent - num_S,
               'total_exposure_time': sum(store.time_with_infected_per_agent.values()),
               'num_contacts_per_agent': num_contacts_per_agent,
               'num_agents_w_contact': len(num_contacts_per_agent),
               'mean_num_cust_in_store': np.mean(list(store.stats['num_customers_in_store'].values())),
               'max_num_cust_in_store': max(list(store.stats['num_customers_in_store'].values())),
               'num_contacts': sum(num_contacts_per_agent),
               'shopping_times': shopping_times,
               'mean_shopping_time': np.mean(shopping_times),
               'num_waiting_people': num_waiting_people,
               'mean_waiting_time': mean_waiting_time,
               'store_open_length': max(list(store.stats['num_customers_in_store'].keys())),
               'df_num_encounters_per_node': df_num_encounters_per_node,
               'df_exposure_time_per_node': df_exposure_time_per_node,
               'total_time_crowded': store.total_time_crowded,
               'exposure_times': exposure_times,
               }

    if floorarea is not None:
        results['mean_num_cust_in_store_per_sqm'] = results['mean_num_cust_in_store'] / floorarea
        results['max_num_cust_in_store_per_sqm'] = results['max_num_cust_in_store'] / floorarea
    results['logs'] = store.logs
    results['runtime'] = timeit.default_timer() - start
    return results


def simulate_several_days(config: dict,
                          G: nx.Graph,
                          # extra_outputs,
                          path_generator_function,
                          path_generator_args: list,
                          num_iterations: int = 1000,
                          use_parallel: bool = False):
    """Run several simulations and return selected number of stats from these simulations"""

    # Run simulations

    # path_generation = config.get('path_generation', 'synthetic')
    # path_generator_function, path_generator_args = get_path_generator(G, path_generation, zone_paths=extra_outputs,
    #                                                                   synthetic_path_generator_args=extra_outputs)
    if use_parallel:
        args = [config, G, path_generator_function, path_generator_args]
        repeated_args = zip(*[repeat(item, num_iterations) for item in args])
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_cores) as p:
            with tqdm(total=num_iterations) as pbar:
                results = []
                for i, results_dict in enumerate(p.istarmap(simulate_one_day, repeated_args)):
                    results.append(results_dict)
                    pbar.update()
                    if config['day'] >= 6:
                        config['day'] = 0
                    else:
                        config['day'] += 1
    else:
        results = []
        for _ in tqdm(range(num_iterations)):
            results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
            results.append(results_dict)
            if config['day'] >= 6:
                config['day'] = 0
            else:
                config['day'] += 1

    # Initialize containers to save any scalar statistics
    df_num_encounters_per_node_list = []
    df_exposure_time_per_node_list = []
    stats_dict = {}
    cols_to_record = [key for key, val in results_dict.items()
                      if isinstance(val, (int, np.integer)) or isinstance(val, (float, np.float))]
    cols_to_record += ['exposure_times']
    cols_not_recording = [key for key in results_dict.keys() if key not in cols_to_record]
    logging.info(f'Recording the scalar stats for {cols_to_record}.')
    logging.info(f'We are not recording {cols_not_recording}.')
    for stat in cols_to_record:
        stats_dict[stat] = []

    # Record encounter stats as well
    logging.info('Recording stats for df_num_encounters_per_node, df_exposure_time_per_node')
    for i in range(num_iterations):
        results_dict = results[i]
        df_num_encounters_per_node_list.append(results_dict['df_num_encounters_per_node'])
        df_exposure_time_per_node_list.append(results_dict['df_exposure_time_per_node'])
        for col in cols_to_record:
            stats_dict[col].append(results_dict[col])
    df_stats = pd.DataFrame(stats_dict)
    df_num_encounter_per_node_stats = pd.concat(df_num_encounters_per_node_list).reset_index(drop=True)
    df_encounter_time_per_node_stats = pd.concat(df_exposure_time_per_node_list).reset_index(drop=True)
    return df_stats, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats

