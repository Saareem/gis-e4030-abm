import os
from pathlib import Path

from flask import Flask, render_template, request, redirect
from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.create_weights import create_weights
from covid19_supermarket_abm.utils.node_visibility import node_visibility
from covid19_supermarket_abm.utils.visualize import visualize_single_day, visualize_multiple_days, chart


app = Flask(__name__)
staff_start_node = 27  # Change the staff start node here or expose possibility for the user to change the node


# Default view. Based on index.html file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Get config parameters from html-form
        config = {'arrival_rate': float(request.form['param1']),
                  'traversal_time': float(request.form['param2']),
                  'infection_proportion': float(request.form['param3']),
                  'logging_enabled': False,
                  'duration_days': int(request.form['param4']),
                  'day': int(request.form['param5']),
                  'customers_together': float(request.form['param6']),
                  'random_weight_range': int(request.form['param7']),
                  'random_weight_seed': int(request.form['param8']),
                  'runtime': 'param9' in request.form,
                  'path_update_interval': int(request.form['param10']),
                  'avoidance_factor': float(request.form['param11']),
                  'avoidance_k': float(request.form['param12']),
                  'staff_start_nodes': tuple([staff_start_node for i in range(0, int(request.form['param13']))]),
                  'staff_traversal_time': float(request.form['param14'])}

        # Config2 for clean up parameters
        config2 = {'Rate at which customers arrive to the store (in customers per minute)': config['arrival_rate'],
                   'Mean wait time at each node (in minutes)': config['traversal_time'],
                   'Proportion of agents that are infected': config['infection_proportion'],
                   'Number of days simulated': config['duration_days'],
                   'Starting week day': config['day'],
                   'Proportion of customers shopping together': config['customers_together'],
                   'Random weights range': config['random_weight_range'],
                   'Random weights seed': config['random_weight_seed'],
                   'Runtime path generator activated': config['runtime'],
                   'Start nodes for the staff members': config['staff_start_nodes'],
                   'Mean wait time at each node for staff members (in minutes)': config['staff_traversal_time']}

        # Initialize model
        data_dir = os.path.join(Path(__file__).parent, f'covid19_supermarket_abm\kmarket_data')
        zone_paths = load_example_paths()
        G = load_kmarket_store_graph()
        shortest_path_dict = get_all_shortest_path_dicts(G)
        weights = create_weights(G=G, weight_range=config['random_weight_range'], seed=config['random_weight_seed'])
        item_nodes = [i for i in range(1, 106) if not i in [1, 2, 3, 23, 51, 52, 53, 54, 55]]
        path_gen_type = 'synthetic'
        path_generator_args = [1, 1, [1], [51, 52], [55], item_nodes, shortest_path_dict, weights]

        # Create paths based on user input
        if config['runtime']:
            path_gen_type = 'runtime'
            del path_generator_args[-2]
            config['shortest_path_dict'] = shortest_path_dict
            config['node_visibility'] = node_visibility(G, data_dir=data_dir)
        path_generator_function, path_generator_args = get_path_generator(path_generation=path_gen_type,
                                                                          zone_paths=zone_paths,
                                                                          G=G,
                                                                          synthetic_path_generator_args=path_generator_args,
                                                                          runtime_path_generator_args=path_generator_args)

        # Simulate one day or several days based on user input
        if config['duration_days'] == 1:
            init_results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
            result_images = visualize_single_day(G, init_results_dict)

            # Change dict keys and select only relevant variables
            results_dict = {
                'Total number of agents': init_results_dict['num_agents'],
                'Number of susceptible agents': init_results_dict['num_S'],
                'Number of infected agents': init_results_dict['num_I'],
                'Total exposure time (minutes)': round(init_results_dict['total_exposure_time'], 2),
                'Number of susceptible agents which have at least one contact with an infectious agent':
                    init_results_dict['num_agents_w_contact'],
                'Mean number of customers in the store during the simulation': round(
                    init_results_dict['mean_num_cust_in_store'], 2),
                'Maximum number of customers in the store during the simulation': init_results_dict[
                    'max_num_cust_in_store'],
                'Total number of contacts between infectious agents and susceptible agents': init_results_dict[
                    'num_contacts'],
                'Mean of the shopping times (minutes)': round(init_results_dict['mean_shopping_time'], 2),
                "Length of the store's opening hours (minutes)": init_results_dict['store_open_length'],
                'Running time per day (seconds)': round(init_results_dict['runtime'], 2)
            }

            # Display new page where you can see results
            return render_template('results.html', config2=config2, result_dict=results_dict,
                                   result_images=result_images)

        else:
            init_results_dict, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats = simulate_several_days(
                config, G, path_generator_function, path_generator_args, num_iterations=config['duration_days'],
                use_parallel=False)
            result_images = visualize_multiple_days(G, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats,
                            days=config['duration_days'])

            # Change dict keys and select only relevant variables, visualize
            results_dict = {
                'Total number of agents': chart(init_results_dict['num_agents'], config['duration_days']),
                'Number of susceptible agents': chart(init_results_dict['num_S'], config['duration_days']),
                'Number of infected agents': chart(init_results_dict['num_I'], config['duration_days']),
                'Total exposure time (minutes)': chart(round(init_results_dict['total_exposure_time'], 2), config['duration_days']),
                'Number of susceptible agents which have at least one contact with an infectious agent':
                    chart(init_results_dict['num_agents_w_contact'], config['duration_days']),
                'Mean number of customers in the store during the simulation': chart(round(
                    init_results_dict['mean_num_cust_in_store'], 2), config['duration_days']),
                'Maximum number of customers in the store during the simulation': chart(init_results_dict[
                    'max_num_cust_in_store'], config['duration_days']),
                'Total number of contacts between infectious agents and susceptible agents': chart(init_results_dict[
                    'num_contacts'], config['duration_days']),
                'Mean of the shopping times (minutes)': chart(round(init_results_dict['mean_shopping_time'], 2), config['duration_days']),
                "Length of the store's opening hours (minutes)": chart(init_results_dict['store_open_length'], config['duration_days']),
                'Running time per day (seconds)': chart(round(init_results_dict['runtime'], 2), config['duration_days'])
            }
            # Display new page where you can see results
            return render_template('results2.html', config2=config2, result_dict=results_dict,
                                   result_images=result_images)



    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
