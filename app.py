from flask import Flask, render_template, request, redirect
from covid19_supermarket_abm.utils.load_kmarket_data import load_kmarket_store_graph, load_example_paths
from covid19_supermarket_abm.path_generators import get_path_generator
from covid19_supermarket_abm.utils.create_synthetic_baskets import get_all_shortest_path_dicts
from covid19_supermarket_abm.simulator import simulate_one_day, simulate_several_days
from covid19_supermarket_abm.utils.create_weights import create_weights
from covid19_supermarket_abm.utils.node_visibility import node_visibility

app = Flask(__name__)


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
                  'realtime': 'param7' in request.form}
        config2 = config.copy()

        # Initialize model
        zone_paths = load_example_paths()
        G = load_kmarket_store_graph()
        shortest_path_dict = get_all_shortest_path_dicts(G)
        weights = create_weights(random_weights=True)
        item_nodes = [i for i in range(1, 106) if not i in [1, 2, 3, 23, 51, 52, 53, 54, 55]]
        path_gen_type = 'synthetic'
        path_generator_args = [1, 1, [1], [51, 52], [55], item_nodes, shortest_path_dict, weights]

        # Create paths based on user input
        if config['realtime']:
            path_gen_type = 'realtime'
            del path_generator_args[-2]
            config['shortest_path_dict'] = shortest_path_dict
            config['node_visibility'] = node_visibility(G)
        path_generator_function, path_generator_args = get_path_generator(path_generation=path_gen_type,
                                                                          zone_paths=zone_paths,
                                                                          G=G,
                                                                          synthetic_path_generator_args=path_generator_args,
                                                                          realtime_path_generator_args=path_generator_args)

        # Simulate one day or several days based on user input
        if config['duration_days'] == 1:
            init_results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
        else:
            init_results_dict, df_num_encounter_per_node_stats, df_encounter_time_per_node_stats = simulate_several_days(
                config, G, path_generator_function, path_generator_args, num_iterations=config['duration_days'],
                use_parallel=False)

        # Change dict keys and select only relevant variables
        results_dict = {
            'Total number of customers': init_results_dict['num_cust'],
            'Number of susceptible customers': init_results_dict['num_S'],
            'Number of infected customers': init_results_dict['num_I'],
            'Total exposure time': round(init_results_dict['total_exposure_time'],2),
            'Number of susceptible customers which have at least one contact with an infectious customer': init_results_dict['num_cust_w_contact'],
            'Mean number of customers in the store during the simulation': round(init_results_dict['mean_num_cust_in_store'],2),
            'Maximum number of customers in the store during the simulation': init_results_dict['max_num_cust_in_store'],
            'Total number of contacts between infectious customers and susceptible customers': init_results_dict['num_contacts'],
            'Mean of the shopping times': round(init_results_dict['mean_shopping_time'],2),
            "Length of the store's opening hours (in minutes)": init_results_dict['store_open_length'],
        }


        # Get visualizations
        result_images = ['static/images/cat-meme.jpg']  # TODO: Hermanni: Add visualizations

        # Display new page where you can see results
        return render_template('results.html', config2=config2, result_dict=results_dict, result_images=result_images)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
