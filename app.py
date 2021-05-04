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
        # Set parameters
        config = {'arrival_rate': float(request.form['param1']),
                  'traversal_time': float(request.form['param2']),
                  'infection_proportion': float(request.form['param3']),
                  "logging_enabled": 'param4' in request.form,
                  'day': int(request.form['param5']),
                  'customers_together': float(request.form['param6']),
                  'realtime': 'param7' in request.form}

        print(config)

        zone_paths = load_example_paths()
        G = load_kmarket_store_graph()
        shortest_path_dict = get_all_shortest_path_dicts(G)

        weights = create_weights(random_weights=True)
        item_nodes = [i for i in range(1, 106) if not i in [1, 2, 3, 23, 51, 52, 53, 54, 55]]

        path_gen_type = 'synthetic'
        path_generator_args = [1,
                               1,
                               [1],
                               [51, 52],
                               [55],
                               item_nodes,
                               shortest_path_dict,
                               weights]

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

        results_dict = simulate_one_day(config, G, path_generator_function, path_generator_args)
        del results_dict['df_num_encounters_per_node']
        del results_dict['df_exposure_time_per_node']
        del results_dict['shopping_times']
        del results_dict['logs']  # TODO: Show only selected variables instead of deleting

        result_images = ['static/images/cat-meme.jpg']  # TODO: Read all images from folder instead of just one

        # Display new page where you can see simulation results
        return render_template('results.html', result_dict=results_dict, result_images=result_images)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
