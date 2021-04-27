from flask import Flask, render_template, request, redirect
import os

app = Flask(__name__)


# Default view. Based on index.html file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Set parameters
        config = {'arrival_rate': request.form['param1'],
                  'traversal_time': request.form['param2'],
                  'infection_proportion': request.form['param3'],
                  "logging_enabled": 'param4' in request.form,
                  'day': request.form['param5'],
                  'customers_together': request.form['param6']}
        print(config)


        parameters_ok = True # TODO: Create function that check if parameters are ok. Return true/false.


        if parameters_ok == True:
            # Run simulation and get results
            result_values = {'key1': 'result1',
                             'key2': 'result2',
                             'key3': 'result3'}
            result_images = ['covid19_supermarket_abm/static/images/cat-meme.jpg']  #TODO: Fix the path

            # Display new page where you can see simulation results
            return render_template('results.html', result_values=result_values, result_images=result_images)
        else:
            # If parameters are erroneous, re-input parameters (This is probably not the most convenient way...)
            return redirect('/')

    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)