from flask import Flask, render_template, request, redirect

app = Flask(__name__)


# Default view. Based on index.html file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Set parameters
        config = {'arrival_rate': request.form['param1'],
                  'traversal_time': request.form['param2'],
                  'infection_proportion': request.form['param3'],
                  "logging_enabled": request.form['param4'],
                  'day': request.form['param5'],
                  'customers_together': request.form['param6']}
        print(config)

        parameters_ok = True
        if parameters_ok == True:
            # Run simulation ans show results
            return render_template('results.html')
        else:
            # Re-input parameters
            return redirect('/')

    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)