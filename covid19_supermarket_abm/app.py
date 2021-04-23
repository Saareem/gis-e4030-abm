from flask import Flask, render_template, request, redirect

app = Flask(__name__)


# Default view. Based on index.html file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        parameter1 = request.form['content']
        print(parameter1)
        return redirect('/')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)