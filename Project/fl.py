from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input1 = request.form['input1']
        input2 = request.form['input2']
        input3 = request.form['input3']
        input4 = request.form['input4']

        # Replace with your script logic
        result = subprocess.run(
            ['python', 'o.py', input1, input2, input3, input4],
            capture_output=True,
            text=True
        )
        output = result.stdout
        return render_template('index', output=output)
    else:
        return render_template('index')

if __name__ == '__main__':
    app.run(debug=True)
