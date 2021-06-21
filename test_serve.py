from flask import Flask
from flask import request
import pandas as pd

app = Flask(__name__)
@app.route('/', methods=['GET'])
def ping():
    return "hello"


@app.route('/home/<string:name>')
def hello(name):
    return "hello" + name


@app.route('/predict', methods=['POST'])
def kethuc():
    return "oki"




if __name__ == "__main__":
    app.run(host="0.0.0.0")
