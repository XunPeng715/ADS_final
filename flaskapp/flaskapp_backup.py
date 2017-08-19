import sqlite3
import os.path
import ast
import GeospatialSearch3

from flask import Flask, Response, request, g

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello from Flask!'

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)

@app.route('/buildYear', methods=['GET'])
def metrics1():  # pragma: no cover
    content = get_file('buildYearDensity.html')
    return Response(content, mimetype="text/html")

@app.route('/location', methods=['GET'])
def metrics():  # pragma: no cover
    print('Form location url')
    content = get_file('location.html')
    return Response(content, mimetype="text/html")

@app.route('/returnlocations', methods=['POST'])
def returnlocations():  # pragma: no cover
    longitide = ast.literal_eval(request.form.get("longitude"))
    latitude = ast.literal_eval(request.form.get("latitude"))
    number = int(ast.literal_eval(request.form.get("number")))
    location = [longitide, latitude]
    GeospatialSearch3.getLocations(location, number, [])
    content = get_file("locations.html")
    return Response(content, mimetype="text/html")

if __name__ == '__main__':
  app.run()
