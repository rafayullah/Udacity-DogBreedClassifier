import json
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify, url_for
import os
import re

from DogBreedClassifier import *

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    output_string = ''
    path = ''
    
    if request.method == 'POST':
            if 'file1' not in request.files:
                print( 'there is no file1 in form!')
            file1 = request.files['file1']
            if file1.filename == '':
                print('No selected file')
                return render_template('index.html',img_path=path, output_string=output_string)
            else:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
                file1.save(path)
                output_string = get_prediction(path)
    return render_template('index.html', img_path=path, output_string=output_string)


def main():
    app.run(host='0.0.0.0', port=3001, debug=False,threaded=False)

if __name__ == '__main__':
    main()
