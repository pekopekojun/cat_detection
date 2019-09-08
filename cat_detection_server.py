# -*- coding: utf-8 -*-
from flask import Flask, request, make_response, jsonify
import os
import werkzeug
import json
from datetime import datetime

# flask
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
UPLOAD_DIR = ".\\"

@app.route('/ssd', methods=['POST'])
def upload_multipart():
    if 'image' not in request.files:
        make_response(jsonify({'result': 'No image file'}))

    file = request.files['image']
    fileName = file.filename
    if '' == fileName:
        make_response(jsonify({'result': 'filename must not empty.'}))

    saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
        + werkzeug.utils.secure_filename(fileName)
    print(saveFileName)
    file.save(os.path.join(UPLOAD_DIR, saveFileName))
    return make_response(jsonify({'result': 'upload OK.'}))

@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'


# main
if __name__ == "__main__":
    print(app.url_map)
    app.run(host='localhost', port=8008)
