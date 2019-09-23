#!/usr/bin/env python3
# coding: utf-8

import os
import io
import sys
from flask import Flask, request, make_response, jsonify, Response
import werkzeug

# flask
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
UPLOAD_DIR = ".\\"
counter = 0

@app.route('/det/<name>')
def api_image(name):
    f = open(name, 'rb')
    image = f.read()
    return Response(response=image, content_type='image/jpeg')

@app.route('/now.jpg')
def api_now():
    f = open("ramdisk/latest.jpg", 'rb')
    image = f.read()
    return Response(response=image, content_type='image/jpeg')

@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'

if __name__ == "__main__":
    print("=================== ENV Start")
    print(os.environ.get('HTTPS_CRT_FILE'))
    print(os.environ.get('HTTPS_KEY_FILE'))
    print("=================== ENV End")
    sys.stdout.flush()
    app.run(host='0.0.0.0', port=5001, ssl_context=(os.environ.get('HTTPS_CRT_FILE'), os.environ.get('HTTPS_KEY_FILE')), threaded=True, debug=False)
