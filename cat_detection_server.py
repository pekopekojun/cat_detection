#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from flask import Flask, request, make_response, jsonify
import werkzeug
import json
from datetime import datetime

LIB_DIR='SSD/'
sys.path.append(LIB_DIR)

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

slim = tf.contrib.slim

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(
        image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = LIB_DIR+'checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
tf_saver = tf.train.Saver()
tf_saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(
        rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(
        rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# flask
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
UPLOAD_DIR = ".\\"


@app.route('/ssd', methods=['POST'])
def upload_multipart():
    if 'image' not in request.files:
        make_response(jsonify({'result': 'No image file'}))
    file = request.files['image']
    print(file)
    fileName = file.filename
    if '' == fileName:
        make_response(jsonify({'result': 'filename must not empty.'}))

    saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_") \
        + werkzeug.utils.secure_filename(fileName)
    print(saveFileName)
    file.save(os.path.join(UPLOAD_DIR, saveFileName))

    img_pil = Image.open(file.stream)
    img_numpy = np.asarray(img_pil)
    frame = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)

    #cv2.imshow("Flame", frame) #無くてもいい
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB順に
    rclasses, rscores, rbboxes = process_image(img)
    for i in range(rclasses.shape[0]):
        cls_id = int(rclasses[i])
        if cls_id == 8:
            visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

    return make_response(jsonify({'result': 'upload OK.'}))


@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'

# main
if __name__ == "__main__":
    print(app.url_map)
    app.run(host='', port=5000)
