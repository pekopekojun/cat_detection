#!/usr/bin/env python3
# coding: utf-8

import os
import io
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

slim = tf.contrib.slim

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.compat.v1.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.compat.v1.placeholder(tf.uint8, shape=(None, None, 3))
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
isess.run(tf.compat.v1.global_variables_initializer())
tf_saver = tf.compat.v1.train.Saver()
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

def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    # height, width = img.shape[:2]

    #img = cv2.rectangle(img, center, radius, color[, thickness[, lineType[, shift]]])
    # 引数
    #   img: 描画対象の画像 (inplace で描画される)
    #   pt1: 長方形の左上の点(x,y)
    #   pt2: 長方形の右下の点(x,y)
    #   color: 色
    #   thickness: 太さ。負の値で塗りつぶし。
    #   line_type: 線の種類
    #   shift: オフセット
    # 返り値
    #   img: 描画結果
    
    shape = img.shape
    colors = dict()
    info = []
    for i in range(bboxes.shape[0]):
        cls_id = int(classes[i])
        if cls_id not in colors:
            colors[cls_id] = (int(255*random.random()), int(255*random.random()), int(255*random.random()))
        color = colors[cls_id]
        bbox = bboxes[i]
        # bbox[0] : y, bbox[1] : x
        # Draw bounding box...
        pic_y, pic_x = img.shape[:2]
        
        p1 = (int(bbox[1] * pic_x), int(bbox[0] * pic_y))
        p2 = (int(bbox[3] * pic_x), int(bbox[2] * pic_y))
        cv2.rectangle(img, p1, p2, color, thickness)

        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        cv2.putText(img, s, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
        center = (
            int(p1[0] + ((p2[0] - p1[0]) / 2)), 
            int(p1[1] + ((p2[1] - p1[1]) / 2))
            )

        info.append({
            "class":int(classes[i]), 
            "score":int(scores[i]*100), 
            "size":(pic_x, pic_y),
            "bbox":(p1,p2), 
            "center":center
            })
        #print(center)
        #cv2.circle(img,center, 5, (0,0,255), -1)
    return info

# flask
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_DIR = ".\\"
counter = 0
@app.route('/ssd', methods=['POST'])
def upload():
    cat_info = []
    try:
        if 'image' not in request.files:
            return make_response(jsonify({'result': 'No image file'}))

        file = request.files['image']

        img_pil = Image.open(file.stream)
        img_numpy = np.asarray(img_pil)
        img = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)

        rclasses, rscores, rbboxes = process_image(img)
        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id == 8:
                cat_info = bboxes_draw_on_img(img, rclasses, rscores, rbboxes)
                saveFileName = datetime.now().strftime("%Y%m%d_%H%M%S_")
                cv2.imwrite(saveFileName + 'nora_cat.jpg', img)

        global counter
        txt = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        cv2.putText(img, txt, (0, 28), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        cv2.imwrite('ramdisk/image_' + str(counter) +'.jpg', img)
        counter += 1
        if(counter >= 16):
            counter = 0

        if(len(cat_info) != 0):
            return make_response(jsonify({'result': 'nyan!', 'info': cat_info}))
    except:
        import traceback
        traceback.print_exc()
        pass
    
    return make_response(jsonify({'result': 'no cat'}))

@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'

# main
if __name__ == "__main__":
    print(app.url_map)
    #app.run(host='192.168.10.120', port=5000, debug=False)
    #app.run(host='192.168.10.204', port=5000, debug=False)
    app.run(host='', port=5000)
