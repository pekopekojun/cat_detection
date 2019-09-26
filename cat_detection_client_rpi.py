# -*- coding: utf-8 -*-
import requests
import cv2
import picamera
import io
import time
import numpy as np
from PIL import Image

if __name__ == "__main__":
    URL = "http://192.168.10.204:5000/ssd"
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.resolution = (1920, 1080)
        camera.framerate = 15
        stream = io.BytesIO()

        while True:
            camera.capture(stream, format="jpeg", use_video_port=True)
            stream.seek(0)
            #frame = np.fromstring(stream.getvalue(), dtype=np.uint8)
            #frame = cv2.imdecode(frame, 1)
            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            #result, jpg_image = cv2.imencode('.jpg', frame, encode_param)

            files = {'image': ('000001.jpg', stream.getvalue(), 'image/jpeg')}
            try:
                r = requests.post(URL, files=files, timeout=10.0)
                print(json.loads(r.text))
                #time.sleep(2)
            except:
                import traceback
                traceback.print_exc()
                pass
