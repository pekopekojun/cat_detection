# -*- coding: utf-8 -*-
import requests
import cv2
import numpy as np
import json
from datetime import datetime

def send_win(url, buf):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    result, encimg = cv2.imencode('.jpg', buf, encode_param)
    files = {'image': ('000001.jpg', encimg, 'image/jpeg')}
    try:
        r = requests.post(url, files=files, timeout = 1)
        print(json.loads(r.text))
    except:
        pass

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()

        txt = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        cv2.putText(frame, txt, (0, 28), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        cv2.imshow("Flame", frame) 
        send_win("http://localhost:5000/ssd", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
