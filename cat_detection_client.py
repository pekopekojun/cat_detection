# -*- coding: utf-8 -*-
import requests
import cv2
from requests.exceptions import Timeout

def send(url, filename):
    files = {'image': ('000001.jpg', open(filename, 'rb'), 'image/jpeg')}
    r = requests.post(url, files=files)

def send_win(url, buf):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    result, encimg = cv2.imencode('.jpg', buf, encode_param)
    files = {'image': ('000001.jpg', encimg, 'image/jpeg')}
    try:
        r = requests.post(url, files=files, timeout = 1)
    except:
        pass
    #except Timeout:
    #    print("timeout!")
    #    pass

if __name__ == "__main__":

    # 動画の読み込み
    cap = cv2.VideoCapture(0)
    # 動画終了まで繰り返し
    while(cap.isOpened()):
        # フレームを取得
        ret, frame = cap.read()
        # フレームを表示
        cv2.imshow("Flame", frame)  # 無くてもいい
        send_win("http://localhost:5000/ssd", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

#    send("http://localhost:8008/ssd", '000001.jpg')
    send("http://localhost:5000/ssd", '000001.jpg')
