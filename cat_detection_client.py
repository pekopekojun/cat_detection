# -*- coding: utf-8 -*-
import requests


def send(url, filename):
    files = {'image': ('000001.jpg', open(filename, 'rb'), 'image/jpeg')}
    r = requests.post(url, files=files)


if __name__ == "__main__":

#    send("http://localhost:8008/ssd", '000001.jpg')
    send("http://localhost:5000/ssd", '000001.jpg')
