#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2021-08-06 11:53

@author: CLB-Tianyi
@description: Test server by sending a random image for testing

"""
import os
import io
import cv2
import base64
import random
import argparse
import requests
from glob import glob
from PIL import Image

def b64_to_file(base64_string, outfile):
    imgdata = base64.b64decode(base64_string)
    pillow = Image.open(io.BytesIO(imgdata))
    pillow.save(outfile)
    return outfile

def test_server(addr, port, payload):
    res = requests.post("http://{}:{}/api/getPrediction".format(addr, port), json=payload)
    response = res.json()
    if response['data']:
        b64_to_file(response['data'], "test_response.jpg")
    return response['message']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Folder to input data")
    parser.add_argument("--CALL",
                        type = str,
                        choices = ["LOCAL", "CHA_DEV"],
                        required = True,
                        help = "test server from 'LOCAL', 'CHA_DEV', etc"
                        )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        exit("Error: Input path {} doesn't seem to be a folder!".format(args.input))

    image_paths = glob("{}/**/*.*".format(args.input), recursive=True)

    test_image_path = random.choice(image_paths)
    test_shape = random.choice(["r", "s", "t"])

    img = cv2.imread(test_image_path)
    _, arr = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(arr.tobytes()).decode()

    payload = {
        'gt': test_shape,
        'img_b64': img_b64,
    }

    print("Test shape: {}, test image: {}".format(test_shape, test_image_path))

    if args.CALL == "LOCAL":
        res = test_server("0.0.0.0", "1337", payload)

    print(res)