#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2021-08-06 11:19

@author: CLB-Tianyi
@description: Server to get input and return prediction

"""
import io
import os
import sys
import json
import base64
from PIL import Image
from flask import Flask, request, jsonify

pwd = os.path.dirname(os.path.realpath(__file__))
proj_path = os.path.dirname(pwd)
core_path = os.path.join(proj_path, "core")
sys.path.append(core_path)
os.chdir(core_path)
import starchild

server = Flask(__name__)
app = starchild.StarChild(starchild.TEMPLATES)

def b64_to_file(base64_string, outfile):
    imgdata = base64.b64decode(base64_string)
    pillow = Image.open(io.BytesIO(imgdata))
    pillow.save(outfile)
    return outfile

@server.route("/index")
def index():
    return jsonify({
            "message":"StarChild shape recognizer is running",
            "interface":"/api/getPrediction",
            })

@server.route("/api/getPrediction", methods=["GET", "POST"])
def get_info():
    res = {
        "status": 0,
        "message": None
    }
    try:
        data = request.json
        # Parsing data
        gt = data['gt']
        base64_string = data['img_b64']
        # Saving image to temp location and get path
        img_path = b64_to_file(base64_string, "__temp.jpg")
        # Perform evaluation and save result
        res['message'] = app.evaluate_one_image(img_path, gt)
        return jsonify(res)
    except Exception as e:
        res['status'] = 2
        res['message'] = str(e)
        return jsonify(res)

if __name__ == '__main__':
    server.run(host="0.0.0.0", port=1337)