#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 2021-08-11 22:45

@author: CLB-Tianyi
@description: Shape recognizer plaything for autistic children using deep learning

"""
import os
import sys
import cv2
import math
import torch
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch.nn.functional as F
from multiprocessing import Pool
from torch.autograd import Variable
from model.ResNet_models import SCRN
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = SCRN()
model.load_state_dict(torch.load(os.path.join('model', 'model.pth'), map_location=torch.device('cpu')))
model.eval()

TEMPLATES = {
    'r': 'round.png',
    's': 'square.png',
    't': 'triangle.png',
}

class StarChild:

    def __init__(self, templates):
        self.one_side = 128
        self.testsize = 352
        self.images = []
        self.processed_images = []
        self.preview = None
        self.method = "shape"
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Load templates
        self.templates = {}
        for key in templates:
            _, self.templates[key] = self.load_and_preprocess_nn(templates[key])

    def load_and_preprocess_nn(self, image_path):
        # Load
        img = Image.open(image_path)
        img = img.convert('RGB')
        # Preprocess
        img = self._crop_square(img, PIL=True)
        img = img.resize((self.one_side, self.one_side))
        t_img = self.img_transform(img).unsqueeze(0)
        img = img.convert('L')
        # Infer salient mask
        with torch.no_grad():
            image = Variable(t_img).cpu()
            res, edge = model(image)
            res = F.interpolate(res, size=img.size, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu().numpy().squeeze()
        res *= 255
        res = np.round(res, 0).astype(np.uint8)
        # Threshold mask to make edges sharper
        _, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return (np.array(img), res)

    def _crop_square(self, img, PIL=False):
        # Crop to 1:1 aspect ratio
        if PIL:
            w, h = img.size
        else:
            h, w = img.shape
        min_side = min(h, w)
        if h > min_side:
            diff = h - min_side
            top = int(diff / 2)
            if PIL:
                return img.crop((0, top, w, top+min_side))
            else:
                return img[top:top+min_side, :]
        else:
            diff = w - min_side
            left = int(diff / 2)
            if PIL:
                return img.crop((left, 0, left+min_side, h))
            else:
                return img[:, left:left+min_side]

    def ORB(self, key, img):
        orb = cv2.ORB_create()

        # Get keypoints and their descriptors
        kp_templ, des_templ = orb.detectAndCompute(self.templates[key], None)
        kp_img, des_img =  orb.detectAndCompute(img, None)

        # Apply matching algorithm
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = matcher.match(des_templ, des_img)
        except:
            return 0

        return len(matches)

    def calculate_similarities(self, img):
        # Calculate the similarity between query and template images based on various methods and return scores
        res = {}
        for key in self.templates:

            if self.method == "ORB":
                # ORB matching method
                res[key] = self.ORB(key, img)

            if self.method == "shape":
                # Shape matching algorithm
                diff = sys.float_info.epsilon + cv2.matchShapes(self.templates[key], img, 1, 0.0)
                res[key] = round(1/diff, 3)
                res[key] = round(1 - cv2.matchShapes(self.templates[key], img, 2, 0.0), 3)

        return res

    def generate_collage(self, images, data=None, debug=False):
        # Get the number of tiles per side
        n_tiles_side = math.ceil(math.sqrt(len(images)))

        # Generate empty canvas
        collage = np.zeros(
            (n_tiles_side * self.one_side, n_tiles_side * self.one_side), np.uint8
        )
        # Fill the canvas with individual images
        cnt = 0
        for i in range(n_tiles_side):
            for j in range(n_tiles_side):
                top = i * self.one_side
                left = j * self.one_side
                bottom = top + self.one_side
                right = left + self.one_side
                # print("Filling pixels: ({}, {}) - ({}, {})...".format(top, left, bottom, right))
                collage[top:bottom, left:right] = images[cnt]
                if data:
                    cur_data = data[cnt]
                    x, y0 = left + 5, top + 15
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.4
                    color = (255, 255, 255)
                    if debug:
                        overlay = ""
                        for key in cur_data:
                            overlay += "{}: {}\n".format(key, cur_data[key])
                    else:
                        pred = max(cur_data, key=cur_data.get)
                        prob = cur_data[pred]
                        if prob == "nan":
                            p_text = "none"
                        elif pred == "r":
                            p_text = "round"
                        elif pred == "s":
                            p_text = "square"
                        elif pred == "t":
                            p_text = "triangle"
                        overlay = "Prediction:\n{} ({})".format(p_text, prob)
                    text_size, _ = cv2.getTextSize(overlay, font, scale, 1)
                    line_height = text_size[1] + 5
                    for k, line in enumerate(overlay.split("\n")):
                        y = y0 + k * line_height
                        cv2.putText(collage, line, (x, y), font, scale, (0, 0, 0), thickness=2)
                        cv2.putText(collage, line, (x, y), font, scale, color, thickness=1)
                cnt += 1
                if cnt >= len(images):
                    return collage
        return collage

    def create_blend(self, inp):
        image, processed_image, similarity = inp
        # Add some blur on the mask
        processed_image = cv2.GaussianBlur(processed_image, (7, 7), 3)
        # Blend image
        alpha = 0.5     # How much original image is retained (between 0...1)
        blended = cv2.addWeighted(image, alpha, processed_image, (1-alpha), 0.0)

        # Add prediction text
        x, y0 = 5, 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        color = (255, 255, 255)
        pred = max(similarity, key=similarity.get)
        prob = similarity[pred]
        if prob <= 0 or prob == "nan":
            p_text = "none"
        elif pred == "r":
            p_text = "round"
        elif pred == "s":
            p_text = "square"
        elif pred == "t":
            p_text = "triangle"
        if p_text == "none":
            overlay = "Prediction:\n{}".format(p_text)
        else:
            overlay = "Prediction:\n{} ({})".format(p_text, prob)
        text_size, _ = cv2.getTextSize(overlay, font, scale, 1)
        line_height = text_size[1] + 5
        for k, line in enumerate(overlay.split("\n")):
            y = y0 + k * line_height
            cv2.putText(blended, line, (x, y), font, scale, (0, 0, 0), thickness=2)
            cv2.putText(blended, line, (x, y), font, scale, color, thickness=1)
        return blended

    def run(self, image_paths, debug=False):
        # template_collage = self.generate_collage(list(self.templates.values()))
        # cv2.imshow('image', template_collage)
        # cv2.waitKey(0)

        # Load up bunch of input images and preprocess them
        with Pool() as p:
            res = list(tqdm(p.imap(self.load_and_preprocess_nn, image_paths), total=len(image_paths)))
        p.join()

        for item in res:
            self.images.append(item[0])
            self.processed_images.append(item[1])

        # Calculate similarity metric for each individual image against all templates
        with Pool() as p:
            similarities = list(tqdm(p.imap(self.calculate_similarities, self.processed_images), total=len(self.processed_images)))
        p.join()

        # Blend the processed result onto original image
        blend_inp = list(zip(self.images, self.processed_images, similarities))

        with Pool() as p:
            blended_images = list(tqdm(p.imap(self.create_blend, blend_inp), total=len(blend_inp)))
        p.join()

        # Generate the preview collage and fill it with images and the calculation results
        self.preview = self.generate_collage(self.images, similarities, debug)
        blended_collage = self.generate_collage(blended_images)

        concat = np.concatenate([self.preview, blended_collage], axis=1)

        cv2.imshow('image', concat)
        cv2.waitKey(0)

    def evaluate_one_image(self, image_path, gt=None):
        img, processed_img = self.load_and_preprocess_nn(image_path)
        similarities = self.calculate_similarities(processed_img)
        blended = self.create_blend((img, processed_img, similarities))
        pred = max(similarities, key=similarities.get)
        prob = similarities[pred]
        if prob <= 0:
            return False, blended
        elif gt:
            return pred == gt, blended
        else:
            return None, blended

def run_evaluation(gt, image_path):
    # gt should be one of: r, s, t (round, square, triangle)
    starchild = StarChild(TEMPLATES)
    return starchild.evaluate_one_image(image_path, gt)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test code")
    parser.add_argument("input", help="Path to folder where images exist")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_one", action="store_true", help="Test with one random shape and one random image")
    parser.add_argument("--method", default="shape", help="Matching method")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        exit("Error: Input path {} doesn't seem to be a folder!".format(args.input))

    image_paths = []
    accepted_exts = [".jpg", ".JPG", ".png", ".PNG", ".tiff", ".TIFF", ".bmp", ".BMP"]
    for ext in accepted_exts:
        matches = glob("{}/**/*{}".format(args.input, ext), recursive=True)
        for match in matches:
            if match not in image_paths:
                image_paths.append(match)

    if args.test_one:
        one_image_path = random.choice(image_paths)
        test_shape = random.choice(["r", "s", "t"])
        res = run_evaluation(test_shape, one_image_path)
        print("Random shape: {}, test image: {}, match: {}".format(test_shape, one_image_path, res))
        quit()

    # Load templates and images
    starchild = StarChild(TEMPLATES)

    # Change matching method (default = "SSIM")
    starchild.method = args.method

    # if args.debug:
    #     template_collage = starchild.generate_collage(list(starchild.templates.values()))
    #     cv2.imshow('image', template_collage)
    #     cv2.waitKey(0)

    starchild.run(image_paths, args.debug)