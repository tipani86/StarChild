"""
Created on 2021-07-31 12:51

@author: CLB-Tianyi
@description: Shape recognizer plaything for autistic children

"""
import os
import sys
import cv2
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
# import paddlehub as hub
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim

TEMPLATES = {
    'r': 'round.png',
    's': 'square.png',
    't': 'triangle.png',
}

class StarChild:

    def __init__(self, templates):
        self.one_side = 160
        self.images = []
        self.processed_images = []
        self.preview = None
        self.method = "shape"
        # Load templates
        self.templates = {}
        for key in templates:
            img = self.load(templates[key])
            self.templates[key] = self.preprocess(img)
            # self.templates[key] = img

    def load_and_preprocess(self, image_path):
        img = self.load(image_path)
        # return (img, img)
        return (img, self.preprocess(img))

    def load(self, image_path):
        # Load, crop and resize image
        img = cv2.imread(image_path, 0)
        img = self._crop_square(img)
        img = cv2.resize(img, (self.one_side, self.one_side), interpolation=cv2.INTER_AREA)
        return img

    def preprocess(self, img):
        return self._preprocess_one_image(img)

    def _get_contours(self, img):
        kernel = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ], np.uint8)
        gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 1)
        wide = cv2.Canny(gaussian_blurred, 10, 200)
        mid = cv2.Canny(gaussian_blurred, 30, 150)
        tight = cv2.Canny(gaussian_blurred, 240, 250)
        edges = wide

        edges = cv2.dilate(edges, kernel, iterations=4)

        canvas = np.zeros(img.shape, np.uint8)

        cnts, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        new_cnts = []
        for cnt in cnts:
            # Skip too small contours
            if cv2.contourArea(cnt) < 15:
                continue

            # Find length of contours
            param = cv2.arcLength(cnt, True)

            # Approximate what type of shape this is
            approx = cv2.approxPolyDP(cnt, 0.01 * param, True)
            # print(len(approx))
            new_cnts.append(approx)

        # Draw contours
        # -1 signifies drawing all contours
        cv2.drawContours(canvas, new_cnts, 0, 255, -1)
        if len(new_cnts) > 1:
            cv2.drawContours(canvas, new_cnts, 1, 255, -1)
        canvas = cv2.erode(canvas, kernel, iterations=4)
        return canvas

    def _preprocess_one_image(self, img):
        rows = img.shape[0]

        # Extract first contours
        canvas = self._get_contours(img)

        # Find circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 2, rows / 3, param1=100, param2=0.8, minRadius=int(0.2 * self.one_side), maxRadius=int(0.9 * self.one_side))

        qualified_circles = []
        if circles is not None:
            # canvas = np.zeros(img.shape, np.uint8)
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                draw = True
                center = (i[0], i[1])
                radius = i[2]
                for dim in center:
                    if int(dim) - radius < 0:
                        draw = False
                        break
                    if int(dim) + radius > rows:
                        draw = False
                        break
                # circle outline
                if draw:
                    qualified_circles.append((center, radius))
        if len(qualified_circles) > 0:
            canvas = np.zeros(img.shape, np.uint8)
            for center, radius in qualified_circles:
                cv2.circle(canvas, center, radius, (255, 0, 255), -1)

        return canvas

    def _crop_square(self, img):
        # Crop to 1:1 aspect ratio
        h, w = img.shape
        min_side = min(h, w)
        if h > min_side:
            diff = h - min_side
            top = int(diff / 2)
            return img[top:top+min_side, :]
        else:
            diff = w - min_side
            left = int(diff / 2)
            return img[:, left:left+min_side]

    def histogram(self, key, img):
        template_histogram = cv2.calcHist([self.templates[key]], [0], None, [256], [0, 256])
        image_histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
        return round(cv2.compareHist(template_histogram, image_histogram, cv2.HISTCMP_CHISQR), 3)

    def SSIM_all_angles(self, key, img, method="max"):
        height, width = img.shape
        center = (width/2, height/2)
        rot_angle = 30
        template = self.templates[key].copy()
        similarities = []
        for i in range(int(360/rot_angle)):
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=i * rot_angle, scale=1)
            rotated_template = cv2.warpAffine(src=template, M=rotate_matrix, dsize=(width, height))
            similarities.append(ssim(rotated_template, img))
        if method == "max":
            return round(max(similarities), 3)
        if method == "mean":
            return round(np.mean(similarities), 3)

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

            if self.method == "histogram":
                # Histogram method
                res[key] = self.histogram(key, img)

            if self.method == "SSIM_max":
                # SSIM method (max of all angles)
                res[key] = self.SSIM_all_angles(key, img, "max")

            if self.method == "SSIM_mean":
                # SSIM method (mean of all angles)
                res[key] = self.SSIM_all_angles(key, img, "mean")

            if self.method == "SSIM":
                # SSIM method (direct)
                res[key] = round(ssim(self.templates[key], img), 3)

            if self.method == "ORB":
                # ORB matching method
                res[key] = self.ORB(key, img)

            if self.method == "shape":
                # Shape matching algorithm
                diff = sys.float_info.epsilon + cv2.matchShapes(self.templates[key], img, 1, 0.0)
                res[key] = round(1/diff, 3)

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

    def run(self, image_paths, debug=False):
        # template_collage = self.generate_collage(list(self.templates.values()))
        # cv2.imshow('image', template_collage)
        # cv2.waitKey(0)

        # Load up bunch of input images and preprocess them
        with Pool() as p:
            res = list(tqdm(p.imap(self.load_and_preprocess, image_paths), total=len(image_paths)))
        p.join()

        for item in res:
            self.images.append(item[0])
            self.processed_images.append(item[1])

        # Calculate similarity metric for each individual image against all templates
        with Pool() as p:
            similarities = list(tqdm(p.imap(self.calculate_similarities, self.processed_images), total=len(self.processed_images)))
        p.join()

        # Generate the preview collage and fill it with images and the calculation results
        self.preview = self.generate_collage(self.images, similarities, debug)
        processed_collage = self.generate_collage(self.processed_images)

        concat = np.concatenate([self.preview, processed_collage], axis=1)

        cv2.imshow('image', concat)
        cv2.waitKey(0)

    def evaluate_one_image(self, image_path, gt=None):
        img, processed_img = self.load_and_preprocess(image_path)
        similarities = self.calculate_similarities(processed_img)
        pred = max(similarities, key=similarities.get)
        if gt:
            return pred == gt
        else:
            return None

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

    # model = hub.Module(name="resnet50_vd_imagenet_ssld")