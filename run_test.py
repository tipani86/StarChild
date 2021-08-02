"""
Created on 2021-07-31 12:51

@author: CLB-Tianyi
@description: Shape recognizer plaything for autistic children

"""
import os
import cv2
import math
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import paddlehub as hub
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim

class StarChild:

    def __init__(self, templates):
        self.one_side = 160
        self.images = []
        self.processed_images = []
        self.preview = None
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

    def _preprocess_one_image(self, img):
        rows = img.shape[0]
        gaussian_blurred = cv2.GaussianBlur(img, (9, 9), 0)
        median_blurred = cv2.medianBlur(img, 5)
        wide = cv2.Canny(gaussian_blurred, 10, 200)
        mid = cv2.Canny(gaussian_blurred, 30, 150)
        tight = cv2.Canny(gaussian_blurred, 240, 250)
        edges = mid

        canvas = np.zeros(img.shape, np.uint8)

        cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        # Draw all contours
        # -1 signifies drawing all contours
        cv2.drawContours(canvas, cnts, -1, 255, 3)

        # Find circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 3,
                                  param1=150, param2=30,
                                  minRadius=int(0.3 * self.one_side), maxRadius=int(0.9 * self.one_side))

        qualified_circles = []
        if circles is not None:
            canvas = np.zeros(img.shape, np.uint8)
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
                cv2.circle(canvas, center, radius, (255, 0, 255), 3)

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

    def calculate_similarities(self, img):
        res = {}
        height, width = img.shape
        center = (width/2, height/2)
        rot_angle = 30
        for key in self.templates:
            # # Histogram method
            # template_histogram = cv2.calcHist([self.templates[key]], [0], None, [256], [0, 256])
            # image_histogram = cv2.calcHist([self.templates[key]], [0], None, [256], [0, 256])
            # res[key] = cv2.compareHist(template_histogram, image_histogram, cv2.HISTCMP_CHISQR)

            # SSIM method (max of all angles)
            # template = self.templates[key].copy()
            # max_similarity = 0
            # for i in range(int(360/rot_angle)):
            #     rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=i * rot_angle, scale=1)
            #     rotated_template = cv2.warpAffine(src=template, M=rotate_matrix, dsize=(width, height))
            #     ssim_score = round(ssim(rotated_template, img), 3)
            #     if ssim_score > max_similarity:
            #         max_similarity = ssim_score
            # res[key] = ssim_score

            # # SSIM method (direct)
            res[key] = round(ssim(self.templates[key], img), 3)
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test code")
    parser.add_argument("input", help="Path to folder where images exist")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        exit("Error: Input path {} doesn't seem to be a folder!".format(args.input))

    image_paths = glob("{}/**/*.*".format(args.input), recursive=True)

    templates = {
        'r': 'round.png',
        's': 'square.png',
        't': 'triangle.png',
    }

    # Load templates and images
    starchild = StarChild(templates)

    if args.debug:
        template_collage = starchild.generate_collage(list(starchild.templates.values()))
        cv2.imshow('image', template_collage)
        cv2.waitKey(0)

    starchild.run(image_paths, args.debug)

    model = hub.Module(name="resnet50_vd_imagenet_ssld")