import random
import time
from os import wait
from pathlib import Path
from time import sleep

import cv2
import torch
from matplotlib import pyplot as plt

from utils import torch_utils
import os

from utils.datasets import LoadWebcam, LoadImages
from utils.parse_config import parse_data_cfg
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
from utils.video import bgr_to_rgb
from models import ONNX_EXPORT, Darknet, load_darknet_weights
import shutil


def show(frame):
    plt.imshow(frame)
    plt.show()


def capture(cap):
    if not cap.isOpened(): raise Exception("Failed to open CV2")
    _, n = cap.read()
    return bgr_to_rgb(n)


def detect(
        cfg,
        classes_file,
        weights,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5
):
    model = Darknet(cfg, img_size)

    # Load weights
    _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to('cpu').eval()

    cap = LoadWebcam(img_size=img_size)
    # Get classes and colors
    classes = load_classes(classes_file)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in enumerate(cap):
        t = time.time()
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to('cpu')
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            detected_classes = []
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                detected_classes.append({classes[int(c)]: int(n)})

            for *xyxy, conf, cls_conf, cls in det:
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # TODO: Do something with the detected classes
            print(detected_classes)

        print('(%.3fs)' % (time.time() - t))
        plt.imshow(bgr_to_rgb(im0))
        plt.show()


if __name__ == '__main__':
    detect('./ml-data/yolov3.cfg', './ml-data/classes.txt', './ml-data/weights/yolov3.weights', img_size=512)
