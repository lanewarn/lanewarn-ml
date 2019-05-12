import os
import random
import subprocess
import freenect
import time
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt

from models import Darknet, load_darknet_weights
from utils.datasets import LoadWebcam, LoadKinect
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
from utils.video import bgr_to_rgb

import socket

from settings import RISKY_CLASSES, CAMERA_FOV
import numpy as np


def get_depth():
    raw = np.array(freenect.sync_get_depth()[0])
    np.clip(raw, 0, 2 ** 10 - 1, raw)
    raw >>= 2
    return raw.astype(np.uint8)


def stream(cfg, classes_file, weights, socket_ip, socket_port, image_size=128, confidence_threshold=0.6, nms_thres=0.5):
    print('+ Initializing model')
    model = Darknet(cfg, image_size)
    print('+ Loading model')
    load_darknet_weights(model, weights)
    print('+ Fusing model')
    model.fuse()
    print('+ Loading model to CPU')
    model.to('cpu').eval()
    print('+ Loading webcam')
    cap = LoadKinect(img_size=image_size)
    print('+ Loading classes')
    classes = load_classes(classes_file)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    print('+ Connecting to remote socket')
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((socket_ip, socket_port))
    print('+ Enumerating cam')
    for counter, (path, img, im0, vid_cap) in enumerate(cap):
        t = time.time()

        print('+ Loading image to CPU')
        img = torch.from_numpy(img).unsqueeze(0).to('cpu')
        pred, _ = model(img)
        print('+ Detecting objects')
        det = non_max_suppression(pred, confidence_threshold, nms_thres)[0]

        if det is not None and len(det) > 0:
            detected_classes = []
            print('+ Rescaling model')
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            print('+ Reading depth')

            depth = get_depth()
            depth_swap = np.swapaxes(depth, 0, 1)

            depth_strip1d = np.array([np.sort(stripe)[100] for stripe in depth_swap]).astype(np.uint8)
            depth_strip2d_swap = np.array([np.ones(depth_swap.shape[1]) * depth for depth in depth_strip1d]).astype(np.uint8)
            depth_strip2d = np.swapaxes(depth_strip2d_swap, 0, 1)

            depth_edge1d = np.zeros(depth_strip1d.shape)

            state = False
            for counter, _ in np.ndenumerate(depth_edge1d[:-1]):
                state = True if not state and depth_strip1d[counter] < 230 else False
                depth_edge1d[counter[0]] = not state

            state = False
            state_cnt = 0
            for counter, _ in np.ndenumerate(depth_edge1d[:-1]):
                counter = counter[0]
                if depth_edge1d[counter] == state:
                    state_cnt += 1
                else:
                    if state_cnt < 10:
                        for r in range(max(0, counter - 10), counter):
                            depth_edge1d[counter] = state
                    state_cnt = 0
                    state = depth_edge1d[counter]

            depth_edge1d = depth_edge1d * 255

            depth_edge2d_swap = np.array([np.ones(100) * awddawd for awddawd in depth_edge1d]).astype(np.uint8)
            depth_edge2d = np.swapaxes(depth_edge2d_swap, 0, 1)

            for *coordinates, conf, cls_conf, cls in det:
                if classes[int(cls)] in RISKY_CLASSES:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(coordinates, im0, label=label, color=colors[int(cls)])
                    print(f"+ Detected {classes[int(cls)]}")
                    x_avg_depth = np.mean(depth[coordinates[0] - 5:coordinates[0] + 5])
                    y_avg_depth = np.mean(depth[coordinates[1] - 5:coordinates[1] + 5])
                    detected_classes.append({classes[int(cls)]: {'x': coordinates[0], 'y': coordinates[1], 'z': np.average(np.array([x_avg_depth, y_avg_depth]))}})

            n = []
            for counter in detected_classes:
                width = im0.shape[1]
                x, y, z = counter[list(counter.keys())[0]].values()
                phi = (x / width * 2 - 1) * (CAMERA_FOV / 2)
                n.append(f"{list(counter.keys())[0]};{phi};{z}|")
            sock.send(''.join(str(x) for x in n)[:-1].encode('utf-8'))
        print('+ Cycle took %.3fs' % (time.time() - t))
        plt.imshow(bgr_to_rgb(im0))
        plt.show(block=False)
        plt.pause(.001)


if __name__ == '__main__':
    parser = ArgumentParser(description='LaneWarn machine learning executable')
    parser.add_argument('-s', '--host', default='127.0.0.1', help='Sound module host')
    parser.add_argument('-q', '--quality', default=128, type=int, help='NN image size')
    parser.add_argument('-p', '--port', default=1337, type=int, help='Sound module port')

    args = parser.parse_args()

    try:
        print('+ Attempting to flash firmware')
        subprocess.run('freenect-flashfirmware')

        stream('./ml-data/yolov3.cfg',
               './ml-data/classes.txt',
               './ml-data/weights/yolov3.weights',
               args.host,
               args.port,
               image_size=args.quality, )
    except KeyboardInterrupt:
        if sock: sock.close()
