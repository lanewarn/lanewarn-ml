import random
import time
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt

from models import Darknet, load_darknet_weights
from utils.datasets import LoadWebcam
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
from utils.video import bgr_to_rgb

import socket

from settings import RISKY_CLASSES, CAMERA_FOV


def stream(cfg, classes_file, weights, socket_ip, socket_port, camera=0, image_size=128, confidence_threshold=0.6, nms_thres=0.5):
    print('+ Initializing model')
    model = Darknet(cfg, image_size)
    print('+ Loading model')
    load_darknet_weights(model, weights)
    print('+ Fusing model')
    model.fuse()
    print('+ Loading model to CPU')
    model.to('cpu').eval()
    print('+ Loading webcam')
    cap = LoadWebcam(img_size=image_size, camera=camera)
    print('+ Loading classes')
    classes = load_classes(classes_file)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    print('+ Connecting to remote socket')
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((socket_ip, socket_port))
    # (x1, y1, x2, y2, object_conf, class_conf, class)
    for i, (path, img, im0, vid_cap) in enumerate(cap):
        t = time.time()

        img = torch.from_numpy(img).unsqueeze(0).to('cpu')
        pred, _ = model(img)
        det = non_max_suppression(pred, confidence_threshold, nms_thres)[0]

        if det is not None and len(det) > 0:
            detected_classes = []

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *coordinates, conf, cls_conf, cls in det:
                if classes[int(cls)] in RISKY_CLASSES:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(coordinates, im0, label=label, color=colors[int(cls)])
                    print(f"+ Detected {classes[int(cls)]}")
                    detected_classes.append({classes[int(cls)]: {'x': (coordinates[0] + coordinates[2]) / 2, 'y': (coordinates[1] + coordinates[3]) / 2}})

            n = []
            for c in detected_classes:
                width = im0.shape[1]
                x, y = c[list(c.keys())[0]].values()
                phi = (x / width * 2 - 1) * (CAMERA_FOV / 2)
                n.append(f"{list(c.keys())[0]};{phi};-1|")
            sock.send(''.join(str(x) for x in n)[:-1].encode('utf-8'))
        print('+ Cycle took %.3fs' % (time.time() - t))
        plt.imshow(bgr_to_rgb(im0))
        plt.show(block=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='LaneWarn machine learning executable')
    parser.add_argument('-c', '--camera', default=0, type=int, help='Camera ID (/dev/video[n]), default 0')
    parser.add_argument('-s', '--host', default='127.0.0.1', help='Sound module host')
    parser.add_argument('-q', '--quality', default=128, type=int, help='NN image size')
    parser.add_argument('-p', '--port', default=1337, type=int, help='Sound module port')

    args = parser.parse_args()

    try:
        stream('./ml-data/yolov3.cfg',
               './ml-data/classes.txt',
               './ml-data/weights/yolov3.weights',
               args.host,
               args.port,
               camera=args.camera,
               image_size=args.quality)
    except KeyboardInterrupt:
        if sock: sock.close()
