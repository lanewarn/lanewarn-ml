import random
import time

import torch
from matplotlib import pyplot as plt

from models import Darknet, load_darknet_weights
from utils.datasets import LoadWebcam
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
from utils.video import bgr_to_rgb

import socket

RISKY_CLASSES = [
    'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat',
    'cell phone', 'person'
]


def stream(cfg, classes_file, weights, socket_ip, socket_port, cam=0, img_size=128, conf_thres=0.6, nms_thres=0.5):
    model = Darknet(cfg, img_size)

    load_darknet_weights(model, weights)

    model.fuse()

    model.to('cpu').eval()

    cap = LoadWebcam(img_size=img_size, camera=cam)
    # Get classes and colors
    classes = load_classes(classes_file)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((socket_ip, socket_port))

    sock.send(b'CLR;')

    # (x1, y1, x2, y2, object_conf, class_conf, class)
    for i, (path, img, im0, vid_cap) in enumerate(cap):
        t = time.time()
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to('cpu')
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            detected_classes = []

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # for c in det[:, -1].unique():
            #    n = (det[:, -1] == c).sum()

            for *xyxy, conf, cls_conf, cls in det:
                print(classes[int(cls)])
                if classes[int(cls)] in RISKY_CLASSES:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                    detected_classes.append({classes[int(cls)]: {'x': (xyxy[0] + xyxy[2]) / 2, 'y': (xyxy[1] + xyxy[3]) / 2}})

            n = []
            for c in detected_classes:
                print(c)
                fov = 62
                width = im0.shape[1]
                x, y = c[list(c.keys())[0]].values()
                phi = (x / width * 2 - 1) * (fov / 2)
                n.append(f"{list(c.keys())[0]};{phi};-1|")
            sock.send(''.join(str(x) for x in n).encode('utf-8'))

        print('(%.3fs)' % (time.time() - t))
        plt.imshow(bgr_to_rgb(im0))
        plt.show(block=False)


if __name__ == '__main__':
    try:
        stream('./ml-data/yolov3.cfg',
               './ml-data/classes.txt',
               './ml-data/weights/yolov3.weights',
               '127.0.0.1',
               1234,
               cam=0,
               img_size=128)
    except KeyboardInterrupt:
        sock.close()
