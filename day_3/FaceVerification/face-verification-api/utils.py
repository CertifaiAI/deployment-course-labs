
import numpy as np
from math import ceil
from itertools import product as product
from PIL import Image
from io import BytesIO
import base64

def base64_to_image(image_string):
    bytesObj = BytesIO(base64.b64decode(image_string.encode('utf-8')))
    return Image.open(bytesObj)

def bytes_to_image(image_bytes):
    return Image.open(BytesIO(image_bytes))

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def prior_box(image_size, steps, clip, min_sizes):
    """
    Create prior anchor boxes
    """
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        min_s = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_s:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    output = np.array(anchors).reshape(-1, 4)
    if clip:
        output.clip(min=0, max=1)
    return output

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors,4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate(
        (
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
        ), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes